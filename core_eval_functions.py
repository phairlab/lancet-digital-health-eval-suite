import warnings
from typing import Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

# Suppress sklearn warnings about penalty/C parameters
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')

# ============================================================================
# STATISTICAL COMPARISON (Nadeau & Bengio, 1999)
# ============================================================================

def bengio_nadeau_test(diffs: List[float], n_test: float, n_train: float,
                       ci_level: float = 0.95) -> Tuple[float, float, float, float]:
    """
    Corrected paired t-test for comparing cross-validated models (Nadeau & Bengio, 1999).

    A naive paired t-test on fold-level differences underestimates variance because
    folds share training data. This correction multiplies the sample variance by
    (1/k + n_test/n_train) before computing the t-statistic.

    The same corrected standard error yields a confidence interval on the mean
    difference. It is built from the identical variance estimate as the p-value,
    so the interval excludes zero exactly when p < (1 - ci_level).

    Args:
        diffs:    Per-fold metric differences (model_A - model_B), length k.
        n_test:   Average number of test samples per fold.
        n_train:  Average number of training samples per fold.
        ci_level: Confidence level for the interval (default 0.95).

    Returns:
        (t_stat, p_value, ci_lo, ci_hi), two-tailed. Returns all-nan if undetermined.
    """
    k = len(diffs)
    if k < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean_d = float(np.mean(diffs))
    var_d = np.var(diffs, ddof=1)
    corrected_var = (1 / k + n_test / n_train) * var_d
    if corrected_var <= 0:
        # Every fold gave an identical difference: the t-statistic is undefined,
        # but the interval collapses to the point estimate.
        return np.nan, np.nan, mean_d, mean_d
    se = float(np.sqrt(corrected_var))
    t_stat = mean_d / se
    p_value = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=k - 1)
    return float(t_stat), float(p_value), mean_d - t_crit * se, mean_d + t_crit * se


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================

# Rates and scores, which keep their meaning under resampling. Raw counts
# (n, tp, tn, fp, fn) are deliberately excluded: a count computed on a resample
# of different size is not comparable to the count on the original data.
BOOTSTRAP_METRICS = (
    'prevalence_pct', 'auroc', 'calibration_slope', 'brier_score',
    'alert_rate', 'sensitivity', 'specificity', 'ppv', 'npv',
)


def _fast_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank statistic.

    Equivalent to sklearn's roc_auc_score (including tie handling, since
    scipy's rankdata assigns average ranks) but avoids rebuilding an ROC
    curve on every bootstrap resample.
    """
    n_pos = int(y_true.sum())
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = stats.rankdata(y_prob)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _fast_calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Cox calibration slope: unregularised logistic regression on the logits."""
    if np.unique(y_true).size < 2:
        return np.nan
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    # C=inf is the unregularised fit; spelled this way rather than penalty=None,
    # which sklearn deprecated in 1.8.
    lr = LogisticRegression(C=np.inf, solver='lbfgs', max_iter=1000)
    lr.fit(logit.reshape(-1, 1), y_true)
    return float(lr.coef_[0][0])


def resample_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                     threshold: Optional[float] = None) -> dict:
    """Compute the bootstrappable metrics for one (re)sample.

    Definitions mirror evaluate_model() in ldh_eval.py, except that degenerate
    denominators return nan rather than 0 so that such draws can be dropped
    from the percentile calculation instead of biasing it toward zero.
    """
    out = {
        'prevalence_pct': float(np.mean(y_true) * 100),
        'auroc': _fast_auroc(y_true, y_prob),
        'calibration_slope': _fast_calibration_slope(y_true, y_prob),
        'brier_score': float(np.mean((y_prob - y_true) ** 2)),
    }

    if threshold is not None:
        pred = y_prob >= threshold
        tp = int(np.sum(pred & (y_true == 1)))
        tn = int(np.sum(~pred & (y_true == 0)))
        fp = int(np.sum(pred & (y_true == 0)))
        fn = int(np.sum(~pred & (y_true == 1)))
        out.update({
            'alert_rate': float((tp + fp) / y_true.size),
            'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
            'ppv': float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
            'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else np.nan,
        })

    return out


def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray,
                 cluster_ids: Optional[np.ndarray] = None,
                 n_boot: int = 2000, threshold: Optional[float] = None,
                 ci_level: float = 0.95, seed: int = 0) -> Tuple[dict, dict]:
    """Percentile bootstrap confidence intervals for the evaluation metrics.

    If cluster_ids is given, resampling draws *clusters* with replacement and
    keeps every row belonging to a drawn cluster (the cluster, or block,
    bootstrap). This is required whenever one unit can contribute more than one
    row -- a patient with several encounters, an eye with several images, a site
    contributing many cases. Resampling rows independently in that situation
    treats correlated rows as independent and yields intervals that are too
    narrow.

    Args:
        y_true:      Binary outcomes.
        y_prob:      Predicted probabilities.
        cluster_ids: Per-row cluster label. None resamples rows independently.
        n_boot:      Number of bootstrap resamples.
        threshold:   If given, also bootstrap the threshold-dependent metrics.
        ci_level:    Confidence level (default 0.95).
        seed:        Seed for reproducible resampling.

    Returns:
        (results, meta) where results maps metric -> {point, ci_lo, ci_hi,
        boot_sd, n_valid_draws} and meta describes how the bootstrap was run.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = int(y_true.size)
    rng = np.random.default_rng(seed)

    groups = None
    if cluster_ids is not None:
        cluster_ids = np.asarray(cluster_ids)
        if cluster_ids.size != n:
            raise ValueError(
                f"cluster_ids has {cluster_ids.size} entries but there are {n} predictions"
            )
        # Row indices grouped by cluster, so a draw is a concatenation of blocks
        order = np.argsort(cluster_ids, kind='stable')
        _, starts = np.unique(cluster_ids[order], return_index=True)
        groups = np.split(order, starts[1:])

    draws: dict = {}
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, n, n)
        else:
            picked = rng.integers(0, len(groups), len(groups))
            idx = np.concatenate([groups[k] for k in picked])
        for key, val in resample_metrics(y_true[idx], y_prob[idx], threshold).items():
            draws.setdefault(key, []).append(val)

    lo_pct = (1 - ci_level) / 2 * 100
    hi_pct = 100 - lo_pct
    point = resample_metrics(y_true, y_prob, threshold)

    results = {}
    for key, vals in draws.items():
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        results[key] = {
            'point': point.get(key),
            'ci_lo': float(np.percentile(arr, lo_pct)),
            'ci_hi': float(np.percentile(arr, hi_pct)),
            'boot_sd': float(arr.std(ddof=1)) if arr.size > 1 else np.nan,
            'n_valid_draws': int(arr.size),
        }

    meta = {
        'n_boot': n_boot,
        'ci_level': ci_level,
        'resample_unit': 'cluster' if groups is not None else 'row',
        'n_rows': n,
        'n_clusters': int(len(groups)) if groups is not None else None,
    }
    if groups is not None:
        sizes = np.array([g.size for g in groups], dtype=float)
        meta['mean_cluster_size'] = float(sizes.mean())
        meta['max_cluster_size'] = int(sizes.max())
        # Variance inflation under clustering tracks this, not the mean:
        # a long tail of large clusters matters more than the average.
        meta['effective_cluster_size'] = float((sizes ** 2).sum() / sizes.sum())

    return results, meta


# ============================================================================
# CORE EVALUATION FUNCTIONS (for direct import)
# ============================================================================

def auroc(y_true: np.ndarray, y_prob: np.ndarray, 
          save_path: Optional[str] = None) -> float:
    """Calculate AUROC and optionally plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'AUROC = {auc:.3f}', linewidth=2, color='#1f77b4')
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return auc


def calibration(y_true: np.ndarray, y_prob: np.ndarray,
                n_bins: int = 10, save_path: Optional[str] = None, 
                method: str = 'loess') -> float:
    """Generate calibration plot with loess smoothing or binned calibration and return calibration slope
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for binned calibration (default: 10)
        save_path: Path to save figure (optional)
        method: 'loess' or 'binned' (default: 'loess')
    """
    
    # Calculate calibration slope using unregularized logistic regression on logits
    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
    logit_pred = np.log(y_prob_clipped / (1 - y_prob_clipped))
    
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(logit_pred.reshape(-1, 1), y_true)
    calibration_slope = lr.coef_[0][0]
    calibration_intercept = lr.intercept_[0]
    
    brier = brier_score_loss(y_true, y_prob)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if method == 'loess':
        # Sort data for smoothing
        sort_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sort_idx]
        y_true_sorted = y_true[sort_idx].astype(float)
        
        # Apply loess with more conservative settings
        try:
            smoothed = lowess(y_true_sorted, y_prob_sorted, 
                             frac=0.25, it=0, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2.5, 
                    label='Model (loess)', color='#1f77b4')
        except Exception as e:
            print(f"Warning: Loess smoothing failed ({e}), falling back to binned calibration")
            # Fallback to binned
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
            ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                    label='Model (binned)', color='#1f77b4')
    
    elif method == 'binned':
        # Use binned calibration
        from sklearn.calibration import calibration_curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8,
                label='Model (binned)', color='#1f77b4')
    
    else:
        raise ValueError(f"Invalid method '{method}'. Must be 'loess' or 'binned'.")
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')

    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Proportion', fontsize=12)
    ax.set_title(f'Calibration Plot (Slope = {calibration_slope:.3f}, Intercept = {calibration_intercept:.3f}, Brier = {brier:.3f})', 
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return calibration_slope


def decision_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold_range: Tuple[float, float] = (0.0, 0.5),
                   save_path: Optional[str] = None,
                   threshold: Optional[float] = None) -> None:
    """Generate decision curve analysis"""
    thresholds = np.linspace(threshold_range[0], threshold_range[1], 100)
    net_benefits = []
    prevalence = y_true.mean()
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
        net_benefits.append(net_benefit)
    
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw vertical line at threshold if provided
    if threshold is not None and threshold_range[0] <= threshold <= threshold_range[1]:
        ax.axvline(threshold, linestyle='-', color='red', linewidth=1.5, alpha=0.6, label=f'Threshold ({threshold:.2f})', zorder=4)
    
    ax.plot(thresholds, net_benefits, label='Model', linewidth=2.5, color='#1f77b4', zorder=3)
    ax.plot(thresholds, treat_all, '--', label='Treat All', linewidth=2, alpha=0.7, color='#ff7f0e', zorder=2)
    ax.axhline(0, linestyle='--', color='gray', label='Treat None', linewidth=2, alpha=0.7, zorder=2)
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('Decision Curve Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Focus on clinically relevant range
    max_nb = max(net_benefits)
    ax.set_ylim(-0.05, max_nb * 1.15)
    ax.set_xlim(threshold_range[0], threshold_range[1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def risk_distribution(y_true: np.ndarray, y_prob: np.ndarray,
                      save_path: Optional[str] = None) -> None:
    """Plot probability distributions by outcome using violin + strip plots"""
    import pandas as pd
    import seaborn as sns

    # Create dataframe for plotting
    df = pd.DataFrame({
        'Predicted Probability': y_prob,
        'Outcome': ['Positive' if y else 'Negative' for y in y_true]
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot dots FIRST (so they're behind), make them more transparent
    sns.stripplot(data=df, x='Outcome', y='Predicted Probability',
                  order=['Negative', 'Positive'],
                  alpha=0.2, size=2, color='black', zorder=1, ax=ax)

    # Plot violin on top with some transparency and cut=0 to prevent bleeding
    sns.violinplot(data=df, x='Outcome', y='Predicted Probability',
                   order=['Negative', 'Positive'],
                   inner=None, palette=['#1f77b4', '#ff7f0e'], 
                   alpha=0.6, zorder=2, cut=0, hue='Outcome',
                   hue_order=['Negative', 'Positive'], legend=False, ax=ax)

    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Predicted Probability', fontsize=12)
    ax.set_xlabel('True Outcome', fontsize=12)
    ax.set_title('Risk Distribution by Outcome', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
