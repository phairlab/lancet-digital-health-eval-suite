import warnings
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

# Suppress sklearn warnings about penalty/C parameters
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')

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
        
