#!/usr/bin/env python3
"""
PHAIR Model Evaluation Suite
Implements Van Calster et al. (2025) recommendations for clinical ML model evaluation
Written by Sacha Davis (sdavis1@ualberta.ca) + Copilot (multiple models)
"""

import os
import json
import argparse
import warnings

from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from core_eval_functions import auroc, calibration, decision_curve, risk_distribution, bengio_nadeau_test
from helpers import risk_distribution_grid, convert_to_serializable

# Suppress sklearn warnings about penalty/C parameters
# IDK why this isn't working -- sorry lol
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')

# ============================================================================
# FOR RECURSIVE EVALUATION ACROSS MULTIPLE EXPERIMENTS
# Controls which experiments appear in overlay plots, in what order, and under
# what legend label. Experiments not named here are excluded from the overlays.
#
# The default is empty, meaning: use every subdirectory, alphabetically, labeled
# by directory name. To pin an ordering, pass --ordering with a JSON file (see
# example_ordering.json), or set the tuple below for a project-local default.
# Element 1 of each tuple is the directory name, element 2 is the plot label.

consistent_ordering = ()

# consistent_ordering = (
#     ("LACE","LACE"),
#     ("LACE-C","LACE-C"),
#     ("A0","A0"),
#     ("A1","A1"),
#     ("AH","AH"),
#     ("D0","D0"),
#     ("D1","D1"),
# )  # example 1

# consistent_ordering = (
#     ("AH","AH"),
#     ("D1_removedtop0percent","D1: Remove Top 0%"),
#     ("D1_removedtop5percent","D1: Remove Top 5%"),
#     ("D1_removedtop10percent","D1: Remove Top 10%"),
#     ("D1_removedtop15percent","D1: Remove Top 15%"),
#     ("D1_removedtop20percent","D1: Remove Top 20%"),
#     ("D1_removedtop25percent","D1: Remove Top 25%"),
#     ("D1_removedtop30percent","D1: Remove Top 30%"),
#     ("D1_removedtop35percent","D1: Remove Top 35%"),
#     ("D1_removedtop40percent","D1: Remove Top 40%"),
#     ("D1_removedtop45percent","D1: Remove Top 45%"),
#     ("D1_removedtop50percent","D1: Remove Top 50%"),
# )  # example 2


def resolve_input_dir(input_dir: str) -> Path:
    """Resolve an input directory, tolerating an absolute path with its leading '/' stripped.

    Tries the path as given first (absolute, or relative to the current working
    directory). If that does not exist and the path is relative, retries it as
    root-anchored -- so 'Users/me/experiments' still finds '/Users/me/experiments'.
    """
    path = Path(input_dir).expanduser()

    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Input directory not found: {resolved}")
        return resolved

    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.is_dir():
        return cwd_candidate

    root_candidate = Path('/').joinpath(path).resolve()
    if root_candidate.is_dir():
        print(f"Note: reading '{input_dir}' as the absolute path {root_candidate}")
        return root_candidate

    raise FileNotFoundError(
        f"Input directory '{input_dir}' not found. Tried:\n"
        f"  {cwd_candidate}\n"
        f"  {root_candidate}"
    )


def load_ordering(ordering_path: Optional[str]) -> tuple:
    """Load experiment ordering/labels from JSON, falling back to consistent_ordering.

    Accepts either an object mapping directory name -> plot label (key order is
    preserved) or a list of [directory_name, plot_label] pairs.
    """
    if ordering_path is None:
        return consistent_ordering

    path = Path(ordering_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Ordering file not found: {path}")

    with open(path, 'r') as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        return tuple(raw.items())
    if isinstance(raw, list):
        try:
            return tuple((str(name), str(label)) for name, label in raw)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{path}: list entries must be [directory_name, plot_label] pairs ({e})"
            ) from e
    raise ValueError(f"{path}: expected a JSON object or a list of pairs, got {type(raw).__name__}")


# ============================================================================

def evaluate_model(y_test_true: np.ndarray, y_test_prob: np.ndarray,
                   y_train_true: Optional[np.ndarray] = None, y_train_prob: Optional[np.ndarray] = None,
                   threshold_range: Tuple[float, float] = (0.0, 0.5),
                   output_dir: Optional[str] = None,
                   threshold: Optional[float] = None,
                   recalibrate: bool = False) -> Tuple[Dict[str, float], np.ndarray]:
    """Generate all recommended evaluation plots and metrics"""

    if recalibrate:
        # Perform logistic recalibration (Platt scaling)
        # Train on training predictions, then apply to test predictions
        if y_train_true is None or y_train_prob is None:
            raise ValueError("y_train_true and y_train_prob are required when recalibrate=True")

        y_train_prob_clipped = np.clip(y_train_prob, 1e-7, 1 - 1e-7)
        train_logit_pred = np.log(y_train_prob_clipped / (1 - y_train_prob_clipped))
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(train_logit_pred.reshape(-1, 1), y_train_true)

        y_test_prob_clipped = np.clip(y_test_prob, 1e-7, 1 - 1e-7)
        test_logit_pred = np.log(y_test_prob_clipped / (1 - y_test_prob_clipped))
        y_test_prob = lr.predict_proba(test_logit_pred.reshape(-1, 1))[:, 1]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    auc = auroc(y_test_true, y_test_prob, 
                save_path=os.path.join(output_dir, 'auroc.png') if output_dir else None)
    cal_slope = calibration(y_test_true, y_test_prob,
                           save_path=os.path.join(output_dir, 'calibration.png') if output_dir else None)
    decision_curve(y_test_true, y_test_prob, threshold_range=threshold_range,
                  save_path=os.path.join(output_dir, 'decision_curve.png') if output_dir else None,
                  threshold=threshold)
    risk_distribution(y_test_true, y_test_prob,
                     save_path=os.path.join(output_dir, 'risk_distribution.png') if output_dir else None)

    # Calculate Brier score
    brier = brier_score_loss(y_test_true, y_test_prob)

    metrics = {
        'n': len(y_test_true),
        'prevalence_pct': float(np.mean(y_test_true) * 100),
        'auroc': auc,
        'calibration_slope': cal_slope,
        'brier_score': brier
    }

    if threshold is not None:
        # Calculate additional metrics at the given threshold
        y_pred = (y_test_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_test_true == 1))
        tn = np.sum((y_pred == 0) & (y_test_true == 0))
        fp = np.sum((y_pred == 1) & (y_test_true == 0))
        fn = np.sum((y_pred == 0) & (y_test_true == 1))

        alert_rate = (tp + fp) / len(y_test_true)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        metrics.update({
            'alert_rate': alert_rate,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        })

    # Convert metrics to serializable types
    metrics = {k: convert_to_serializable(v) for k, v in metrics.items()}

    if output_dir:
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Results saved to {output_dir}")

    return metrics, y_test_prob


# ============================================================================
# CROSS-VALIDATION AND MULTIPLE EXPERIMENT SUPPORT (for command line usage)
# ============================================================================

def evaluate_cross_validation(input_dir: str, recalibrate: bool = False, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate all folds and aggregate results"""
    input_path = resolve_input_dir(input_dir)

    json_files = sorted(input_path.rglob('fold_*_predictions.json'))
    # print(json_files)

    if not json_files:
        raise ValueError(f"No fold_*_predictions.json files found in {input_path}")

    print(f"Found {len(json_files)} folds")
    all_metrics = []
    pooled_y_true = []
    pooled_y_prob = []

    # Evaluate each fold
    for json_file in json_files:
        fold_name = json_file.stem.replace('_predictions', '')
        print(f"Evaluating {fold_name}...")

        with open(json_file, 'r') as f:
            data_test = json.load(f)
        y_true = np.array(data_test['y_true'])
        y_prob = np.array(data_test['y_proba'])

        # Training predictions are only needed to fit the recalibration map
        y_train_true = y_train_prob = None
        if recalibrate:
            train_file = json_file.with_name(json_file.name.replace('fold_', 'train_', 1))
            if not train_file.exists():
                raise FileNotFoundError(
                    f"--recalibrate needs training predictions, but '{train_file.name}' was not "
                    f"found in {train_file.parent}. Either save train_*_predictions.json alongside "
                    f"your fold_*_predictions.json (see README), or drop --recalibrate."
                )
            with open(train_file, 'r') as f:
                data_train = json.load(f)
            y_train_true = np.array(data_train['y_true'])
            y_train_prob = np.array(data_train['y_proba'])

        fold_dir = os.path.join(str(input_path), fold_name)
        metrics, y_prob = evaluate_model(y_true, y_prob, y_train_true, y_train_prob, output_dir=fold_dir, threshold=threshold, recalibrate=recalibrate) 
        all_metrics.append(metrics)

        pooled_y_true.extend(y_true)
        pooled_y_prob.extend(y_prob)

    # Aggregate across folds
    print(f"\n=== Aggregate Results for {input_dir} ===")
    aggregate = {}

    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics if metric in m]
        mean, std = np.mean(values), np.std(values)
        aggregate[metric] = {'mean': convert_to_serializable(mean), 'std': convert_to_serializable(std)}
        print(f"{metric}: {mean:.3f} ± {std:.3f}")

    # Save aggregate metrics to JSON
    with open(os.path.join(str(input_path), 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate, f, indent=4)


    # Generate pooled plots
    pooled_y_true = np.array(pooled_y_true)
    pooled_y_prob = np.array(pooled_y_prob)

    pooled_dir = str(input_path)  # Save in the same directory as aggregate_metrics.json
    os.makedirs(pooled_dir, exist_ok=True)

    auroc(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_auroc.png'))
    calibration(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_calibration.png'))
    decision_curve(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_decision_curve.png'), threshold=threshold)
    risk_distribution(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_risk_distribution.png'))

    print("\n")

    return pooled_y_prob, pooled_y_true


def evaluate_recursive(input_dir: str, recalibrate: bool = False, threshold: Optional[float] = None,
                       bengio_correction: bool = False, ordering: Optional[tuple] = None) -> None:
    """Evaluate multiple experiments recursively and aggregate results."""
    input_path = resolve_input_dir(input_dir)

    if ordering is None:
        ordering = consistent_ordering

    # 'overlay_results*' dirs hold this script's own output, not experiments
    experiment_dirs = [
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith('overlay_results')
    ]

    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in {input_path}")

    # Use the configured ordering if defined, otherwise sort alphabetically
    if ordering:
        found_names = {d.name for d in experiment_dirs}
        ordered_dirs = [
            (next(d for d in experiment_dirs if d.name == dir_name), legend_name)
            for dir_name, legend_name in ordering
            if dir_name in found_names
        ]
        if not ordered_dirs:
            raise ValueError(
                f"None of the {len(ordering)} experiment names in the configured ordering matched "
                f"a subdirectory of {input_path}.\n"
                f"  Ordering expects: {sorted(name for name, _ in ordering)}\n"
                f"  Directories found: {sorted(found_names)}\n"
                f"Pass --ordering with a JSON file matching your directory names, or clear the "
                f"ordering to use every subdirectory alphabetically."
            )
        missing = [name for name, _ in ordering if name not in found_names]
        if missing:
            print(f"Note: no directory found for ordering entries {missing}; skipping them.")
        experiment_dirs_with_labels = ordered_dirs
    else:
        # Sort experiment directories alphabetically and use dir name as label
        experiment_dirs = sorted(experiment_dirs, key=lambda d: d.name)
        experiment_dirs_with_labels = [(d, d.name) for d in experiment_dirs]

    print(f"Found {len(experiment_dirs_with_labels)} experiments")

    all_experiment_metrics = []
    pooled_y_trues = []
    pooled_y_probs = []

    for experiment_dir, legend_name in experiment_dirs_with_labels:
        print(f"Processing experiment: {experiment_dir.name}")
        try:
            pooled_y_prob, pooled_y_true = evaluate_cross_validation(str(experiment_dir), recalibrate=recalibrate, threshold=threshold)

            # Collect aggregate metrics from each experiment
            with open(Path(experiment_dir) / 'aggregate_metrics.json', 'r') as f:
                experiment_metrics = json.load(f)
                all_experiment_metrics.append({
                    'name': legend_name,
                    'metrics': experiment_metrics
                })

            pooled_y_trues.append(np.array(pooled_y_true))
            pooled_y_probs.append(np.array(pooled_y_prob))

        except Exception as e:
            print(f"Warning: Failed to process {experiment_dir.name} ({type(e).__name__}: {e})")

    if not all_experiment_metrics:
        raise ValueError(
            f"None of the {len(experiment_dirs_with_labels)} experiment(s) in {input_path} could be "
            f"evaluated — see the warnings above for the per-experiment cause."
        )

    # Generate overlay plots across all experiments
    print("\n=== Generating Overlay Plots Across All Experiments ===")
    overlay_dir = input_path / 'overlay_results'
    os.makedirs(overlay_dir, exist_ok=True)

    # ROC curves overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_trues, pooled_y_probs)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        # Use mean and std from aggregate metrics instead of pooled AUROC
        auc_mean = all_experiment_metrics[i]['metrics']['auroc']['mean']
        auc_std = all_experiment_metrics[i]['metrics']['auroc']['std']
        ax.plot(fpr, tpr, label=f'{all_experiment_metrics[i]["name"]} (AUC={auc_mean:.3f}±{auc_std:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1.5)
    ax.set_xlabel('1 - Specificity', fontsize=12)
    ax.set_ylabel('Sensitivity', fontsize=12)
    ax.set_title('ROC Curves - All Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(overlay_dir / 'overlay_auroc.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calibration curves overlay
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_trues, pooled_y_probs)):
        sort_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sort_idx]
        y_true_sorted = y_true[sort_idx].astype(float)
        try:
            smoothed = lowess(y_true_sorted, y_prob_sorted, frac=0.25, it=0, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2, 
                    label=all_experiment_metrics[i]['name'])
        except Exception as e:
            print(f"Warning: Loess smoothing failed for {all_experiment_metrics[i]['name']} ({e})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Proportion', fontsize=12)
    ax.set_title('Calibration Curves - All Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.savefig(overlay_dir / 'overlay_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Decision curves overlay
    fig, ax = plt.subplots(figsize=(10, 6))
    thresh_range = np.linspace(0.0, 0.5, 100)
    all_net_benefits = []
    
    # Draw vertical line at threshold if provided
    if threshold is not None and 0.0 <= threshold <= 0.5:
        ax.axvline(threshold, linestyle='-', color='red', linewidth=1.5, alpha=0.6, label=f'Threshold ({threshold:.2f})', zorder=4)
    
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_trues, pooled_y_probs)):
        net_benefits = []
        for thresh in thresh_range:
            y_pred = (y_prob >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            net_benefit = (tp / n) - (fp / n) * (thresh / (1 - thresh))
            net_benefits.append(net_benefit)
        all_net_benefits.append(net_benefits)
        ax.plot(thresh_range, net_benefits, linewidth=2, 
                label=all_experiment_metrics[i]['name'], zorder=3)

    prevalence = np.mean([y.mean() for y in pooled_y_trues])
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresh_range]
    ax.plot(thresh_range, treat_all, '--', label='Treat All', linewidth=2, alpha=0.7, color='#1f77b4', zorder=2)
    ax.axhline(0, linestyle='--', color='gray', label='Treat None', linewidth=2, alpha=0.7, zorder=2)

    max_y = max(max(nb) for nb in all_net_benefits)
    max_y = max(max_y, max(treat_all), 0)
    ax.set_ylim(-0.05, max_y * 1.15)
    ax.set_xlim(0, 0.5)
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title('Decision Curves - All Experiments', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.savefig(overlay_dir / 'overlay_decision_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Risk distribution grid
    experiment_names = [exp['name'] for exp in all_experiment_metrics]
    risk_distribution_grid(
        pooled_y_trues, 
        pooled_y_probs,
        experiment_names,
        save_path=overlay_dir / 'overlay_risk_distribution.png'
    )

    # Save combined metrics dataframe
    print("\n=== Saving Combined Metrics DataFrame ===")
    metrics_data = []
    for exp_metrics in all_experiment_metrics:
        row = {'experiment': exp_metrics['name']}
        for metric_name, metric_values in exp_metrics['metrics'].items():
            row[f'{metric_name}_mean'] = metric_values['mean']
            row[f'{metric_name}_std'] = metric_values['std']
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(overlay_dir / 'combined_metrics.csv', index=False)
    print(f"✓ Combined metrics saved to {overlay_dir / 'combined_metrics.csv'}")

    # Save combined metrics dataframe (value ± std format) -- for C+P into spreadsheets
    print("\n=== Saving Combined Metrics DataFrame (± format) ===")
    metrics_data_formatted = []
    for exp_metrics in all_experiment_metrics:
        row = {'experiment': exp_metrics['name']}
        for metric_name, metric_values in exp_metrics['metrics'].items():
            mean = metric_values['mean']
            std = metric_values['std']
            # Format n as integer, prevalence with 1 decimal, others with 3 decimals
            if metric_name == 'n':
                row[metric_name] = f"{mean:.0f} (±{std:.0f})"
            elif metric_name == 'prevalence_pct':
                row[metric_name] = f"{mean:.1f}% (±{std:.1f}%)"
            else:
                row[metric_name] = f"{mean:.3f} (±{std:.3f})"
        metrics_data_formatted.append(row)
    
    metrics_df_formatted = pd.DataFrame(metrics_data_formatted)
    metrics_df_formatted.to_csv(overlay_dir / 'combined_metrics_formatted.tsv', index=False, sep='\t')
    print(f"✓ Combined metrics (formatted) saved to {overlay_dir / 'combined_metrics_formatted.tsv'}")

    if bengio_correction:
        bengio_correction_analysis(experiment_dirs_with_labels, overlay_dir)



# ============================================================================
# BENGIO-NADEAU CORRECTION FOR MULTI-EXPERIMENT COMPARISON
# ============================================================================

# Metrics excluded from statistical comparison (not continuous performance measures)
_SKIP_METRICS = {'n', 'prevalence_pct', 'tp', 'tn', 'fp', 'fn'}

def bengio_correction_analysis(experiment_dirs_with_labels: list, overlay_dir: Path) -> None:
    """
    Pairwise Nadeau-Bengio corrected t-tests across experiments.

    Reads per-fold metrics.json files saved during evaluate_cross_validation,
    pairs folds by name across experiments, and produces a CSV with corrected
    t-statistics and p-values for every performance metric.

    Output files written to overlay_dir:
        bengio_correction.csv           -- long-form pairwise comparison table
        bengio_correction_auroc_pvals.csv -- square p-value matrix for AUROC
    """
    print("\n=== Running Bengio-Nadeau Corrected Comparisons ===")

    # Collect per-fold metrics for each experiment
    experiment_fold_metrics: Dict[str, dict] = {}
    for exp_dir, exp_name in experiment_dirs_with_labels:
        fold_files = sorted(Path(exp_dir).glob('fold_*/metrics.json'))
        if not fold_files:
            print(f"Warning: No fold metrics found for {exp_name}, skipping.")
            continue
        fold_metrics = {}
        for ff in fold_files:
            with open(ff) as f:
                fold_metrics[ff.parent.name] = json.load(f)
        experiment_fold_metrics[exp_name] = fold_metrics

    if len(experiment_fold_metrics) < 2:
        print("Warning: Need at least 2 experiments with fold metrics for comparison.")
        return

    # Find folds present in all experiments
    all_fold_sets = [set(fm.keys()) for fm in experiment_fold_metrics.values()]
    common_folds = sorted(set.intersection(*all_fold_sets))
    k = len(common_folds)

    if k < 2:
        print(f"Warning: Only {k} common fold(s) across experiments — need ≥2 for the test.")
        return

    missing = set.union(*all_fold_sets) - set(common_folds)
    if missing:
        print(f"Note: Folds {missing} not present in all experiments; using {k} common folds.")

    # Estimate average n_test and n_train across folds
    # In k-fold CV: n_total ≈ k * n_test, n_train ≈ (k-1) * n_test
    all_n = [
        fold_metrics[fold]['n']
        for fold_metrics in experiment_fold_metrics.values()
        for fold in common_folds
        if 'n' in fold_metrics.get(fold, {})
    ]
    n_test = float(np.mean(all_n))
    n_train = n_test * (k - 1)  # standard k-fold assumption

    print(f"Folds: {k}  |  avg n_test: {n_test:.0f}  |  avg n_train: {n_train:.0f}")

    # Determine which metrics to compare
    first_exp = next(iter(experiment_fold_metrics.values()))
    first_fold = next(iter(first_exp.values()))
    compare_metrics = [m for m in first_fold if m not in _SKIP_METRICS]

    # Pairwise corrected t-tests
    exp_names = list(experiment_fold_metrics.keys())
    rows = []

    for i, name_a in enumerate(exp_names):
        for j, name_b in enumerate(exp_names):
            if j <= i:
                continue
            row = {'experiment_A': name_a, 'experiment_B': name_b, 'k_folds': k,
                   'avg_n_test': round(n_test), 'avg_n_train': round(n_train)}
            for metric in compare_metrics:
                diffs = []
                for fold in common_folds:
                    ma = experiment_fold_metrics[name_a].get(fold, {}).get(metric)
                    mb = experiment_fold_metrics[name_b].get(fold, {}).get(metric)
                    if ma is not None and mb is not None:
                        diffs.append(ma - mb)
                if len(diffs) >= 2:
                    t_stat, p_val = bengio_nadeau_test(diffs, n_test, n_train)
                    row[f'{metric}_mean_diff(A-B)'] = round(float(np.mean(diffs)), 5)
                    row[f'{metric}_t'] = round(t_stat, 4) if not np.isnan(t_stat) else np.nan
                    row[f'{metric}_p'] = round(p_val, 5) if not np.isnan(p_val) else np.nan
                    row[f'{metric}_sig_p05'] = (p_val < 0.05) if not np.isnan(p_val) else False
            rows.append(row)

    results_df = pd.DataFrame(rows)
    out_path = overlay_dir / 'bengio_correction.csv'
    results_df.to_csv(out_path, index=False)
    print(f"✓ Pairwise comparison table saved to {out_path}")

    # Square p-value matrix for AUROC
    if 'auroc' in compare_metrics:
        auroc_matrix = pd.DataFrame(
            np.eye(len(exp_names), dtype=float),
            index=exp_names,
            columns=exp_names,
        )
        for row in rows:
            na, nb = row['experiment_A'], row['experiment_B']
            p = row.get('auroc_p', np.nan)
            auroc_matrix.loc[na, nb] = p
            auroc_matrix.loc[nb, na] = p  # symmetric
        matrix_path = overlay_dir / 'bengio_correction_auroc_pvals.csv'
        auroc_matrix.to_csv(matrix_path)
        print(f"✓ AUROC p-value matrix saved to {matrix_path}")

    # Print summary to console
    print("\nPairwise AUROC comparisons (Bengio-Nadeau corrected, two-tailed):")
    for row in rows:
        diff = row.get('auroc_mean_diff(A-B)', np.nan)
        p = row.get('auroc_p', np.nan)
        sig = '*' if row.get('auroc_sig_p05', False) else ' '
        print(f"  {sig} {row['experiment_A']} vs {row['experiment_B']}: "
              f"Δ={diff:+.4f}, p={p:.4f}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PHAIR Model Evaluation Suite'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing fold prediction JSON files')
    parser.add_argument('--recurse', action='store_true',
                        help='Set to True to evaluate multiple experiments recursively and generate pooled plots')
    parser.add_argument('--recalibrate', action='store_true',
                        help='Perform logistic recalibration before evaluation')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Threshold for classification metrics (e.g., sensitivity, specificity)')
    parser.add_argument('--bengio-correction', action='store_true',
                        help='Run Nadeau-Bengio corrected pairwise t-tests across experiments (requires --recurse)')
    parser.add_argument('--ordering', type=str, default=None,
                        help='JSON file mapping experiment directory name -> plot label, which also '
                             'defines overlay plot ordering and filters to just those experiments '
                             '(requires --recurse). Omit to use every subdirectory alphabetically.')

    args = parser.parse_args()

    if args.bengio_correction and not args.recurse:
        parser.error('--bengio-correction requires --recurse')

    if args.ordering and not args.recurse:
        parser.error('--ordering requires --recurse')

    if args.recurse:
        evaluate_recursive(args.input_dir, recalibrate=args.recalibrate, threshold=args.threshold,
                           bengio_correction=args.bengio_correction,
                           ordering=load_ordering(args.ordering))
    else:
        evaluate_cross_validation(args.input_dir, recalibrate=args.recalibrate, threshold=args.threshold)