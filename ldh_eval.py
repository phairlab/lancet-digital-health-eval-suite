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

from core_eval_functions import auroc, calibration, decision_curve, risk_distribution
from helpers import risk_distribution_grid, convert_to_serializable

# Suppress sklearn warnings about penalty/C parameters
# IDK why this isn't working -- sorry lol
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.linear_model')

# ============================================================================
# FOR RECURSIVE EVALUATION ACROSS MULTIPLE EXPERIMENTS
# Define consistent ordering and naming for experiments when generating overlay plots
#    shorten names for legend labels
#    use ordering in tuples
#    filtering to use only experiments defined in consistent ordering
# element 1 of each tuple is the experiment directory name, element 2 is how you want it labeled in plots

consistent_ordering = ()
# consistent_ordering = ( 
#     ("LACE","LACE"),
#     ("LACE-C","LACE-C"),
#     ("A0","A0"),
#     ("A1","A1"),
#     ("A2","A2"),
#     ("AH","AH"),
#     ("AH, 2y Lookback","AH, 2y Lookback"),
#     ("AH+","AH+"),
#     ("AH+, 2y Lookback","AH+, 2y Lookback"),
#     ("D0","D0"),
#     ("D1","D1"),
#     ("D2","D2"),
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

# ============================================================================

def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold_range: Tuple[float, float] = (0.0, 0.5),
                   output_dir: Optional[str] = None,
                   threshold: Optional[float] = None,
                   recalibrate: bool = False) -> Tuple[Dict[str, float], np.ndarray]:
    """Generate all recommended evaluation plots and metrics"""

    if recalibrate:
        # Perform logistic recalibration (Platt scaling)
        # Uses unregularized logistic regression on logit-transformed predictions
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logit_pred = np.log(y_prob_clipped / (1 - y_prob_clipped))
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(logit_pred.reshape(-1, 1), y_true)
        y_prob = lr.predict_proba(logit_pred.reshape(-1, 1))[:, 1]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    auc = auroc(y_true, y_prob, 
                save_path=os.path.join(output_dir, 'auroc.png') if output_dir else None)
    cal_slope = calibration(y_true, y_prob,
                           save_path=os.path.join(output_dir, 'calibration.png') if output_dir else None)
    decision_curve(y_true, y_prob, threshold_range=threshold_range,
                  save_path=os.path.join(output_dir, 'decision_curve.png') if output_dir else None)
    risk_distribution(y_true, y_prob,
                     save_path=os.path.join(output_dir, 'risk_distribution.png') if output_dir else None)

    # Calculate Brier score
    brier = brier_score_loss(y_true, y_prob)

    metrics = {
        'n': len(y_true),
        'prevalence_pct': float(np.mean(y_true) * 100),
        'auroc': auc,
        'calibration_slope': cal_slope,
        'brier_score': brier
    }

    if threshold is not None:
        # Calculate additional metrics at the given threshold
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        alert_rate = (tp + fp) / len(y_true)
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

    return metrics, y_prob


# ============================================================================
# CROSS-VALIDATION AND MULTIPLE EXPERIMENT SUPPORT (for command line usage)
# ============================================================================

def evaluate_cross_validation(input_dir: str, recalibrate: bool = False, threshold: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate all folds and aggregate results"""
    json_files = sorted(Path(input_dir).rglob('fold_*_predictions.json'))
    # print(json_files)

    if not json_files:
        raise ValueError(f"No fold_*_predictions.json files found in {input_dir}")

    print(f"Found {len(json_files)} folds")
    all_metrics = []
    pooled_y_true = []
    pooled_y_prob = []

    # Evaluate each fold
    for json_file in json_files:
        fold_name = json_file.stem.replace('_predictions', '')
        print(f"Evaluating {fold_name}...")

        with open(json_file, 'r') as f:
            data = json.load(f)
        y_true = np.array(data['y_true'])
        y_prob = np.array(data['y_proba'])

        fold_dir = os.path.join(input_dir, fold_name)
        metrics, y_prob = evaluate_model(y_true, y_prob, output_dir=fold_dir, threshold=threshold, recalibrate=recalibrate) 
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
    with open(os.path.join(input_dir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate, f, indent=4)


    # Generate pooled plots
    pooled_y_true = np.array(pooled_y_true)
    pooled_y_prob = np.array(pooled_y_prob)

    pooled_dir = input_dir  # Save in the same directory as aggregate_metrics.json
    os.makedirs(pooled_dir, exist_ok=True)

    auroc(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_auroc.png'))
    calibration(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_calibration.png'))
    decision_curve(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_decision_curve.png'))
    risk_distribution(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_risk_distribution.png'))

    print("\n")

    return pooled_y_prob, pooled_y_true


def evaluate_recursive(input_dir: str, recalibrate: bool = False, threshold: Optional[float] = None) -> None:
    """Evaluate multiple experiments recursively and aggregate results."""
    experiment_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]

    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in {input_dir}")

    # Use consistent_ordering if defined, otherwise sort alphabetically
    if consistent_ordering:
        # Create a mapping from dir name to legend name
        ordering_dict = {dir_name: legend_name for dir_name, legend_name in consistent_ordering}
        # Filter and order experiment_dirs based on consistent_ordering
        ordered_dirs = []
        for dir_name, legend_name in consistent_ordering:
            matching = [d for d in experiment_dirs if d.name == dir_name]
            if matching:
                ordered_dirs.append((matching[0], legend_name))
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
            print(f"Warning: Failed to process {experiment_dir.name} ({e})")


    # Generate overlay plots across all experiments
    print("\n=== Generating Overlay Plots Across All Experiments ===")
    overlay_dir = Path(input_dir) / 'overlay_results'
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
    thresholds = np.linspace(0.0, 0.5, 100)
    all_net_benefits = []
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_trues, pooled_y_probs)):
        net_benefits = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
        all_net_benefits.append(net_benefits)
        ax.plot(thresholds, net_benefits, linewidth=2, 
                label=all_experiment_metrics[i]['name'])

    prevalence = np.mean([y.mean() for y in pooled_y_trues])
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    ax.plot(thresholds, treat_all, '--', label='Treat All', linewidth=2, alpha=0.7, color='#1f77b4')
    ax.axhline(0, linestyle='--', color='gray', label='Treat None', linewidth=2, alpha=0.7)

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

    args = parser.parse_args()

    if args.recurse:
        evaluate_recursive(args.input_dir, recalibrate=args.recalibrate, threshold=args.threshold)
    else:
        evaluate_cross_validation(args.input_dir, recalibrate=args.recalibrate, threshold=args.threshold)