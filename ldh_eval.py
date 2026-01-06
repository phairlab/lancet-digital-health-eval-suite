#!/usr/bin/env python3
"""
PHAIR Model Evaluation Suite
Implements Van Calster et al. (2025) recommendations for clinical ML model evaluation
Written by Sacha Davis (sdavis1@ualberta.ca)
"""
import os
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.isotonic import IsotonicRegression

from core_eval_functions import auroc, calibration, decision_curve, risk_distribution


def convert_to_serializable(obj):
    """Convert NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold_range: Tuple[float, float] = (0.0, 0.5),
                   output_dir: Optional[str] = None,
                   threshold: Optional[float] = None) -> Dict[str, float]:
    """Generate all recommended evaluation plots and metrics"""

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

    return metrics


# ============================================================================
# CROSS-VALIDATION SUPPORT (for command line usage)
# ============================================================================

def evaluate_cross_validation(input_dir: str, recalibrate: bool = False, threshold: Optional[float] = None) -> None:
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

        if recalibrate:
            # Perform logistic recalibration
            y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
            logit_pred = np.log(y_prob_clipped / (1 - y_prob_clipped))
            lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
            lr.fit(logit_pred.reshape(-1, 1), y_true)
            y_prob = lr.predict_proba(logit_pred.reshape(-1, 1))[:, 1]

        fold_dir = os.path.join(input_dir, fold_name)
        metrics = evaluate_model(y_true, y_prob, output_dir=fold_dir, threshold=threshold)  # Already recalibrated
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


def evaluate_recursive(input_dir: str, recalibrate: bool = False, threshold: Optional[float] = None) -> None:
    """Evaluate multiple experiments recursively and aggregate results."""
    experiment_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]

    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in {input_dir}")

    # Sort experiment directories to ensure consistent order
    experiment_dirs = sorted(experiment_dirs, key=lambda d: d.name)

    print(f"Found {len(experiment_dirs)} experiments")

    all_experiment_metrics = []
    pooled_y_true = []
    pooled_y_prob = []

    for experiment_dir in experiment_dirs:
        print(f"Processing experiment: {experiment_dir.name}")
        try:
            evaluate_cross_validation(str(experiment_dir), recalibrate=recalibrate, threshold=threshold)

            # Collect aggregate metrics from each experiment
            with open(Path(experiment_dir) / 'aggregate_metrics.json', 'r') as f:
                experiment_metrics = json.load(f)
                all_experiment_metrics.append({
                    'name': experiment_dir.name,
                    'metrics': experiment_metrics
                })

            # Collect pooled predictions from each experiment (for separate curves)
            exp_y_true = []
            exp_y_prob = []
            json_files = sorted(Path(experiment_dir).rglob('fold_*_predictions.json'))
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                y_true = np.array(data['y_true'])
                y_prob = np.array(data['y_proba'])

                if recalibrate:
                    # Perform logistic recalibration
                    y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
                    logit_pred = np.log(y_prob_clipped / (1 - y_prob_clipped))
                    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
                    lr.fit(logit_pred.reshape(-1, 1), y_true)
                    y_prob = lr.predict_proba(logit_pred.reshape(-1, 1))[:, 1]

                exp_y_true.extend(y_true)
                exp_y_prob.extend(y_prob)

            pooled_y_true.append(np.array(exp_y_true))
            pooled_y_prob.append(np.array(exp_y_prob))

        except Exception as e:
            print(f"Warning: Failed to process {experiment_dir.name} ({e})")




    # Generate overlay plots across all experiments
    print("\n=== Generating Overlay Plots Across All Experiments ===")
    overlay_dir = Path(input_dir) / 'overlay_results'
    os.makedirs(overlay_dir, exist_ok=True)

    # ROC curves overlay
    plt.figure(figsize=(8, 6))
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_true, pooled_y_prob)):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{all_experiment_metrics[i]["name"]} (AUC={auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curves - All Experiments')
    plt.legend(loc='best', title='Experiments', fontsize=10, title_fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(overlay_dir / 'overlay_auroc.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Calibration curves overlay
    plt.figure(figsize=(8, 6))
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_true, pooled_y_prob)):
        sort_idx = np.argsort(y_prob)
        y_prob_sorted = y_prob[sort_idx]
        y_true_sorted = y_true[sort_idx].astype(float)
        try:
            smoothed = lowess(y_true_sorted, y_prob_sorted, frac=0.25, it=0, return_sorted=True)
            plt.plot(smoothed[:, 0], smoothed[:, 1], linewidth=2, 
                    label=all_experiment_metrics[i]['name'])
        except Exception as e:
            print(f"Warning: Loess smoothing failed for {all_experiment_metrics[i]['name']} ({e})")
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Proportion')
    plt.title('Calibration Curves - All Experiments')
    plt.legend(loc='best', title='Experiments', fontsize=10, title_fontsize=11)
    plt.grid(alpha=0.3)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.savefig(overlay_dir / 'overlay_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Decision curves overlay
    plt.figure(figsize=(10, 6))
    thresholds = np.linspace(0.0, 0.5, 100)
    for i, (y_true, y_prob) in enumerate(zip(pooled_y_true, pooled_y_prob)):
        net_benefits = []
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n = len(y_true)
            net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
            net_benefits.append(net_benefit)
        plt.plot(thresholds, net_benefits, linewidth=2, 
                label=all_experiment_metrics[i]['name'])

    prevalence = np.mean([y.mean() for y in pooled_y_true])
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    plt.plot(thresholds, treat_all, '--', label='Treat All', linewidth=2, alpha=0.7)
    plt.axhline(0, linestyle='--', color='gray', label='Treat None', linewidth=2, alpha=0.7)
    plt.xlim(0, 0.5)
    plt.ylim(-0.2, 0.2)
    plt.xlabel('Decision Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curves - All Experiments')
    plt.legend(loc='best', title='Experiments', fontsize=10, title_fontsize=11)
    plt.grid(alpha=0.3)
    plt.savefig(overlay_dir / 'overlay_decision_curve.png', dpi=300, bbox_inches='tight')
    plt.close()


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