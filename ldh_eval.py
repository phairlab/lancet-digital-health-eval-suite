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
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from statsmodels.nonparametric.smoothers_lowess import lowess


# ============================================================================
# CORE EVALUATION FUNCTIONS (for direct import)
# ============================================================================

def auroc(y_true: np.ndarray, y_prob: np.ndarray, 
          save_path: Optional[str] = None) -> float:
    """Calculate AUROC and optionally plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    
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
    
    # Calculate calibration slope
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
                 fontsize=13)
    ax.legend(loc='upper left')
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
                   save_path: Optional[str] = None) -> None:
    """Generate decision curve analysis"""
    thresholds = np.linspace(threshold_range[0], threshold_range[1], 100)
    net_benefits = []
    prevalence = y_true.mean()
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        net_benefits.append(net_benefit)
    
    treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
    
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits, label='Model', linewidth=2.5, color='#1f77b4')
    plt.plot(thresholds, treat_all, '--', label='Treat All', linewidth=2, alpha=0.7, color='#ff7f0e')
    plt.axhline(0, linestyle='--', color='gray', label='Treat None', linewidth=2, alpha=0.7)
    
    plt.xlabel('Decision Threshold', fontsize=12)
    plt.ylabel('Net Benefit', fontsize=12)
    plt.title('Decision Curve Analysis', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    # Focus on clinically relevant range
    max_nb = max(net_benefits)
    plt.ylim(-0.05, max_nb * 1.15)  # Small negative buffer, space above max
    plt.xlim(threshold_range[0], threshold_range[1])
    
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
    
    plt.figure(figsize=(10, 6))
    
    # Plot dots FIRST (so they're behind), make them more transparent
    sns.stripplot(data=df, x='Outcome', y='Predicted Probability', 
                  alpha=0.2, size=2, color='black', zorder=1)
    
    # Plot violin on top with some transparency and cut=0 to prevent bleeding
    sns.violinplot(data=df, x='Outcome', y='Predicted Probability', 
                   inner=None, palette=['#1f77b4', '#ff7f0e'], 
                   alpha=0.6, zorder=2, cut=0)  # cut=0 prevents KDE bleeding
    
    plt.ylim(-0.05, 1.05)
    plt.ylabel('Predicted Probability', fontsize=12)
    plt.xlabel('True Outcome', fontsize=12)
    plt.title('Risk Distribution by Outcome', fontsize=13)
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        

def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray,
                   threshold_range: Tuple[float, float] = (0.0, 0.5),
                   output_dir: Optional[str] = None) -> Dict[str, float]:
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
    
    metrics = {'auroc': auc, 'calibration_slope': cal_slope}
    
    if output_dir:
        with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"✓ Results saved to {output_dir}")
    
    return metrics


# ============================================================================
# CROSS-VALIDATION SUPPORT (for command line usage)
# ============================================================================

def evaluate_cross_validation(input_dir: str) -> None:
    """Evaluate all folds and aggregate results"""
    # json_files = sorted(Path(input_dir).glob('fold_*_predictions.json'))
    json_files = sorted(Path(input_dir).rglob('fold_*_predictions.json'))
    print(json_files)
    
    if not json_files:
        raise ValueError(f"No fold_*_predictions.json files found in {input_dir}")
    
    print(f"Found {len(json_files)} folds")
    all_metrics = []
    
    # Evaluate each fold
    for json_file in json_files:
        fold_name = json_file.stem.replace('_predictions', '')
        print(f"Evaluating {fold_name}...")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        y_true = np.array(data['y_true'])
        y_prob = np.array(data['y_proba'])
        
        fold_dir = os.path.join(input_dir, fold_name)
        metrics = evaluate_model(y_true, y_prob, output_dir=fold_dir)
        all_metrics.append(metrics)
    
    # Aggregate across folds
    print("\n=== Aggregate Results ===")
    aggregate = {}
    pooled_y_true = []
    pooled_y_prob = []

    for metric in all_metrics[0].keys():
        values = [m[metric] for m in all_metrics]
        mean, std = np.mean(values), np.std(values)
        aggregate[metric] = {'mean': mean, 'std': std}
        print(f"{metric}: {mean:.3f} ± {std:.3f}")

    # Combine predictions from all folds for pooled plots
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        pooled_y_true.extend(data['y_true'])
        pooled_y_prob.extend(data['y_proba'])

    pooled_y_true = np.array(pooled_y_true)
    pooled_y_prob = np.array(pooled_y_prob)

    # Generate pooled plots
    pooled_dir = input_dir  # Save in the same directory as aggregate_metrics.json
    os.makedirs(pooled_dir, exist_ok=True)

    auroc(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_auroc.png'))
    calibration(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_calibration.png'))
    decision_curve(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_decision_curve.png'))
    risk_distribution(pooled_y_true, pooled_y_prob, save_path=os.path.join(pooled_dir, 'pooled_risk_distribution.png'))

    with open(os.path.join(input_dir, 'aggregate_metrics.json'), 'w') as f:
        json.dump(aggregate, f, indent=4)


def evaluate_recursive(input_dir: str) -> None:
    """Evaluate multiple experiments recursively and aggregate results."""
    experiment_dirs = [d for d in Path(input_dir).iterdir() if d.is_dir()]

    if not experiment_dirs:
        raise ValueError(f"No experiment directories found in {input_dir}")

    print(f"Found {len(experiment_dirs)} experiments")

    all_experiment_metrics = []
    pooled_y_true = []
    pooled_y_prob = []

    for experiment_dir in experiment_dirs:
        print(f"Processing experiment: {experiment_dir.name}")
        try:
            evaluate_cross_validation(str(experiment_dir))

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
                exp_y_true.extend(data['y_true'])
                exp_y_prob.extend(data['y_proba'])
            
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
    plt.legend()
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
    plt.legend()
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
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(overlay_dir / 'overlay_decision_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save summary metrics
    summary = {}
    for metric in all_experiment_metrics[0]['metrics'].keys():
        values = [exp['metrics'][metric]['mean'] for exp in all_experiment_metrics]
        mean, std = np.mean(values), np.std(values)
        summary[metric] = {'mean': mean, 'std': std}
        print(f"{metric}: {mean:.3f} ± {std:.3f}")

    with open(overlay_dir / 'summary_metrics.json', 'w') as f:
        json.dump(summary, f, indent=4)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='PHAIR Model Evaluation Suite'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing fold prediction JSON files')
    parser.add_argument('--recursive', type=bool, default=False,
                        help='Set to True to evaluate multiple experiments recursively and generate pooled plots')

    args = parser.parse_args()

    if args.recursive:
        evaluate_recursive(args.input_dir)
    else:
        evaluate_cross_validation(args.input_dir)