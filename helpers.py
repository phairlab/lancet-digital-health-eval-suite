"""
Helper functions for the PHAIR Model Evaluation Suite
"""

import textwrap
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


def get_grid_layout(n: int) -> Tuple[int, int, list]:
    """Determine grid layout for n plots.
    
    Returns:
        (nrows, ncols, row_counts): nrows and ncols define the grid,
        row_counts is a list of how many plots per row
    """
    if n <= 0 or n > 16:
        return (0, 0, [])
    
    layouts = {
        1: (1, 1, [1]),
        2: (1, 2, [2]),
        3: (1, 3, [3]),
        4: (2, 2, [2, 2]),
        5: (2, 3, [3, 2]),  # could also be (1, 5, [5]) - 1 row of 5
        6: (2, 3, [3, 3]),
        7: (2, 4, [4, 3]),
        8: (2, 4, [4, 4]),
        9: (3, 3, [3, 3, 3]),
        10: (2, 5, [5, 5]),
        11: (3, 4, [4, 4, 3]),
        12: (3, 4, [4, 4, 4]),
        13: (3, 5, [5, 4, 4]),
        14: (3, 5, [5, 5, 4]),
        15: (3, 5, [5, 5, 5]),
        16: (4, 4, [4, 4, 4, 4]),
    }
    return layouts[n]


def risk_distribution_grid(y_trues: list, y_probs: list,
                           experiment_names: list,
                           save_path: Optional[str] = None) -> None:
    """Generate a grid of risk distribution plots for multiple experiments.
    
    Args:
        y_trues: List of true label arrays for each experiment
        y_probs: List of predicted probability arrays for each experiment
        experiment_names: List of experiment names for subplot titles
        save_path: Path to save figure (optional)
    """
    n = len(y_trues)
    
    if n > 16:
        print(f"Warning: {n} experiments exceeds maximum of 16 for grid visualization. Skipping.")
        return
    
    if n == 0:
        return
    
    nrows, ncols, row_counts = get_grid_layout(n)
    
    # Calculate figure size based on grid dimensions
    fig_width = 3.5 * ncols
    fig_height = 4 * nrows
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Use GridSpec with extra columns for centering irregular rows
    # We use 2*ncols to allow half-column offsets for centering
    gs = GridSpec(nrows, 2 * ncols, figure=fig, hspace=0.35, wspace=0.5)
    
    plot_idx = 0
    for row_idx, count in enumerate(row_counts):
        # Calculate offset to center this row's plots
        # Each plot spans 2 columns in the doubled grid
        offset = ncols - count  # This gives the left offset in "half-columns"
        
        for col_idx in range(count):
            if plot_idx >= n:
                break
            
            # Calculate grid position with centering offset
            gs_col_start = offset + col_idx * 2
            gs_col_end = gs_col_start + 2
            
            ax = fig.add_subplot(gs[row_idx, gs_col_start:gs_col_end])
            
            y_true = y_trues[plot_idx]
            y_prob = y_probs[plot_idx]
            name = experiment_names[plot_idx]
            
            # Create dataframe for plotting
            df = pd.DataFrame({
                'Predicted Probability': y_prob,
                'Outcome': ['Positive' if y else 'Negative' for y in y_true]
            })
            
            # Plot dots first (behind)
            sns.stripplot(data=df, x='Outcome', y='Predicted Probability',
                          order=['Negative', 'Positive'],
                          alpha=0.15, size=1.5, color='black', zorder=1, ax=ax)
            
            # Plot violin on top
            sns.violinplot(data=df, x='Outcome', y='Predicted Probability',
                           order=['Negative', 'Positive'],
                           inner=None, palette=['#1f77b4', '#ff7f0e'],
                           alpha=0.6, zorder=2, cut=0, hue='Outcome', 
                           hue_order=['Negative', 'Positive'],
                           legend=False, ax=ax)
            
            ax.set_ylim(-0.05, 1.05)
            # Only show y-axis label for leftmost plots in each row
            if col_idx == 0:
                ax.set_ylabel('Predicted Probability', fontsize=10)
            else:
                ax.set_ylabel('')
            ax.set_xlabel('')
            # Wrap long titles to prevent overlap
            wrapped_title = '\n'.join(textwrap.wrap(name, width=25))
            ax.set_title(wrapped_title, fontsize=11, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            plot_idx += 1
    
    # Add overall title
    fig.suptitle('Risk Distribution by Outcome - All Experiments', 
                 fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def convert_to_serializable(obj):
    """Convert NumPy data types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
