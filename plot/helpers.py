import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

from tools.paths import PLOTS_DIR
import tools.physics_constants as const
# from debug_categories import debug_category_assignment

def make_output_dir(name, nruns, star_catalog):
    out_dir = os.path.join(PLOTS_DIR, str(name)+'_'+str(nruns)+'_'+str(star_catalog))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def temp_zone(temp):
    '''
    Assigns a temperature zone based on the temperature value.'''
    if temp > 390:
        return 'hot'
    elif 390>temp > 270:
        return 'habitable'
    else:
        return 'cold'

def assign_category(row):
    """
    Assign planet categories based on radius, temperature, and star type.
    Returns a list of applicable categories for each planet.
    """
    categories = []
    r = row['radius_p']
    temp = row['temp_p']
    stype = row['stype']
    
    # Temperature zone categories (using ranges for efficiency)
    if temp < 125:
        categories.append('Cold planets')
    elif temp <= 305:  # Combined condition for efficiency
        categories.append('Habitable planets')
    else:  # temp > 305
        categories.append('Hot planets')
    
    # Radius-based categories (using if-elif for efficiency)
    if r < 1.5:
        categories.append('Rocky')
        # Star type categories for rocky planets
        if stype == 'M':
            categories.append('Rocky planets around M-type stars')
        elif stype in ['G', 'K']:
            categories.append('Rocky planets around G and K-type stars')
    elif r < 2.0:
        categories.append('Super-Earths')
    elif r < 4.0:
        categories.append('Sub-Neptunes')
    elif r < 8.0:
        categories.append('Sub-Jovians')
    
    return categories if categories else None
    

def get_rejection_reason(row, scenario='best'):
    '''
    Returns the rejection reason for a given row.
    If the row has HWO-style columns (with best/worst suffixes), it will use the specified scenario for rejection reasons.
    Otherwise, it will use the standard column names.
    Returns all failure reasons, not just the first one.

    Args:
        row: pd.Series, a row of a DataFrame
        scenario: str, either 'best' or 'worst' for HWO scenarios
    Returns:
        str, the rejection reason(s) - multiple reasons separated by ' + ' if applicable
    '''
    failure_reasons = []
    
    # Check if we have HWO-style columns (with best/worst suffixes)
    if f'iwa_pass_{scenario}' in row.index:
        # HWO scenario - use specified scenario for rejection reasons
        if not row[f'iwa_pass_{scenario}']:
            failure_reasons.append('IWA')
        if f'flux_pass_{scenario}' in row.index and not row[f'flux_pass_{scenario}']:
            failure_reasons.append('Flux Ratio')
        elif f'flux_ratio_{scenario}' in row.index and not row[f'flux_ratio_{scenario}']:
            failure_reasons.append('Flux Ratio')
        if not row[f'min_photons_pass_{scenario}']:
            failure_reasons.append('# photons hitting detector')
        # Add exozodi constraint check
        if f'exozodi_pass_{scenario}' in row.index and not row[f'exozodi_pass_{scenario}']:
            failure_reasons.append('Exozodi')
        
        if failure_reasons:
            return ' + '.join(failure_reasons)
        else:
            return 'Detected'
    elif 'iwa_pass' in row.index:
        # Non-HWO scenario - use standard column names
        if not row['iwa_pass']:
            failure_reasons.append('IWA')
        if 'flux_pass' in row.index and not row['flux_pass']:
            failure_reasons.append('Flux Ratio')
        elif 'flux_ratio' in row.index and not row['flux_ratio']:
            failure_reasons.append('Flux Ratio')
        if not row['min_photons_pass']:
            failure_reasons.append('# photons hitting detector')
        # Add exozodi constraint check
        if 'exozodi_pass' in row.index and not row['exozodi_pass']:
            failure_reasons.append('Exozodi')
        
        if failure_reasons:
            return ' + '.join(failure_reasons)
        else:
            return 'Detected'
    else:
        # Fallback - check for any available columns
        available_cols = list(row.index)
        print(f"Warning: No standard detection columns found. Available columns: {available_cols}")
        return 'Unknown'
    

def pivot_stats(df, groupby_cols):
    """
    Compute mean and std of counts by run for arbitrary groupby columns.
    Args:
        df: DataFrame
        groupby_cols: list of str, columns to group by
    Returns:
        DataFrame with groupby_cols, 'count' (mean), and 'error' (std).
    """
    # More efficient approach using value_counts and groupby
    if 'run' in df.columns:
        # Multi-run data: compute statistics across runs
        grouped = df.groupby(groupby_cols + ['run']).size().reset_index(name='count')
        stats = grouped.groupby(groupby_cols).agg({'count': ['mean', 'std']}).reset_index()
        stats.columns = groupby_cols + ['count', 'error']
    else:
        # Single run data: just count
        stats = df.groupby(groupby_cols).size().reset_index(name='count')
        stats['error'] = 0.0
    
    return stats

def prep_plot_df_stars(stats, star_order, bin_labels):
    df = stats.pivot(index='stype', columns='radius_bin', values='count').fillna(0)
    errors = stats.pivot(index='stype', columns='radius_bin', values='error').fillna(0)
    df = df.reindex(star_order).reindex(columns=bin_labels, fill_value=0)
    errors = errors.reindex(star_order).reindex(columns=bin_labels, fill_value=0)
    return df, errors


def bar_plot_with_errors(
    x, heights_list, errors_list, bar_width, labels, colors=None, hatches=None,
    xticks=None, xticklabels=None, ylabel=None, title=None, legend_title=None,
    text_offset=2, filename=None, stacked=False, bottom_list=None, alpha_list=None, show=False, figsize=(12, 8),
    bbox_inches='tight' ):
    """
    Generalized bar plot with error bars and text labels. Can handle grouped or stacked bars.
    Args:
        x: np.array, x positions for the groups.
        heights_list: list of lists/arrays, heights for each group/bar.
        errors_list: list of lists/arrays or None, error values for each group/bar.
        bar_width: float, width of each bar.
        labels: list of str, labels for each group/bar.
        colors: list of str or None, colors for each group/bar.
        hatches: list of str or None, hatches for each group/bar.
        xticks: np.array or None, x-tick positions.
        xticklabels: list of str or None, x-tick labels.
        ylabel: str or None, y-axis label.
        title: str or None, plot title.
        legend_title: str or None, legend title.
        text_offset: float, offset for annotation text above bars.
        filename: str or None, if provided, save the plot to this path.
        stacked: bool, if True, stack bars.
        bottom_list: list of lists/arrays or None, bottoms for stacked bars.
        alpha_list: list of floats or None, alpha values for each group/bar.
        show: bool, if True, display the plot interactively.
        figsize: tuple, figure size (width, height) in inches.
        bbox_inches: str, bbox_inches parameter for plt.savefig.
    Returns:
        fig, ax: matplotlib Figure and Axes objects.
    """
    # Clear any existing figures to prevent multiple axes
    plt.close('all')
    fig, ax = plt.subplots(figsize=figsize)
    n_bars = len(heights_list)
    
    # Pre-allocate arrays for better performance
    all_heights = np.array(heights_list)
    all_errors = np.array(errors_list) if errors_list else None
    
    for i in range(n_bars):
        heights = all_heights[i]
        errors = all_errors[i] if all_errors is not None else None
        color = colors[i] if colors else None
        hatch = hatches[i] if hatches else None
        label = labels[i] if labels else None
        alpha = alpha_list[i] if alpha_list else 1.0
        bottom = bottom_list[i] if (bottom_list and stacked) else None
        bar = ax.bar(x + i * bar_width if not stacked else x,
                     heights, width=bar_width,
                     yerr=errors, label=label,
                     color=color, hatch=hatch, edgecolor='black',
                     bottom=bottom, alpha=alpha)
        
        # Vectorized text annotation for better performance
        if errors is not None:
            err_values = errors
        else:
            err_values = np.zeros_like(heights)
            
        for j, (h, err) in enumerate(zip(heights, err_values)):
            h_display = int(h) if not np.isnan(h) else 0
            err_display = int(err) if not np.isnan(err) else 0
            y_pos = h + (bottom[j] if (bottom is not None and stacked and j < len(bottom)) else 0) + text_offset
            x_pos = (x[j] + i * bar_width) if not stacked else x[j]
            ax.text(x_pos, y_pos, f"{h_display}±{err_display}", ha='center', fontsize=8)
    
    if xticks is not None and xticklabels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend_title:
        ax.legend(title=legend_title)
    else:
        ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches=bbox_inches)
    if show:
        plt.show()
    plt.close(fig)
    return fig, ax

def overlay_best_worst(
    ax, x: np.ndarray, bar_width: float, overlay_lists: List[List[float]],
    colors: List[str], labels: List[str]
) -> None:
    """Overlay best/worst bars on an axis."""
    for i, (overlay, color, label) in enumerate(zip(overlay_lists, colors, labels)):
        xpos = x + i * bar_width
        ax.bar(xpos, overlay, width=bar_width, color=color, label=label if i == 0 else "", edgecolor='black', alpha=0.7 if 'Worst' in label else 0.8)

def output_filename(plot_type: str, name: str, nruns: int, star_catalog: str, suffix: Optional[str] = None) -> str:
    """Centralize output filename formatting."""
    base = f"{plot_type}_{name}_nruns{nruns}_{star_catalog}"
    if suffix:
        base += f"_{suffix}"
    return base + ".png"

def scatter_best_worst_overlay(
    ax,
    x_best, y_best, x_worst, y_worst,
    x_best_not, y_best_not, x_worst_not, y_worst_not,
    color_detected='green', color_not_detected='red',
    alpha_best=0.8, alpha_worst=0.4, alpha_best_not=0.6, alpha_worst_not=0.3,
    label_best='Detected (Best)', label_worst='Detected (Worst)',
    label_best_not='Not Detected (Best)', label_worst_not='Not Detected (Worst)'
):
    """
    Overlay best/worst detected and not detected points on a scatter plot.
    Args:
        ax: matplotlib axis
        x_best, y_best: arrays for detected (best)
        x_worst, y_worst: arrays for detected (worst)
        x_best_not, y_best_not: arrays for not detected (best)
        x_worst_not, y_worst_not: arrays for not detected (worst)
        color_detected, color_not_detected: colors for detected/not detected
        alpha_best, alpha_worst, ...: alpha values for overlays
        label_best, label_worst, ...: legend labels
    """
    if x_best_not is not None and y_best_not is not None:
        ax.scatter(x_best_not, y_best_not, color=color_not_detected, alpha=alpha_best_not, label=label_best_not)
    if x_best is not None and y_best is not None:
        ax.scatter(x_best, y_best, color=color_detected, alpha=alpha_best, label=label_best)
    if x_worst_not is not None and y_worst_not is not None:
        ax.scatter(x_worst_not, y_worst_not, color=color_not_detected, alpha=alpha_worst_not, label=label_worst_not)
    if x_worst is not None and y_worst is not None:
        ax.scatter(x_worst, y_worst, color=color_detected, alpha=alpha_worst, label=label_worst)

def get_detection_masks(df: pd.DataFrame, name: str):
    """
    Return detection masks for detected, detected_best, detected_worst depending on scenario.
    Args:
        df: DataFrame with detection columns
        name: scenario name (e.g. 'HWO')
    Returns:
        mask_best, mask_worst (mask_worst is None if not HWO)
    """
    if name == 'HWO':
        mask_best = df['detected_best'] if 'detected_best' in df else df['detected']
        mask_worst = df['detected_worst'] if 'detected_worst' in df else None
        return mask_best, mask_worst
    else:
        mask = df['detected']
        return mask, None