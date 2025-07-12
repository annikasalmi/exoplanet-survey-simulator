import os
import numpy as np
import matplotlib.pyplot as plt
from plot.base_plotter import BasePlotter
from tools.physics_constants import HWOConstants
import pandas as pd
from tools.plotting_constants import REJECTION_COLUMN_MAPPING, REJECTION_COLORS, REJECTION_SCENARIO_LABELS

plt.rcParams.update({'font.size': 16})

class PlanetRejectionPlotter(BasePlotter):
    """
    Class for generating rejection/failure plots (pie chart and histograms) for planet detection.
    Handles HWO logic for detected_best/detected_worst columns.
    """
    
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize with data and metadata. Allow any name."""
        super().__init__(df, nruns, star_catalog, name)

    def plot_all(self, plot_percentages=True) -> None:
        """Generate all rejection/failure plots."""
        if not self._validate_data():
            return
            
        self.plot_combined_failures()

    def _calculate_rejection_counts(self) -> dict:
        """Calculate rejection counts for each reason across all planets."""
        rejection_counts = {}
        # List of required columns for detailed rejection
        required_cols = [
            'min_photons_pass_best',
            'flux_pass_best',
            'iwa_pass_best',
            'z_pass_best',
            'detected_best',
        ]
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns for rejection analysis: {missing}")
        # Use the individual pass/fail columns
        rejection_counts['# photons hitting detector'] = len(self.df[~self.df['min_photons_pass_best']])
        rejection_counts['Flux Ratio'] = len(self.df[~self.df['flux_pass_best']])
        rejection_counts['IWA'] = len(self.df[~self.df['iwa_pass_best']])
        rejection_counts['Exozodi'] = len(self.df[~self.df['z_pass_best']])
        return rejection_counts

    def _create_rejection_bar_plot(self, ax, rejection_counts, total_planets, colors):
        """Create a bar plot showing rejection reasons."""
        reasons = list(rejection_counts.keys())
        
        # Plot total bar (light gray background)
        ax.bar(reasons, [total_planets] * len(reasons), color='lightgray', alpha=0.7, label='Total planets')
        
        # Plot rejection bars (colored)
        for j, reason in enumerate(reasons):
            count = rejection_counts[reason]
            percentage = (count / total_planets) * 100
            ax.bar(reasons[j], count, color=colors[reason], alpha=0.8, label=f'{reason}: {percentage:.1f}%')
            ax.text(j, count + total_planets * 0.01, f'{percentage:.1f}%', 
                   ha='center', va='bottom', fontsize=16)
        # Set custom x-axis labels for the bar plot
        labels = [r if r != '# photons hitting detector' else '# photons\nhitting detector' for r in reasons]
        ax.set_xticklabels(labels)

    def _add_total_rejection_line(self, ax, total_planets):
        """Add a horizontal line showing total rejection percentage."""
        rejected_planets = len(self.df[~self.df['detected_best']])
        rejection_percentage = (rejected_planets / total_planets) * 100
        ax.axhline(y=rejected_planets, color='red', linestyle='--', alpha=0.8, 
                  label=f'Total rejected: {rejection_percentage:.1f}%')
        return rejected_planets

    def _setup_plot_axes(self, ax, total_planets, rejected_planets):
        """Setup plot axes with titles, labels, and formatting."""
        ax.set_title(f"Total: {total_planets}, Rejected: {rejected_planets}")
        ax.set_ylabel("Number of planets")
        ax.set_ylim(0, total_planets * 1.1)
        ax.legend(fontsize=14, loc='upper right')
        ax.tick_params(axis='x', rotation=0)

    def plot_failures_percentages(self) -> None:
        """Plot side-by-side bar charts of rejection reasons as a fraction of all planets."""
        scenario_labels = REJECTION_SCENARIO_LABELS
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        axs = np.array([axs])
        
        plotted = False
        for i, scenario in enumerate(['best']):
            # Check if we have any data to plot
            if self.df.empty:
                continue
            plotted = True
            
            # Calculate data
            total_planets = len(self.df)
            rejection_counts = self._calculate_rejection_counts()
            
            # Create plot
            self._create_rejection_bar_plot(axs[i], rejection_counts, total_planets, REJECTION_COLORS)
            rejected_planets = self._add_total_rejection_line(axs[i], total_planets)
            self._setup_plot_axes(axs[i], total_planets, rejected_planets)
        
        if plotted:
            plt.suptitle(f"Rejection Reasons for All Planets")
            # Set custom x-axis labels for the bar plot
            ax = axs[0]
            labels = [label if label != '# photons hitting detector' else '# photons\nhitting detector' for label in ax.get_xticklabels()]
            ax.set_xticklabels(labels)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            self._save_plot(fig, 'failure_detected', 'best_case')

    def _get_bins_for_reason(self, reason, df, col):
        """Get appropriate bins for histogram based on rejection reason."""
        if reason == 'Flux Ratio':
            arr = np.asarray(pd.to_numeric(df[col], errors='coerce'))
            min_val = np.nanmin(arr)
            max_val = np.nanmax(arr)
            # Handle edge cases
            if not np.isfinite(min_val) or min_val <= 0:
                min_val = 1e-10
            if not np.isfinite(max_val) or max_val <= 0:
                max_val = 1.0
            if min_val >= max_val:
                min_val = 1e-10
                max_val = 1.0
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
            bins = np.sort(bins)
        elif reason == '# photons hitting detector':
            min_val = 1e-20
            max_val = 10**0.8
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
            bins = np.sort(bins)
        elif reason == 'Exozodi':
            # Use log-spaced bins for exozodi since it has a log x-axis
            arr = np.asarray(pd.to_numeric(df['z'], errors='coerce'))
            min_val = np.nanmin(arr)
            max_val = np.nanmax(arr)
            # Handle edge cases
            if not np.isfinite(min_val) or min_val <= 0:
                min_val = 1e-3
            if not np.isfinite(max_val) or max_val <= 0:
                max_val = 1e3
            if min_val >= max_val:
                min_val = 1e-3
                max_val = 1e3
            bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
            bins = np.sort(bins)
        else:
            bins = 40
        return bins

    def _get_pass_fail_column(self, reason):
        """Get the appropriate pass/fail column name for a given reason."""
        column_mapping = {
            '# photons hitting detector': 'min_photons_pass_best',
            'Flux Ratio': 'flux_pass_best',
            'IWA': 'iwa_pass_best',
            'Exozodi': 'z_pass_best'
        }
        return column_mapping.get(reason)

    def _calculate_rejection_percentage(self, df, reason, pass_col_best):
        """Calculate rejection percentage for a given reason."""
        if reason == 'Exozodi':
            mask_best = (df['z'] <= HWOConstants('best').max_z)
        else:
            mask_best = df[pass_col_best].astype(bool)
        rejected_best = len(df[~mask_best])
        total_planets = len(df)
        return (rejected_best / total_planets) * 100

    def _calculate_worst_case_percentage(self, df, reason):
        """Calculate worst case rejection percentage for any reason with a _worst column."""
        # Map reason to the corresponding _worst column
        column_mapping = {
            '# photons hitting detector': 'min_photons_pass_worst',
            'Flux Ratio': 'flux_pass_worst',
            'IWA': 'iwa_pass_worst',
            'Exozodi': None  # handled separately
        }
        if reason == 'Exozodi':
            mask_worst = (df['z'] <= HWOConstants('worst').max_z)
            rejected_worst = len(df[~mask_worst])
            total_planets = len(df)
            return (rejected_worst / total_planets) * 100
        pass_col_worst = column_mapping.get(reason)
        if pass_col_worst and pass_col_worst in df.columns:
            mask_worst = df[pass_col_worst].astype(bool)
            rejected_worst = len(df[~mask_worst])
            total_planets = len(df)
            return (rejected_worst / total_planets) * 100
        return 0

    def _add_rejection_percentage_text(self, ax, pct_best, pct_worst=None):
        """Add rejection percentage text to the plot."""
        ax.text(0.05, 0.05, f'Best: {pct_best:.1f}% rejected',
                transform=ax.transAxes, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if pct_worst is not None and pct_worst > 0:
            ax.text(0.05, 0.15, f'Worst: {pct_worst:.1f}% rejected',
                    transform=ax.transAxes, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    def _add_threshold_lines(self, ax, reason, hwo_best, hwo_worst):
        """Add threshold lines for best and worst case scenarios."""
        if reason == '# photons hitting detector':
            best_threshold = HWOConstants('best').max_z
            worst_threshold = HWOConstants('worst').max_z
            ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, 
                      label=f'Best case cutoff = {best_threshold:.2e}')
            ax.axvline(worst_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Worst case cutoff = {worst_threshold:.2e}')
        elif reason == 'Exozodi':
            best_threshold = HWOConstants('best').max_z
            worst_threshold = HWOConstants('worst').max_z
            ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, 
                      label=f'{best_threshold:.2e}')
            ax.axvline(worst_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'{worst_threshold:.2e}')
        else:
            threshold_name = {
                '# photons hitting detector': 'min_photons',
                'Flux Ratio': 'min_planet_flux_star_ratio',
                'IWA': 'iwa',
            }[reason]
            best_threshold = getattr(hwo_best, threshold_name)
            worst_threshold = getattr(hwo_worst, threshold_name)
            
            # Add best case threshold lines
            if isinstance(best_threshold, tuple):
                thresh_0 = f'{best_threshold[0]:.2f}' if reason == '# photons hitting detector' else f'{best_threshold[0]:.2e}'
                thresh_1 = f'{best_threshold[1]:.2f}' if reason == '# photons hitting detector' else f'{best_threshold[1]:.2e}'
                ax.axvline(best_threshold[0], color='green', linestyle='--', alpha=0.7, 
                          label=f'{thresh_0}')
                ax.axvline(best_threshold[1], color='green', linestyle=':', alpha=0.7, 
                          label=f'Best case max = {thresh_1}')
            else:
                ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, 
                          label=f'{best_threshold:.2e}')
            
            # Add worst case threshold lines
            if isinstance(worst_threshold, tuple):
                thresh_0 = f'{worst_threshold[0]:.2f}' if reason == '# photons hitting detector' else f'{worst_threshold[0]:.2e}'
                thresh_1 = f'{worst_threshold[1]:.2f}' if reason == '# photons hitting detector' else f'{worst_threshold[1]:.2e}'
                ax.axvline(worst_threshold[0], color='red', linestyle='--', alpha=0.7, 
                          label=f'{thresh_0}')
                ax.axvline(worst_threshold[1], color='red', linestyle=':', alpha=0.7, 
                          label=f'Worst case max = {thresh_1}')
            else:
                ax.axvline(worst_threshold, color='red', linestyle='--', alpha=0.7, 
                          label=f'{worst_threshold:.2e}')

    def _get_edgecolor(self, reason):
        """Return edgecolor for a given rejection reason."""
        return {
            'Flux Ratio': 'red',
            'IWA': 'blue',
            'Exozodi': 'gold',
        }.get(reason, 'black')

    def _get_x_label(self, reason):
        if reason == 'IWA':
            return 'Maximum angular separation'
        elif reason == 'Exozodi':
            return 'z (zodis)'
        elif reason == '# photons hitting detector':
            return '# photons\nhitting detector'
        else:
            return reason.replace('_', ' ').capitalize()

    def _plot_histogram(self, ax, df, reason, col, bins):
        edgecolor = self._get_edgecolor(reason)
        ax.hist(df[col], bins=bins, color='lightgray', alpha=0.7, edgecolor=edgecolor,
                    log=True)

    def _add_rejection_percentages(self, ax, df, reason):
        pass_col_best = self._get_pass_fail_column(reason)
        if pass_col_best in df.columns:
            pct_best = self._calculate_rejection_percentage(df, reason, pass_col_best)
            pct_worst = self._calculate_worst_case_percentage(df, reason)
            self._add_rejection_percentage_text(ax, pct_best, pct_worst)

    def _setup_histogram_axes(self, ax, reason):
        ax.set_xlabel(self._get_x_label(reason))
        ax.set_ylabel("Number of planets")
        ax.set_title(f"{reason}")
        if reason not in ['IWA']:
            ax.set_xscale('log')
        ax.legend(fontsize=14, loc='upper right')

    def plot_failures_histogram(self) -> None:
        """Plot histograms of rejection reasons for non-detected planets."""
        df = self.df.copy()
        if df.empty:
            return
        nrows, ncols = 2, 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 6), sharey=True)
        axs = axs.flatten()
        column_mapping = REJECTION_COLUMN_MAPPING
        for i, (reason, col) in enumerate(column_mapping.items()):
            ax = axs[i]
            if reason != 'Exozodi' and col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
            bins = self._get_bins_for_reason(reason, df, col)
            self._plot_histogram(ax, df, reason, col, bins)
            self._add_rejection_percentages(ax, df, reason)
            hwo_best = HWOConstants('best')
            hwo_worst = HWOConstants('worst')
            self._add_threshold_lines(ax, reason, hwo_best, hwo_worst)
            self._setup_histogram_axes(ax, reason)
        for j in range(i+1, nrows*ncols):
            fig.delaxes(axs[j])
        total_planets = len(df)
        if 'detected_best' in df.columns:
            mask_detected_best = df['detected_best'].astype(bool)
            rejected_best = len(df[~mask_detected_best])
            pct_best = (rejected_best / total_planets) * 100
            suptitle = f"Actual Values vs. Cutoff Thresholds for Planet Rejection\nBest case: {pct_best:.1f}% rejected"
        else:
            suptitle = "Actual Values vs. Cutoff Thresholds for Planet Rejection"
        plt.suptitle(suptitle, fontsize=30)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(fig, 'failure_multipanel', 'actual_values')

    def plot_combined_failures(self) -> None:
        """Create a combined figure with 2x2 histograms on the left and the percentages bar plot on the right, sharing a single title."""
        df = self.df.copy()
        if df.empty:
            return
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.1])
        axs_hist = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                    fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        ax_bar = fig.add_subplot(gs[:, 2])
        column_mapping = REJECTION_COLUMN_MAPPING
        for i, (reason, col) in enumerate(column_mapping.items()):
            ax = axs_hist[i]
            if reason != 'Exozodi' and col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
            bins = self._get_bins_for_reason(reason, df, col)
            self._plot_histogram(ax, df, reason, col, bins)
            self._add_rejection_percentages(ax, df, reason)
            hwo_best = HWOConstants('best')
            hwo_worst = HWOConstants('worst')
            self._add_threshold_lines(ax, reason, hwo_best, hwo_worst)
            self._setup_histogram_axes(ax, reason)
        for j in range(i+1, 4):
            fig.delaxes(axs_hist[j])
        scenario = 'best'
        total_planets = len(df)
        rejection_counts = self._calculate_rejection_counts()
        self._create_rejection_bar_plot(ax_bar, rejection_counts, total_planets, REJECTION_COLORS)
        rejected_planets = self._add_total_rejection_line(ax_bar, total_planets)
        self._setup_plot_axes(ax_bar, total_planets, rejected_planets)
        ax_bar.legend(fontsize=14, loc='upper right')
        if 'detected_best' in df.columns:
            mask_detected_best = df['detected_best'].astype(bool)
            rejected_best = len(df[~mask_detected_best])
            pct_best = (rejected_best / total_planets) * 100
            suptitle = f"Actual Values vs. Cutoff Thresholds for Planet Rejection\nBest case: {pct_best:.1f}% rejected"
        else:
            suptitle = "Actual Values vs. Cutoff Thresholds for Planet Rejection"
        plt.suptitle(suptitle, fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(fig, 'failure_combined', 'combined')