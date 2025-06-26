import os
import numpy as np
import matplotlib.pyplot as plt
from plot.base_plotter import BasePlotter
from tools.physics_constants import HWOConstants
import pandas as pd
from tools.plotting_constants import REJECTION_COLUMN_MAPPING, REJECTION_COLORS, REJECTION_SCENARIO_LABELS

class PlanetRejectionPlotter(BasePlotter):
    """
    Class for generating rejection/failure plots (pie chart and histograms) for planet detection.
    Handles HWO logic for detected_best/detected_worst columns.
    """
    
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize with data and metadata. Only supports HWO (name == 'HWO')."""
        if name != 'HWO':
            raise ValueError("PlanetRejectionPlotter only supports name == 'HWO'.")
        super().__init__(df, nruns, star_catalog, name)

    def plot_all(self, plot_percentages=True) -> None:
        """Generate all rejection/failure plots."""
        if not self._validate_data():
            return
            
        self.plot_failures_histogram()
        if plot_percentages:
            self.plot_failures_percentages()

    def _calculate_rejection_counts(self) -> dict:
        """Calculate rejection counts for each reason across all planets."""
        rejection_counts = {}
        if self.name == 'HWO':
            rejection_counts['# photons hitting detector'] = len(self.df[~self.df['min_photons_pass_best']])
            rejection_counts['Flux Ratio'] = len(self.df[~self.df['flux_pass_best']])
            rejection_counts['IWA'] = len(self.df[~self.df['iwa_pass_best']])
            z_mask = self.df['z'] > HWOConstants('best').max_z
            rejection_counts['Exozodi'] = len(self.df[z_mask])
        else:
            rejection_counts['Not Detected'] = len(self.df[~self.df['detected_best']])
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
            ax.bar(reasons[j], count, color=colors[j], alpha=0.8, label=f'{reason}: {percentage:.1f}%')
            ax.text(j, count + total_planets * 0.01, f'{percentage:.1f}%', 
                   ha='center', va='bottom', fontsize=10)

    def _add_total_rejection_line(self, ax, total_planets):
        """Add a horizontal line showing total rejection percentage."""
        rejected_planets = len(self.df[~self.df['detected_best']])
        rejection_percentage = (rejected_planets / total_planets) * 100
        ax.axhline(y=rejected_planets, color='red', linestyle='--', alpha=0.8, 
                  label=f'Total rejected: {rejection_percentage:.1f}%')
        return rejected_planets

    def _setup_plot_axes(self, ax, scenario_labels, scenario, total_planets, rejected_planets):
        """Setup plot axes with titles, labels, and formatting."""
        ax.set_title(f"{scenario_labels[scenario]}\nTotal: {total_planets}, Rejected: {rejected_planets}")
        ax.set_ylabel("Number of planets")
        ax.set_ylim(0, total_planets * 1.1)
        ax.legend(fontsize=8, loc='upper right')
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
            self._setup_plot_axes(axs[i], scenario_labels, scenario, total_planets, rejected_planets)
        
        if plotted:
            plt.suptitle(f"Rejection Reasons for All Planets", fontsize=16)
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
        """Calculate worst case rejection percentage."""
        if reason == 'Exozodi':
            mask_worst = (df['z'] <= HWOConstants('worst').max_z)
            rejected_worst = len(df[~mask_worst])
            total_planets = len(df)
            return (rejected_worst / total_planets) * 100
        elif reason in ['# photons hitting detector', 'Flux Ratio']:
            pass_col_worst = 'min_photons_pass_worst' if reason == '# photons hitting detector' else 'flux_pass_worst'
            if pass_col_worst in df.columns:
                mask_worst = df[pass_col_worst].astype(bool)
                rejected_worst = len(df[~mask_worst])
                total_planets = len(df)
                return (rejected_worst / total_planets) * 100
        return 0

    def _add_rejection_percentage_text(self, ax, pct_best, pct_worst=None):
        """Add rejection percentage text to the plot."""
        ax.text(0.05, 0.95, f'Best: {pct_best:.1f}% rejected',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if pct_worst is not None and pct_worst > 0:
            ax.text(0.05, 0.85, f'Worst: {pct_worst:.1f}% rejected',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    def _add_threshold_lines(self, ax, reason, hwo_best, hwo_worst):
        """Add threshold lines for best and worst case scenarios."""
        if reason == 'Exozodi':
            best_threshold = HWOConstants('best').max_z
            worst_threshold = HWOConstants('worst').max_z
            ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, 
                      label=f'Best case cutoff = {best_threshold:.2e}')
            ax.axvline(worst_threshold, color='red', linestyle='--', alpha=0.7, 
                      label=f'Worst case cutoff = {worst_threshold:.2e}')
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
                          label=f'Best case cutoff = {thresh_0}')
                ax.axvline(best_threshold[1], color='green', linestyle=':', alpha=0.7, 
                          label=f'Best case max = {thresh_1}')
            else:
                ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, 
                          label=f'Best case cutoff = {best_threshold:.2e}')
            
            # Add worst case threshold lines
            if isinstance(worst_threshold, tuple):
                thresh_0 = f'{worst_threshold[0]:.2f}' if reason == '# photons hitting detector' else f'{worst_threshold[0]:.2e}'
                thresh_1 = f'{worst_threshold[1]:.2f}' if reason == '# photons hitting detector' else f'{worst_threshold[1]:.2e}'
                ax.axvline(worst_threshold[0], color='red', linestyle='--', alpha=0.7, 
                          label=f'Worst case cutoff = {thresh_0}')
                ax.axvline(worst_threshold[1], color='red', linestyle=':', alpha=0.7, 
                          label=f'Worst case max = {thresh_1}')
            else:
                ax.axvline(worst_threshold, color='red', linestyle='--', alpha=0.7, 
                          label=f'Worst case cutoff = {worst_threshold:.2e}')

    def _setup_histogram_axes(self, ax, reason):
        """Setup histogram axes with appropriate labels and scales."""
        if reason == 'IWA':
            ax.set_xlabel('Maximum angular separation')
        elif reason == 'Exozodi':
            ax.set_xlabel('z (zodis)')
        else:
            ax.set_xlabel(reason.replace('_', ' ').capitalize())
        
        ax.set_ylabel("Number of planets")
        ax.set_title(f"{reason}")
        
        # Set logarithmic x-axis for all except IWA and Exozodi
        if reason not in ['IWA', 'Exozodi']:
            ax.set_xscale('log')
        
        ax.legend(fontsize=8, loc='upper right')

    def plot_failures_histogram(self) -> None:
        """Plot histograms of rejection reasons for non-detected planets."""
        df = self.df.copy()
        if df.empty:
            return
            
        # Setup subplots
        nrows, ncols = 2, 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
        
        column_mapping = REJECTION_COLUMN_MAPPING
        
        for i, (reason, col) in enumerate(column_mapping.items()):
            ax = axs[i]
            
            # Check column existence
            if reason != 'Exozodi' and col not in df.columns:
                print(f"Warning: Column '{col}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
            
            # Get bins and create histogram
            bins = self._get_bins_for_reason(reason, df, col)
            
            if reason == 'IWA':
                ax.hist(df['maxangsep'], bins=bins, color='lightblue', alpha=0.7, edgecolor='black',
                        log=True, label='Maximum angular separation')
            else:
                ax.hist(df[col], bins=bins, color='lightblue', alpha=0.7, edgecolor='black', 
                        log=True, label='Best case')
            
            # Calculate and display rejection percentages
            if reason != 'IWA':
                pass_col_best = self._get_pass_fail_column(reason)
                if pass_col_best in df.columns:
                    pct_best = self._calculate_rejection_percentage(df, reason, pass_col_best)
                    pct_worst = self._calculate_worst_case_percentage(df, reason)
                    self._add_rejection_percentage_text(ax, pct_best, pct_worst)
            
            # Add threshold lines
            hwo_best = HWOConstants('best')
            hwo_worst = HWOConstants('worst')
            self._add_threshold_lines(ax, reason, hwo_best, hwo_worst)
            
            # Setup axes
            self._setup_histogram_axes(ax, reason)
        
        # Remove unused subplots
        for j in range(i+1, nrows*ncols):
            fig.delaxes(axs[j])
        
        # Add overall title
        total_planets = len(df)
        if 'detected_best' in df.columns:
            mask_detected_best = df['detected_best'].astype(bool)
            rejected_best = len(df[~mask_detected_best])
            pct_best = (rejected_best / total_planets) * 100
            suptitle = f"Actual Values vs. Cutoff Thresholds for Planet Rejection\nBest case: {pct_best:.1f}% rejected"
        else:
            suptitle = "Actual Values vs. Cutoff Thresholds for Planet Rejection"
        
        plt.suptitle(suptitle, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self._save_plot(fig, 'failure_multipanel', 'actual_values')