import os
import numpy as np
import matplotlib.pyplot as plt
from plot.helpers import make_output_dir, get_rejection_reason, output_filename
from tools.physics_constants import HWOConstants
import pandas as pd

class PlanetRejectionPlotter:
    """
    Class for generating rejection/failure plots (pie chart and histograms) for planet detection.
    Handles HWO logic for detected_best/detected_worst columns.
    """
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize with data and metadata. Only supports HWO (name == 'HWO')."""
        if name != 'HWO':
            raise ValueError("PlanetRejectionPlotter only supports name == 'HWO'.")
        self.df = df.copy()
        self.nruns = nruns
        self.star_catalog = star_catalog
        self.name = name
        self.data_dir = make_output_dir(name, nruns, star_catalog)

    def plot_all(self, plot_percentages=False) -> None:
        """Generate all rejection/failure plots."""
        self.plot_failures_histogram()
        if plot_percentages:
            self.plot_failures_percentages()

    def _get_rejection_df(self, scenario: str = 'best') -> pd.DataFrame:
        """
        Return a DataFrame filtered for the appropriate detection scenario.
        For HWO, scenario is 'best' or 'worst'.
        """
        df = self.df.copy()
        if scenario == 'best':
            df = df[~df['detected_best']]
        elif scenario == 'worst':
            df = df[~df['detected_worst']]
        else:
            raise ValueError("scenario must be 'best' or 'worst' for HWO")
        return pd.DataFrame(df)

    def plot_failures_percentages(self) -> None:
        """
        Plot side-by-side bar charts of rejection reasons for non-detected planets for best and worst case scenarios.
        Shows total planets as bar height with colored sections for rejection percentages.
        """
        scenario_labels = {'best': 'Best Case Scenario', 'worst': 'Worst Case Scenario'}
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
        
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plotted = False
        
        for i, scenario in enumerate(['best', 'worst']):
            df = self.df.copy()
            label = scenario_labels[scenario]
            
            if df.empty:
                axs[i].axis('off')
                continue
                
            plotted = True
            
            # Get total planets and rejected planets
            total_planets = len(df)
            if scenario == 'best':
                rejected_planets = len(df[~df['detected_best']])
            else:
                rejected_planets = len(df[~df['detected_worst']])
            
            # Get rejection reasons for rejected planets
            rejected_df = df[~df[f'detected_{scenario}']] if f'detected_{scenario}' in df.columns else pd.DataFrame()
            if not rejected_df.empty:
                rejected_df['rejection_reason'] = rejected_df.apply(lambda row: get_rejection_reason(row, scenario), axis=1)
                
                # Handle combined rejection reasons (e.g., "IWA + Flux Ratio")
                all_reasons = []
                for reason in rejected_df['rejection_reason']:
                    if ' + ' in reason:
                        # Split combined reasons and add each individually
                        all_reasons.extend(reason.split(' + '))
                    else:
                        all_reasons.append(reason)
                
                # Count each individual reason
                rejection_counts = pd.Series(all_reasons).value_counts()
            else:
                rejection_counts = pd.Series()
            
            # Create bar plot
            reasons = ['# photons hitting detector', 'Flux Ratio', 'IWA', 'Exozodi']
            bar_height = total_planets
            
            # Plot the total bar (light gray)
            axs[i].bar(reasons, [bar_height] * 4, color='lightgray', alpha=0.7, label='Total planets')
            
            # Plot rejection sections (colored)
            bottom = np.zeros(4)
            for j, reason in enumerate(reasons):
                if reason in rejection_counts.index:
                    count = rejection_counts[reason]
                    percentage = (count / total_planets) * 100
                    axs[i].bar(reasons[j], count, bottom=bottom[j], color=colors[j], 
                              alpha=0.8, label=f'{reason}: {percentage:.1f}%')
                    bottom[j] += count
            
            # Add percentage line in the middle
            rejection_percentage = (rejected_planets / total_planets) * 100
            axs[i].axhline(y=rejected_planets, color='red', linestyle='--', alpha=0.8, 
                          label=f'Total rejected: {rejection_percentage:.1f}%')
            
            axs[i].set_title(f"{label}\nTotal: {total_planets}, Rejected: {rejected_planets}")
            axs[i].set_ylabel("Number of planets")
            axs[i].set_ylim(0, total_planets * 1.1)
            axs[i].legend(fontsize=8, loc='upper right')
            
            # Rotate x-axis labels for better readability
            axs[i].tick_params(axis='x', rotation=0)
            
        if plotted:
            plt.suptitle(f"Rejection Reasons for Non-Detected Planets (Best vs. Worst Case)", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.93])
            filename = output_filename('failure_detected', self.name, self.nruns, self.star_catalog, 'best_vs_worst')
            plt.savefig(os.path.join(self.data_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_failures_histogram(self) -> None:
        """
        Plot histograms of rejection reasons for non-detected planets.
        Shows actual values with cutoff lines overlaid.
        """
        # Get the main dataframe with actual values
        df = self.df.copy()
        
        # If no data, skip
        if df.empty:
            return
            
        # Map rejection reasons to actual column names (without best/worst suffixes)
        column_mapping = {
            '# photons hitting detector': 'photon_rate_value_best',
            'Flux Ratio': 'flux_ratio_value_best', 
            'IWA': 'maxangsep',
            'Exozodi': 'exozodi_surface_brightness_ratio_best'
        }
        
        _, axs = plt.subplots(1, 4, figsize=(24, 5), sharey=True)
        
        for ax, (reason, column) in zip(axs, column_mapping.items()):
            # Check if column exists
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
            
            # Use better binning for wide ranges
            if reason in ['Flux Ratio', 'Min Photons', 'Exozodi']:
                # Use log-spaced bins for flux ratio, photon rates, and exozodi surface brightness ratios
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
            else:
                # Use linear bins for IWA
                bins = 40
            
            # Plot histogram of all planets
            ax.hist(df[column], bins=bins, color='lightblue', alpha=0.7, edgecolor='black', 
                   log=True, label='All planets')
            
            # Calculate and display rejection percentages
            total_planets = len(df)
            
            # Get the appropriate pass/fail columns based on the reason
            if reason == '# photons hitting detector':
                pass_col_best = 'min_photons_pass_best'
                pass_col_worst = 'min_photons_pass_worst'
            elif reason == 'Flux Ratio':
                pass_col_best = 'flux_pass_best'
                pass_col_worst = 'flux_pass_worst'
            elif reason == 'IWA':
                pass_col_best = 'iwa_pass_best'
                pass_col_worst = 'iwa_pass_worst'
            elif reason == 'Exozodi':
                pass_col_best = 'exozodi_pass_best'
                pass_col_worst = 'exozodi_pass_worst'
            
            # Calculate rejection percentages
            if pass_col_best in df.columns and pass_col_worst in df.columns:
                rejected_best = len(df[~df[pass_col_best]])
                rejected_worst = len(df[~df[pass_col_worst]])
                pct_best = (rejected_best / total_planets) * 100
                pct_worst = (rejected_worst / total_planets) * 100
                
                # Add text annotation with rejection percentages
                ax.text(0.05, 0.95, f'Best case: {pct_best:.1f}% rejected\nWorst case: {pct_worst:.1f}% rejected', 
                       transform=ax.transAxes, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Draw cutoff lines for both best and worst case scenarios
            hwo_best = HWOConstants('best')
            hwo_worst = HWOConstants('worst')
            
            # Handle exozodi threshold (fixed at 1.0)
            if reason == 'Exozodi':
                best_threshold = 1.0
                worst_threshold = 1.0
            else:
                threshold_name = {
                    '# photons hitting detector': 'min_photons',
                    'Flux Ratio': 'min_planet_flux_star_ratio',
                    'IWA': 'iwa',
                }[reason]
                best_threshold = getattr(hwo_best, threshold_name)
                worst_threshold = getattr(hwo_worst, threshold_name)
            
            # Plot cutoff lines
            if isinstance(best_threshold, tuple):
                if reason == '# photons hitting detector':
                    thresh_0 = f'{best_threshold[0]:.2f}'
                    thresh_1 = f'{best_threshold[1]:.2f}'
                else:
                    thresh_0 = f'{best_threshold[0]:.2e}'
                    thresh_1 = f'{best_threshold[1]:.2e}'
                ax.axvline(best_threshold[0], color='green', linestyle='--', alpha=0.7, 
                          label=f'Best case cutoff = {thresh_0}')
                ax.axvline(best_threshold[1], color='green', linestyle=':', alpha=0.7, 
                          label=f'Best case max = {thresh_1}')
            else:
                ax.axvline(best_threshold, color='green', linestyle='--', alpha=0.7, 
                          label=f'Best case cutoff = {best_threshold:.2e}')
                
            if isinstance(worst_threshold, tuple):
                if reason == '# photons hitting detector':
                    thresh_0 = f'{worst_threshold[0]:.2f}'
                    thresh_1 = f'{worst_threshold[1]:.2f}'
                else:
                    thresh_0 = f'{worst_threshold[0]:.2e}'
                    thresh_1 = f'{worst_threshold[1]:.2e}'
                ax.axvline(worst_threshold[0], color='red', linestyle='--', alpha=0.7, 
                          label=f'Worst case cutoff = {thresh_0}')
                ax.axvline(worst_threshold[1], color='red', linestyle=':', alpha=0.7, 
                          label=f'Worst case max = {thresh_1}')
            else:
                ax.axvline(worst_threshold, color='red', linestyle='--', alpha=0.7, 
                          label=f'Worst case cutoff = {worst_threshold:.2e}')
            
            # Set logarithmic x-axis for flux ratio
            if reason == 'IWA':
                pass
            else:
                ax.set_xscale('log')
            
            ax.set_title(f"{reason}")
            ax.set_xlabel(reason.replace('_', ' ').capitalize())
            ax.set_ylabel("Number of planets")
            ax.legend(fontsize=8, loc='upper right')
            
        # Calculate overall rejection percentages for subtitle
        total_planets = len(df)
        if 'detected_best' in df.columns and 'detected_worst' in df.columns:
            rejected_best = len(df[~df['detected_best']])
            rejected_worst = len(df[~df['detected_worst']])
            pct_best = (rejected_best / total_planets) * 100
            pct_worst = (rejected_worst / total_planets) * 100
            suptitle = f"Actual Values vs. Cutoff Thresholds for Planet Rejection\nBest case: {pct_best:.1f}% rejected, Worst case: {pct_worst:.1f}% rejected"
        else:
            suptitle = "Actual Values vs. Cutoff Thresholds for Planet Rejection"
        
        plt.suptitle(suptitle, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = output_filename('failure_multipanel', self.name, self.nruns, self.star_catalog, 'actual_values')
        plt.savefig(os.path.join(self.data_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()