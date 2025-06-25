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

    def plot_all(self, plot_percentages=True) -> None:
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
            
            # Calculate Exozodi (z) rejection count and add to rejection_counts if not present
            if scenario == 'best':
                z_mask = df['z'] > HWOConstants('best').max_z
            else:
                z_mask = df['z'] > HWOConstants('worst').max_z
            exozodi_count = z_mask.sum()
            if 'Exozodi' not in rejection_counts:
                rejection_counts['Exozodi'] = exozodi_count
            
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
            
        # 2x2 subplot: best & worst for all except Exozodi (only best)
        nrows, ncols = 2, 2
        fig, axs = plt.subplots(nrows, ncols, figsize=(12, 10), sharey=True)
        axs = axs.flatten()
        
        column_mapping = {
            '# photons hitting detector': ('photon_rate_value_best', 'photon_rate_value_worst'),
            'Flux Ratio': ('flux_ratio_value_best', 'flux_ratio_value_worst'),
            'IWA': ('maxangsep', 'maxangsep'),
            'Exozodi': ('z', None),  # Now mapping z for Exozodi
        }
        
        for i, (reason, (col_best, col_worst)) in enumerate(column_mapping.items()):
            ax = axs[i]
            # Only check for column existence for reasons other than Exozodi
            if reason != 'Exozodi' and col_best not in df.columns:
                print(f"Warning: Column '{col_best}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
            
            # Use better binning for wide ranges
            if reason in ['Flux Ratio', 'Min Photons']:
                arr = np.asarray(pd.to_numeric(df[col_best], errors='coerce'))
                min_val = np.nanmin(arr)
                max_val = np.nanmax(arr)
                # If not finite or <=0, set to a small positive value or default
                if not np.isfinite(min_val) or min_val <= 0:
                    min_val = 1e-10
                if not np.isfinite(max_val) or max_val <= 0:
                    max_val = 1.0
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
            else:
                bins = 40
            
            if reason == 'IWA':
                # Only plot the distribution of maxangsep, no best/worst bars
                ax.hist(df['maxangsep'], bins=bins, color='lightblue', alpha=0.7, edgecolor='black',
                        log=True, label='Maximum angular separation')
            else:
                # Plot best case
                ax.hist(df[col_best], bins=bins, color='lightblue', alpha=0.7, edgecolor='black', 
                        log=True, label='Best case')
                # Plot worst case if applicable (not for Exozodi)
                if col_worst and col_worst in df.columns:
                    ax.hist(df[col_worst], bins=bins, color='orange', alpha=0.5, edgecolor='black', 
                            log=True, label='Worst case')
            
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
                pass_col_best = 'z_pass_best'
                pass_col_worst = 'z_pass_worst'
            
            # Calculate rejection percentages (skip for IWA)
            if reason != 'IWA':
                if pass_col_best in df.columns:
                    if reason == 'Exozodi':
                        mask_best = (df['z'] <= HWOConstants('best').max_z)
                    else:
                        mask_best = df[pass_col_best].astype(bool)
                    rejected_best = len(df[~mask_best])
                    pct_best = (rejected_best / total_planets) * 100
                    ax.text(0.05, 0.95, f'Best: {pct_best:.1f}% rejected',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                if col_worst and pass_col_worst and pass_col_worst in df.columns:
                    if reason == 'Exozodi':
                        mask_worst = (df['z'] <= HWOConstants('worst').max_z)
                    else:
                        mask_worst = df[pass_col_worst].astype(bool)
                    rejected_worst = len(df[~mask_worst])
                    pct_worst = (rejected_worst / total_planets) * 100
                    ax.text(0.05, 0.85, f'Worst: {pct_worst:.1f}% rejected',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
            
            # Draw cutoff lines for best and worst case scenario
            hwo_best = HWOConstants('best')
            hwo_worst = HWOConstants('worst')
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
            # Set logarithmic x-axis for all except IWA and Exozodi
            if reason == 'IWA':
                pass
            elif reason == 'Exozodi':
                pass  # Do not set log scale for Exozodi
            else:
                ax.set_xscale('log')
            ax.set_title(f"{reason}")
            if reason == 'IWA':
                ax.set_xlabel('Maximum angular separation')
            elif reason == 'Exozodi':
                ax.set_xlabel('z (zodis)')
            else:
                ax.set_xlabel(reason.replace('_', ' ').capitalize())
            ax.set_ylabel("Number of planets")
            ax.legend(fontsize=8, loc='upper right')
        # Remove any unused subplots (if fewer than 4)
        for j in range(i+1, nrows*ncols):
            fig.delaxes(axs[j])
        # Calculate overall rejection percentages for subtitle
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
        filename = output_filename('failure_multipanel', self.name, self.nruns, self.star_catalog, 'actual_values')
        plt.savefig(os.path.join(self.data_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()