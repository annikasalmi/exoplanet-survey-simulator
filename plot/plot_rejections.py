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

    def plot_all(self) -> None:
        """Generate all rejection/failure plots."""
        self.plot_failures_histogram()
        self.plot_failures_piechart()

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

    def plot_failures_piechart(self) -> None:
        """
        Plot a side-by-side pie chart of rejection reasons for non-detected planets for best and worst case scenarios.
        """
        scenario_labels = {'best': 'Best Case Scenario', 'worst': 'Worst Case Scenario'}
        colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plotted = False
        for i, scenario in enumerate(['best', 'worst']):
            df = self._get_rejection_df(scenario)
            label = scenario_labels[scenario]
            if df.empty:
                axs[i].axis('off')
                continue
            plotted = True
            df['rejection_reason'] = df.apply(get_rejection_reason, axis=1)
            counts = df['rejection_reason'].value_counts()
            axs[i].pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=colors[:len(counts)])
            axs[i].set_title(label + f"\nTotal planets: {len(df)}")
            axs[i].axis('equal')
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
        
        # Add rejection_reason column
        df['rejection_reason'] = df.apply(get_rejection_reason, axis=1)
        
        # If no data, skip
        if df.empty:
            return
            
        # Map rejection reasons to actual column names (without best/worst suffixes)
        column_mapping = {
            'Number of photons hitting detector': 'photon_rate_value_best',
            'Flux Ratio': 'flux_ratio_value_best', 
            'IWA': 'maxangsep'
        }
        
        _, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        
        for ax, (reason, column) in zip(axs, column_mapping.items()):
            # Check if column exists
            if column not in df.columns:
                print(f"Warning: Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
                continue
                
            reason_df = df[df['rejection_reason'] == reason]
            
            # Debug prints
            print(f"Debug {reason}:")
            print(f"  Total planets in df: {len(df)}")
            print(f"  Planets rejected for {reason}: {len(reason_df)}")
            print(f"  Column '{column}' exists: {column in df.columns}")
            if column in df.columns:
                print(f"  Column '{column}' has data: {df[column].notna().sum()} non-null values")
                print(f"  Column '{column}' range: {df[column].min():.2e} to {df[column].max():.2e}")
            
            # Use better binning for wide ranges
            if reason in ['Flux Ratio', 'Min Photons']:
                # Use log-spaced bins for flux ratio and photon rates
                min_val = float(df[column].min())
                max_val = float(df[column].max())
                bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
            else:
                # Use linear bins for IWA
                bins = 40
            
            # Plot histogram of all planets
            ax.hist(df[column], bins=bins, color='lightblue', alpha=0.7, edgecolor='black', 
                   log=True, label='All planets')
            
            # Plot histogram of rejected planets for this reason
            if len(reason_df) > 0:
                print(f"  Plotted histogram for {reason} with {len(reason_df)} rejected points")
            else:
                print(f"  No data to plot for {reason}")
            
            # Draw cutoff lines for both best and worst case scenarios
            hwo_best = HWOConstants('best')
            hwo_worst = HWOConstants('worst')
            
            threshold_name = {
                'Number of photons hitting detector': 'min_photons',
                'Flux Ratio': 'min_planet_flux_star_ratio',
                'IWA': 'iwa',
            }[reason]
            
            best_threshold = getattr(hwo_best, threshold_name)
            worst_threshold = getattr(hwo_worst, threshold_name)
            
            # Plot cutoff lines
            if isinstance(best_threshold, tuple):
                ax.axvline(best_threshold[0], color='blue', linestyle='--', alpha=0.7, 
                          label=f'Best case cutoff = {best_threshold[0]:.2e}')
                ax.axvline(best_threshold[1], color='blue', linestyle=':', alpha=0.7, 
                          label=f'Best case max = {best_threshold[1]:.2e}')
            else:
                ax.axvline(best_threshold, color='blue', linestyle='--', alpha=0.7, 
                          label=f'Best case cutoff = {best_threshold:.2e}')
                
            if isinstance(worst_threshold, tuple):
                ax.axvline(worst_threshold[0], color='orange', linestyle='--', alpha=0.7, 
                          label=f'Worst case cutoff = {worst_threshold[0]:.2e}')
                ax.axvline(worst_threshold[1], color='orange', linestyle=':', alpha=0.7, 
                          label=f'Worst case max = {worst_threshold[1]:.2e}')
            else:
                ax.axvline(worst_threshold, color='orange', linestyle='--', alpha=0.7, 
                          label=f'Worst case cutoff = {worst_threshold:.2e}')
            
            ax.set_title(f"{reason}")
            ax.set_xlabel(reason.replace('_', ' ').capitalize())
            ax.set_ylabel("Number of planets")
            ax.legend(fontsize=8)
            
        suptitle = "Actual Values vs. Cutoff Thresholds for Planet Rejection"
        plt.suptitle(suptitle, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = output_filename('failure_multipanel', self.name, self.nruns, self.star_catalog, 'actual_values')
        plt.savefig(os.path.join(self.data_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()