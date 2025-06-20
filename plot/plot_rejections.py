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
        Overlays best and worst case scenarios in the same plot for each reason.
        """
        # Prepare data for both scenarios
        dfs = {
            'Best': self._get_rejection_df('best'),
            'Worst': self._get_rejection_df('worst')
        }
        colors = {'Best': '#1f77b4', 'Worst': '#ff7f0e'}
        hwo_consts = {
            'Best': HWOConstants('best'),
            'Worst': HWOConstants('worst')
        }
        # Map rejection reasons to actual column names with suffixes
        column_mapping = {
            'Min Photons': {
                'Best': 'photon_rate_value_best',
                'Worst': 'photon_rate_value_worst'
            },
            'Flux Ratio': {
                'Best': 'flux_ratio_value_best', 
                'Worst': 'flux_ratio_value_worst'
            },
            'IWA': {
                'Best': 'maxangsep',  # Fixed: use maxangsep instead of angsep
                'Worst': 'maxangsep'
            }
        }
        # Add rejection_reason column to both dfs
        for key in dfs:
            if not dfs[key].empty:
                dfs[key] = dfs[key].copy()
                dfs[key]['rejection_reason'] = dfs[key].apply(get_rejection_reason, axis=1)
        # If both are empty, skip
        if all(df.empty for df in dfs.values()):
            return
        fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
        for ax, (reason, scenario_columns) in zip(axs, column_mapping.items()):
            for scenario in ['Best', 'Worst']:
                df = dfs[scenario]
                if df.empty:
                    continue
                column = scenario_columns[scenario]
                # Check if column exists
                if column not in df.columns:
                    print(f"Warning: Column '{column}' not found in DataFrame. Available columns: {list(df.columns)}")
                    continue
                reason_df = df[df['rejection_reason'] == reason]
                # Debug prints
                print(f"Debug {reason} - {scenario}:")
                print(f"  Total planets in rejection df: {len(df)}")
                print(f"  Planets rejected for {reason}: {len(reason_df)}")
                print(f"  Column '{column}' exists: {column in df.columns}")
                if column in df.columns:
                    print(f"  Column '{column}' has data: {df[column].notna().sum()} non-null values")
                    print(f"  Column '{column}' range: {df[column].min():.2e} to {df[column].max():.2e}")
                
                # Use better binning for wide ranges
                if reason == 'Flux Ratio':
                    # Use log-spaced bins for flux ratio
                    min_val = float(df[column].min())
                    max_val = float(df[column].max())
                    bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
                elif reason == 'Min Photons':
                    # Use log-spaced bins for photon rates
                    min_val = float(df[column].min())
                    max_val = float(df[column].max())
                    bins = np.logspace(np.log10(min_val), np.log10(max_val), 40)
                else:
                    # Use linear bins for IWA
                    bins = 40
                
                # Plot histograms
                ax.hist(df[column], bins=bins, color=colors[scenario], alpha=0.5, edgecolor='black', log=True, label=f'{scenario} (all)')
                if len(reason_df) > 0:
                    ax.hist(reason_df[column], bins=bins, color=colors[scenario], alpha=0.9, edgecolor='black', log=True, label=f'{scenario} ({reason})', histtype='step')
                    print(f"  Plotted histogram for {reason} - {scenario} with {len(reason_df)} points")
                else:
                    print(f"  No data to plot for {reason} - {scenario}")
                
                # Draw cutoffs for each scenario
                threshold = getattr(hwo_consts[scenario], {
                    'Min Photons': 'min_photons',
                    'Flux Ratio': 'min_planet_flux_star_ratio',
                    'IWA': 'iwa',
                }[reason])
                if isinstance(threshold, tuple):
                    ax.axvline(threshold[0], color=colors[scenario], linestyle='--', label=f'{scenario} Min cutoff = {threshold[0]:.2e}')
                    ax.axvline(threshold[1], color=colors[scenario], linestyle=':', label=f'{scenario} Max cutoff = {threshold[1]:.2e}')
                else:
                    ax.axvline(threshold, color=colors[scenario], linestyle='--', label=f'{scenario} Cutoff = {threshold:.2e}')
            ax.set_title(f"{reason}")
            ax.set_xlabel(reason.replace('_', ' ').capitalize())
            ax.set_ylabel("Number of planets")
            ax.legend(fontsize=8)
        suptitle = f"Reasons for Planet Non-Detection (Best vs. Worst Case)"
        plt.suptitle(suptitle, fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        filename = output_filename('failure_multipanel', self.name, self.nruns, self.star_catalog, 'best_vs_worst')
        plt.savefig(os.path.join(self.data_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()