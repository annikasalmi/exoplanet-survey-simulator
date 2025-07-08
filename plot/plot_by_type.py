import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.base_plotter import BasePlotter
from tools.plotting_constants import (
    STAR_ORDER, TEMP_ZONES, DISTANCE_LABELS,
    TEMP_COLORS,  BAR_WIDTH_TEMP, BAR_WIDTH_DIST, HATCHES,
    STAR_COLORS, STAR_HATCHES, BIN_LABELS, BAR_WIDTH_STAR
)

plt.rcParams.update({'font.size': 16})

class PlotPlanetType(BasePlotter):
    """
    Class for generating grouped bar plots of planet statistics by star type, planet type, and distance.
    Handles best/worst case overlays for HWO scenarios.
    """
    
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize the plotter with data and metadata."""
        super().__init__(df, nruns, star_catalog, name)
        # Ensure temp_zone column is present for all plotting methods that need it
        if 'temp_zone' not in self.df.columns:
            self.df['temp_zone'] = self.df['temp_p'].apply(self._temp_zone)

    def _temp_zone(self, temp):
        """Assigns a temperature zone based on the temperature value."""
        if temp > 390:
            return 'hot'
        elif 390 > temp > 270:
            return 'habitable'
        else:
            return 'cold'

    def plot_all(self) -> None:
        """Generate all planet type plots."""
        if not self._validate_data():
            return
            
        self.plot_by_star()
        self.plot_by_planet()
        self.plot_by_distance()

    def _assign_category(self, row):
        """
        Assign planet categories based on radius, temperature, and star type.
        Returns a list of applicable categories for each planet.
        """
        categories = []
        r = row['radius_p']
        temp = row['temp_p']
        stype = row['stype']
        
        # Radius-based categories
        if r < 1.5:
            categories.append('Rocky')
            # Star type categories for rocky planets
            if stype == 'M':
                categories.append('Rocky, M stars')
            elif stype in ['G', 'K']:
                categories.append('Rocky, G and K stars')
        elif r < 2.0:
            categories.append('Super-Earths')
        elif r < 4.0:
            categories.append('Sub-Neptunes')
        elif r < 8.0:
            categories.append('Sub-Jovians')
        
        return categories if categories else None

    def plot_by_star(self) -> None:
        """Grouped bar plots by star type and radius bin. For HWO, uses detected_best/worst logic."""
        df = self.df.copy()
        x = np.arange(len(STAR_ORDER))
        
        # Add radius bins and ensure 'Rocky HZ' is a category
        df['radius_bin'] = pd.cut(df['radius_p'], 
                                 bins=[0, 1.5, 3.0, 6.0, float('inf')], 
                                 labels=['<1.5', '1.5–3.0', '3.0–6.0', '>6.0'],
                                 include_lowest=True)
        df['radius_bin'] = df['radius_bin'].cat.add_categories(['Rocky HZ'])
        
        # Filter for rocky habitable zone planets and assign them to 'Rocky HZ' bin
        rocky_hz_mask = (df['habitable'] == True) & (df['radius_p'] < 1.5)
        df.loc[rocky_hz_mask, 'radius_bin'] = 'Rocky HZ'
        
        # Only keep bins in BIN_LABELS (removes '>6.0')
        df = df[df['radius_bin'].isin(BIN_LABELS)]
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]
        detected_df = df[mask_best].copy()
        
        # Calculate statistics
        detected_per_run = detected_df.groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
        detected_pivot = detected_per_run.pivot_table(index=['stype', 'radius_bin'], 
                                                   columns='run', values='count', fill_value=0)
        detected_mean = detected_pivot.mean(axis=1).reset_index(name='count')
        detected_std = detected_pivot.std(axis=1).reset_index(name='count')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = BAR_WIDTH_STAR
        
        # Pre-compute data matrix
        data_matrix = np.zeros((len(BIN_LABELS), len(STAR_ORDER)))
        error_matrix = np.zeros((len(BIN_LABELS), len(STAR_ORDER)))
        
        for i, bin_label in enumerate(BIN_LABELS):
            bin_data = detected_mean[detected_mean['radius_bin'] == bin_label]
            bin_std = detected_std[detected_std['radius_bin'] == bin_label]
            
            for j, star in enumerate(STAR_ORDER):
                star_data = bin_data[bin_data['stype'] == star]
                star_std = bin_std[bin_std['stype'] == star]
                
                data_matrix[i, j] = star_data['count'].iloc[0] if len(star_data) > 0 else 0
                error_matrix[i, j] = star_std['count'].iloc[0] if len(star_std) > 0 else 0
        
        # Plot bars
        for i, bin_label in enumerate(BIN_LABELS):
            heights = data_matrix[i, :]
            errors = error_matrix[i, :]
            
            ax.bar(x + i * bar_width, heights, width=bar_width, color=STAR_COLORS[i], 
                   alpha=0.8, edgecolor='black', yerr=errors, capsize=3, 
                   ecolor='black', hatch=STAR_HATCHES[i], label=bin_label)
            
            # Add error labels
            for j, (height, error) in enumerate(zip(heights, errors)):
                if height > 0 and not np.isnan(height) and not np.isnan(error):
                    label_text = f'{int(round(height))}±{int(round(error))}'
                    ax.text(x[j] + i * bar_width, height + error + 0.5, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Setup plot
        ax.set_xlabel('Star Type')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Star Type for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.legend(title='Radius Bin')
        ax.set_ylim(bottom=0)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'stellar_type_overlay')

    def plot_by_planet(self) -> None:
        """Planet type by subcategory: detected planets only, error bars, integer labels."""
        df = self.df.copy()
        
        # Vectorized category assignment
        conditions = [
            (df['radius_p'] >= 0.5) & (df['radius_p'] < 1.4) & (df['habitable'] == True),
            (df['radius_p'] >= 1.0) & (df['radius_p'] < 1.4),
            (df['radius_p'] >= 1.4) & (df['radius_p'] < 2.6),
            (df['radius_p'] >= 2.6) & (df['radius_p'] < 4.0)
        ]
        choices = ['Rocky eHZ', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df['categories'] = np.select(conditions, choices, default='')
        
        # Filter for valid categories
        categories = ['Rocky eHZ', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df = df[df['categories'].isin(categories)]
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]
        df_detected = df[mask_best].copy()
        
        # Calculate stats
        detected_stats = self._pivot_stats(df_detected, ['categories', 'temp_zone'])
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        bar_width = 0.2
        
        # Pre-compute data
        bar_data = {}
        for cat in categories:
            cat_data = detected_stats[detected_stats['categories'] == cat]
            bar_data[cat] = {'count': cat_data['count'].sum(), 'error': cat_data['error'].sum()}
        
        # Plot temperature-separated categories
        for i_cat, cat in enumerate(categories):
            idx = i_cat
            for j, (temp_zone, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
                cat_temp_data = detected_stats[(detected_stats['categories'] == cat) & 
                                             (detected_stats['temp_zone'] == temp_zone)]
                detected = round(cat_temp_data['count'].iloc[0]) if len(cat_temp_data) > 0 else 0
                detected_error = cat_temp_data['error'].iloc[0] if len(cat_temp_data) > 0 else 0
                
                ax.bar(x[idx] + j * bar_width, detected, width=bar_width, color=color, 
                       edgecolor='black', alpha=0.8, hatch=hatch, yerr=detected_error, 
                       capsize=3, ecolor='black',
                       label=f'{cat} ({temp_zone})' if i_cat == 0 else None)
        
        # Setup plot
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Subcategory for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        
        # Add error bar labels
        for i_cat, cat in enumerate(categories):
            idx = i_cat
            for j, (temp_zone, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
                cat_temp_data = detected_stats[(detected_stats['categories'] == cat) & 
                                             (detected_stats['temp_zone'] == temp_zone)]
                detected = round(cat_temp_data['count'].iloc[0]) if len(cat_temp_data) > 0 else 0
                detected_error = cat_temp_data['error'].iloc[0] if len(cat_temp_data) > 0 else 0
                
                if detected > 0 and not np.isnan(detected) and not np.isnan(detected_error):
                    label_text = f'{detected}±{round(detected_error)}'
                    ax.text(x[idx] + j * bar_width, detected + detected_error + 0.5, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Finalize plot
        ax.legend(title='Temperature Zone', loc='upper right')
        ax.set_ylim(bottom=0)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_type_subcategory')

    def plot_by_distance(self) -> None:
        """Distance-based analysis."""
        df = self.df.copy()
        
        # Create distance bins
        df['distance_bin'] = pd.cut(df['distance_s'], 
                                   bins=[0, 5, 10, 15, 20, float('inf')], 
                                   labels=['0-5', '5-10', '10-15', '15-20', '>20'], 
                                   include_lowest=True)
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]
        detected_df = df[mask_best].copy()
        
        # Calculate statistics
        detected_per_run = detected_df.groupby(['run', 'distance_bin']).size().reset_index(name='count')
        detected_pivot = detected_per_run.pivot_table(index='distance_bin', 
                                                   columns='run', values='count', fill_value=0)
        detected_mean = detected_pivot.mean(axis=1).reset_index(name='count')
        detected_std = detected_pivot.std(axis=1).reset_index(name='count')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(detected_mean))
        
        heights = detected_mean['count'].values
        errors = detected_std['count'].values
        
        ax.bar(x, heights, width=0.8, color='green', alpha=0.8, 
               edgecolor='black', yerr=errors, capsize=3, ecolor='black')
        
        # Add error labels
        for i, (height, error) in enumerate(zip(heights, errors)):
            if height > 0 and not np.isnan(height) and not np.isnan(error):
                label_text = f'{int(round(height))}±{int(round(error))}'
                ax.text(x[i], height + error + 0.5, label_text, 
                       ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Setup plot
        ax.set_xlabel('Distance [pc]')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Distance for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x)
        ax.set_xticklabels(detected_mean['distance_bin'].values)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        self._save_plot(fig, 'distance_analysis')

    def _pivot_stats(self, df, groupby_cols):
        """Calculate statistics for grouped data."""
        if df.empty:
            return pd.DataFrame(columns=groupby_cols + ['count', 'error'])
        
        # Group by specified columns and calculate statistics
        grouped = df.groupby(groupby_cols + ['run']).size().reset_index(name='count')
        pivot_table = grouped.pivot_table(index=groupby_cols, columns='run', values='count', fill_value=0)
        
        # Calculate mean and standard deviation
        mean_counts = pivot_table.mean(axis=1).reset_index(name='count')
        std_counts = pivot_table.std(axis=1).reset_index(name='error')
        
        # Merge mean and std
        result = mean_counts.merge(std_counts, on=groupby_cols)
        return result
