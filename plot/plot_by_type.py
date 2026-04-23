import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.base_plotter import BasePlotter
from tools.plotting_constants import (
    STAR_ORDER, TEMP_ZONES,
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

    def _calculate_star_stats(self, df):
        """Calculate statistics for star-based plots."""
        # Calculate statistics
        stats = self._pivot_stats(df, ['stype', 'temp_zone'])
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]  # Align mask with filtered df
        df_detected = df[mask_best].copy()
        
        # Calculate detected statistics
        detected_stats = self._pivot_stats(df_detected, ['stype', 'temp_zone'])
        
        return stats, detected_stats

    def _get_star_data_for_bin(self, stats, detected_stats, bin_label):
        """Get star data for a specific bin."""
        bin_stats = stats[stats['temp_zone'] == bin_label]
        bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
        
        # Align data
        bin_stats = bin_stats.set_index('stype').reindex(STAR_ORDER).fillna(0)
        bin_detected = bin_detected.set_index('stype').reindex(STAR_ORDER).fillna(0)
        
        return bin_stats['count'].values, bin_stats['error'].values, bin_detected['count'].values, bin_detected['error'].values

    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize the plotter with data and metadata."""
        super().__init__(df, nruns, star_catalog, name)
        # Ensure temp_zone column is present for all plotting methods that need it
        self.df['temp_zone'] = self.df['temp_p'].apply(self._temp_zone)

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
        
        # Drop any existing radius_bin column to avoid contamination from input data
        if 'radius_bin' in df.columns:
            df = df.drop(columns=['radius_bin'])
        # Always recalculate radius_bin from scratch
        df['radius_bin'] = pd.cut(
            df['radius_p'],
            bins=[0, 1.5, 3.0, 6.0, float('inf')],
            labels=['<1.5', '1.5–3.0', '3.0–6.0', '>6.0'],
            include_lowest=True
        )
        df['radius_bin'] = df['radius_bin'].cat.add_categories(['Rocky HZ'])
        
        # Assign Rocky HZ as before
        rocky_hz_mask = (df['habitable'] == True) & (df['radius_p'] < 1.5)
        df.loc[rocky_hz_mask, 'radius_bin'] = 'Rocky HZ'
        
        # Only keep bins in BIN_LABELS (removes '>6.0')
        df = df[df['radius_bin'].isin(BIN_LABELS)]
        
        # Get detection mask BEFORE any further filtering to ensure alignment
        mask_best, _ = self._get_detection_masks()
        if hasattr(mask_best, 'reindex'):
            mask_best = mask_best.reindex(self.df.index, fill_value=False)
        detected_df = self.df[mask_best]

        # Total stats - simple groupby and sum across runs
        total_per_run = self.df.groupby(['run', 'stype', 'radius_bin']).size().reset_index()
        total_per_run.columns = ['run', 'stype', 'radius_bin', 'count']
        total_pivot = total_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        total_mean = total_pivot.mean(axis=1)
        if isinstance(total_mean, pd.Series):
            total_mean = total_mean.to_frame('count').reset_index()
        total_std = total_pivot.std(axis=1)
        if isinstance(total_std, pd.Series):
            total_std = total_std.to_frame('count').reset_index()

        # Detected stats - use the mask to filter the DataFrame
        detected_per_run = detected_df.groupby(['run', 'stype', 'radius_bin']).size().reset_index()
        detected_per_run.columns = ['run', 'stype', 'radius_bin', 'count']
        detected_pivot = detected_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        detected_mean = detected_pivot.mean(axis=1)
        if isinstance(detected_mean, pd.Series):
            detected_mean = detected_mean.to_frame('count').reset_index()
        detected_std = detected_pivot.std(axis=1)
        if isinstance(detected_std, pd.Series):
            detected_std = detected_std.to_frame('count').reset_index()
        
        # Create the overlay plot
        fig, ax = plt.subplots(figsize=(8,6))
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
        if self.name == 'HWO_exoplanets':
            name = '\nHWO detection ability of exoplanet catalog'
        elif self.name == 'LIFEsim_exoplanets':
            name = '\nLIFEsim detection ability of exoplanet catalog'
        else:
            name = self.name
        ax.set_title(f'Planet Detection by Star Type for {name}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.legend(title='Radius Bin')
        ax.set_ylim(bottom=0)
        
        def get_count(df, star=None, bin_label=None):
            if not isinstance(df, pd.DataFrame):
                return 0
            if star is not None and bin_label is not None:
                filtered = df[(df['radius_bin'] == bin_label) & (df['stype'] == star)] if 'radius_bin' in df.columns and 'stype' in df.columns else pd.DataFrame()
            elif star is not None:
                filtered = df[df['stype'] == star] if 'stype' in df.columns else pd.DataFrame()
            else:
                filtered = df
            return filtered['count'].iloc[0] if isinstance(filtered, pd.DataFrame) and not filtered.empty and 'count' in filtered.columns else 0

        # Add percentage labels
        for i, bin_label in enumerate(BIN_LABELS):
            bin_data = total_mean[total_mean['radius_bin'] == bin_label] if isinstance(total_mean, pd.DataFrame) else pd.DataFrame()
            bin_detected = detected_mean[detected_mean['radius_bin'] == bin_label] if isinstance(detected_mean, pd.DataFrame) else pd.DataFrame()
            total_heights, detected_heights = [], []
            for star in STAR_ORDER:
                total_heights.append(get_count(bin_data, star=star))
                detected_heights.append(get_count(bin_detected, star=star))
            
            self._add_percentage_labels(ax, x + i * bar_width, detected_heights, total_heights, offset=8)
        
        # Add detected/total labels above each stellar type
        for idx, star in enumerate(STAR_ORDER):
            def sum_count(df):
                return df[df['stype'] == star]['count'].sum() if isinstance(df, pd.DataFrame) and 'stype' in df.columns and 'count' in df.columns else 0
            total_count = sum_count(total_mean)
            detected_count = sum_count(detected_mean)
            # Find the tallest bar for this star
            heights = [get_count(total_mean, star=star, bin_label=bin_label) + get_count(detected_mean, star=star, bin_label=bin_label) for bin_label in BIN_LABELS]
            max_height = max(heights) if heights else 0
            # if self.name == 'HWO':
            #     if star == 'M':
            #         ax.text(
            #                 x[idx] + 1.5 * bar_width, max_height + 5,
            #                 f"{int(detected_count)} detected out of \n {int(total_count)} ",
            #                 ha='center', va='bottom', fontsize=12
            #             )
            #     elif star == 'K':
            #         ax.text(
            #                 x[idx] + 1.5 * bar_width, max_height-5,
            #                 f"{int(detected_count)} detected out of \n {int(total_count)}",
            #                 ha='center', va='bottom', fontsize=12
            #             )
            #     else:
            #         ax.text(
            #                 x[idx] + 1.5 * bar_width, max_height - 15,
            #                 f"{int(detected_count)} detected out of \n {int(total_count)}",
            #                 ha='center', va='bottom', fontsize=12
            #             )
            # else:
            #     if star == 'K':
            #         ax.text(
            #                 x[idx] + 1.5 * bar_width, max_height-50,
            #                 f"{int(detected_count)} detected out of \n {int(total_count)}",
            #                 ha='center', va='bottom', fontsize=12
            #             )
            #     else:
            #         ax.text(
            #                 x[idx] + 1.5 * bar_width, max_height,
            #                 f"{int(detected_count)} detected out of \n {int(total_count)}",
            #                 ha='center', va='bottom', fontsize=12
            #             )
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.data_dir, 
                                self._output_filename('stellar_type_overlay')),
                   bbox_inches='tight')
        plt.close(fig)

    def _calculate_planet_stats(self, df):
        """Calculate statistics for planet-based plots."""
        df = df.copy()
        df['categories'] = df.apply(self._assign_category, axis=1)
        df = df.explode('categories')

        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]

        stats = self._pivot_stats(df, ['categories', 'temp_zone'])
        df_detected = df[mask_best].copy()
        detected_stats = self._pivot_stats(df_detected, ['categories', 'temp_zone'])

        return stats, detected_stats

    def _get_planet_data_for_bin(self, stats, detected_stats, bin_label):
        """Get planet data for a specific temperature zone bin."""
        categories = ['Rocky', 'Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        bin_stats = stats[stats['temp_zone'] == bin_label].set_index('categories').reindex(categories).fillna(0)
        bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label].set_index('categories').reindex(categories).fillna(0)
        return bin_stats['count'].values, bin_stats['error'].values, bin_detected['count'].values, bin_detected['error'].values

    def plot_by_planet(self) -> None:
        """Plot planet counts by planet type with temperature zone breakdown."""
        stats, detected_stats = self._calculate_planet_stats(self.df)

        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 6))
        categories = ['Rocky', 'Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        x = np.arange(len(categories))
        # Ensure ax is a matplotlib Axes, not an ndarray
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]

        # Plot each temperature zone
        for i, (bin_label, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
            total_heights, total_errors, detected_heights, detected_errors = self._get_planet_data_for_bin(
                stats, detected_stats, bin_label
            )

            add_total_label = (i == 0)
            self._create_overlay_bars(
                ax, x + i * BAR_WIDTH_TEMP, total_heights, detected_heights,
                total_errors, detected_errors, BAR_WIDTH_TEMP, 'lightgray', color, hatch,
                add_total_label=add_total_label, detected_label=bin_label
            )

        if self.name == 'HWO_exoplanets':
            name = '\nHWO detection ability of exoplanet catalog'
        elif self.name == 'LIFEsim_exoplanets':
            name = '\nLIFEsim detection ability of exoplanet catalog'
        else:
            name = self.name

        self._setup_plot_style(
            ax, 'Planet Type', 'Number of Planets',
            f'Planet Detection by Type for {name}',
            'Temperature Zone'
        )

        ax.set_xticks(x + BAR_WIDTH_TEMP)
        ax.set_xticklabels(categories)

        for i, bin_label in enumerate(TEMP_ZONES):
            total_heights, _, detected_heights, _ = self._get_planet_data_for_bin(
                stats, detected_stats, bin_label
            )
            self._add_percentage_labels(
                ax, x + i * BAR_WIDTH_TEMP, detected_heights, total_heights, offset=16
            )
        plt.ylim(0, 7000)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_type')

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
        """Compute mean and std of counts by run for arbitrary groupby columns."""
        if df.empty:
            return pd.DataFrame(columns=groupby_cols + ['count', 'error'])
        if 'run' in df.columns:
            grouped = df.groupby(groupby_cols + ['run']).size().reset_index(name='count')
            stats = grouped.groupby(groupby_cols).agg({'count': ['mean', 'std']}).reset_index()
            stats.columns = groupby_cols + ['count', 'error']
        else:
            stats = df.groupby(groupby_cols).size().reset_index(name='count')
            stats['error'] = 0.0
        return stats
