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
        if 'temp_zone' not in self.df.columns:
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
        
        # Total stats - simple groupby and sum across runs
        total_per_run = df.groupby(['run', 'stype', 'radius_bin']).size().reset_index()
        total_per_run.columns = ['run', 'stype', 'radius_bin', 'count']
        total_pivot = total_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        total_mean = total_pivot.mean(axis=1)
        if isinstance(total_mean, pd.Series):
            total_mean = total_mean.to_frame('count').reset_index()
        total_std = total_pivot.std(axis=1)
        if isinstance(total_std, pd.Series):
            total_std = total_std.to_frame('count').reset_index()
        
        # Detected stats
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]  # Align mask with filtered df
        if self.name == 'HWO':
            df['detected_flag_best'] = mask_best.astype(bool)
            detected_df = df[df['detected_flag_best']]
        else:
            df['detected_flag'] = mask_best.astype(bool)
            detected_df = df[df['detected_flag']]
        if not isinstance(detected_df, pd.DataFrame):
            detected_df = pd.DataFrame(columns=df.columns)
        
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
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = BAR_WIDTH_STAR
        # Ensure ax is a matplotlib Axes, not an ndarray
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]
        
        # Helper for safe count extraction
        def get_count(df, star=None, bin_label=None):
            if not isinstance(df, pd.DataFrame):
                return 0
            if star is not None and bin_label is not None:
                filtered = df[(df['radius_bin'] == bin_label) & (df['stype'] == star)] if 'radius_bin' in df.columns and 'stype' in df.columns else pd.DataFrame()
            elif star is not None:
                filtered = df[df['stype'] == star] if 'stype' in df.columns else pd.DataFrame()
            else:
                filtered = df
            return filtered['count'].iloc[0] if isinstance(filtered, pd.DataFrame) and not filtered.empty and not isinstance(filtered, np.ndarray) and 'count' in filtered.columns else 0
        # Plot bars for each radius bin
        for i, bin_label in enumerate(BIN_LABELS):
            bin_data = total_mean[total_mean['radius_bin'] == bin_label] if isinstance(total_mean, pd.DataFrame) else pd.DataFrame()
            bin_std = total_std[total_std['radius_bin'] == bin_label] if isinstance(total_std, pd.DataFrame) else pd.DataFrame()
            bin_detected = detected_mean[detected_mean['radius_bin'] == bin_label] if isinstance(detected_mean, pd.DataFrame) else pd.DataFrame()
            bin_detected_std = detected_std[detected_std['radius_bin'] == bin_label] if isinstance(detected_std, pd.DataFrame) else pd.DataFrame()
            total_heights, total_errors, detected_heights, detected_errors = [], [], [], []
            for star in STAR_ORDER:
                total_heights.append(get_count(bin_data, star=star))
                total_errors.append(get_count(bin_std, star=star))
                detected_heights.append(get_count(bin_detected, star=star))
                detected_errors.append(get_count(bin_detected_std, star=star))
            add_total_label = (i == 0)
            self._create_overlay_bars(
                ax, x + i * bar_width, total_heights, detected_heights,
                total_errors, detected_errors, bar_width, 'lightgray', STAR_COLORS[i], STAR_HATCHES[i],
                add_total_label=add_total_label, detected_label=bin_label
            )
        
        ax.set_xlabel('Star Type')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Star Type for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.legend(title='Radius Bin')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
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
            if self.name == 'HWO':
                if star == 'M':
                    ax.text(
                            x[idx] + 1.5 * bar_width, max_height + 5,
                            f"{int(detected_count)} detected out of \n {int(total_count)} simulated",
                            ha='center', va='bottom', fontsize=12
                        )
                elif star == 'K':
                    ax.text(
                            x[idx] + 1.5 * bar_width, max_height-5,
                            f"{int(detected_count)} detected out of \n {int(total_count)} simulated",
                            ha='center', va='bottom', fontsize=12
                        )
                else:
                    ax.text(
                            x[idx] + 1.5 * bar_width, max_height - 15,
                            f"{int(detected_count)} detected out of \n {int(total_count)} simulated",
                            ha='center', va='bottom', fontsize=12
                        )
            else:
                if star == 'K':
                    ax.text(
                            x[idx] + 1.5 * bar_width, max_height-50,
                            f"{int(detected_count)} detected out of \n {int(total_count)} simulated",
                            ha='center', va='bottom', fontsize=12
                        )
                else:
                    ax.text(
                            x[idx] + 1.5 * bar_width, max_height,
                            f"{int(detected_count)} detected out of \n {int(total_count)} simulated",
                            ha='center', va='bottom', fontsize=12
                        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(self.data_dir, 
                                self._output_filename('stellar_type_overlay')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _calculate_planet_stats(self, df):
        """Calculate statistics for planet-based plots."""
        # Add categories using vectorized operations (much faster than apply)
        df = df.copy()
        
        # Vectorized category assignment
        conditions = [
            (df['radius_p'] < 1.5),
            (df['radius_p'] < 1.5) & (df['stype'] == 'M'),
            (df['radius_p'] < 1.5) & (df['stype'].isin(['G', 'K'])),
            (df['radius_p'] >= 1.5) & (df['radius_p'] < 2.0),
            (df['radius_p'] >= 2.0) & (df['radius_p'] < 4.0),
            (df['radius_p'] >= 4.0) & (df['radius_p'] < 8.0)
        ]
        choices = ['Rocky', 'Rocky, M stars', 'Rocky, G and K stars', 'Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df['categories'] = np.select(conditions, choices, default='')
        
        # Filter for valid categories only
        valid_categories = ['Rocky', 'Rocky, M stars', 'Rocky, G and K stars', 'Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df = df[df['categories'].isin(valid_categories)]
        
        # Calculate statistics
        stats = self._pivot_stats(df, ['categories', 'temp_zone'])
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]  # Align mask with filtered df
        df_detected = df[mask_best].copy()
        
        # Calculate detected statistics
        detected_stats = self._pivot_stats(df_detected, ['categories', 'temp_zone'])
        
        return stats, detected_stats

    def _get_planet_data_for_bin(self, stats, detected_stats, bin_label):
        """Get planet data for a specific bin."""
        bin_stats = stats[stats['temp_zone'] == bin_label]
        bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
        
        # Get unique categories
        categories = ['Rocky', 'Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        
        # Align data
        bin_stats = bin_stats.set_index('categories').reindex(categories).fillna(0)
        bin_detected = bin_detected.set_index('categories').reindex(categories).fillna(0)
        
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
            
            # Create overlay bars - only add "Total" label for the first temperature zone
            add_total_label = (i == 0)  # Only add "Total" label for the first iteration
            self._create_overlay_bars(
                ax, x + i * BAR_WIDTH_TEMP, total_heights, detected_heights,
                total_errors, detected_errors, BAR_WIDTH_TEMP, 'lightgray', color, hatch,
                add_total_label=add_total_label, detected_label=bin_label
            )
        
        # Setup plot
        self._setup_plot_style(
            ax, 'Planet Type', 'Number of Planets', 
            f'Planet Detection by Type for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}',
            'Temperature Zone'
        )
        
        # Set x-axis
        ax.set_xticks(x + BAR_WIDTH_TEMP)
        ax.set_xticklabels(categories)
        
        # Add percentage labels
        for i, bin_label in enumerate(TEMP_ZONES):
            total_heights, _, detected_heights, _ = self._get_planet_data_for_bin(
                stats, detected_stats, bin_label
            )
            self._add_percentage_labels(
                ax, x + i * BAR_WIDTH_TEMP, detected_heights, total_heights, offset=16
            )
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_type')

    def _calculate_distance_stats(self, df):
        """Calculate statistics for distance-based plots."""
        # Create proper distance bins that match the labels
        distance_bins = [0, 3, 5, 7, 9, 11, 13, 15, 20]
        
        # Check if there are any distances outside our bin range and handle them
        min_dist = df['distance_s'].min()
        max_dist = df['distance_s'].max()
        
        # Filter out any distances outside our range or handle them
        df_filtered = df.copy()
        if min_dist < 0 or max_dist > 20:
            print(f"Warning: Distance range is {min_dist:.2f} to {max_dist:.2f}, filtering to 0-20 range")
            df_filtered = df_filtered[(df_filtered['distance_s'] >= 0) & (df_filtered['distance_s'] <= 20)]
        
        # Add distance bins
        df_filtered['distance_bin'] = pd.cut(df_filtered['distance_s'], bins=distance_bins, labels=DISTANCE_LABELS)
        
        # Calculate statistics
        stats = self._pivot_stats(df_filtered, ['distance_bin', 'temp_zone'])
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df_filtered.index]  # Align mask with filtered df
        df_detected = df_filtered[mask_best].copy()
        df_detected['distance_bin'] = pd.cut(df_detected['distance_s'], bins=distance_bins, labels=DISTANCE_LABELS)
        
        # Calculate detected statistics
        detected_stats = self._pivot_stats(df_detected, ['distance_bin', 'temp_zone'])
        
        return stats, detected_stats

    
    def _pivot_stats(self, df, groupby_cols):
        """
        Compute mean and std of counts by run for arbitrary groupby columns.
        Args:
            df: DataFrame
            groupby_cols: list of str, columns to group by
        Returns:
            DataFrame with groupby_cols, 'count' (mean), and 'error' (std).
        """
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

    def _get_distance_data_for_bin(self, stats, detected_stats, bin_label):
        """Get distance data for a specific bin."""
        bin_stats = stats[stats['temp_zone'] == bin_label]
        bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
        
        # Align data
        bin_stats = bin_stats.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
        bin_detected = bin_detected.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
        
        return bin_stats['count'].values, bin_stats['error'].values, bin_detected['count'].values, bin_detected['error'].values

    def plot_by_distance(self) -> None:
        """Plot planet counts by distance with detected vs total overlay."""
        stats, detected_stats = self._calculate_distance_stats(self.df)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(DISTANCE_LABELS))
        # Ensure ax is a matplotlib Axes, not an ndarray
        if isinstance(ax, np.ndarray):
            ax = ax.flatten()[0]
        
        # Calculate total and detected counts across all temperature zones
        total_counts = np.zeros(len(DISTANCE_LABELS))
        detected_counts = np.zeros(len(DISTANCE_LABELS))
        total_errors = np.zeros(len(DISTANCE_LABELS))
        detected_errors = np.zeros(len(DISTANCE_LABELS))
        
        # Sum across all temperature zones
        for bin_label in TEMP_ZONES:
            bin_stats = stats[stats['temp_zone'] == bin_label]
            bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
            
            # Align data
            bin_stats = bin_stats.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
            bin_detected = bin_detected.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
            
            total_counts += bin_stats['count'].values
            detected_counts += bin_detected['count'].values
            # For errors, we'll use the maximum error (conservative approach)
            total_errors = np.maximum(total_errors, bin_stats['error'].values)
            detected_errors = np.maximum(detected_errors, bin_detected['error'].values)
        
        # Create overlay bars - total vs detected
        self._create_overlay_bars(
            ax, x, total_counts, detected_counts,
            total_errors, detected_errors, 0.6, 'lightgray', 'green', None
        )
        
        # Setup plot
        self._setup_plot_style(
            ax, 'Distance Bin', 'Number of Planets', 
            f'Planet Detection by Distance Bin for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}'
        )
       
        # Set x-axis
        ax.set_xticks(x)
        ax.set_xticklabels(DISTANCE_LABELS)
        
        # Add percentage labels
        self._add_percentage_labels(
            ax, x, detected_counts, total_counts, offset=12
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_distance')

    # After all subplots are created and twin_axes is filled:
    def _create_legend(self, fig, axs, twin_axes):
        handles, labels = [], []
        for ax, ax2 in zip(axs, twin_axes):
            if ax2 is not None:
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h1 + h2
                labels += l1 + l2
        # Remove duplicates
        unique = dict(zip(labels, handles))
        fig.legend(unique.values(), unique.keys(), loc='upper left', bbox_to_anchor=(0.08, 0.92), fontsize=14)
