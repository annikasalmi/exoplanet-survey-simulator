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
        total_mean = total_pivot.mean(axis=1).reset_index()
        total_mean = total_mean.rename(columns={total_mean.columns[-1]: 'count'})
        total_std = total_pivot.std(axis=1).reset_index()
        total_std = total_std.rename(columns={total_std.columns[-1]: 'count'})
        
        # Detected stats
        mask_best, _ = self._cache['detection_masks']
        if self.name == 'HWO':
            df['detected_flag_best'] = mask_best.astype(bool)
            detected_df = df[df['detected_flag_best']]
        else:
            df['detected_flag'] = mask_best.astype(bool)
            detected_df = df[df['detected_flag']]
        
        detected_per_run = detected_df.groupby(['run', 'stype', 'radius_bin']).size().reset_index()
        detected_per_run.columns = ['run', 'stype', 'radius_bin', 'count']
        detected_pivot = detected_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        detected_mean = detected_pivot.mean(axis=1).reset_index()
        detected_mean = detected_mean.rename(columns={detected_mean.columns[-1]: 'count'})
        detected_std = detected_pivot.std(axis=1).reset_index()
        detected_std = detected_std.rename(columns={detected_std.columns[-1]: 'count'})
        
        # Create the overlay plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = BAR_WIDTH_STAR
        
        # Plot bars for each radius bin
        for i, bin_label in enumerate(BIN_LABELS):
            # Get data for this radius bin
            bin_data = total_mean[total_mean['radius_bin'] == bin_label]
            bin_std = total_std[total_std['radius_bin'] == bin_label]
            bin_detected = detected_mean[detected_mean['radius_bin'] == bin_label]
            bin_detected_std = detected_std[detected_std['radius_bin'] == bin_label]
            
            # Align with STAR_ORDER
            total_heights = []
            total_errors = []
            detected_heights = []
            detected_errors = []
            for star in STAR_ORDER:
                # Total data
                star_data = bin_data[bin_data['stype'] == star]
                star_std_data = bin_std[bin_std['stype'] == star]
                if len(star_data) > 0:
                    total_heights.append(star_data.iloc[0]['count'])
                    total_errors.append(star_std_data.iloc[0]['count'])
                else:
                    total_heights.append(0)
                    total_errors.append(0)
                # Detected data
                star_detected = bin_detected[bin_detected['stype'] == star]
                star_detected_std = bin_detected_std[bin_detected_std['stype'] == star]
                if len(star_detected) > 0:
                    detected_heights.append(star_detected.iloc[0]['count'])
                    detected_errors.append(star_detected_std.iloc[0]['count'])
                else:
                    detected_heights.append(0)
                    detected_errors.append(0)
            # Create overlay bars - only add "Total" label for the first radius bin
            add_total_label = (i == 0)  # Only add "Total" label for the first iteration
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
            # Get data for this radius bin
            bin_data = total_mean[total_mean['radius_bin'] == bin_label]
            bin_detected = detected_mean[detected_mean['radius_bin'] == bin_label]
            
            # Align with STAR_ORDER
            total_heights = []
            detected_heights = []
            for star in STAR_ORDER:
                # Total data
                star_data = bin_data[bin_data['stype'] == star]
                if len(star_data) > 0:
                    total_heights.append(star_data.iloc[0]['count'])
                else:
                    total_heights.append(0)
                # Detected data
                star_detected = bin_detected[bin_detected['stype'] == star]
                if len(star_detected) > 0:
                    detected_heights.append(star_detected.iloc[0]['count'])
                else:
                    detected_heights.append(0)
            
            self._add_percentage_labels(
                ax, x + i * bar_width, detected_heights, total_heights
            )
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 
                                self._output_filename('stellar_type_overlay')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _calculate_planet_stats(self, df):
        """Calculate statistics for planet-based plots."""
        # Add categories
        df['categories'] = df.apply(self._assign_category, axis=1)
        df = df.explode('categories')
        
        # Calculate statistics
        stats = self._pivot_stats(df, ['categories', 'temp_zone'])
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        df_detected = df[mask_best].copy()
        df_detected['categories'] = df_detected.apply(self._assign_category, axis=1)
        df_detected = df_detected.explode('categories')
        
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
        fig, ax = plt.subplots(figsize=(12, 8))
        categories = ['Rocky', 'Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        x = np.arange(len(categories))
        
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
                ax, x + i * BAR_WIDTH_TEMP, detected_heights, total_heights
            )
        
        self._save_plot(fig, 'planet_detection_by_type')

    def _calculate_distance_stats(self, df):
        """Calculate statistics for distance-based plots."""
        # Create proper distance bins that match the labels
        # DISTANCE_LABELS: ['< 3', '3 - 5', '5 - 7', '7 - 9', '9 - 11', '11 - 13', '13 - 15', '15 - 20']
        # Need 9 bin edges for 8 labels: [0, 3, 5, 7, 9, 11, 13, 15, 20]
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
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(DISTANCE_LABELS))
        
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
            ax, x, detected_counts, total_counts
        )
        
        self._save_plot(fig, 'planet_detection_by_distance')
