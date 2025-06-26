import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot.base_plotter import BasePlotter
from tools.plotting_constants import (
    STAR_ORDER, TEMP_ZONES, DISTANCE_LABELS,
    TEMP_COLORS,  BAR_WIDTH_TEMP, BAR_WIDTH_DIST, HATCHES
)

class PlotPlanetType(BasePlotter):
    """
    Class for generating grouped bar plots of planet statistics by star type, planet type, and distance.
    Handles best/worst case overlays for HWO scenarios.
    """
    
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize the plotter with data and metadata."""
        super().__init__(df, nruns, star_catalog, name)

    def plot_all(self) -> None:
        """Generate all planet type plots."""
        if not self._validate_data():
            return
            
        self.plot_by_star()
        self.plot_by_planet()
        self.plot_by_distance()

    def _calculate_star_stats(self, df):
        """Calculate statistics for star-based plots."""
        # Add temperature zones
        df['temp_zone'] = df['temp_p'].apply(self._temp_zone)
        
        # Calculate statistics
        stats = self._pivot_stats(df, ['stype', 'temp_zone'])
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        df_detected = df[mask_best].copy()
        df_detected['temp_zone'] = df_detected['temp_p'].apply(self._temp_zone)
        
        # Calculate detected statistics
        detected_stats = self._pivot_stats(df_detected, ['stype', 'temp_zone'])
        
        return stats, detected_stats


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
    
    
    def _temp_zone(self, temp):
        """Assigns a temperature zone based on the temperature value."""
        if temp > 390:
            return 'hot'
        elif 390 > temp > 270:
            return 'habitable'
        else:
            return 'cold'

    def _get_star_data_for_bin(self, stats, detected_stats, bin_label):
        """Get star data for a specific bin."""
        bin_stats = stats[stats['temp_zone'] == bin_label]
        bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
        
        # Align data
        bin_stats = bin_stats.set_index('stype').reindex(STAR_ORDER).fillna(0)
        bin_detected = bin_detected.set_index('stype').reindex(STAR_ORDER).fillna(0)
        
        return bin_stats['count'].values, bin_stats['error'].values, bin_detected['count'].values, bin_detected['error'].values

    def plot_by_star(self) -> None:
        """Plot planet counts by star type with temperature zone breakdown."""
        stats, detected_stats = self._calculate_star_stats(self.df)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(STAR_ORDER))
        
        # Plot each temperature zone
        for i, (bin_label, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
            total_heights, total_errors, detected_heights, detected_errors = self._get_star_data_for_bin(
                stats, detected_stats, bin_label
            )
            
            # Create overlay bars
            self._create_overlay_bars(
                ax, x + i * BAR_WIDTH_TEMP, total_heights, detected_heights,
                total_errors, detected_errors, BAR_WIDTH_TEMP, 'lightgray', color, hatch
            )
        
        # Setup plot
        self._setup_plot_style(
            ax, 'Star Type', 'Number of Planets', 
            f'Planet Detection by Star Type for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}',
            'Temperature Zone'
        )
        
        # Set x-axis
        ax.set_xticks(x + BAR_WIDTH_TEMP)
        ax.set_xticklabels(STAR_ORDER)
        
        # Add percentage labels
        for i, bin_label in enumerate(TEMP_ZONES):
            total_heights, _, detected_heights, _ = self._get_star_data_for_bin(
                stats, detected_stats, bin_label
            )
            self._add_percentage_labels(
                ax, x + i * BAR_WIDTH_TEMP, detected_heights, total_heights
            )
        
        self._save_plot(fig, 'planet_detection_by_star')

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
            
            # Create overlay bars
            self._create_overlay_bars(
                ax, x + i * BAR_WIDTH_TEMP, total_heights, detected_heights,
                total_errors, detected_errors, BAR_WIDTH_TEMP, 'lightgray', color, hatch
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
        # Add distance bins
        df['distance_bin'] = pd.cut(df['distance_s'], bins=5, labels=DISTANCE_LABELS)
        
        # Calculate statistics
        stats = self._pivot_stats(df, ['distance_bin', 'temp_zone'])
        
        # Get detection masks
        mask_best, _ = self._get_detection_masks()
        df_detected = df[mask_best].copy()
        df_detected['distance_bin'] = pd.cut(df_detected['distance_s'], bins=5, labels=DISTANCE_LABELS)
        
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
        """Plot planet counts by distance with temperature zone breakdown."""
        stats, detected_stats = self._calculate_distance_stats(self.df)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(12, 8))
        x = np.arange(len(DISTANCE_LABELS))
        
        # Plot each temperature zone
        for i, (bin_label, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
            total_heights, total_errors, detected_heights, detected_errors = self._get_distance_data_for_bin(
                stats, detected_stats, bin_label
            )
            
            # Create overlay bars
            self._create_overlay_bars(
                ax, x + i * BAR_WIDTH_DIST, total_heights, detected_heights,
                total_errors, detected_errors, BAR_WIDTH_DIST, 'lightgray', color, hatch
            )
        
        # Setup plot
        self._setup_plot_style(
            ax, 'Distance Bin', 'Number of Planets', 
            f'Planet Detection by Distance Bin for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}',
            'Temperature Zone'
        )
        
        # Set x-axis
        ax.set_xticks(x + BAR_WIDTH_DIST)
        ax.set_xticklabels(DISTANCE_LABELS)
        
        # Add percentage labels
        for i, bin_label in enumerate(TEMP_ZONES):
            total_heights, _, detected_heights, _ = self._get_distance_data_for_bin(
                stats, detected_stats, bin_label
            )
            self._add_percentage_labels(
                ax, x + i * BAR_WIDTH_DIST, detected_heights, total_heights
            )
        
        self._save_plot(fig, 'planet_detection_by_distance')
