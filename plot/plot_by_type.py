import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot.helpers import (
    make_output_dir, temp_zone, assign_category, prep_plot_df_stars, pivot_stats,
    bar_plot_with_errors, overlay_best_worst, output_filename, get_detection_masks
)
from tools.plotting_constants import (
    STAR_ORDER, BIN_LABELS, TEMP_ZONES, DISTANCE_LABELS,
    STAR_COLORS, STAR_HATCHES, TEMP_COLORS, BAR_WIDTH_STAR, BAR_WIDTH_TEMP, BAR_WIDTH_DIST
)

class PlotPlanetType:
    """
    Class for generating grouped bar plots of planet statistics by star type, planet type, and distance.
    Handles best/worst case overlays for HWO scenarios.
    """
    def __init__(self, df: pd.DataFrame, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize the plotter with data and metadata."""
        self.df = df.copy()
        self.nruns = nruns
        self.star_catalog = star_catalog
        self.name = name
        self.data_dir = make_output_dir(name, nruns, star_catalog)
        if self.name == 'HWO':
            self.case = ' (best case)'
        else:
            self.case = ''

    def plot_all(self) -> None:
        """Main entry point to generate all plots based on the name parameter."""
        self.plot_by_planet()
        self.plot_by_star()
        self.plot_distances()

    def plot_by_star(self) -> None:
        """Grouped bar plots by star type and radius bin. For HWO, uses detected_best/worst logic."""
        df = self.df.copy()
        x = np.arange(len(STAR_ORDER))
        
        # Total stats - simple groupby and sum across runs
        total_counts = df.groupby(['stype', 'radius_bin']).size().reset_index()
        total_counts.columns = ['stype', 'radius_bin', 'count']
        
        # Calculate means and stds across runs
        total_per_run = df.groupby(['run', 'stype', 'radius_bin']).size().reset_index()
        total_per_run.columns = ['run', 'stype', 'radius_bin', 'count']
        total_pivot = total_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        
        total_mean = total_pivot.mean(axis=1).reset_index()
        total_mean = total_mean.rename(columns={total_mean.columns[-1]: 'count'})
        
        total_std = total_pivot.std(axis=1).reset_index()
        total_std = total_std.rename(columns={total_std.columns[-1]: 'count'})
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = BAR_WIDTH_STAR
        
        # Plot bars for each radius bin
        for i, bin_label in enumerate(BIN_LABELS):
            # Get data for this radius bin
            bin_data = total_mean[total_mean['radius_bin'] == bin_label]
            bin_std = total_std[total_std['radius_bin'] == bin_label]
            
            # Align with STAR_ORDER
            heights = []
            errors = []
            for star in STAR_ORDER:
                star_data = bin_data[bin_data['stype'] == star]
                star_std_data = bin_std[bin_std['stype'] == star]
                if len(star_data) > 0:
                    heights.append(star_data.iloc[0]['count'])
                    errors.append(star_std_data.iloc[0]['count'])
                else:
                    heights.append(0)
                    errors.append(0)
            
            ax.bar(x + i * bar_width, heights, bar_width, 
                   label=bin_label, color=STAR_COLORS[i], 
                   hatch=STAR_HATCHES[i], edgecolor='black',
                   yerr=errors, capsize=3)
            
            # Add text annotations for counts
            for j, (h, err) in enumerate(zip(heights, errors)):
                if h > 0:  # Only add text if there are planets
                    # Handle NaN values
                    h_display = int(h) if not np.isnan(h) else 0
                    err_display = int(err) if not np.isnan(err) else 0
                    ax.text(x[j] + i * bar_width, h + err + 0.5, f"{h_display}±{err_display}", 
                           ha='center', fontsize=8)
        
        ax.set_xlabel('Star Type')
        ax.set_ylabel('Total Planets')
        ax.set_title(f'Total Planets by Star Type for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.legend(title='Radius Bin')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 
                                output_filename('stellar_type_total', self.name, self.nruns, self.star_catalog)), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Detected stats
        mask_best, mask_worst = get_detection_masks(df, self.name)
        if self.name == 'HWO':
            df['detected_flag_best'] = mask_best.astype(bool)
            detected_df = df[df['detected_flag_best']]
        else:
            df['detected_flag'] = mask_best.astype(bool)
            detected_df = df[df['detected_flag']]
        
        # Calculate detected counts across runs
        detected_per_run = detected_df.groupby(['run', 'stype', 'radius_bin']).size().reset_index()
        detected_per_run.columns = ['run', 'stype', 'radius_bin', 'count']
        detected_pivot = detected_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
 
        detected_mean = detected_pivot.mean(axis=1).reset_index()
        detected_mean = detected_mean.rename(columns={detected_mean.columns[-1]: 'count'})
        
        detected_std = detected_pivot.std(axis=1).reset_index()
        detected_std = detected_std.rename(columns={detected_std.columns[-1]: 'count'})
        
        # Create the detected plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot bars for each radius bin
        for i, bin_label in enumerate(BIN_LABELS):
            # Get data for this radius bin
            bin_data = detected_mean[detected_mean['radius_bin'] == bin_label]
            bin_std = detected_std[detected_std['radius_bin'] == bin_label]
            
            # Align with STAR_ORDER
            heights = []
            errors = []
            for star in STAR_ORDER:
                star_data = bin_data[bin_data['stype'] == star]
                star_std_data = bin_std[bin_std['stype'] == star]
                if len(star_data) > 0:
                    heights.append(star_data.iloc[0]['count'])
                    errors.append(star_std_data.iloc[0]['count'])
                else:
                    heights.append(0)
                    errors.append(0)
            
            ax.bar(x + i * bar_width, heights, bar_width, 
                   label=bin_label, color=STAR_COLORS[i], 
                   hatch=STAR_HATCHES[i], edgecolor='black',
                   yerr=errors, capsize=3)
            
            # Add text annotations for counts
            for j, (h, err) in enumerate(zip(heights, errors)):
                if h > 0:  # Only add text if there are planets
                    # Handle NaN values
                    h_display = int(h) if not np.isnan(h) else 0
                    err_display = int(err) if not np.isnan(err) else 0
                    ax.text(x[j] + i * bar_width, h + err + 0.1, f"{h_display}±{err_display}", 
                           ha='center', fontsize=8)
        
        ax.set_xlabel('Star Type')
        ax.set_ylabel('Detected Planets')
        ax.set_title(f'Detected Planets by Star Type for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.set_ylim(bottom=0)  # Set y-axis to start at 0
        ax.legend(title='Radius Bin')
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 
                                output_filename('stellar_type_detected', self.name, self.nruns, self.star_catalog)), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_by_planet(self, detected_only=False) -> None:
        """Bar plots by planet category and temperature zone. Best/worst overlays for HWO."""
        df = self.df.copy()
        df['temp_zone'] = df['temp_p'].apply(temp_zone)
        df['category'] = df.apply(assign_category, axis=1)
        df = df.dropna(subset=['category'])
        
        # Define all possible categories from assign_category function
        all_possible_categories = [
            'Habitable Rocky',
            'Rocky', 
            'Exo-Earth Candidates',
            'Super-Earths',
            'Habitable Super-Earths',
            'Habitable Sub-Neptunes',
            'Sub-Neptunes',
            'Sub-Jovians',
            'Giant planets'
        ]
        
        # Get categories that actually exist in the data
        detected_categories = sorted(df['category'].unique())
        print(f"Detected categories in data: {detected_categories}")
        
        # Use all possible categories for consistent plotting
        plot_categories = all_possible_categories
        x = np.arange(len(plot_categories))
        
        # Total
        total_stats = pivot_stats(df, ['category', 'temp_zone'])
        heights_list, errors_list = [], []
        for zone in TEMP_ZONES:
            data = total_stats[total_stats['temp_zone'] == zone].set_index('category').reindex(plot_categories)
            heights_list.append(data['count'].fillna(0).values)
            errors_list.append(data['error'].fillna(0).values)
        bar_plot_with_errors(
            x, heights_list, errors_list, BAR_WIDTH_TEMP, TEMP_ZONES, colors=TEMP_COLORS,
            xticks=x, xticklabels=plot_categories, ylabel='Planet Count',
            title=f'Total Planets by Type and Temp Zone\n{self.name}, {self.nruns} Runs — {self.star_catalog}',
            legend_title='Temp Zone', filename=os.path.join(self.data_dir, output_filename('planets_by_type_total', self.name, self.nruns, self.star_catalog)),
            text_offset=1
        )
        if detected_only:
            # Detected
            mask_best, mask_worst = get_detection_masks(df, self.name)
            if self.name == 'HWO':
                df['detected_flag_best'] = mask_best.astype(bool)
                df['detected_flag_worst'] = mask_worst.astype(bool)
                detected_df_best = df[df['detected_flag_best']]
                detected_df_worst = df[df['detected_flag_worst']]
                
                # Best case
                best_stats = pivot_stats(detected_df_best, ['category', 'temp_zone'])
                heights_list_best, errors_list_best = [], []
                for zone in TEMP_ZONES:
                    data = best_stats[best_stats['temp_zone'] == zone].set_index('category').reindex(plot_categories)
                    heights_list_best.append(data['count'].fillna(0).values)
                    errors_list_best.append(data['error'].fillna(0).values)
                
                # Worst case
                worst_stats = pivot_stats(detected_df_worst, ['category', 'temp_zone'])
                heights_list_worst, errors_list_worst = [], []
                for zone in TEMP_ZONES:
                    data = worst_stats[worst_stats['temp_zone'] == zone].set_index('category').reindex(plot_categories)
                    heights_list_worst.append(data['count'].fillna(0).values)
                    errors_list_worst.append(data['error'].fillna(0).values)
                
                # Create overlay plot
                fig, ax = plt.subplots(figsize=(15, 8))
                bar_width = BAR_WIDTH_TEMP
                
                for i, zone in enumerate(TEMP_ZONES):
                    overlay_best_worst(
                        ax, x + i * bar_width * 3, bar_width,
                        [heights_list_best[i], heights_list_worst[i]],
                        [TEMP_COLORS[i], TEMP_COLORS[i]],
                        [f'{zone} (Best)', f'{zone} (Worst)']
                    )
                
                ax.set_xlabel('Planet Category')
                ax.set_ylabel('Detected Planet Count')
                ax.set_title(f'Detected Planets by Type and Temp Zone\n{self.name}, {self.nruns} Runs — {self.star_catalog}')
                ax.set_xticks(x + bar_width * 3)
                ax.set_xticklabels(plot_categories, rotation=45, ha='right')
                ax.legend(title='Temp Zone')
                plt.tight_layout()
                plt.savefig(os.path.join(self.data_dir, 
                                        output_filename('planets_by_type_detected', self.name, self.nruns, self.star_catalog)), 
                           dpi=300, bbox_inches='tight')
                plt.close(fig)
            else:
                df['detected_flag'] = mask_best.astype(bool)
                detected_df = df[df['detected_flag']]
                detected_stats = pivot_stats(detected_df, ['category', 'temp_zone'])
                heights_list, errors_list = [], []
                for zone in TEMP_ZONES:
                    data = detected_stats[detected_stats['temp_zone'] == zone].set_index('category').reindex(plot_categories)
                    heights_list.append(data['count'].fillna(0).values)
                    errors_list.append(data['error'].fillna(0).values)
                bar_plot_with_errors(
                    x, heights_list, errors_list, BAR_WIDTH_TEMP, TEMP_ZONES, colors=TEMP_COLORS,
                    xticks=x, xticklabels=plot_categories, ylabel='Detected Planet Count',
                    title=f'Detected Planets by Type and Temp Zone\n{self.name}, {self.nruns} Runs — {self.star_catalog}',
                    legend_title='Temp Zone', filename=os.path.join(self.data_dir, output_filename('planets_by_type_detected', self.name, self.nruns, self.star_catalog)),
                    text_offset=1
                )

    def plot_distances(self, detected_only=False) -> None:
        """Bar plots by distance bin. Best/worst overlays for HWO."""
        df = self.df.copy()
        bins = [0, 3, 5, 7, 9, 11, 13, 15, np.inf]
        df['distance_bin'] = pd.cut(df['distance_s'], bins=bins, labels=DISTANCE_LABELS, right=False)
        x = np.arange(len(DISTANCE_LABELS))
        # Detected logic
        mask_best, mask_worst = get_detection_masks(df, self.name)
        if self.name == 'HWO':
            df['detected_flag_best'] = mask_best.astype(bool)
            detected_per_run = df[df['detected_flag_best']].groupby(['run', 'distance_bin']).size().unstack(fill_value=0).reindex(columns=DISTANCE_LABELS, fill_value=0)
        else:
            df['detected_flag'] = mask_best.astype(bool)
            detected_per_run = df[df['detected_flag']].groupby(['run', 'distance_bin']).size().unstack(fill_value=0).reindex(columns=DISTANCE_LABELS, fill_value=0)
        total_per_run = df.groupby(['run', 'distance_bin']).size().unstack(fill_value=0).reindex(columns=DISTANCE_LABELS, fill_value=0)
        total_mean = total_per_run.mean()
        total_std = total_per_run.std()
        detected_mean = detected_per_run.mean()
        detected_std = detected_per_run.std()
        undetected_mean = total_mean - detected_mean
        # Stacked bar plot
        heights_list = [undetected_mean.values, detected_mean.values]
        errors_list = [total_std.values, detected_std.values]
        bottom_list = [None, undetected_mean.values]
        alpha_list = [0.5, 1.0]
        bar_plot_with_errors(
            x, heights_list, [None, None], BAR_WIDTH_DIST, ['Not detected', 'Detected'],
            colors=['lightgray', 'seagreen'],
            xticks=x, xticklabels=DISTANCE_LABELS, ylabel='Planet Count',
            title=f'Planet Detection by Distance Bin\n{self.name}, {self.nruns} Runs — {self.star_catalog}',
            legend_title=None, filename=os.path.join(self.data_dir, output_filename('planet_distance', self.name, self.nruns, self.star_catalog)),
            stacked=True, bottom_list=bottom_list, alpha_list=alpha_list, text_offset=2
        )
        if detected_only:
            # Detected-only bar plot
            heights_list = [detected_mean.values]
            errors_list = [detected_std.values]
            bar_plot_with_errors(
                x, heights_list, errors_list, BAR_WIDTH_DIST, ['Detected'], colors=['seagreen'],
                xticks=x, xticklabels=DISTANCE_LABELS, ylabel='Detected Planet Count',
                title=f'Detected Planets by Distance Bin\n{self.name}, {self.nruns} Runs — {self.star_catalog}',
                legend_title=None, filename=os.path.join(self.data_dir, output_filename('planet_distance_detected', self.name, self.nruns, self.star_catalog)),
                text_offset=2
            )
            # Best/worst overlays if HWO
            if self.name == 'HWO' and mask_worst is not None:
                df['detected_flag_worst'] = mask_worst.astype(bool)
                detected_worst_per_run = df[df['detected_flag_worst']].groupby(['run', 'distance_bin']).size().unstack(fill_value=0).reindex(columns=DISTANCE_LABELS, fill_value=0)
                detected_best_per_run = df[df['detected_flag_best']].groupby(['run', 'distance_bin']).size().unstack(fill_value=0).reindex(columns=DISTANCE_LABELS, fill_value=0)
                detected_worst_mean = detected_worst_per_run.mean()
                detected_best_mean = detected_best_per_run.mean()
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(x, detected_worst_mean.values, width=BAR_WIDTH_DIST, color='green', label='Worst Case (Green)', edgecolor='black', alpha=0.7)
                ax.bar(x, detected_best_mean.values, width=BAR_WIDTH_DIST, color='lightgreen', label='Best Case (Light Green)', edgecolor='black', alpha=0.8)
                ax.set_ylabel('Detected Planet Count (Best/Worst)')
                ax.set_xlabel('Distance [pc]')
                ax.set_xticks(x)
                ax.set_xticklabels(DISTANCE_LABELS)
                ax.set_title(f'Best/Worst Detected Planets by Distance Bin\n{self.name}, {self.nruns} Runs — {self.star_catalog}')
                ax.legend(title='Overlay', fontsize=9)
                plt.tight_layout()
                plt.savefig(os.path.join(self.data_dir, output_filename('planet_distance_detected', self.name, self.nruns, self.star_catalog, 'best_worst')), dpi=300, bbox_inches='tight')
                plt.close(fig)
