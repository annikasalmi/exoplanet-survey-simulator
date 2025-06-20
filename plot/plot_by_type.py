import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from plot.helpers import (
    make_output_dir, temp_zone, assign_category, prep_plot_df_stars, pivot_stats,
    bar_plot_with_errors, overlay_best_worst, output_filename, get_detection_masks
)
from tools.plotting_constants import (
    STAR_ORDER, BIN_LABELS, CATEGORY_LABELS, TEMP_ZONES, DISTANCE_LABELS,
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
        # Total stats
        total = df.groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
        total_stats = pivot_stats(total, ['stype', 'radius_bin'])
        total_df, total_err = prep_plot_df_stars(total_stats, STAR_ORDER, BIN_LABELS)
        heights_list = prepare_bar_lists(total_df, BIN_LABELS)
        errors_list = prepare_bar_lists(total_err, BIN_LABELS)
        bar_plot_with_errors(
            x, heights_list, errors_list, BAR_WIDTH_STAR, BIN_LABELS, colors=STAR_COLORS, hatches=STAR_HATCHES,
            xticks=x + 1.5 * BAR_WIDTH_STAR, xticklabels=STAR_ORDER, ylabel='Total Planets',
            title=f'Total Planets by Star Type for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}',
            legend_title='Radius Bin', filename=os.path.join(self.data_dir, output_filename('stellar_type_total', self.name, self.nruns, self.star_catalog)),
            figsize=(12, 8)
        )
        # Detected stats
        mask_best, _ = get_detection_masks(df, self.name)
        if self.name == 'HWO':
            df['detected_flag_best'] = mask_best.astype(bool)
            detected = df[df['detected_flag_best']].groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
        else:
            df['detected_flag'] = mask_best.astype(bool)
            detected = df[df['detected_flag']].groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
        detected_stats = pivot_stats(detected, ['stype', 'radius_bin'])
        det_df, det_err = prep_plot_df_stars(detected_stats, STAR_ORDER, BIN_LABELS)
        heights_list = prepare_bar_lists(det_df, BIN_LABELS)
        errors_list = prepare_bar_lists(det_err, BIN_LABELS)
        bar_plot_with_errors(
            x, heights_list, errors_list, BAR_WIDTH_STAR, BIN_LABELS, colors=STAR_COLORS, hatches=STAR_HATCHES,
            xticks=x + 1.5 * BAR_WIDTH_STAR, xticklabels=STAR_ORDER, ylabel='Detected Planets',
            title=f'Detected Planets by Star Type for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}',
            legend_title='Radius Bin', filename=os.path.join(self.data_dir, output_filename('stellar_type_detected', self.name, self.nruns, self.star_catalog)),
            figsize=(12, 8)
        )
        # Best/worst overlays if HWO
        if self.name == 'HWO' and mask_worst is not None:
            df['detected_flag_worst'] = mask_worst.astype(bool)
            detected_worst = df[df['detected_flag_worst']].groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
            detected_best = df[df['detected_flag_best']].groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
            worst_stats = pivot_stats(detected_worst, ['stype', 'radius_bin'])
            best_stats = pivot_stats(detected_best, ['stype', 'radius_bin'])
            worst_df, _ = prep_plot_df_stars(worst_stats, STAR_ORDER, BIN_LABELS)
            best_df, _ = prep_plot_df_stars(best_stats, STAR_ORDER, BIN_LABELS)
            worst_list = prepare_bar_lists(worst_df, BIN_LABELS)
            best_list = prepare_bar_lists(best_df, BIN_LABELS)
            fig, ax = plt.subplots(figsize=(10, 6))
            overlay_best_worst(
                ax, x, BAR_WIDTH_STAR,
                worst_list + best_list,
                ['green'] * len(worst_list) + ['lightgreen'] * len(best_list),
                ['Worst Case (Green)'] * len(worst_list) + ['Best Case (Light Green)'] * len(best_list)
            )
            ax.set_xticks(x + 1.5 * BAR_WIDTH_STAR)
            ax.set_xticklabels(STAR_ORDER)
            ax.set_ylabel('Detected Planets (Best/Worst)')
            ax.set_title(f'Best/Worst Detected Planets by Star Type\n{self.name}, {self.nruns} Runs — {self.star_catalog}')
            ax.legend(title='Overlay', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, output_filename('stellar_type_detected', self.name, self.nruns, self.star_catalog, 'best_worst')), dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_by_planet(self) -> None:
        """Bar plots by planet category and temperature zone. Best/worst overlays for HWO."""
        df = self.df.copy()
        df['temp_zone'] = df['temp_p'].apply(temp_zone)
        df['category'] = df.apply(assign_category, axis=1)
        df = df.dropna(subset=['category'])
        x = np.arange(len(CATEGORY_LABELS))
        # Total
        total_stats = pivot_stats(df, ['category', 'temp_zone'])
        heights_list, errors_list = [], []
        for zone in TEMP_ZONES:
            data = total_stats[total_stats['temp_zone'] == zone].set_index('category').reindex(CATEGORY_LABELS)
            heights_list.append(data['count'].fillna(0).values)
            errors_list.append(data['error'].fillna(0).values)
        bar_plot_with_errors(
            x, heights_list, errors_list, BAR_WIDTH_TEMP, TEMP_ZONES, colors=TEMP_COLORS,
            xticks=x, xticklabels=CATEGORY_LABELS, ylabel='Planet Count',
            title=f'Total Planets by Type and Temp Zone\n{self.name}, {self.nruns} Runs — {self.star_catalog}',
            legend_title='Temp Zone', filename=os.path.join(self.data_dir, output_filename('planets_by_type_total', self.name, self.nruns, self.star_catalog)),
            text_offset=1
        )
        # Detected
        mask_best, mask_worst = get_detection_masks(df, self.name)
        if self.name == 'HWO':
            df['detected_flag_best'] = mask_best.astype(bool)
            detected_stats = pivot_stats(df[df['detected_flag_best']], ['category', 'temp_zone'])
        else:
            df['detected_flag'] = mask_best.astype(bool)
            detected_stats = pivot_stats(df[df['detected_flag']], ['category', 'temp_zone'])
        heights_list, errors_list = [], []
        for zone in TEMP_ZONES:
            data = detected_stats[detected_stats['temp_zone'] == zone].set_index('category').reindex(CATEGORY_LABELS)
            heights_list.append(data['count'].fillna(0).values)
            errors_list.append(data['error'].fillna(0).values)
        bar_plot_with_errors(
            x, heights_list, errors_list, BAR_WIDTH_TEMP, TEMP_ZONES, colors=TEMP_COLORS,
            xticks=x, xticklabels=CATEGORY_LABELS, ylabel='Detected Planet Count',
            title=f'Detected Planets by Type and Temp Zone\n{self.name}, {self.nruns} Runs — {self.star_catalog}',
            legend_title='Temp Zone', filename=os.path.join(self.data_dir, output_filename('planets_by_type_detected', self.name, self.nruns, self.star_catalog)),
            text_offset=1
        )
        # Best/worst overlays if HWO
        if self.name == 'HWO' and mask_worst is not None:
            df['detected_flag_worst'] = mask_worst.astype(bool)
            worst_stats = pivot_stats(df[df['detected_flag_worst']], ['category', 'temp_zone'])
            best_stats = pivot_stats(df[df['detected_flag_best']], ['category', 'temp_zone'])
            worst_list, best_list = [], []
            for zone in TEMP_ZONES:
                data_worst = worst_stats[worst_stats['temp_zone'] == zone].set_index('category').reindex(CATEGORY_LABELS)
                data_best = best_stats[best_stats['temp_zone'] == zone].set_index('category').reindex(CATEGORY_LABELS)
                worst_list.append(data_worst['count'].fillna(0).values)
                best_list.append(data_best['count'].fillna(0).values)
            fig, ax = plt.subplots(figsize=(10, 6))
            overlay_best_worst(
                ax, x, BAR_WIDTH_TEMP,
                worst_list + best_list,
                ['green'] * len(worst_list) + ['lightgreen'] * len(best_list),
                ['Worst Case (Green)'] * len(worst_list) + ['Best Case (Light Green)'] * len(best_list)
            )
            ax.set_xticks(x)
            ax.set_xticklabels(CATEGORY_LABELS, rotation=15, ha='right')
            ax.set_ylabel('Detected Planets (Best/Worst)')
            ax.set_title(f'Best/Worst Detected Planets by Type and Temp Zone\n{self.name}, {self.nruns} Runs — {self.star_catalog}')
            ax.legend(title='Overlay', fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(self.data_dir, output_filename('planets_by_type_detected', self.name, self.nruns, self.star_catalog, 'best_worst')), dpi=300, bbox_inches='tight')
            plt.close(fig)

    def plot_distances(self) -> None:
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
