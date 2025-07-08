import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot.plot_by_type import PlotPlanetType
from plot.plot_detections import PlanetDetectionPlotter
from tools.plotting_constants import (
    TEMP_ZONES, TEMP_COLORS, HATCHES, STAR_ORDER, BIN_LABELS, STAR_COLORS, STAR_HATCHES, BAR_WIDTH_STAR, DISTANCE_LABELS
)

class PlanetDetectionPlotterLTC3(PlanetDetectionPlotter):
    """LTC3-specific detection plotter: single panel for rocky habitable zone planets."""
    def plot_detection_efficiency_by_planet_type(self) -> None:
        df_filtered = self.df[(self.df['habitable'] == True) & (self.df['radius_p'] < 1.5) & (self.df['stype'].isin(['F', 'G', 'K', 'M']))].copy()
        if len(df_filtered) == 0:
            print("No rocky planets in habitable zone found for LTC3 detection efficiency")
            return
        config = {'col': 'temp_p', 'label': 'Temperature [K]', 'range': (125, 305)}
        x_col, x_label, x_range = config['col'], config['label'], config['range']
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bins = np.linspace(x_range[0], x_range[1], 40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        total_counts, detected_counts, efficiency, _ = self._calculate_efficiency_data(df_filtered, x_col, bins)
        total_planets = len(df_filtered) / self.nruns
        detected_planets = np.sum(detected_counts)
        title = f"Rocky Planets in Habitable Zone\nTotal: {total_planets:.1f}, Detected: {detected_planets:.1f}"
        ax2 = self._setup_bar_and_efficiency_axes(ax, bin_centers, total_counts, detected_counts, efficiency, x_label, title, bins)
        handles, labels = self._collect_legend_handles(ax)
        ax.legend(handles, labels, loc='upper left', fontsize=14)
        ax.set_ylabel("Number of Planets")
        ax2.set_ylabel("Detection Efficiency")
        ax.set_xlim([bins[0], 305])
        ax2.set_xlim([bins[0], 305])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.data_dir, self._output_filename('detection_efficiency_by_type_temp')), dpi=300, bbox_inches='tight')
        plt.close(fig)

class PlotPlanetTypeLTC3(PlotPlanetType):
    """LTC3-specific plotter with custom habitability and styling."""
    def _create_overlay_bars(self, ax, x, total_heights, detected_heights, total_errors=None, detected_errors=None, bar_width=0.8, total_color='lightgray', detected_color='green', detected_hatch=None, add_total_label=True, detected_label='Detected'):
        ax.bar(x, detected_heights, width=bar_width, color=detected_color, alpha=0.8, edgecolor='black', yerr=detected_errors, capsize=3, bottom=None, hatch=detected_hatch, label=detected_label, ecolor='black')
        if detected_errors is not None:
            for i, (height, error) in enumerate(zip(detected_heights, detected_errors)):
                if height > 0:
                    label_text = f'{int(height)}±{int(error)}'
                    ax.text(x[i], height + error + 0.3, label_text, ha='center', va='bottom', rotation=90)
        return (detected_heights, detected_errors) if detected_errors is not None else (detected_heights, None)

    def _assign_category_LTC3(self, row):
        r = row['radius_p']
        stype = row['stype']
        hab = row.get('habitable', False)
        categories = []
        if 0.5 <= r < 1.4 and hab:
            categories.append('Rocky eHZ')
        if 0.5 <= r < 1.4 and hab and stype in ['G', 'K']:
            categories.append('Exo-Earths')
        if 1.0 <= r < 1.4:
            categories.append('Rocky + Super-Earths')
        if 1.4 <= r < 2.6:
            categories.append('Sub-Neptunes')
        if 2.6 <= r < 4.0:
            categories.append('Sub-Jovians')
        return categories if categories else None

    def plot_by_star(self) -> None:
        """Stellar type overlay: detected planets only, error bars, integer labels."""
        df = self.df.copy()
        x = np.arange(len(STAR_ORDER))
        df['radius_bin'] = pd.cut(df['radius_p'], bins=[0, 1.5, 3.0, 6.0, float('inf')], labels=['<1.5', '1.5–3.0', '3.0–6.0', '>6.0'], include_lowest=True)
        df['radius_bin'] = df['radius_bin'].cat.add_categories(['Rocky HZ'])
        rocky_hz_mask = (df['habitable'] == True) & (df['radius_p'] < 1.5)
        df.loc[rocky_hz_mask, 'radius_bin'] = 'Rocky HZ'
        df = df[df['radius_bin'].isin(BIN_LABELS)]
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]
        detected_df = df[mask_best].copy()
        detected_per_run = detected_df.groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
        detected_pivot = detected_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        detected_mean = detected_pivot.mean(axis=1).reset_index(name='count')
        detected_std = detected_pivot.std(axis=1).reset_index(name='count')
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = BAR_WIDTH_STAR
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
        for i, bin_label in enumerate(BIN_LABELS):
            heights = data_matrix[i, :]
            errors = error_matrix[i, :]
            ax.bar(x + i * bar_width, heights, width=bar_width, color=STAR_COLORS[i], alpha=0.8, edgecolor='black', yerr=errors, capsize=3, ecolor='black', hatch=STAR_HATCHES[i], label=bin_label)
            for j, (height, error) in enumerate(zip(heights, errors)):
                if height > 0:
                    label_text = f'{int(round(height))}±{int(round(error))}'
                    ax.text(x[j] + i * bar_width, height + error + 0.3, label_text, ha='center', va='bottom', rotation=90, fontsize=10)
        ax.set_xlabel('Star Type')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Star Type for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.legend(title='Radius Bin')
        ax.set_ylim(bottom=0)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.3)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(self.data_dir, self._output_filename('stellar_type_overlay')), dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_by_planet(self) -> None:
        """Planet type by subcategory: detected planets only, error bars, integer labels."""
        # Pre-compute categories once to avoid repeated apply operations
        df = self.df.copy()
        
        # Vectorized category assignment (much faster than apply)
        conditions = [
            (df['radius_p'] >= 0.5) & (df['radius_p'] < 1.4) & (df['habitable'] == True),
            (df['radius_p'] >= 0.5) & (df['radius_p'] < 1.4) & (df['habitable'] == True) & (df['stype'].isin(['G', 'K'])),
            (df['radius_p'] >= 1.0) & (df['radius_p'] < 1.4),
            (df['radius_p'] >= 1.4) & (df['radius_p'] < 2.6),
            (df['radius_p'] >= 2.6) & (df['radius_p'] < 4.0)
        ]
        choices = ['Rocky eHZ', 'Exo-Earths', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df['categories'] = np.select(conditions, choices, default=None)
        
        # Filter for valid categories only
        categories = ['Rocky eHZ', 'Exo-Earths', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df = df[df['categories'].isin(categories)]
        
        # Get detection masks once
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]
        df_detected = df[mask_best].copy()
        
        # Calculate stats efficiently
        detected_stats = self._pivot_stats(df_detected, ['categories', 'temp_zone'])
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        bar_width = 0.2
        
        # Pre-compute data efficiently
        bar_data = {}
        for cat in categories:
            cat_data = detected_stats[detected_stats['categories'] == cat]
            bar_data[cat] = {'count': cat_data['count'].sum(), 'error': cat_data['error'].sum()}
        
        # Plot Rocky eHZ
        rocky_ehz_idx = 0
        rocky_ehz_detected = round(bar_data['Rocky eHZ']['count'])
        rocky_ehz_error = bar_data['Rocky eHZ']['error']
        ax.bar(x[rocky_ehz_idx], rocky_ehz_detected, width=bar_width, color='darkgreen', 
               edgecolor='black', yerr=rocky_ehz_error, capsize=3, ecolor='black', label='Rocky eHZ')
        
        # Plot Exo-Earths
        exo_earth_idx = 1
        exo_earth_detected = round(bar_data['Exo-Earths']['count'])
        exo_earth_error = bar_data['Exo-Earths']['error']
        ax.bar(x[exo_earth_idx], exo_earth_detected, width=bar_width, color='limegreen', 
               edgecolor='black', yerr=exo_earth_error, capsize=3, ecolor='black', label='Exo-Earths')
        
        # Plot temperature-separated categories
        for i_cat, cat in enumerate(['Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']):
            idx = i_cat + 2
            for j, (temp_zone, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
                cat_temp_data = detected_stats[(detected_stats['categories'] == cat) & 
                                             (detected_stats['temp_zone'] == temp_zone)]
                detected = round(cat_temp_data['count'].iloc[0]) if len(cat_temp_data) > 0 else 0
                detected_error = cat_temp_data['error'].iloc[0] if len(cat_temp_data) > 0 else 0
                
                ax.bar(x[idx] + j * bar_width, detected, width=bar_width, color=color, 
                       edgecolor='black', alpha=0.8, hatch=hatch, yerr=detected_error, capsize=3, ecolor='black',
                       label=f'{cat} ({temp_zone})' if i_cat == 0 else None)
        
        # Setup plot
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Subcategory for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        
        # Add error bar labels efficiently
        if rocky_ehz_detected > 0:
            label_text = f'{rocky_ehz_detected}±{round(rocky_ehz_error)}'
            ax.text(rocky_ehz_idx, rocky_ehz_detected + rocky_ehz_error + 0.3, label_text, 
                   ha='center', va='bottom', rotation=90, fontsize=10)
        
        if exo_earth_detected > 0:
            label_text = f'{exo_earth_detected}±{round(exo_earth_error)}'
            ax.text(exo_earth_idx, exo_earth_detected + exo_earth_error + 0.3, label_text, 
                   ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Add error bar labels for temperature zone bars
        for i_cat, cat in enumerate(['Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']):
            idx = i_cat + 2
            for j, (temp_zone, color, hatch) in enumerate(zip(TEMP_ZONES, TEMP_COLORS, HATCHES)):
                cat_temp_data = detected_stats[(detected_stats['categories'] == cat) & 
                                             (detected_stats['temp_zone'] == temp_zone)]
                detected = round(cat_temp_data['count'].iloc[0]) if len(cat_temp_data) > 0 else 0
                detected_error = cat_temp_data['error'].iloc[0] if len(cat_temp_data) > 0 else 0
                
                if detected > 0:
                    label_text = f'{detected}±{round(detected_error)}'
                    ax.text(idx + j * bar_width, detected + detected_error + 0.3, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Custom legend
        handles, labels = [], []
        for color, temp_zone in zip(TEMP_COLORS, TEMP_ZONES):
            handles.append(plt.Rectangle((0,0),1,1, color=color,ec='black'))
            labels.append(temp_zone)
        ax.legend(handles, labels, loc='upper left')
        
        # Extend y-axis for labels
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_type_LTC3')

    def plot_by_distance(self) -> None:
        """Distance plot: detected bars only, error bars, integer labels."""
        stats, detected_stats = self._calculate_distance_stats(self.df)
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(DISTANCE_LABELS))
        detected_counts = np.zeros(len(DISTANCE_LABELS))
        detected_errors = np.zeros(len(DISTANCE_LABELS))
        for bin_label in ['hot', 'habitable', 'cold']:
            bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
            bin_detected = bin_detected.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
            detected_counts += bin_detected['count'].values
            detected_errors = np.maximum(detected_errors, bin_detected['error'].values)
        ax.bar(x, detected_counts, width=0.6, color='#c2b280', edgecolor='black', hatch='/', alpha=0.8, yerr=detected_errors, capsize=3, ecolor='black')
        for i, (count, error) in enumerate(zip(detected_counts, detected_errors)):
            if count > 0:
                label_text = f'{int(round(count))}±{int(round(error))}'
                ax.text(i, count + error + 0.3, label_text, ha='center', va='bottom', rotation=90, fontsize=10)
        ax.set_xlabel('Distance [pc]')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Distance Bin for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x)
        ax.set_xticklabels(DISTANCE_LABELS)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_distance_LTC3')

    def plot_detection_efficiency_rocky_habitable(self) -> None:
        df_filtered = self.df[(self.df['habitable'] == True) & (self.df['radius_p'] < 1.5)].copy()
        if len(df_filtered) == 0:
            print("No rocky planets in habitable zone found for LTC3 detection efficiency")
            return
        detection_plotter = PlanetDetectionPlotter(df_filtered, self.nruns, self.star_catalog, self.name)
        config = {'col': 'temp_p', 'label': 'Temperature [K]', 'range': (125, 305)}
        x_col, x_label, x_range = config['col'], config['label'], config['range']
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bins = np.linspace(x_range[0], x_range[1], 40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        total_counts, detected_counts, efficiency, _ = detection_plotter._calculate_efficiency_data(df_filtered, x_col, bins)
        total_planets = len(df_filtered) / self.nruns
        detected_planets = np.sum(detected_counts)
        title = f"Rocky Planets in Habitable Zone\nTotal: {total_planets:.1f}, Detected: {detected_planets:.1f}"
        ax2 = detection_plotter._setup_bar_and_efficiency_axes(ax, bin_centers, total_counts, detected_counts, efficiency, x_label, title, bins)
        handles, labels = detection_plotter._collect_legend_handles(ax)
        ax.legend(handles, labels, loc='upper left', fontsize=14)
        ax.set_ylabel("Number of Planets")
        ax2.set_ylabel("Detection Efficiency")
        ax.set_xlim([bins[0], 305])
        ax2.set_xlim([bins[0], 305])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(self.data_dir, self._output_filename('detection_efficiency_rocky_habitable')), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_by_type_LTC3(df, nruns=1, star_catalog='LTC_3', name='LIFEsim'):
    plotter = PlotPlanetTypeLTC3(df=df, nruns=nruns, star_catalog=star_catalog, name=name)
    plotter.plot_all()
    plotter.plot_detection_efficiency_rocky_habitable() 