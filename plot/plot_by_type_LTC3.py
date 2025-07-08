import matplotlib.pyplot as plt
from plot.plot_by_type import PlotPlanetType
import numpy as np
import pandas as pd

class PlotPlanetTypeLTC3(PlotPlanetType):
    def _create_overlay_bars(self, ax, x, total_heights, detected_heights, 
                           total_errors=None, detected_errors=None,
                           bar_width=0.8, total_color='lightgray', 
                           detected_color='green', detected_hatch=None,
                           add_total_label=True, detected_label='Detected'):
        """Override to only plot detected bars (no gray background)."""
        # Only plot detected bars
        ax.bar(x, detected_heights, width=bar_width, color=detected_color,
               alpha=0.8, edgecolor='black', yerr=detected_errors, capsize=3,
               bottom=None, hatch=detected_hatch,
               label=detected_label, ecolor='darkgray')
        if detected_errors is not None:
            return detected_heights, detected_errors
        else:
            return detected_heights, None

    def _assign_category_LTC3(self, row):
        r = row['radius_p']
        stype = row['stype']
        hab = row.get('habitable', False)
        categories = []
        # Rocky eHZ: rocky (0.5-1.4 R_Earth) in habitable zone
        if 0.5 <= r < 1.4 and hab:
            categories.append('Rocky eHZ')
        # Exo-Earth Candidates: rocky in habitable zone of G-type stars
        if 0.5 <= r < 1.4 and hab and stype == 'G':
            categories.append('Exo-Earth Candidates')
        # Rocky + Super-Earths: 1-1.4 R_Earth
        if 1.0 <= r < 1.4:
            categories.append('Rocky + Super-Earths')
        # Sub-Neptunes: 1.4-2.6 R_Earth
        if 1.4 <= r < 2.6:
            categories.append('Sub-Neptunes')
        # Sub-Jovians: 2.6-4 R_Earth
        if 2.6 <= r < 4.0:
            categories.append('Sub-Jovians')
        return categories if categories else None

    def plot_by_planet(self) -> None:
        """Custom plot_by_planet for LTC_3 with new subcategories and custom coloring/stacking."""
        df = self.df.copy()
        df['categories'] = df.apply(self._assign_category_LTC3, axis=1)
        df = df.explode('categories')
        categories = ['Rocky eHZ', 'Exo-Earth Candidates', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df = df[df['categories'].isin(categories)]
        # Calculate stats for all
        stats = self._pivot_stats(df, ['categories', 'temp_zone'])
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]  # Align mask with filtered df
        df_detected = df[mask_best].copy()
        df_detected['categories'] = df_detected.apply(self._assign_category_LTC3, axis=1)
        df_detected = df_detected.explode('categories')
        df_detected = df_detected[df_detected['categories'].isin(categories)]
        detected_stats = self._pivot_stats(df_detected, ['categories', 'temp_zone'])
        # Setup plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(categories))
        bar_width = 0.2
        # Colors for bars
        bar_colors = ['darkgreen', 'limegreen', None, None, None]
        bar_labels = ['Rocky eHZ', 'Exo-Earth Candidates', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        from tools.plotting_constants import TEMP_ZONES, TEMP_COLORS, HATCHES
        # Plot Rocky eHZ (not separated by temperature)
        rocky_ehz_idx = 0
        rocky_ehz_detected = detected_stats[detected_stats['categories'] == 'Rocky eHZ']['count'].sum()
        ax.bar(x[rocky_ehz_idx], rocky_ehz_detected, width=bar_width, color='darkgreen', edgecolor='black', label='Rocky eHZ')
        # Plot Exo-Earth Candidates (not separated by temperature)
        exo_earth_idx = 1
        exo_earth_detected = detected_stats[detected_stats['categories'] == 'Exo-Earth Candidates']['count'].sum()
        ax.bar(x[exo_earth_idx], exo_earth_detected, width=bar_width, color='limegreen', edgecolor='black', label='Exo-Earth Candidates')
        # Plot last 3 categories, separated by temperature
        for i_cat, (cat, color, hatch) in enumerate(zip(['Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians'], TEMP_COLORS, HATCHES)):
            idx = i_cat + 2
            for j, temp_zone in enumerate(TEMP_ZONES):
                detected = detected_stats[(detected_stats['categories'] == cat) & (detected_stats['temp_zone'] == temp_zone)]['count']
                detected = detected.values[0] if len(detected) > 0 else 0
                ax.bar(x[idx] + j * bar_width, detected, width=bar_width, color=color, edgecolor='black', hatch=hatch, alpha=0.8, label=f'{cat} ({temp_zone})' if i_cat == 0 else None)
        # X-axis and labels
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Subcategory for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        # Custom legend
        handles, labels = [], []
        for bar, label in zip([plt.Rectangle((0,0),1,1, color='darkgreen',ec='black'),
                               plt.Rectangle((0,0),1,1, color='limegreen',ec='black')],
                              ['Rocky eHZ', 'Exo-Earth Candidates']):
            handles.append(bar)
            labels.append(label)
        for color, hatch, temp_zone in zip(TEMP_COLORS, HATCHES, TEMP_ZONES):
            handles.append(plt.Rectangle((0,0),1,1, color=color,ec='black',hatch=hatch,alpha=0.8))
            labels.append(temp_zone)
        ax.legend(handles, labels, title='Category / Temp. Zone', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_type_LTC3')

    def plot_by_distance(self) -> None:
        """Custom plot_by_distance for LTC_3: detected bars are light brown with '/' hatching, no gray background."""
        stats, detected_stats = self._calculate_distance_stats(self.df)
        from tools.plotting_constants import DISTANCE_LABELS
        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(DISTANCE_LABELS))
        # Calculate total and detected counts across all temperature zones
        detected_counts = np.zeros(len(DISTANCE_LABELS))
        detected_errors = np.zeros(len(DISTANCE_LABELS))
        # Sum across all temperature zones
        for bin_label in ['hot', 'habitable', 'cold']:
            bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
            bin_detected = bin_detected.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
            detected_counts += bin_detected['count'].values
            detected_errors = np.maximum(detected_errors, bin_detected['error'].values)
        # Plot detected bars only, in light brown with '/' hatching
        ax.bar(x, detected_counts, width=0.6, color='#c2b280', edgecolor='black', hatch='/', alpha=0.8, label='Detected')
        # Setup plot
        self._setup_plot_style(
            ax, 'Distance Bin', 'Number of Planets',
            f'Planet Detection by Distance Bin for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(DISTANCE_LABELS)
        # Add percentage labels (since no total, just show detected as 100%)
        for i, count in enumerate(detected_counts):
            if count > 0:
                ax.text(x[i], count + detected_errors[i] + 2, f'{int(count)}', ha='center', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_distance_LTC3')

# The following function can be called from plot.py or run_sim.py

def plot_by_type_LTC3(df, nruns=1, star_catalog='LTC_3', name='LIFEsim'):
    plotter = PlotPlanetTypeLTC3(df=df, nruns=nruns, star_catalog=star_catalog, name=name)
    plotter.plot_all() 