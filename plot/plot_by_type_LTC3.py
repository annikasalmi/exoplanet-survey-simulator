import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from plot.plot_by_type import PlotPlanetType
from plot.plot_detections import PlanetDetectionPlotter
from tools.plotting_constants import (TEMP_ZONES, TEMP_COLORS, HATCHES, STAR_ORDER, 
                                     BIN_LABELS, STAR_COLORS, STAR_HATCHES, BAR_WIDTH_STAR, DISTANCE_LABELS)


class PlanetDetectionPlotterLTC3(PlanetDetectionPlotter):
    """LTC3-specific detection plotter that creates single panel plots for rocky habitable zone planets."""
    
    def plot_detection_efficiency_by_planet_type(self) -> None:
        """Override to create single panel for LTC3: rocky habitable zone planets only."""
        # Filter for rocky planets in habitable zone using habitable=True flag
        df_filtered = self.df[(self.df['habitable'] == True) & (self.df['radius_p'] < 1.5)].copy()
        
        if len(df_filtered) == 0:
            print("No rocky planets in habitable zone found for LTC3 detection efficiency")
            return
        
        # Create single panel plot for temperature only
        config = {'col': 'temp_p', 'label': 'Temperature [K]', 'range': (125, 305)}
        x_col = config['col']
        x_label = config['label']
        x_range = config['range']
        
        # Setup figure - single panel instead of 2x2
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bins = np.linspace(x_range[0], x_range[1], 40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Calculate data
        total_counts, detected_counts, efficiency, mask_best = self._calculate_efficiency_data(df_filtered, x_col, bins)
        total_planets = len(df_filtered) / self.nruns
        detected_planets = np.sum(detected_counts)
        
        # Create plot
        title = f"Rocky Planets in Habitable Zone\nTotal: {total_planets:.1f}, Detected: {detected_planets:.1f}"
        ax2 = self._setup_bar_and_efficiency_axes(
            ax, bin_centers, total_counts, detected_counts, efficiency, x_label, title, bins
        )
        
        # Add legend
        handles, labels = self._collect_legend_handles(ax)
        ax.legend(handles, labels, loc='upper left', fontsize=14)
        
        # Set y-labels
        ax.set_ylabel("Number of Planets")
        ax2.set_ylabel("Detection Efficiency")
        
        # Set xlim for temperature plots
        ax.set_xlim([bins[0], 305])
        ax2.set_xlim([bins[0], 305])
        
        # Finalize plot - no duplicate title
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save plot with the original filename that the pipeline expects
        plt.savefig(os.path.join(self.data_dir, 
                                self._output_filename('detection_efficiency_by_type_temp')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)


class PlotPlanetTypeLTC3(PlotPlanetType):
    """LTC3-specific plotter with custom habitability and styling."""
    
    def _create_overlay_bars(self, ax, x, total_heights, detected_heights, 
                           total_errors=None, detected_errors=None,
                           bar_width=0.8, total_color='lightgray', 
                           detected_color='green', detected_hatch=None,
                           add_total_label=True, detected_label='Detected'):
        """Override to use black error bars, no gray background, and add error bar labels for LTC3 plots."""
        # For LTC3, only plot detected bars with error bars, no gray background
        ax.bar(x, detected_heights, width=bar_width, color=detected_color,
               alpha=0.8, edgecolor='black', yerr=detected_errors, capsize=3,
               bottom=None, hatch=detected_hatch,
               label=detected_label, ecolor='black')
        
        # Add error bar labels showing number ± error, rotated 90 degrees
        if detected_errors is not None:
            for i, (height, error) in enumerate(zip(detected_heights, detected_errors)):
                if height > 0:  # Only add labels for bars with data
                    label_text = f'{int(height)}±{int(error)}'
                    ax.text(x[i], height + error + 0.1, label_text, 
                           ha='center', va='bottom', rotation=90)
        
        if detected_errors is not None:
            return detected_heights, detected_errors
        else:
            return detected_heights, None

    def _assign_category_LTC3(self, row):
        """Assign custom categories for LTC3 plot using habitable=True flag."""
        r = row['radius_p']
        stype = row['stype']
        hab = row.get('habitable', False)
        categories = []
        
        # Rocky eHZ: rocky (0.5-1.4 R_Earth) in habitable zone
        if 0.5 <= r < 1.4 and hab:
            categories.append('Rocky eHZ')
        # Exo-Earth Candidates: rocky in habitable zone of G-type stars
        if 0.5 <= r < 1.4 and hab and stype == 'G':
            categories.append('Exo-Earths')
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

    def plot_by_star(self) -> None:
        """Override plot_by_star for LTC3: remove detection percentages and add error bar labels."""
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
        
        # Detected stats only (no total stats for LTC3)
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]  # Align mask with filtered df
        detected_df = df[mask_best].copy()
        
        # Optimize data processing
        detected_per_run = detected_df.groupby(['run', 'stype', 'radius_bin']).size().reset_index(name='count')
        detected_pivot = detected_per_run.pivot_table(index=['stype', 'radius_bin'], columns='run', values='count', fill_value=0)
        detected_mean = detected_pivot.mean(axis=1).reset_index(name='count')
        detected_std = detected_pivot.std(axis=1).reset_index(name='count')
        
        # Create the overlay plot
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = BAR_WIDTH_STAR
        
        # Pre-compute data for efficiency
        data_matrix = np.zeros((len(BIN_LABELS), len(STAR_ORDER)))
        error_matrix = np.zeros((len(BIN_LABELS), len(STAR_ORDER)))
        
        # Fill data matrices efficiently
        for i, bin_label in enumerate(BIN_LABELS):
            bin_data = detected_mean[detected_mean['radius_bin'] == bin_label]
            bin_std = detected_std[detected_std['radius_bin'] == bin_label]
            
            for j, star in enumerate(STAR_ORDER):
                star_data = bin_data[bin_data['stype'] == star]
                star_std = bin_std[bin_std['stype'] == star]
                
                data_matrix[i, j] = star_data['count'].iloc[0] if len(star_data) > 0 else 0
                error_matrix[i, j] = star_std['count'].iloc[0] if len(star_std) > 0 else 0
        
        # Plot bars efficiently
        for i, bin_label in enumerate(BIN_LABELS):
            heights = data_matrix[i, :]
            errors = error_matrix[i, :]
            
            # Plot only detected bars with error bars
            ax.bar(x + i * bar_width, heights, width=bar_width, 
                   color=STAR_COLORS[i], alpha=0.8, edgecolor='black', 
                   yerr=errors, capsize=3, ecolor='black',
                   hatch=STAR_HATCHES[i], label=bin_label)
            
            # Add error bar labels rotated 90 degrees, ± error to nearest whole number, normal font
            for j, (height, error) in enumerate(zip(heights, errors)):
                if height > 0:
                    label_text = f'{int(round(height))}±{int(round(error))}'
                    ax.text(x[j] + i * bar_width, height + error + 0.1, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        ax.set_xlabel('Star Type')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Star Type for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(STAR_ORDER)
        ax.legend(title='Radius Bin')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Extend y-axis to make room for error bar labels
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.3)  # Increase y-axis by 30% to accommodate labels
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(self.data_dir, 
                                self._output_filename('stellar_type_overlay')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_by_planet(self) -> None:
        """Custom plot_by_planet for LTC_3 with new subcategories and custom coloring/stacking."""
        df = self.df.copy()
        df['categories'] = df.apply(self._assign_category_LTC3, axis=1)
        df = df.explode('categories')
        categories = ['Rocky eHZ', 'Exo-Earth Candidates', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df = df[df['categories'].isin(categories)]
        
        # Calculate stats efficiently
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
        
        # Pre-compute data for efficiency
        bar_data = {}
        for cat in categories:
            cat_data = detected_stats[detected_stats['categories'] == cat]
            bar_data[cat] = {
                'count': cat_data['count'].sum(),
                'error': cat_data['error'].sum()
            }
        
        # Plot Rocky eHZ (not separated by temperature)
        rocky_ehz_idx = 0
        rocky_ehz_detected = round(bar_data['Rocky eHZ']['count'])
        rocky_ehz_error = bar_data['Rocky eHZ']['error']
        ax.bar(x[rocky_ehz_idx], rocky_ehz_detected, width=bar_width, color='darkgreen', 
               edgecolor='black', yerr=rocky_ehz_error, capsize=3, ecolor='black', label='Rocky eHZ')
        
        # Plot Exo-Earth Candidates (not separated by temperature)
        exo_earth_idx = 1
        exo_earth_detected = round(bar_data['Exo-Earth Candidates']['count'])
        exo_earth_error = bar_data['Exo-Earth Candidates']['error']
        ax.bar(x[exo_earth_idx], exo_earth_detected, width=bar_width, color='limegreen', 
               edgecolor='black', yerr=exo_earth_error, capsize=3, ecolor='black', label='Exo-Earth Candidates')
        
        # Plot last 3 categories, separated by temperature
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
        
        # X-axis and labels
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Subcategory for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        
        # Add error bar labels efficiently
        if rocky_ehz_detected > 0:
            label_text = f'{rocky_ehz_detected}±{round(rocky_ehz_error)}'
            ax.text(rocky_ehz_idx, rocky_ehz_detected + rocky_ehz_error + 0.1, label_text, 
                   ha='center', va='bottom', rotation=90, fontsize=10)
        
        if exo_earth_detected > 0:
            label_text = f'{exo_earth_detected}±{round(exo_earth_error)}'
            ax.text(exo_earth_idx, exo_earth_detected + exo_earth_error + 0.1, label_text, 
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
                    ax.text(idx + j * bar_width, detected + detected_error + 0.1, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Custom legend - only temperature zones, no title
        handles, labels = [], []
        # Add temperature zone colors only
        for color, temp_zone in zip(TEMP_COLORS, TEMP_ZONES):
            handles.append(plt.Rectangle((0,0),1,1, color=color,ec='black'))
            labels.append(temp_zone)
        ax.legend(handles, labels, loc='upper left')
        
        # Extend y-axis to make room for error bar labels
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.3)  # Increase y-axis by 30% to accommodate labels
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_type_LTC3')

    def plot_by_distance(self) -> None:
        """Custom plot_by_distance for LTC_3: detected bars are light brown with '/' hatching, no gray background."""
        stats, detected_stats = self._calculate_distance_stats(self.df)
        
        # Setup plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(DISTANCE_LABELS))
        
        # Calculate total and detected counts across all temperature zones efficiently
        detected_counts = np.zeros(len(DISTANCE_LABELS))
        detected_errors = np.zeros(len(DISTANCE_LABELS))
        
        # Sum across all temperature zones efficiently
        for bin_label in ['hot', 'habitable', 'cold']:
            bin_detected = detected_stats[detected_stats['temp_zone'] == bin_label]
            bin_detected = bin_detected.set_index('distance_bin').reindex(DISTANCE_LABELS).fillna(0)
            detected_counts += bin_detected['count'].values
            detected_errors = np.maximum(detected_errors, bin_detected['error'].values)
        
        # Plot detected bars only, in light brown with '/' hatching and black error bars
        ax.bar(x, detected_counts, width=0.6, color='#c2b280', edgecolor='black', hatch='/', alpha=0.8,
               yerr=detected_errors, capsize=3, ecolor='black')
        
        # Add labels to error bars showing number +/- error, rotated 90 degrees
        for i, (count, error) in enumerate(zip(detected_counts, detected_errors)):
            if count > 0:  # Only add labels for bars with data
                label_text = f'{count:.1f}±{error:.1f}'
                ax.text(i, count + error + 0.1, label_text, ha='center', va='bottom', 
                       rotation=90, fontsize=10)
        
        # Setup plot without legend
        ax.set_xlabel('Distance [pc]')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Distance Bin for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x)
        ax.set_xticklabels(DISTANCE_LABELS)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        self._save_plot(fig, 'planet_detection_by_distance_LTC3')

    def plot_detection_efficiency_rocky_habitable(self) -> None:
        """Custom detection efficiency plot for LTC3: only rocky planets in habitable zone."""
        # Filter for rocky planets in habitable zone using habitable=True flag
        df_filtered = self.df[(self.df['habitable'] == True) & (self.df['radius_p'] < 1.5)].copy()
        
        if len(df_filtered) == 0:
            print("No rocky planets in habitable zone found for LTC3 detection efficiency")
            return
        
        # Create detection plotter with filtered data
        detection_plotter = PlanetDetectionPlotter(df_filtered, self.nruns, self.star_catalog, self.name)
        
        # Create single panel plot for temperature
        config = {'col': 'temp_p', 'label': 'Temperature [K]', 'range': (125, 305)}
        x_col = config['col']
        x_label = config['label']
        x_range = config['range']
        
        # Setup figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bins = np.linspace(x_range[0], x_range[1], 40)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        # Calculate data
        total_counts, detected_counts, efficiency, mask_best = detection_plotter._calculate_efficiency_data(df_filtered, x_col, bins)
        total_planets = len(df_filtered) / self.nruns
        detected_planets = np.sum(detected_counts)
        
        # Create plot
        title = f"Rocky Planets in Habitable Zone\nTotal: {total_planets:.1f}, Detected: {detected_planets:.1f}"
        ax2 = detection_plotter._setup_bar_and_efficiency_axes(
            ax, bin_centers, total_counts, detected_counts, efficiency, x_label, title, bins
        )
        
        # Add legend
        handles, labels = detection_plotter._collect_legend_handles(ax)
        ax.legend(handles, labels, loc='upper left', fontsize=14)
        
        # Set y-labels
        ax.set_ylabel("Number of Planets")
        ax2.set_ylabel("Detection Efficiency")
        
        # Set xlim for temperature plots
        ax.set_xlim([bins[0], 305])
        ax2.set_xlim([bins[0], 305])
        
        # Finalize plot
        title = f"Detection Efficiency for Rocky Planets in Habitable Zone - {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}"
        fig.suptitle(title, fontsize=20, y=0.96)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save plot
        plt.savefig(os.path.join(self.data_dir, self._output_filename('detection_efficiency_rocky_habitable')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)


def plot_by_type_LTC3(df, nruns=1, star_catalog='LTC_3', name='LIFEsim'):
    """Main function to create LTC3-specific plots."""
    plotter = PlotPlanetTypeLTC3(df=df, nruns=nruns, star_catalog=star_catalog, name=name)
    plotter.plot_all()
    # Add the custom detection efficiency plot for rocky planets in habitable zone
    plotter.plot_detection_efficiency_rocky_habitable() 