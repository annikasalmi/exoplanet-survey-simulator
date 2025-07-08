import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot.plot_by_type import PlotPlanetType
from plot.plot_detections import PlanetDetectionPlotter
from tools.plotting_constants import (
    TEMP_ZONES, TEMP_COLORS, HATCHES, STAR_ORDER, BIN_LABELS, STAR_COLORS, 
    STAR_HATCHES, BAR_WIDTH_STAR, DISTANCE_LABELS, ERROR_LABEL_OFFSET
)

class PlanetDetectionPlotterLTC3(PlanetDetectionPlotter):
    """LTC3-specific detection plotter: single panel for rocky habitable zone planets."""
    
    def plot_detection_efficiency_by_planet_type(self) -> None:
        """Plot detection efficiency for rocky habitable zone planets (G/K stars only)."""
        df_filtered = self.df[
            (self.df['habitable'] == True) & 
            (self.df['radius_p'] < 1.5) & 
            (self.df['stype'].isin(['G', 'K']))
        ].copy()
        
        if len(df_filtered) == 0:
            print("No rocky planets in habitable zone found for LTC3 detection efficiency (G/K stars only)")
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
        plt.savefig(os.path.join(self.data_dir, self._output_filename('detection_efficiency_by_type_temp')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

class PlotPlanetTypeLTC3(PlotPlanetType):
    """LTC3-specific plotter with custom habitability and styling."""
    
    def _create_overlay_bars(self, ax, x, total_heights, detected_heights, 
                           total_errors=None, detected_errors=None, bar_width=0.8, 
                           total_color='lightgray', detected_color='green', 
                           detected_hatch=None, add_total_label=True, detected_label='Detected'):
        """Create overlay bars with error labels."""
        ax.bar(x, detected_heights, width=bar_width, color=detected_color, 
               alpha=0.8, edgecolor='black', yerr=detected_errors, capsize=3, 
               bottom=None, hatch=detected_hatch, label=detected_label, ecolor='black')
        
        if detected_errors is not None:
            for i, (height, error) in enumerate(zip(detected_heights, detected_errors)):
                if height > 0:
                    label_text = f'{int(height)}±{int(error)}'
                    ax.text(x[i], height + error + ERROR_LABEL_OFFSET, label_text, 
                           ha='center', va='bottom', rotation=90)
        
        return (detected_heights, detected_errors) if detected_errors is not None else (detected_heights, None)

    def _assign_category_LTC3(self, row):
        """Assign planet categories for LTC3 analysis."""
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
        
        # Create radius bins
        df['radius_bin'] = pd.cut(df['radius_p'], 
                                 bins=[0, 1.5, 3.0, 6.0, float('inf')], 
                                 labels=['<1.5', '1.5–3.0', '3.0–6.0', '>6.0'], 
                                 include_lowest=True)
        df['radius_bin'] = df['radius_bin'].cat.add_categories(['Rocky HZ'])
        
        # Mark rocky habitable zone planets
        rocky_hz_mask = (df['habitable'] == True) & (df['radius_p'] < 1.5)
        df.loc[rocky_hz_mask, 'radius_bin'] = 'Rocky HZ'
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
                if height > 0:
                    label_text = f'{int(round(height))}±{int(round(error))}'
                    ax.text(x[j] + i * bar_width, height + error + ERROR_LABEL_OFFSET, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Setup plot
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
        plt.savefig(os.path.join(self.data_dir, self._output_filename('stellar_type_overlay')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_by_planet(self) -> None:
        """Planet type by subcategory: detected planets only, error bars, integer labels."""
        df = self.df.copy()
        
        # Vectorized category assignment
        conditions = [
            (df['radius_p'] >= 0.5) & (df['radius_p'] < 1.4) & (df['habitable'] == True) & (df['stype'].isin(['G', 'K'])),
            (df['radius_p'] >= 0.5) & (df['radius_p'] < 1.4) & (df['habitable'] == True) & (df['stype'].isin(['G', 'K'])),
            (df['radius_p'] >= 1.0) & (df['radius_p'] < 1.4),
            (df['radius_p'] >= 1.4) & (df['radius_p'] < 2.6),
            (df['radius_p'] >= 2.6) & (df['radius_p'] < 4.0)
        ]
        choices = ['Rocky eHZ', 'Exo-Earths', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
        df['categories'] = np.select(conditions, choices, default=None)
        
        # Filter for valid categories
        categories = ['Rocky eHZ', 'Exo-Earths', 'Rocky + Super-Earths', 'Sub-Neptunes', 'Sub-Jovians']
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
                       edgecolor='black', alpha=0.8, hatch=hatch, yerr=detected_error, 
                       capsize=3, ecolor='black',
                       label=f'{cat} ({temp_zone})' if i_cat == 0 else None)
        
        # Setup plot
        ax.set_xticks(x + bar_width)
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Subcategory for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        
        # Add error bar labels
        if rocky_ehz_detected > 0:
            label_text = f'{rocky_ehz_detected}±{round(rocky_ehz_error)}'
            ax.text(rocky_ehz_idx, rocky_ehz_detected + rocky_ehz_error + ERROR_LABEL_OFFSET, label_text, 
                   ha='center', va='bottom', rotation=90, fontsize=10)
        
        if exo_earth_detected > 0:
            label_text = f'{exo_earth_detected}±{round(exo_earth_error)}'
            ax.text(exo_earth_idx, exo_earth_detected + exo_earth_error + ERROR_LABEL_OFFSET, label_text, 
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
                    ax.text(x[idx] + j * bar_width, detected + detected_error + ERROR_LABEL_OFFSET, label_text, 
                           ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Finalize plot
        ax.legend(title='Temperature Zone', loc='upper right')
        ax.set_ylim(bottom=0)
        y_max = ax.get_ylim()[1]
        ax.set_ylim(0, y_max * 1.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(os.path.join(self.data_dir, self._output_filename('planet_type_subcategory')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_by_distance(self) -> None:
        """Distance-based analysis for LTC3."""
        df = self.df.copy()
        
        # Create distance bins
        df['distance_bin'] = pd.cut(df['distance'], 
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
            if height > 0:
                label_text = f'{int(round(height))}±{int(round(error))}'
                ax.text(x[i], height + error + ERROR_LABEL_OFFSET, label_text, 
                       ha='center', va='bottom', rotation=90, fontsize=10)
        
        # Setup plot
        ax.set_xlabel('Distance [pc]')
        ax.set_ylabel('Number of Planets')
        ax.set_title(f'Planet Detection by Distance for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.set_xticks(x)
        ax.set_xticklabels(detected_mean['distance_bin'].values)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, self._output_filename('distance_analysis')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_detection_efficiency_rocky_habitable(self) -> None:
        """Plot detection efficiency specifically for rocky habitable zone planets."""
        df_filtered = self.df[
            (self.df['habitable'] == True) & 
            (self.df['radius_p'] < 1.5)
        ].copy()
        
        if len(df_filtered) == 0:
            print("No rocky habitable zone planets found for efficiency analysis")
            return
        
        # Calculate efficiency by stellar type
        stellar_types = ['M', 'K', 'G']
        efficiencies = []
        errors = []
        
        for stype in stellar_types:
            stype_data = df_filtered[df_filtered['stype'] == stype]
            if len(stype_data) == 0:
                efficiencies.append(0)
                errors.append(0)
                continue
            
            mask_best, _ = self._get_detection_masks()
            mask_best = mask_best[stype_data.index]
            detected_count = np.sum(mask_best)
            total_count = len(stype_data)
            
            efficiency = detected_count / total_count if total_count > 0 else 0
            efficiencies.append(efficiency)
            
            # Simple error estimate
            error = np.sqrt(efficiency * (1 - efficiency) / total_count) if total_count > 0 else 0
            errors.append(error)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        x = np.arange(len(stellar_types))
        
        ax.bar(x, efficiencies, width=0.8, color='green', alpha=0.8, 
               edgecolor='black', yerr=errors, capsize=3, ecolor='black')
        
        # Add percentage labels
        for i, (eff, err) in enumerate(zip(efficiencies, errors)):
            if eff > 0:
                label_text = f'{eff:.1%}±{err:.1%}'
                ax.text(x[i], eff + err + 0.02, label_text, 
                       ha='center', va='bottom', fontsize=10)
        
        # Setup plot
        ax.set_xlabel('Stellar Type')
        ax.set_ylabel('Detection Efficiency')
        ax.set_title(f'Detection Efficiency for Rocky Habitable Zone Planets\n{self.name} ({self.nruns} runs)')
        ax.set_xticks(x)
        ax.set_xticklabels(stellar_types)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, self._output_filename('detection_efficiency_rocky_habitable')), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_by_type_LTC3(df, nruns=1, star_catalog='LTC_3', name='LIFEsim'):
    """Main function to run LTC3 planet type analysis."""
    plotter = PlotPlanetTypeLTC3(df, nruns, star_catalog, name)
    plotter.plot_all()
    
    # Also run detection efficiency analysis
    detection_plotter = PlanetDetectionPlotterLTC3(df, nruns, star_catalog, name)
    detection_plotter.plot_detection_efficiency_by_planet_type() 