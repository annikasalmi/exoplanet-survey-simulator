import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from plot.base_plotter import BasePlotter
from tools.plotting_constants import PLOT_CONFIGS, PANEL_CONFIGS

class PlanetDetectionPlotter(BasePlotter):
    """
    Class for generating detection efficiency plots for HWO scenarios.
    Uses detected_best columns for overlays.
    """
    
    def plot_all(self, temp_plots=False) -> None:
        """Generate all detection efficiency plots."""
        if not self._validate_data():
            return
            
        self.plot_detection_efficiency_by_planet_type()
        self.plot_detection_efficiency_3d_panels()
        if temp_plots:
            self.plot_detection_scatter_by_parameter()

    def _calculate_efficiency_data(self, df, x_col, bins):
        """Calculate detection efficiency data for given dataframe and bins."""
        total_counts, _ = np.histogram(df[x_col], bins=bins)
        total_counts = total_counts / self.nruns
        
        mask_best, _ = self._get_detection_masks()
        detected_counts, _ = np.histogram(df[mask_best][x_col], bins=bins)
        detected_counts = detected_counts / self.nruns
        
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = np.true_divide(detected_counts, total_counts)
            efficiency[np.isnan(efficiency)] = 0.0
        
        return total_counts, detected_counts, efficiency, mask_best

    def _create_bar_plot_with_efficiency(self, ax1, bin_centers, total_counts, detected_counts, 
                                        efficiency, x_label, title, bins):
        """Create a bar plot with efficiency overlay."""
        # Plot bars
        ax1.bar(bin_centers, total_counts, width=np.diff(bins), 
                color='lightgrey', align='center', label='Total')
        ax1.bar(bin_centers, detected_counts, width=np.diff(bins), 
                color='green', alpha=0.8, align='center', label='Detected')
        
        # Setup primary axis
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Number of Planets")
        ax1.set_title(title)
        
        # Add efficiency line on secondary axis
        ax2 = ax1.twinx()
        ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Efficiency')
        ax2.set_ylabel("Detection Efficiency")
        ax2.set_ylim(0, 1.0)
        
        # Combine legends
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper left')
        
        return ax2

    def plot_detection_efficiency_by_planet_type(self) -> None:
        """Plot detection efficiency for different planet types in a 2x2 multipanel figure."""
        for x_axis, config in PLOT_CONFIGS.items():
            x_col = config['col']
            x_label = config['label']
            x_range = config['range']
            
            # Setup figure and bins
            fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharey=False)
            axs = axs.flatten()
            bins = np.linspace(x_range[0], x_range[1], 40)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            
            # Define combined categories: stellar type + planet type
            combined_categories = {
                'M dwarf + Rocky HZ': lambda df: (df['habitable'] == True) & 
                                                 (df['radius_p'] < 1.5) & 
                                                 (df['stype'].str.contains('M')) & 
                                                 (df['temp_p'] <= 350),
                'G/K star + Rocky HZ': lambda df: (df['habitable'] == True) & 
                                                  (df['radius_p'] < 1.5) & 
                                                  (df['stype'].isin(['G', 'K'])) & 
                                                  (df['temp_p'] <= 350),
                'M dwarf + Hycean HZ': lambda df: (df['habitable'] == True) & 
                                                  (df['radius_p'] >= 1.1) & 
                                                  (df['radius_p'] <= 2.6) & 
                                                  (df['stype'].str.contains('M')) & 
                                                  (df['temp_p'] <= 350),
                'All Planets (≤350K)': lambda df: (df['temp_p'] <= 350)
            }
            
            # Plot each combined category
            for i, (label, filter_func) in enumerate(combined_categories.items()):
                if i >= 4:  # Only plot first 4 categories for 2x2
                    break
                    
                subset = self.df[filter_func(self.df)]
                if len(subset) == 0:
                    continue
                
                # Calculate data
                total_counts, detected_counts, efficiency, mask_best = self._calculate_efficiency_data(subset, x_col, bins)
                total_planets = len(subset) / self.nruns
                detected_planets = np.sum(detected_counts)
                
                # Create subplot
                title = f"{label}\nTotal: {total_planets:.1f}, Detected: {detected_planets:.1f}"
                ax2 = self._create_bar_plot_with_efficiency(
                    axs[i], bin_centers, total_counts, detected_counts, efficiency, 
                    x_label, title, bins
                )
                
                # Set y-labels only for leftmost subplots (0 and 2)
                if i in [0, 2]:
                    axs[i].set_ylabel("Number of Planets")
                    ax2.set_ylabel("Detection Efficiency")
                
                # Set xlim to 350 for temperature plots
                if x_col == 'temp_p':
                    axs[i].set_xlim([bins[0], 305])
                    ax2.set_xlim([bins[0], 305])
            
            # Finalize plot
            fig.suptitle(f"Detection Efficiency by {x_axis.capitalize()} "
                        f"for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}", 
                        fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            self._save_plot(fig, f'detection_efficiency_by_type_{x_axis}')

    def plot_detection_efficiency_3d_panels(self, category_column=None, category_label=None) -> None:
        """Plot detection efficiency as 2D panel plots with shared colorbar."""
        df = self.df.copy()
        if category_column and category_label:
            df = df[df[category_column] == category_label]
        
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df.index]  # Align mask with filtered df
        category_str = f" ({category_label})" if category_label else ""
        
        fig = plt.figure(figsize=(15, 4))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.3)

        axs = [fig.add_subplot(gs[i]) for i in range(3)]

        all_meshes = []
        for ax, panel in zip(axs, PANEL_CONFIGS):
            mesh = self._create_efficiency_panel(ax, df, mask_best, panel)
            all_meshes.append(mesh)

        self._enhance_panel_layout(axs)

        # Shared colorbar in the 4th column
        cbar_ax = fig.add_subplot(gs[3])
        cbar = fig.colorbar(all_meshes[0], cax=cbar_ax)
        cbar.set_label('Detection Efficiency')

        fig.suptitle(f'Detection Efficiency {category_str} for {self.name} ({self.nruns} runs)')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        self._save_plot(fig, 'detection_efficiency_3d_panels')

    def _create_efficiency_panel(self, ax, df, mask_best, panel):
        """Create a single efficiency panel."""
        x, y = panel['x'], panel['y']
        xbins, ybins = panel['xbins'], panel['ybins']
        xlabel, ylabel = panel['xlabel'], panel['ylabel']
        xscale, title = panel['xscale'], panel['title']
        
        # Use dynamic binning for temperature to include all data
        if x == 'temp_p':
            x_min = df[x].min()
            x_max = 440  
            xbins = np.linspace(x_min, x_max, 40)
        
        # Calculate efficiency
        total_counts, _, _ = np.histogram2d(df[x], df[y], bins=[xbins, ybins])
        detected_counts, _, _ = np.histogram2d(df[mask_best][x], df[mask_best][y], bins=[xbins, ybins])
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = np.true_divide(detected_counts, total_counts)
            efficiency[~np.isfinite(efficiency)] = 0.0
        
        # Create mesh
        mesh = ax.imshow(efficiency.T, origin='lower', aspect='auto',
                        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                        cmap='viridis', vmin=0, vmax=1)
        
        # Setup axes
        if xscale == 'log':
            ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        
        # Add reference lines
        self._add_reference_lines(ax, x, y)
        ax.legend(loc='upper right', fontsize=10)
        
        return mesh

    def _add_reference_lines(self, ax, x, y):
        """Add reference lines to panels."""
        if y == 'radius_p':
            ax.axhline(y=1.5, color='black', linestyle='--', alpha=0.5, 
                      label='Rocky/Super-Earth boundary')
            ax.axhline(y=4.0, color='black', linestyle=':', alpha=0.5, 
                      label='Sub-Neptune boundary')
        if x == 'temp_p':
            ax.axvline(x=270, color='blue', linestyle='--', alpha=0.5, 
                      label='Cold/Habitable boundary')
            ax.axvline(x=390, color='red', linestyle='--', alpha=0.5, 
                      label='Habitable/Hot boundary')

    def _enhance_panel_layout(self, axs):
        """Add visual enhancements to panel layout."""
        axs2d = np.atleast_2d(axs)
        for ax in axs2d[0, :]:
            ax.spines['bottom'].set_linewidth(2)
        axs2d[0, 0].spines['left'].set_linewidth(2)
        axs2d[0, 2].spines['right'].set_linewidth(2)

    def plot_detection_scatter_by_parameter(self) -> None:
        """Plot scatter plots of detection parameters vs. distance to star."""
        # Define parameters to plot
        if self.name == 'HWO':
            parameters = {
                'maxangsep': 'Max angular separation (arcsec)', 
                'flux_ratio_value_best': 'Flux ratio', 
                'photon_rate_value_best': 'Number of photons hitting detector'
            }
        else:
            parameters = {'maxangsep': 'Max angular separation (arcsec)'}
        
        # Create plots for each parameter
        for var, title in parameters.items():
            self._create_parameter_scatter_plot(var, title)

    def _create_parameter_scatter_plot(self, var, title):
        """Create a single parameter scatter plot."""
        # Filter data
        df_plot = self.df[(self.df[var] > 0) & (self.df['distance_s'] > 0)]
        if len(df_plot) == 0:
            return
        
        # Setup plot
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        mask_best, _ = self._get_detection_masks()
        mask_best = mask_best[df_plot.index]  # Align mask with filtered df_plot
        
        # Split data by detection status
        detected = df_plot[mask_best == 1]
        not_detected = df_plot[mask_best == 0]
        
        # Plot data
        if len(not_detected) > 0:
            ax.scatter(not_detected[var], not_detected['distance_s'], 
                      color='red', alpha=0.1, label='Total planets', s=10)
        if len(detected) > 0:
            ax.scatter(detected[var], detected['distance_s'], 
                      color='green', alpha=0.8, label='Detected', s=20)
        
        # Setup plot properties
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(title)
        plt.ylabel("Distance to Star (pc)")
        plt.title(f"{title} vs. Distance to Star")
        plt.legend()
        
        # Save plot
        self._save_plot(plt.gcf(), f"{var}_vs_distance")