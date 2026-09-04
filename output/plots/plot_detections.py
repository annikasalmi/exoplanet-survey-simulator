import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from output.plots.base_plotter import BasePlotter
from tools.plotting_constants import PLOT_CONFIGS, PANEL_CONFIGS

plt.rcParams.update({'font.size': 16})

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

    def _collect_legend_handles(self, ax):
        """Helper to collect legend handles and labels from two axes."""
        h1, l1 = ax.get_legend_handles_labels()
        return h1, l1 

    def _setup_bar_and_efficiency_axes(self, ax, bin_centers, total_counts, detected_counts, efficiency, x_label, title, bins):
        """Helper to setup bar plot and efficiency overlay."""
        # Plot bars
        ax.bar(bin_centers, total_counts, width=np.diff(bins), color='lightgrey', 
               align='center', label='Total', edgecolor='black')
        ax.bar(bin_centers, detected_counts, width=np.diff(bins), color='green', 
               alpha=0.8, align='center', label='Detected', edgecolor='black')
        
        # Setup primary axis
        ax.set_xlabel(x_label)
        ax.set_ylabel("Number of Planets")
        ax.set_title(title)
        
        # Add efficiency line on secondary axis
        ax2 = ax.twinx()
        ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Efficiency')
        ax2.set_ylabel("Detection Efficiency")
        ax2.set_ylim(0, 1.0)
        
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
                'M dwarf + Rocky HZ': lambda df: (df['habitable'] == True) & (df['radius_p'] < 1.5) & (df['stype'].str.contains('M')),
                'G/K star + Rocky HZ': lambda df: (df['habitable'] == True) & (df['radius_p'] < 1.5) & (df['stype'].isin(['G', 'K'])),
                'M dwarf + Hycean HZ': lambda df: (df['habitable'] == True) & (df['radius_p'] >= 1.1) & (df['radius_p'] <= 2.6) & (df['stype'].str.contains('M')),
                'All Planets': lambda df: np.ones(len(df), dtype=bool)
            }
            
            twin_axes = []
            
            # Plot each combined category
            for i, (label, filter_func) in enumerate(combined_categories.items()):
                if i >= 4:
                    break
                    
                subset = self.df[filter_func(self.df)]
                if len(subset) == 0:
                    twin_axes.append(None)
                    continue
                
                # Calculate data
                total_counts, detected_counts, efficiency, mask_best = self._calculate_efficiency_data(subset, x_col, bins)
                total_planets = len(subset) / self.nruns
                detected_planets = np.sum(detected_counts)
                
                # Create subplot
                title = f"{label}\nTotal: {total_planets:.1f}, Detected: {detected_planets:.1f}"
                ax2 = self._setup_bar_and_efficiency_axes(
                    axs[i], bin_centers, total_counts, detected_counts, efficiency, x_label, title, bins
                )
                twin_axes.append(ax2)
                
                # Set y-labels only for leftmost subplots (0 and 2)
                if i in [0, 2]:
                    axs[i].set_ylabel("Number of Planets")
                    ax2.set_ylabel("Detection Efficiency")
                
                # Set xlim to 350 for temperature plots
                if x_col == 'temp_p':
                    axs[i].set_xlim([bins[0], 305])
                    ax2.set_xlim([bins[0], 305])
            
            # Add a single legend for the whole figure using the first subplot and its twin axis
            handles, labels = self._collect_legend_handles(axs[0])
            axs[0].legend(handles, labels, loc='upper left', fontsize=14)
            
            # Finalize plot
            title = f"Detection Efficiency by {x_axis.capitalize()} for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}"
            fig.suptitle(title, fontsize=20, y=0.96)
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
        
        fig = plt.figure(figsize=(20, 6))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.04], wspace=0.3)

        axs = [fig.add_subplot(gs[i]) for i in range(3)]

        all_meshes = []
        for ax, panel in zip(axs, PANEL_CONFIGS):
            mesh = self._create_efficiency_panel(ax, df, mask_best, panel)
            all_meshes.append(mesh)

        self._enhance_panel_layout(axs)

        # Add legend to just the far left axis
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].legend(handles, labels, loc='upper left', fontsize=14)

        # Shared colorbar in the 4th column
        cbar_ax = fig.add_subplot(gs[3])
        cbar = fig.colorbar(all_meshes[0], cax=cbar_ax)
        cbar.set_label('Detection Efficiency')

        fig.suptitle(f'Detection Efficiency for {self.name} ({self.nruns} runs)', fontsize=20, y=1)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
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
        """Enhance panel layout with consistent styling."""
        for ax in axs:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('lightgray')

    def plot_detection_scatter_by_parameter(self) -> None:
        """Plot detection scatter plots by parameter."""
        for var, title in [('temp_p', 'Planet Temperature'), ('radius_p', 'Planet Radius')]:
            self._create_parameter_scatter_plot(var, title)

    def _create_parameter_scatter_plot(self, var, title):
        """Create a scatter plot for a specific parameter."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        mask_best, _ = self._get_detection_masks()
        
        # Plot detected and non-detected points
        detected = self.df[mask_best]
        non_detected = self.df[~mask_best]
        
        if len(detected) > 0:
            ax.scatter(detected[var], detected['distance'], c='green', alpha=0.6, 
                      label='Detected', s=20)
        if len(non_detected) > 0:
            ax.scatter(non_detected[var], non_detected['distance'], c='red', alpha=0.6, 
                      label='Not Detected', s=20)
        
        ax.set_xlabel(title)
        ax.set_ylabel('Distance [pc]')
        ax.set_title(f'Detection Scatter Plot: {title} vs Distance')
        ax.legend()
        
        self._save_plot(fig, f'detection_scatter_{var}')