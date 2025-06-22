import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from plot.helpers import make_output_dir, output_filename, scatter_best_worst_overlay, get_detection_masks

class PlanetDetectionPlotter:
    """
    Class for generating detection efficiency and detection plots for HWO scenarios.
    Uses detected_best and detected_worst columns for overlays.
    """
    def __init__(self, df, nruns: int = 1, star_catalog: str = 'Gaia', name: str = 'HWO'):
        """Initialize with data and metadata. Supports both HWO and non-HWO scenarios."""
        self.df = df.copy()
        self.nruns = nruns
        self.star_catalog = star_catalog
        self.name = name
        self.data_dir = make_output_dir(name, nruns, star_catalog)

    def plot_all(self) -> None:
        """Generate all detection plots."""
        self.plot_detection_efficiency()
        self.plot_efficiency_multipanel()
        self.plot_detection_mr()
        self.plot_detection_vs_temp_color()
        self.plot_detection_efficiency_radius()

    def plot_efficiency_multipanel(self) -> None:
        """Plot detection efficiency for several planet types in a multipanel figure, with best/worst overlays for HWO, or just detected for others."""
        # Create both temperature and distance plots
        for x_axis, x_col, x_label, x_range in [
            ('temp', 'temp_p', 'Temperature [K]', (125, 305)),
            ('distance', 'distance_s', 'Distance [pc]', (0, 15))
        ]:
            fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
            filters = {
                'Rocky HZ': self.df[
                    (self.df['habitable'] == True) &
                    (self.df['radius_p'] < 1.5)
                ],
                'HZ Rocky around G-type (Sun-like) stars': self.df[
                    (self.df['habitable'] == True) &
                    (self.df['radius_p'] <= 1.5) &
                    (self.df['stype'].str.contains('G'))
                ],
                'HZ around M dwarfs': self.df[
                    (self.df['habitable'] == True) &
                    (self.df['stype'].str.contains('M'))
                ]
            }
            bins = np.linspace(x_range[0], x_range[1], 40)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            for i, (label, subset) in enumerate(filters.items()):
                ax1 = axs[i]
                total_counts, _ = np.histogram(subset[x_col], bins=bins)
                mask_best, mask_worst = get_detection_masks(subset, self.name)
                
                # Calculate total and detected counts
                total_planets = len(subset)
                detected_best = np.sum(mask_best)
                detected_worst = np.sum(mask_worst) if mask_worst is not None else 0
                
                if self.name == 'HWO':
                    detected_counts_best, _ = np.histogram(subset[mask_best][x_col], bins=bins)
                    detected_counts_worst = np.zeros_like(total_counts)
                    if mask_worst is not None:
                        detected_counts_worst, _ = np.histogram(subset[mask_worst][x_col], bins=bins)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        efficiency_best = np.true_divide(detected_counts_best, total_counts)
                        efficiency_best[np.isnan(efficiency_best)] = 0.0
                        efficiency_worst = np.true_divide(detected_counts_worst, total_counts)
                        efficiency_worst[np.isnan(efficiency_worst)] = 0.0
                    ax1.bar(bin_centers, total_counts, width=np.diff(bins), color='lightgrey', align='center', label='Total')
                    ax1.bar(bin_centers, detected_counts_worst, width=np.diff(bins), color='green', alpha=0.4, align='center', label='Detected (Worst)')
                    ax1.bar(bin_centers, detected_counts_best, width=np.diff(bins), color='green', alpha=0.8, align='center', label='Detected (Best)')
                    
                    ax1.set_xlabel(x_label)
                    ax1.set_xlim(bins[0], bins[-1])
                    ax1.set_title(f"{label}\nTotal: {total_planets}, Best: {detected_best}, Worst: {detected_worst}")
                    ax2 = ax1.twinx()
                    ax2.plot(bin_centers, efficiency_worst, 'g:', linewidth=2, label='Efficiency (Worst)')
                    ax2.plot(bin_centers, efficiency_best, 'r--', linewidth=2, label='Efficiency (Best)')
                    ax2.set_ylim(0, 1.0)
                else:
                    detected_counts, _ = np.histogram(subset[mask_best][x_col], bins=bins)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        efficiency = np.true_divide(detected_counts, total_counts)
                        efficiency[np.isnan(efficiency)] = 0.0
                    ax1.bar(bin_centers, total_counts, width=np.diff(bins), color='lightgrey', align='center', label='Total')
                    ax1.bar(bin_centers, detected_counts, width=np.diff(bins), color='green', alpha=0.8, align='center', label='Detected')
                    
                    ax1.set_xlabel(x_label)
                    ax1.set_xlim(bins[0], bins[-1])
                    ax1.set_title(f"{label}\nTotal: {total_planets}, Detected: {detected_best}")
                    ax2 = ax1.twinx()
                    ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Efficiency')
                    ax2.set_ylim(0, 1.0)
                h1, l1 = ax1.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                ax1.legend(h1 + h2, l1 + l2, loc='upper left')
                if i == 0:
                    ax1.set_ylabel("Number of Planets")
                    ax2.set_ylabel("Detection Efficiency")
            fig.suptitle(f"Detection Efficiency Across Planet Types by {x_axis.capitalize()} for {self.name} ({self.nruns} Runs)\nStar Catalog: {self.star_catalog}", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            outfile = output_filename(f'detection_efficiency_multipanel_{x_axis}', self.name, self.nruns, self.star_catalog)
            plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
            plt.close()

    def plot_detection_efficiency(self, category_column=None, category_label=None) -> None:
        """Plot detection efficiency for planetary candidates based on temperature and distance, with best/worst overlays for HWO, or just detected for others."""
        df = self.df.copy()
        if category_column and category_label:
            df = df[df[category_column] == category_label]
        
        # Create both temperature and distance plots
        for x_axis, x_col, x_label, x_range in [
            ('temp', 'temp_p', 'Temperature [K]', (125, 305)),
            ('distance', 'distance_s', 'Distance [pc]', (0, 15))
        ]:
            bins = np.linspace(x_range[0], x_range[1], 40)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            total_counts, _ = np.histogram(df[x_col], bins=bins)
            mask_best, mask_worst = get_detection_masks(df, self.name)
            
            # Calculate total and detected counts
            total_planets = len(df)
            detected_best = np.sum(mask_best)
            detected_worst = np.sum(mask_worst) if mask_worst is not None else 0
            
            if self.name == 'HWO':
                detected_counts_best, _ = np.histogram(df[mask_best][x_col], bins=bins)
                detected_counts_worst = np.zeros_like(total_counts)
                if mask_worst is not None:
                    detected_counts_worst, _ = np.histogram(df[mask_worst][x_col], bins=bins)
                with np.errstate(divide='ignore', invalid='ignore'):
                    efficiency_best = np.true_divide(detected_counts_best, total_counts)
                    efficiency_best[np.isnan(efficiency_best)] = 0.0
                    efficiency_worst = np.true_divide(detected_counts_worst, total_counts)
                    efficiency_worst[np.isnan(efficiency_worst)] = 0.0
                _, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', color='lightgrey', label='All planets present')
                ax1.bar(bin_centers, detected_counts_worst, width=np.diff(bins), align='center', color='green', alpha=0.4, label='Planets detectable (Worst)')
                ax1.bar(bin_centers, detected_counts_best, width=np.diff(bins), color='green', alpha=0.8, align='center', label='Planets detectable (Best)')
                
                ax1.set_ylabel("Number of planets")
                ax1.set_xlabel(x_label)
                ax1.set_xlim(bins[0], bins[-1])
                ax2 = ax1.twinx()
                ax2.plot(bin_centers, efficiency_worst, 'g:', linewidth=2, label='Detection efficiency (Worst)')
                ax2.plot(bin_centers, efficiency_best, 'r--', linewidth=2, label='Detection efficiency (Best)')
                ax2.set_ylabel("Detection efficiency")
                ax2.set_ylim(0, 1.0)
            else:
                detected_counts, _ = np.histogram(df[mask_best][x_col], bins=bins)
                with np.errstate(divide='ignore', invalid='ignore'):
                    efficiency = np.true_divide(detected_counts, total_counts)
                    efficiency[np.isnan(efficiency)] = 0.0
                _, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', color='lightgrey', label='All planets present')
                ax1.bar(bin_centers, detected_counts, width=np.diff(bins), color='green', alpha=0.8, align='center', label='Detected')
                
                ax1.set_ylabel("Number of planets")
                ax1.set_xlabel(x_label)
                ax1.set_xlim(bins[0], bins[-1])
                ax2 = ax1.twinx()
                ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Detection efficiency')
                ax2.set_ylim(0, 1.0)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='upper left')
            category_str = f" ({category_label})" if category_label else ""
            
            # Add statistics to title
            if self.name == 'HWO':
                title = f'Detection Efficiency by {x_axis.capitalize()}{category_str} for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}\nTotal: {total_planets}, Best: {detected_best}, Worst: {detected_worst}'
            else:
                title = f'Detection Efficiency by {x_axis.capitalize()}{category_str} for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}\nTotal: {total_planets}, Detected: {detected_best}'
            
            ax1.set_title(title)
            plt.tight_layout()
            outfile = output_filename(f'detection_efficiency_{x_axis}', self.name, self.nruns, self.star_catalog)
            plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
            plt.close()
    

    def plot_detection_efficiency_radius(self, category_column=None, category_label=None) -> None:
        """Plot detection efficiency as a 2D temperature vs radius plot with best/worst overlays for HWO."""
        df = self.df.copy()
        if category_column and category_label:
            df = df[df[category_column] == category_label]
        
        # Define temperature and radius ranges
        temp_range = (125, 500)
        radius_range = (0, 8)
        
        # Create 2D bins
        temp_bins = np.linspace(temp_range[0], temp_range[1], 40)
        radius_bins = np.linspace(radius_range[0], radius_range[1], 30)
        
        # Calculate total counts in each 2D bin
        total_counts, _, _ = np.histogram2d(df['temp_p'], df['radius_p'], bins=[temp_bins, radius_bins])
        
        # Get detection masks
        mask_best, mask_worst = get_detection_masks(df, self.name)
        
        # Calculate total and detected countsx
        total_planets = len(df)
        detected_best = np.sum(mask_best)
        detected_worst = np.sum(mask_worst) if mask_worst is not None else 0
        
        if self.name == 'HWO':
            # Calculate detected counts for best and worst cases
            detected_counts_best, _, _ = np.histogram2d(df[mask_best]['temp_p'], df[mask_best]['radius_p'], 
                                                       bins=[temp_bins, radius_bins])
            
            # Create a mask for bins that have total planets but no detected planets
            zero_detection_mask = (total_counts > 0) & (detected_counts_best == 0)
            
            # Create single plot for best case
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot best case (number of detected planets)
            vmax_best = max(detected_counts_best.max(), 1.0)  # Ensure vmax is at least 1.0
            if detected_counts_best.max() == 0:
                # No detected planets - use linear scale with dark red for zeros
                norm_best = matplotlib.colors.Normalize(vmin=0, vmax=1)
                # Create a custom colormap that goes from dark red to green
                colors_best = ['darkred', 'green']
                cmap_best = matplotlib.colors.LinearSegmentedColormap.from_list('custom_red_green_best', colors_best)
                # Set zero detection bins to a special value to show as dark red
                plot_data = detected_counts_best.copy()
                plot_data[zero_detection_mask] = 0.5  # Special value for dark red
            else:
                # Use log scale for detected planets
                norm_best = matplotlib.colors.LogNorm(vmin=0.1, vmax=vmax_best)
                cmap_best = 'RdYlGn'
                plot_data = detected_counts_best.copy()
                # Set zero detection bins to a special value to show as dark red
                plot_data[zero_detection_mask] = 0.05  # Special value for dark red in log scale
            im = ax.imshow(plot_data.T, origin='lower', aspect='auto', 
                            extent=[temp_bins[0], temp_bins[-1], radius_bins[0], radius_bins[-1]],
                            cmap=cmap_best, norm=norm_best)
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Radius [Rearth]')
            ax.set_title(f'Number of Detected Planets (Best Case)\n{detected_best}/{total_planets} planets detected')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of detected planets')
            
            # Add reference lines
            ax.axhline(y=1.5, color='black', linestyle='--', alpha=0.5, label='Rocky/Super-Earth boundary')
            ax.axhline(y=4.0, color='black', linestyle=':', alpha=0.5, label='Sub-Neptune boundary')
            ax.axvline(x=270, color='blue', linestyle='--', alpha=0.5, label='Cold/Habitable boundary')
            ax.axvline(x=390, color='red', linestyle='--', alpha=0.5, label='Habitable/Hot boundary')
            ax.legend(loc='upper right', fontsize=10)
        else:
            # Non-HWO case - single plot
            detected_counts, _, _ = np.histogram2d(df[mask_best]['temp_p'], df[mask_best]['radius_p'], 
                                                  bins=[temp_bins, radius_bins])
            
            # Create a mask for bins that have total planets but no detected planets
            zero_detection_mask = (total_counts > 0) & (detected_counts == 0)
            
            vmax_detected = max(detected_counts.max(), 1.0)  # Ensure vmax is at least 1.0
            if detected_counts.max() == 0:
                # No detected planets - use linear scale with dark red for zeros
                norm_detected = matplotlib.colors.Normalize(vmin=0, vmax=1)
                # Create a custom colormap that goes from dark red to green
                colors_detected = ['darkred', 'green']
                cmap_detected = matplotlib.colors.LinearSegmentedColormap.from_list('custom_red_green_detected', colors_detected)
                # Set zero detection bins to a special value to show as dark red
                plot_data = detected_counts.copy()
                plot_data[zero_detection_mask] = 0.5  # Special value for dark red
            else:
                # Use log scale for detected planets
                norm_detected = matplotlib.colors.LogNorm(vmin=0.1, vmax=vmax_detected)
                cmap_detected = 'RdYlGn'
                plot_data = detected_counts.copy()
                # Set zero detection bins to a special value to show as dark red
                plot_data[zero_detection_mask] = 0.05  # Special value for dark red in log scale
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(plot_data.T, origin='lower', aspect='auto',
                          extent=[temp_bins[0], temp_bins[-1], radius_bins[0], radius_bins[-1]],
                          cmap=cmap_detected, norm=norm_detected)
            ax.set_xlabel('Temperature [K]')
            ax.set_ylabel('Radius [Rearth]')
            ax.set_title(f'Number of Detected Planets\n{detected_best}/{total_planets} planets detected')
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Number of detected planets')
            
            # Add reference lines
            ax.axhline(y=1.5, color='black', linestyle='--', alpha=0.5, label='Rocky/Super-Earth boundary')
            ax.axhline(y=4.0, color='black', linestyle=':', alpha=0.5, label='Sub-Neptune boundary')
            ax.axvline(x=270, color='blue', linestyle='--', alpha=0.5, label='Cold/Habitable boundary')
            ax.axvline(x=390, color='red', linestyle='--', alpha=0.5, label='Habitable/Hot boundary')
            ax.legend(loc='upper right', fontsize=10)
        
        # Add category information to title
        category_str = f" ({category_label})" if category_label else ""
        fig.suptitle(f'Detection Efficiency: Temperature vs Radius{category_str} for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}', 
                     fontsize=14, y=0.98)
        
        plt.tight_layout()
        outfile = output_filename('detection_efficiency_temp_radius', self.name, self.nruns, self.star_catalog)
        plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_detection_mr(self) -> None:
        """Plot detection likelihood by planet radius and mass (log scale), with best/worst overlays for HWO, or just detected for others."""
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        mask_best, mask_worst = get_detection_masks(self.df, self.name)
        if self.name == 'HWO':
            detected_best = self.df[mask_best == 1]
            not_detected_best = self.df[mask_best == 0]
            detected_worst = self.df[mask_worst == 1] if mask_worst is not None else None
            not_detected_worst = self.df[mask_worst == 0] if mask_worst is not None else None
            # Plot undetected first (background)
            if not_detected_best is not None and len(not_detected_best) > 0:
                ax.scatter(not_detected_best['radius_p'], not_detected_best['mass_p'], color='red', alpha=0.1, label='Not Detected (Best)', s=10)
            if not_detected_worst is not None and len(not_detected_worst) > 0:
                ax.scatter(not_detected_worst['radius_p'], not_detected_worst['mass_p'], color='red', alpha=0.1, label='Not Detected (Worst)', s=10)
            # Overlay detected on top
            if detected_best is not None and len(detected_best) > 0:
                ax.scatter(detected_best['radius_p'], detected_best['mass_p'], color='green', alpha=0.8, label='Detected (Best)', s=20)
            if detected_worst is not None and len(detected_worst) > 0:
                ax.scatter(detected_worst['radius_p'], detected_worst['mass_p'], color='green', alpha=0.4, label='Detected (Worst)', s=20)
        else:
            detected = self.df[mask_best == 1]
            not_detected = self.df[mask_best == 0]
            # Plot undetected first (background)
            ax.scatter(not_detected['radius_p'], not_detected['mass_p'], color='red', alpha=0.1, label='Not Detected', s=10)
            # Overlay detected on top
            ax.scatter(detected['radius_p'], detected['mass_p'], color='green', alpha=0.8, label='Detected', s=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel("Planet Radius ($R_\\oplus$)")
        plt.ylabel("Planet Mass ($M_\\oplus$)")
        plt.title("Detection Likelihood by Planet Radius and Mass (Log Scale)")
        plt.legend()
        plt.tight_layout()
        outfile = output_filename('detection_mr_log', self.name, self.nruns, self.star_catalog)
        plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_detection_vs_temp_color(self) -> None:
        """Plot temperature vs. distance to star, colored by detection status, with best/worst overlays for HWO, or just detected for others."""
        if self.name == 'HWO':
            xtitles = {'maxangsep': 'Max angular separation (arcsec)', 'flux_ratio_value_best': 'Flux ratio', 'photon_rate_value_best': 'Number of photons hitting detector'}
            xvars = ['maxangsep', 'flux_ratio_value_best', 'photon_rate_value_best']
        else:
            xtitles = {'maxangsep': 'Max angular separation (arcsec)'}
            xvars = ['maxangsep']
        for var in xvars:
            df_plot = self.df[(self.df[var] > 0) & (self.df['distance_s'] > 0)]
            plt.figure(figsize=(8,6))
            ax = plt.gca()
            mask_best, mask_worst = get_detection_masks(df_plot, self.name)
            if self.name == 'HWO':
                detected_best = df_plot[mask_best == 1]
                not_detected_best = df_plot[mask_best == 0]
                detected_worst = df_plot[mask_worst == 1] if mask_worst is not None else None
                not_detected_worst = df_plot[mask_worst == 0] if mask_worst is not None else None
                # Plot undetected first (background)
                if not_detected_best is not None and len(not_detected_best) > 0:
                    ax.scatter(not_detected_best[var], not_detected_best['distance_s'], color='red', alpha=0.1, label='Not Detected (Best)', s=10)
                if not_detected_worst is not None and len(not_detected_worst) > 0:
                    ax.scatter(not_detected_worst[var], not_detected_worst['distance_s'], color='red', alpha=0.1, label='Not Detected (Worst)', s=10)
                # Overlay detected on top
                if detected_best is not None and len(detected_best) > 0:
                    ax.scatter(detected_best[var], detected_best['distance_s'], color='green', alpha=0.8, label='Detected (Best)', s=20)
                if detected_worst is not None and len(detected_worst) > 0:
                    ax.scatter(detected_worst[var], detected_worst['distance_s'], color='green', alpha=0.4, label='Detected (Worst)', s=20)
            else:
                detected = df_plot[mask_best == 1]
                not_detected = df_plot[mask_best == 0]
                # Plot undetected first (background)
                ax.scatter(not_detected[var], not_detected['distance_s'], color='red', alpha=0.1, label='Not Detected', s=10)
                # Overlay detected on top
                ax.scatter(detected[var], detected['distance_s'], color='green', alpha=0.8, label='Detected', s=20)
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(xtitles.get(var, var))
            plt.ylabel("Distance to Star (pc)")
            plt.title(f"{xtitles.get(var, var)} vs. Distance to Star")
            plt.legend()
            plt.tight_layout()
            outfile = output_filename(f"{var}_vs_distance", self.name, self.nruns, self.star_catalog)
            plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
            plt.close()