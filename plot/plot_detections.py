import os
import numpy as np
import matplotlib.pyplot as plt
from plot.helpers import make_output_dir, output_filename, get_detection_masks

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

    def plot_all(self, temp_plots=False) -> None:
        """Generate all detection plots."""
        self.plot_detection_efficiency()
        self.plot_efficiency_multipanel()
        self.plot_efficiency_type_multipanel()
        self.plot_detection_efficiency_radius()
        if temp_plots:
            self.plot_detection_vs_temp_color()

    def plot_efficiency_multipanel(self) -> None:
        """Plot detection efficiency as a 2D color map for several planet types, with best/worst overlays for HWO. Plots both radius vs. period and radius vs. distance. Results are normalized by nruns (per run)."""
        # Define axes and bins for both plots
        plot_configs = [
            {
                'x_col': 'p_orb',
                'y_col': 'radius_p',
                'x_label': 'Orbital Period [days]',
                'y_label': 'Planet Radius [R$_\oplus$]',
                'x_range': (0.5, 500),
                'y_range': (0.5, 5.0),
                'x_bins': list(np.logspace(np.log10(0.5), np.log10(500), 40)),
                'y_bins': list(np.linspace(0.5, 5.0, 40)),
                'suffix': 'radius_period'
            },
            {
                'x_col': 'distance_s',
                'y_col': 'radius_p',
                'x_label': 'Distance [pc]',
                'y_label': 'Planet Radius [R$_\oplus$]',
                'x_range': (0, 15),
                'y_range': (0.5, 5.0),
                'x_bins': list(np.linspace(0, 15, 40)),
                'y_bins': list(np.linspace(0.5, 5.0, 40)),
                'suffix': 'radius_distance'
            }
        ]
        planet_types = {}
        for config in plot_configs:
            x_col = config['x_col']
            y_col = config['y_col']
            x_label = config['x_label']
            y_label = config['y_label']
            x_bins = config['x_bins']
            y_bins = config['y_bins']
            suffix = config['suffix']
            for planet_type, subset in planet_types.items():
                fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
                for i, (scenario, mask_label) in enumerate(zip(['Best', 'Worst'], ['best', 'worst'])):
                    ax = axs[i]
                    mask_best, mask_worst = get_detection_masks(subset, self.name)
                    mask = mask_best if mask_label == 'best' else mask_worst
                    if mask is None:
                        continue
                    # 2D histograms (normalize by nruns)
                    total_hist, xedges, yedges = np.histogram2d(subset[x_col], subset[y_col], bins=(x_bins, y_bins))
                    detected_hist, _, _ = np.histogram2d(subset[mask][x_col], subset[mask][y_col], bins=(x_bins, y_bins))
                    total_hist = total_hist / self.nruns
                    detected_hist = detected_hist / self.nruns
                    with np.errstate(divide='ignore', invalid='ignore'):
                        efficiency = np.true_divide(detected_hist, total_hist)
                        efficiency[~np.isfinite(efficiency)] = 0.0
                    mesh = ax.pcolormesh(xedges, yedges, efficiency.T, cmap='viridis', vmin=0, vmax=1)
                    fig.colorbar(mesh, ax=ax, label='Detection Efficiency (per run)')
                    if x_col == 'p_orb':
                        ax.set_xscale('log')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f'{planet_type}\n{scenario} Case')
                fig.suptitle(f'Detection Efficiency ({y_label} vs. {x_label}) for {planet_type}\n{self.name} ({self.nruns} Runs), Star Catalog: {self.star_catalog}', fontsize=14)
                plt.tight_layout(rect=[0, 0, 1, 0.93])
                outfile = output_filename(f'detection_efficiency_{suffix}_{planet_type.replace(" ", "_").replace("-", "_")}', self.name, self.nruns, self.star_catalog)
                plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
                plt.close()

    def plot_detection_efficiency(self, category_column=None, category_label=None) -> None:
        """Plot detection efficiency for planetary candidates based on temperature and distance, with best/worst overlays for HWO, or just detected for others. Results are normalized by nruns (per run)."""
        df = self.df.copy()
        if category_column and category_label:
            df = df[df[category_column] == category_label]
        for x_axis, x_col, x_label, x_range in [
            ('temp', 'temp_p', 'Temperature [K]', (125, 305)),
            ('distance', 'distance_s', 'Distance [pc]', (0, 15))
        ]:
            bins = np.linspace(x_range[0], x_range[1], 40)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            total_counts, _ = np.histogram(df[x_col], bins=bins)
            mask_best, mask_worst = get_detection_masks(df, self.name)
            total_counts = total_counts / self.nruns
            # Calculate total and detected counts
            total_planets = len(df) / self.nruns
            detected_best = np.sum(mask_best) / self.nruns
            detected_worst = np.sum(mask_worst) / self.nruns if mask_worst is not None else 0
            if self.name == 'HWO':
                detected_counts_best, _ = np.histogram(df[mask_best][x_col], bins=bins)
                detected_counts_worst = np.zeros_like(total_counts)
                if mask_worst is not None:
                    detected_counts_worst, _ = np.histogram(df[mask_worst][x_col], bins=bins)
                detected_counts_best = detected_counts_best / self.nruns
                detected_counts_worst = detected_counts_worst / self.nruns
                with np.errstate(divide='ignore', invalid='ignore'):
                    efficiency_best = np.true_divide(detected_counts_best, total_counts)
                    efficiency_best[np.isnan(efficiency_best)] = 0.0
                    efficiency_worst = np.true_divide(detected_counts_worst, total_counts)
                    efficiency_worst[np.isnan(efficiency_worst)] = 0.0
                _, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', color='lightgrey', label='All planets present (per run)')
                ax1.bar(bin_centers, detected_counts_worst, width=np.diff(bins), align='center', color='green', alpha=0.4, label='Planets detectable (Worst, per run)')
                ax1.bar(bin_centers, detected_counts_best, width=np.diff(bins), color='green', alpha=0.8, align='center', label='Planets detectable (Best, per run)')
                ax1.set_ylabel("Number of planets (per run)")
                ax1.set_xlabel(x_label)
                ax1.set_xlim(bins[0], bins[-1])
                ax2 = ax1.twinx()
                ax2.plot(bin_centers, efficiency_worst, 'g:', linewidth=2, label='Detection efficiency (Worst)')
                ax2.plot(bin_centers, efficiency_best, 'r--', linewidth=2, label='Detection efficiency (Best)')
                ax2.set_ylabel("Detection efficiency")
                ax2.set_ylim(0, 1.0)
            else:
                detected_counts, _ = np.histogram(df[mask_best][x_col], bins=bins)
                detected_counts = detected_counts / self.nruns
                with np.errstate(divide='ignore', invalid='ignore'):
                    efficiency = np.true_divide(detected_counts, total_counts)
                    efficiency[np.isnan(efficiency)] = 0.0
                _, ax1 = plt.subplots(figsize=(10, 6))
                ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', color='lightgrey', label='All planets present (per run)')
                ax1.bar(bin_centers, detected_counts, width=np.diff(bins), color='green', alpha=0.8, align='center', label='Detected (per run)')
                ax1.set_ylabel("Number of planets (per run)")
                ax1.set_xlabel(x_label)
                ax1.set_xlim(bins[0], bins[-1])
                ax2 = ax1.twinx()
                ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Detection efficiency')
                ax2.set_ylim(0, 1.0)
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2, loc='upper left')
            category_str = f" ({category_label})" if category_label else ""
            if self.name == 'HWO':
                title = f'Detection Efficiency by {x_axis.capitalize()}{category_str} for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}\nTotal: {total_planets}, Best: {detected_best}, Worst: {detected_worst}'
            else:
                title = f'Detection Efficiency by {x_axis.capitalize()}{category_str} for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}\nTotal: {total_planets}, Detected: {detected_best}'
            ax1.set_title(title)
            plt.tight_layout()
            outfile = output_filename(f'detection_efficiency_{x_axis}', self.name, self.nruns, self.star_catalog)
            plt.savefig(os.path.join(self.data_dir, outfile), dpi=300, bbox_inches='tight')
            plt.close()

        
    def plot_efficiency_type_multipanel(self) -> None:
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
                total_counts = total_counts / self.nruns
                mask_best, mask_worst = get_detection_masks(subset, self.name)
                
                # Calculate total and detected counts
                total_planets = len(subset) / self.nruns
                detected_best = np.sum(mask_best) / self.nruns
                detected_worst = np.sum(mask_worst) / self.nruns if mask_worst is not None else 0
                
                if self.name == 'HWO':
                    detected_counts_best, _ = np.histogram(subset[mask_best][x_col], bins=bins)
                    detected_counts_best = detected_counts_best / self.nruns
                    detected_counts_worst = np.zeros_like(total_counts)
                    if mask_worst is not None:
                        detected_counts_worst, _ = np.histogram(subset[mask_worst][x_col], bins=bins)
                        detected_counts_worst = detected_counts_worst / self.nruns
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
                    detected_counts = detected_counts / self.nruns
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

    
    def plot_detection_efficiency_radius(self, category_column=None, category_label=None) -> None:
        """Plot detection efficiency as a 2D panel plot: temp vs radius, radius vs period, radius vs distance, mass vs period. 
        All panels in one figure with shared colorbar."""

        df = self.df.copy()
        if category_column and category_label:
            df = df[df[category_column] == category_label]
        mask_best, mask_worst = get_detection_masks(df, self.name)
        category_str = f" ({category_label})" if category_label else ""
        # Panel configs: (x, y, xbins, ybins, xlabel, ylabel, xscale, title)
        panels = [
            {
                'x': 'temp_p', 'y': 'radius_p',
                'xbins': np.linspace(125, 500, 40), 'ybins': np.linspace(0, 8, 30),
                'xlabel': 'Temperature [K]', 'ylabel': 'Radius [Rearth]', 'xscale': 'linear',
                'title': 'Temp vs Radius'
            },
            {
                'x': 'distance_s', 'y': 'radius_p',
                'xbins': np.linspace(0, 15, 40), 'ybins': np.linspace(0, 8, 30),
                'xlabel': 'Distance [pc]', 'ylabel': 'Radius [Rearth]', 'xscale': 'linear',
                'title': 'Radius vs Distance'
            },
            {
                'x': 'p_orb', 'y': 'mass_p',
                'xbins': np.linspace(0.5, 500, 40), 'ybins': np.linspace(0, 10, 30),
                'xlabel': 'Orbital Period [days]', 'ylabel': 'Mass [Mearth]', 'xscale': 'linear',
                'title': 'Mass vs Period'
            },
        ]
        fig, axs = plt.subplots(1, 3, figsize=(21, 6))
        vmin, vmax = 0, 1
        all_meshes = []
        for ax, panel in zip(axs.flat, panels):
            x = panel['x']
            y = panel['y']
            xbins = panel['xbins']
            ybins = panel['ybins']
            xlabel = panel['xlabel']
            ylabel = panel['ylabel']
            xscale = panel['xscale']
            title = panel['title']
            total_counts, _, _ = np.histogram2d(df[x], df[y], bins=[xbins, ybins])
            detected_counts, _, _ = np.histogram2d(df[mask_best][x], df[mask_best][y], bins=[xbins, ybins])
            with np.errstate(divide='ignore', invalid='ignore'):
                efficiency = np.true_divide(detected_counts, total_counts)
                efficiency[~np.isfinite(efficiency)] = 0.0
            mesh = ax.imshow(efficiency.T, origin='lower', aspect='auto',
                             extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
                             cmap='viridis', vmin=vmin, vmax=vmax)
            all_meshes.append(mesh)
            if xscale == 'log':
                ax.set_xscale('log')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            # Add reference lines for radius panels
            if y == 'radius_p':
                ax.axhline(y=1.5, color='black', linestyle='--', alpha=0.5, label='Rocky/Super-Earth boundary')
                ax.axhline(y=4.0, color='black', linestyle=':', alpha=0.5, label='Sub-Neptune boundary')
            if x == 'temp_p':
                ax.axvline(x=270, color='blue', linestyle='--', alpha=0.5, label='Cold/Habitable boundary')
                ax.axvline(x=390, color='red', linestyle='--', alpha=0.5, label='Habitable/Hot boundary')
            ax.legend(loc='upper right', fontsize=10)
        # Add lines between panels for visual separation
        for ax in axs[0, :]:
            ax.spines['bottom'].set_linewidth(2)
        for ax in axs[1, :]:
            ax.spines['top'].set_linewidth(2)
        for ax in axs[:, 0]:
            ax.spines['left'].set_linewidth(2)
        for ax in axs[:, 1]:
            ax.spines['right'].set_linewidth(2)
        # Shared colorbar
        cbar = fig.colorbar(all_meshes[0], ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('Detection Efficiency')
        fig.suptitle(f'Detection Efficiency 2D Panels{category_str} for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}', fontsize=18, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        outfile = output_filename('detection_efficiency_2d_panels', self.name, self.nruns, self.star_catalog)
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