from plot.base_plotter import BasePlotter
from tools import physics_constants as const
from tools.plotting_constants import DETECTION_COLORS
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from scipy.spatial.qhull import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from lifesim.util.habitable import single_habitable_zone
from plot.exoplanet_data_utils import load_exoplanet_luminosity_distance

plt.rcParams.update({'font.size': 16})

class PlotHZLimits(BasePlotter):
    """
    Plotter for M dwarf HZ limits and detectability analysis.
    
    Features:
    - Boundary plots for data analysis
    - Detectability panels with exoplanet overlays
    - Vectorized calculations for performance
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, name: str = 'HWO',
                 nruns: int = 1, star_catalog: str = 'Gaia', 
                 xlim_min: float = 0.001, xlim_max: float = 2.0,
                 ylim_min: float = 0.0, ylim_max: float = 35.0, **kwargs):
        """Initialize with optional dataframe and parameters."""
        if df is None:
            df = self._create_minimal_dataframe()
        super().__init__(df, nruns, star_catalog, name)
        
        # Plot limits
        self.xlim_min = xlim_min
        self.xlim_max = xlim_max
        self.ylim_min = ylim_min
        self.ylim_max = ylim_max
        
        # Cache HWO constants
        self.hwo_best = const.HWOConstants('best')
        self.best_flux_limit = float(self.hwo_best.min_planet_flux_star_ratio)
        self.iwa_limit = float(self.hwo_best.iwa)
        self.max_z = float(self.hwo_best.max_z)
        self.theta_limit_rad = self.iwa_limit * const.arcsec_to_radians
        
        # Detection colors mapping
        self.DETECTION_COLORS = DETECTION_COLORS

    def plot_all(self) -> None:
        """Generate all M dwarf HZ limit plots."""
        if not self._validate_data():
            return
            
        self.plot_boundaries()
        self.plot_luminosity_distance()

    def _create_minimal_dataframe(self) -> pd.DataFrame:
        """Create a minimal dataframe for the plotter to work with."""
        return pd.DataFrame({'dummy': [1]})

    def _get_panel_detection_mask(self, df_panel: pd.DataFrame) -> pd.Series:
        """Get detection mask for a specific panel."""
        mask_best, _ = self._get_detection_masks()
        mask_series = pd.Series(mask_best, index=self.df.index)
        panel_mask = mask_series.reindex(df_panel.index).fillna(False)
        return panel_mask.astype(bool)

    def _get_flux_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to flux ratio."""
        return df['flux_ratio_value_best'] < self.best_flux_limit

    def _get_iwa_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to IWA."""
        return df['maxangsep'] < self.iwa_limit

    def _get_exozodi_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to exozodi."""
        return df['z'] > self.max_z

    def _plot_exoplanet_overlay(self, ax: Axes, region_lum: Tuple[float, float], 
                               region_dist: Tuple[float, float]) -> set:
        """Plot specific exoplanets: TOI-700 and Proxima Cen b."""
        plotted_methods = set()
        
        # Add specific star/planet labels with actual data
        specific_objects = {
            'Proxima Cen b': (0.001567, 1.3),  # (luminosity, distance) - L = 0.001567±0.000020, D=1.3pc
            'TOI-700': (0.023, 31.1)  # The star itself - from CSV: T=3461K, R=0.42R☉, D=31.1pc
        }
        
        # Plot only the specific objects
        for name, (luminosity, distance) in specific_objects.items():
            # Check if the object is within the plot region
            if (region_lum[0] <= luminosity <= region_lum[1] and 
                region_dist[0] <= distance <= region_dist[1]):
                
                # Use a distinct color for these specific objects
                color = 'red' if 'Proxima' in name else 'blue'
                ax.scatter(luminosity, distance, color=color, s=100, alpha=0.8, 
                          marker='*', edgecolors='black', linewidth=1)
                
                # Add labels
                ax.text(luminosity, distance + 0.5, name, 
                       fontsize=10, ha='center', va='bottom', 
                       weight='bold', color=color)
                
                # Add to plotted methods for legend
                method = 'Specific Objects'
                if method not in plotted_methods:
                    plotted_methods.add(method)
        
        return plotted_methods

    def _setup_detectability_plot(self, ax: Axes, L_vals: np.ndarray, D_vals: np.ndarray, 
                                region: np.ndarray, title: str):
        """Setup detectability plot with common elements."""
        L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        
        im = ax.pcolormesh(L_edges, D_edges, region, 
                           cmap=LinearSegmentedColormap.from_list('detectability', ['#ffcccc', '#66cc66']), 
                           shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_title(title)
        self._plot_reference_lines(ax, L_vals, is_au=False)
        return im

    def _plot_reference_lines(self, ax: Axes, xvals: np.ndarray, is_au: bool = False) -> None:
        """Plot reference lines for M dwarf and G-star regions."""
        for L, color in zip([const.L_m_dwarf_max, const.L_g_star_min, const.L_g_star_max], ['red', 'gold', 'gold']):
            val = np.sqrt(L) if is_au else L
            ax.axvline(float(val), color=color, linestyle='--', linewidth=2)

    def plot_boundaries(self) -> None:
        """Plot stellar luminosity vs distance with boundaries."""
        def _safe_float(val):
            try:
                return float(val)
            except Exception:
                return np.nan

        if self.df.empty:
            print("Warning: No data available for panel plot")
            return

        # Use luminosity and distance instead of temperature and semi-major axis
        x_var, y_var = 'luminosity_s', 'distance_s'
        if x_var not in self.df.columns or y_var not in self.df.columns:
            print(f"Warning: Required columns not found: {x_var}, {y_var}")
            return

        # Create data subsets for different detection categories
        detected_mask = self._get_panel_detection_mask(self.df)
        flux_rejected = self._get_flux_rejection_mask(self.df)
        iwa_rejected = self._get_iwa_rejection_mask(self.df)
        exozodi_rejected = self._get_exozodi_rejection_mask(self.df)
        
        data_subsets = {
            'detected': self.df[detected_mask],
            'flux_rejected': self.df[flux_rejected],
            'iwa_rejected': self.df[iwa_rejected],
            'exozodi_rejected': self.df[exozodi_rejected],
        }
        
        boundary_configs = [
            ('flux_rejected', 'red', 'Flux Rejected'),
            ('exozodi_rejected', 'gold', 'Exozodi Rejected'),
            ('iwa_rejected', 'blue', 'IWA Rejected'),
            ('detected', 'green', 'Detected')
        ]
        
        # Compute boundaries using ConvexHull
        boundary_cache = {}
        fig, ax = plt.subplots(figsize=(12, 8))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]
            
        for category, data in data_subsets.items():
            if not isinstance(data, pd.DataFrame):
                continue
            valid_data = data[[x_var, y_var]].dropna()
            if len(valid_data) < 3:
                continue
            points = np.asarray(valid_data.values, dtype=float)
            finite_mask = np.isfinite(points).all(axis=1)
            points = points[finite_mask]
            if len(points) < 3:
                continue
            try:
                hull = ConvexHull(points)
                boundary = np.append(hull.vertices, hull.vertices[0])
                boundary_points = points[boundary]
                
                # Find matching config
                config = next((cfg for cfg in boundary_configs if cfg[0] == category), None)
                if config:
                    color, label = config[1], config[2]
                    ax.plot(boundary_points[:, 0], boundary_points[:, 1], 
                           color=color, linewidth=2, label=label)
                    boundary_cache[category] = boundary_points
            except Exception as e:
                print(f"Warning: Could not compute boundary for {category}: {e}")
                continue
                
        # Plot boundaries
        for category, color, label in boundary_configs:
            if category in boundary_cache:
                boundary_points = boundary_cache[category]
                linewidth = 3 if color == 'green' else 0
                alpha_fill = 0.3 if color in ['red', 'gold', 'blue'] else None
                facecolor = color if color in ['red', 'gold', 'blue'] else None
                if facecolor:
                    ax.fill(boundary_points[:, 0], boundary_points[:, 1], facecolor=facecolor, 
                           alpha=alpha_fill, color=color, linewidth=linewidth, label=label)
                elif linewidth > 0:
                    ax.plot(boundary_points[:, 0], boundary_points[:, 1], color=color, 
                           linewidth=linewidth, label=label)
                    
        # Set labels and title
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_title('Stellar Luminosity vs Distance')
        
        # Set axis limits to match the luminosity-distance plot
        ax.set_xlim(0.001, 2.0)  # Luminosity range
        ax.set_ylim(0.0, 35.0)   # Distance range
        ax.set_xscale('log')      # Log scale for luminosity
                
        # Add reference lines for M dwarf and G star luminosity ranges
        ax.axvline(const.L_m_dwarf_max, color='red', linestyle='--', linewidth=2, 
                  label=f'M dwarf max L ({const.L_m_dwarf_max:.3f} L☉)')
        ax.axvline(const.L_g_star_min, color='gold', linestyle='--', linewidth=2, 
                  label=f'G star min L ({const.L_g_star_min:.3f} L☉)')
        ax.axvline(const.L_g_star_max, color='gold', linestyle='--', linewidth=2, 
                  label=f'G star max L ({const.L_g_star_max:.3f} L☉)')
        
        # Create legend
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Flux Rejected'),
            Patch(facecolor='gold', alpha=0.3, label='Exozodi Rejected'),
            Patch(facecolor='blue', alpha=0.3, label='IWA Rejected'),
            Patch(facecolor='green', alpha=1.0, label='Detected'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='M dwarf L range'),
            Line2D([0], [0], color='gold', linestyle='--', linewidth=2, label='G star L range')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        plt.tight_layout()
        self._save_plot(fig, 'panel_luminosity_vs_distance')

    def _add_habitable_zone_line(self, ax: Axes) -> None:
        """Add habitable zone line to the plot."""
        temp_range = np.logspace(2.5, 3.8, 100)
        hz_distances = []
        
        for temp in temp_range:
            try:
                hz_dist = single_habitable_zone(temp, 1.0)  # 1 Earth mass
                hz_distances.append(hz_dist)
            except:
                hz_distances.append(np.nan)
        
        valid_mask = ~np.isnan(hz_distances)
        if np.any(valid_mask):
            ax.plot(temp_range[valid_mask], np.array(hz_distances)[valid_mask], 
                   color='black', linestyle='--', linewidth=2, label='Habitable Zone (1 M⊕)')

    def _add_stellar_type_reference_lines(self, ax: Axes) -> None:
        """Add reference lines for different stellar types."""
        stellar_types = {
            'M dwarf max': (const.L_m_dwarf_max, 'red'),
            'G star min': (const.L_g_star_min, 'gold'),
            'G star max': (const.L_g_star_max, 'gold')
        }
        
        for name, (L, color) in stellar_types.items():
            # Convert luminosity to temperature (approximate)
            temp = 5778 * (L ** 0.25)  # Using L ∝ T^4
            ax.axvline(temp, color=color, linestyle=':', alpha=0.7, linewidth=1)

    def plot_luminosity_distance(self) -> None:
        """Plot luminosity vs distance with detectability regions."""
        if self.df.empty:
            print("Warning: No data available for luminosity-distance plot")
            return

        # Create grid for detectability analysis
        L_vals = np.logspace(np.log10(self.xlim_min), np.log10(self.xlim_max), 100)
        D_vals = np.linspace(self.ylim_min, self.ylim_max, 100)
        L_grid, D_grid = np.meshgrid(L_vals, D_vals)
        
        # Calculate detectability regions
        detectability = np.zeros_like(L_grid)
        
        # M dwarf region (L < 0.08 L☉)
        m_dwarf_mask = L_grid < const.L_m_dwarf_max
        detectability[m_dwarf_mask] = 1
        
        # G star region (0.6 < L < 1.5 L☉)
        g_star_mask = (L_grid >= const.L_g_star_min) & (L_grid <= const.L_g_star_max)
        detectability[g_star_mask] = 1
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]
            
        # Add pink fill for too faint region - follows the angular separation limit curve
        # Calculate the boundary curve for the pink region
        L_pink = np.logspace(np.log10(0.001), np.log10(0.1), 100)
        D_pink_boundary = (np.sqrt(L_pink) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        
        # Fill the region below the curve (where IWA is too small in habitable zone)
        mask = (D_pink_boundary <= 35.0) & (D_pink_boundary >= 0)
        if np.any(mask):
            ax.fill_between(L_pink[mask], 0, D_pink_boundary[mask], 
                           color='pink', alpha=0.3, label='IWA too small in habitable zone')
        
        # Setup detectability plot
        im = self._setup_detectability_plot(ax, L_vals, D_vals, region, 
                                           f'Detectable Radius Range ({const.R_earth_min_habitable}–{const.R_earth_max_habitable} R⊕)')
        ax.set_ylim(0.0, 35.0)
        ax.set_xlim(0.001, 2.0)
        
        # Add angular separation threshold boundary using basic habitable zone calculation
        D_theta_boundary = (np.sqrt(L_vals) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        
        mask = D_theta_boundary <= self.ylim_max
        ax.plot(L_vals[mask], D_theta_boundary[mask], color='black', linestyle='--', linewidth=2, 
               label='HWO Angular Sep. Limit (1.0 R⊕)')
        
        # Add equation annotation in upper right corner
        # The equation includes habitable zone dependency:
        # D = a_hz / θ_IWA = √L × 1 AU / θ_IWA
        # where a_hz = √L × 1 AU (habitable zone distance)
        ax.text(0.95, 0.95, f'$D = a_{{hz}} / \\theta_{{IWA}} = \\sqrt{{L}} \\times 1\\,AU / \\theta_{{IWA}}$', 
               transform=ax.transAxes, fontsize=10, ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Add reference lines
        self._plot_reference_lines(ax, L_vals, is_au=False)

        # Add M dwarfs observable region - ensure it covers the full range
        L_m_dwarf_range = np.linspace(0, const.L_m_dwarf_max, 100)
        D_theta_boundary_mdwarf = (np.sqrt(L_m_dwarf_range) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        mask = (D_theta_boundary_mdwarf <= 35.0) & (D_theta_boundary_mdwarf >= 0)
        if np.any(mask):
            ax.fill_between(L_m_dwarf_range[mask], 0, D_theta_boundary_mdwarf[mask], 
                           color='darkgreen', alpha=0.7, label='M dwarfs observable by HWO')

        # Add exoplanet overlay with specific planet labels
        plotted_methods = self._plot_exoplanet_overlay(ax, region_lum=(self.xlim_min, self.xlim_max), 
                                                      region_dist=(self.ylim_min, self.ylim_max))
        
        # Add specific labels for Proxima Centauri b and TOI-700 planets
        specific_planets = {
            'Proxima Cen b': (0.0016, 1.3),  # (luminosity, distance)
            'TOI-700 b': (0.023, 31.1),
            'TOI-700 c': (0.023, 31.1),
            'TOI-700 d': (0.023, 31.1),
            'TOI-700 e': (0.023, 31.1)
        }
        
        for planet_name, (lum, dist) in specific_planets.items():
            if (self.xlim_min <= lum <= self.xlim_max and 
                self.ylim_min <= dist <= self.ylim_max):
                ax.annotate(planet_name, (lum, dist), 
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Create legend
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', label=f'M dwarf Region (L = {const.L_m_dwarf_min}, {const.L_m_dwarf_max})'),
            Line2D([0], [0], color='gold', linestyle='--', label=f'G Star Region (L={const.L_g_star_min}, {const.L_g_star_max})'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)'),
            Patch(facecolor='darkgreen', alpha=1, label='M dwarfs with habitable planets observable by HWO'),
            Patch(facecolor='pink', alpha=0.3, label='IWA too small in habitable zone')
        ]
        
        # Note: Specific objects (Proxima Cen b and TOI-700) are plotted but not included in legend
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
        plt.tight_layout()
        self._save_plot(fig, 'distance_luminosity')
