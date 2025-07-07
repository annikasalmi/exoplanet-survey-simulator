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
                 xlim_min: float = 0.01, xlim_max: float = 15.0,
                 ylim_min: float = 4.0, ylim_max: float = 15.0, **kwargs):
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
        """Plot exoplanet overlay with detection method colors."""
        plotted_methods = set()
        try:
            from plot.exoplanet_data_utils import load_exoplanet_luminosity_distance
            exo_df = load_exoplanet_luminosity_distance(region_lum=region_lum, 
                                                       region_dist=region_dist, 
                                                       return_names=True)
            if exo_df is None or exo_df.empty:
                return plotted_methods
                
            # Plot exoplanets by detection method
            for _, row in exo_df.iterrows():
                method = str(row.get('Detection Method', 'Other'))
                color = self.DETECTION_COLORS.get(method, 'gray')
                if method not in plotted_methods:
                    ax.scatter(row['Luminosity'], row['Distance'], color=color, s=8, alpha=0.7, 
                              label=f'R<{const.R_earth_max_habitable}R⊕ found by {method}')
                    plotted_methods.add(method)
                else:
                    ax.scatter(row['Luminosity'], row['Distance'], color=color, s=8, alpha=0.7)
                    
                # Add planet name labels
                if 'Planet Name' in exo_df.columns:
                    ax.text(row['Luminosity'], row['Distance']+0.15, str(row['Planet Name']), 
                           fontsize=7, ha='center', va='bottom', rotation=30)
        except ImportError:
            print("Warning: exoplanet_data_utils not available, skipping exoplanet overlay")
        
        return plotted_methods

    def _setup_detectability_plot(self, ax: Axes, L_vals: np.ndarray, D_vals: np.ndarray, 
                                region: np.ndarray, title: str):
        """Setup detectability plot with common elements."""
        L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        
        im = ax.pcolormesh(L_edges, D_edges, region, 
                           cmap=LinearSegmentedColormap.from_list('pinkgreen', ['#f7cac9', '#88b04b']), 
                           shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_title(title)
        self._plot_reference_lines(ax, L_vals, is_au=False)
        return im

    def _get_axis_label(self, var_name: str) -> str:
        """Get formatted axis label for a variable name."""
        labels = {
            'semimajor_p': 'Semi-major Axis (AU)',
            'p_orb': 'Orbital Period (days)',
            'radius_p': 'Planet Radius (R⊕)',
            'temp': 'Planet Temperature (K)',
            'temp_s': 'Stellar Temperature (K)',
            'mass_s': 'Stellar Mass (M☉)',
            'flux_p': 'Planet Flux (W/m²)'
        }
        return labels.get(var_name, var_name)

    def _plot_reference_lines(self, ax: Axes, xvals: np.ndarray, is_au: bool = False) -> None:
        """Plot reference lines for M dwarf and G-star regions."""
        for L, color in zip([const.L_m_dwarf_max, const.L_g_star_min, const.L_g_star_max], ['red', 'gold', 'gold']):
            val = np.sqrt(L) if is_au else L
            ax.axvline(float(val), color=color, linestyle='--', linewidth=2)

    def plot_boundaries(self) -> None:
        """Plot stellar temperature vs semi-major axis with boundaries."""
        def _safe_float(val):
            try:
                return float(val)
            except Exception:
                return np.nan

        if self.df.empty:
            print("Warning: No data available for panel plot")
            return

        x_var, y_var = 'temp_s', 'semimajor_p'
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
                boundary_cache[category] = boundary_points
            except Exception as e:
                print(f"Warning: Could not compute boundary for {x_var} vs {y_var} ({category}): {str(e)}")
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
        ax.set_xlabel(self._get_axis_label(x_var))
        ax.set_ylabel(self._get_axis_label(y_var))
        ax.set_title(f'{self._get_axis_label(x_var)} vs {self._get_axis_label(y_var)}')
        
        # Set axis limits to focus on actual data range (K, G, F stars)
        # Data range: 3465-7498K, G-star T range: ~5000-6500K
        ax.set_xlim(3000, 8000)  # Cover K, G, F star temperature range
                
        y_data = self.df[y_var]
        if isinstance(y_data, pd.Series):
            y_data = y_data.dropna()
        if len(y_data) > 0:
            y_min = _safe_float(y_data.min())
            y_max = _safe_float(y_data.max())
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                ax.set_ylim(y_min * 0.9, y_max * 1.1)
                
        # Add habitable zone line
        self._add_habitable_zone_line(ax)
        
        # Add M-dwarf and G-star reference lines
        self._add_stellar_type_reference_lines(ax)
        
        # Create legend with all elements
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Flux Rejected'),
            Patch(facecolor='gold', alpha=0.3, label='Exozodi Rejected'),
            Patch(facecolor='blue', alpha=0.3, label='IWA Rejected'),
            Patch(facecolor='green', alpha=1.0, label='Detected'),
            Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Habitable Zone (R=1 R⊕)'),
            Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='M dwarf T range'),
            Line2D([0], [0], color='gold', linestyle='--', linewidth=2, label='G star T range')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
        plt.tight_layout()
        self._save_plot(fig, 'panel_temp_s_vs_semimajor_p')

    def _add_habitable_zone_line(self, ax: Axes) -> None:
        """Add habitable zone line for R=1 R⊕ planet."""
        # For each stellar temperature, calculate the habitable zone distance
        # HZ distance scales as sqrt(L_star), and L scales as T^4
        T_range = np.linspace(2000, 7000, 100)  # Temperature range in K
        L_from_T = (T_range / const.temp_sun)**4  # Luminosity from temperature
        a_hz_au = np.sqrt(L_from_T)  # HZ distance in AU
        
        # Calculate habitable zone curve
        hz_curve_T = []
        hz_curve_a = []
        for i, T in enumerate(T_range):
            a_hz_au_single = a_hz_au[i]
            
            # For this stellar temperature, find the semi-major axis in the habitable zone
            if 0.1 <= a_hz_au_single <= 10.0:  # Reasonable HZ range
                hz_curve_T.append(T)
                hz_curve_a.append(a_hz_au_single)
        
        if hz_curve_T:
            ax.plot(hz_curve_T, hz_curve_a, color='black', linestyle='-', linewidth=2, 
                   label='Habitable Zone (R=1 R⊕)')

    def _add_stellar_type_reference_lines(self, ax: Axes) -> None:
        """Add M-dwarf and G-star reference lines to the plot."""
        # Use actual M-dwarf temperature range from data (hot M-dwarfs M0-M2)
        T_m_dwarf_min = const.T_m_dwarf_data_min
        T_m_dwarf_max = const.T_m_dwarf_data_max
        T_g_star_min = const.T_g_star_min
        T_g_star_max = const.T_g_star_max
        
        # Add vertical lines for stellar type boundaries
        ax.axvline(T_m_dwarf_min, color='red', linestyle='--', linewidth=2, 
                  label=f'M dwarf min T ({T_m_dwarf_min:.0f} K)')
        ax.axvline(T_m_dwarf_max, color='red', linestyle='--', linewidth=2, 
                  label=f'M dwarf max T ({T_m_dwarf_max:.0f} K)')
        ax.axvline(T_g_star_min, color='gold', linestyle='--', linewidth=2, 
                  label=f'G star min T ({T_g_star_min:.0f} K)')
        ax.axvline(T_g_star_max, color='gold', linestyle='--', linewidth=2, 
                  label=f'G star max T ({T_g_star_max:.0f} K)')

    def plot_luminosity_distance(self) -> None:
        """Plot detectability for habitable planet radius range with exoplanet overlay."""
        # Create grid for calculations
        L_vals = np.logspace(np.log10(self.xlim_min), np.log10(self.xlim_max), const.L_GRID_SIZE)
        D_vals = np.linspace(self.ylim_min, self.ylim_max, const.D_GRID_SIZE)
        L_grid, D_grid = np.meshgrid(L_vals, D_vals)
        
        # Calculate angular separations
        a_hz_m = np.sqrt(L_grid) * const.au_to_m
        distance_m = D_grid * const.pc_to_m
        theta_arcsec = (a_hz_m / distance_m) * const.rad_to_arcsec
        
        # Calculate flux ratios for different planet sizes
        Rp_small = const.R_earth_min_habitable * const.R_earth
        Rp_large = const.R_earth_max_habitable * const.R_earth
        
        # Calculate stellar temperature from luminosity
        T_star = (L_grid * const.temp_sun**4)**0.25
        T_planet = const.T_earth
        
        # Use Tp*Rp²/(Ts*Rs²) approximation for contrast ratio
        flux_ratio_small = (T_planet * Rp_small**2) / (T_star * const.R_sun**2) / (distance_m / const.pc_to_m) ** 2
        flux_ratio_large = (T_planet * Rp_large**2) / (T_star * const.R_sun**2) / (distance_m / const.pc_to_m) ** 2
        
        # Determine detectable regions
        detect_small = (flux_ratio_small >= self.best_flux_limit) & (theta_arcsec >= self.theta_limit_rad * const.rad_to_arcsec)
        detect_large = (flux_ratio_large >= self.best_flux_limit) & (theta_arcsec >= self.theta_limit_rad * const.rad_to_arcsec)
        region = np.zeros_like(L_grid, dtype=int)
        region[detect_small | detect_large] = 1

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]
            
        # Add pink fill for too faint region
        ax.fill_betweenx([0, self.ylim_max], self.xlim_min, 0.1, 
                        color='pink', alpha=0.3, label='Too faint to detect')
        
        # Setup detectability plot
        im = self._setup_detectability_plot(ax, L_vals, D_vals, region, 
                                           f'Detectable Radius Range ({const.R_earth_min_habitable}–{const.R_earth_max_habitable} R⊕)')
        ax.set_ylim(self.ylim_min, self.ylim_max)
        ax.set_xlim(self.xlim_min, 2.0)
        
        # Add angular separation threshold boundary
        D_theta_boundary = (np.sqrt(L_vals) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        mask = D_theta_boundary <= self.ylim_max
        ax.plot(L_vals[mask], D_theta_boundary[mask], color='black', linestyle='--', linewidth=2, 
               label='HWO Angular Sep. Limit (1.0 R⊕)')

        # Add reference lines
        self._plot_reference_lines(ax, L_vals, is_au=False)

        # Add M dwarfs observable region
        L_m_dwarf_range = np.linspace(0, const.L_m_dwarf_max, 100)
        D_theta_boundary_mdwarf = (np.sqrt(L_m_dwarf_range) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        mask = (D_theta_boundary_mdwarf <= self.ylim_max) & (D_theta_boundary_mdwarf >= 0)
        if np.any(mask):
            ax.fill_between(L_m_dwarf_range[mask], 0, D_theta_boundary_mdwarf[mask], 
                           color='darkgreen', alpha=1, label='M dwarfs observable by HWO')

        # Add exoplanet overlay
        plotted_methods = self._plot_exoplanet_overlay(ax, region_lum=(self.xlim_min, self.xlim_max), 
                                                      region_dist=(self.ylim_min, self.ylim_max))

        # Create legend
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', label=f'M dwarf Region (L = {const.L_m_dwarf_min}, {const.L_m_dwarf_max})'),
            Line2D([0], [0], color='gold', linestyle='--', label=f'G Star Region (L={const.L_g_star_min}, {const.L_g_star_max})'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)'),
            Patch(facecolor='darkgreen', alpha=1, label='M dwarfs with habitable planets observable by HWO'),
            Patch(facecolor='pink', alpha=0.3, label='Too faint to detect')
        ]
        
        # Add exoplanet detection methods to legend
        for method in plotted_methods:
            legend_elements.append(
                Line2D([0], [0], marker='o', color=self.DETECTION_COLORS[method], markersize=8, 
                       linestyle='', label=f'R<{const.R_earth_max_habitable}R⊕ found by {method}')
            )
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
        plt.tight_layout()
        self._save_plot(fig, 'distance_luminosity')
