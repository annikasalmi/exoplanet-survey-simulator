import pandas as pd
import numpy as np
from typing import Optional, Tuple, Any
from scipy.spatial.qhull import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.axes import Axes
import alphashape
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap

from plot.base_plotter import BasePlotter
from tools import physics_constants as const
from tools.plotting_constants import DETECTION_COLORS

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
        
        # Plot limits - change these to modify x and y axis ranges
        self.xlim_min = xlim_min
        self.xlim_max = xlim_max
        self.ylim_min = ylim_min
        self.ylim_max = ylim_max
        
        # Cache HWO constants to avoid repeated instantiation
        self.hwo_best = const.HWOConstants('best')
        self.best_flux_limit = float(self.hwo_best.min_planet_flux_star_ratio)
        self.iwa_limit = float(self.hwo_best.iwa)
        self.max_z = float(self.hwo_best.max_z)
        self.theta_limit_rad = self.iwa_limit * const.arcsec_to_radians
        
        # Add detection colors mapping
        self.DETECTION_COLORS = DETECTION_COLORS

    def plot_all(self) -> None:
        """Generate all M dwarf HZ limit plots."""
        if not self._validate_data():
            return
            
        self.plot_boundaries()
        self.plot_luminosity_distance()

    def _validate_and_filter_points(self, points: np.ndarray, xcol: str, ycol: str) -> Optional[np.ndarray]:
        """
        Validate and filter points to remove extreme values that cause Qhull precision errors.
        
        Parameters:
            points: numpy array of shape (n, 2) with x and y coordinates
            xcol, ycol: column names for debugging
            
        Returns:
            Filtered points array, or None if insufficient valid points
        """
        if len(points) < 3:
            return None
            
        # Define reasonable bounds for different variable types
        bounds = {
            'semimajor_p': (1e-3, 100),      # AU
            'p_orb': (1e-2, 1e5),           # days
            'radius_p': (1e-2, 100),        # Earth radii
            'temp': (1, 1e4),               # Kelvin
            'temp_s': (1e3, 1e5),           # Kelvin
            'mass_s': (1e-3, 100),          # Solar masses
            'flux_p': (1e-10, 1e4),         # W/m²
            'flux_ratio_value_best': (1e-20, 1e-2),  # dimensionless
            'maxangsep': (1e-6, 1e3),       # arcsec
            'z': (1e-6, 1e3)                # dimensionless
        }
        
        # Get bounds for this specific combination
        x_bounds = bounds.get(xcol, (1e-10, 1e10))
        y_bounds = bounds.get(ycol, (1e-10, 1e10))
        
        # Filter out extreme values
        x_valid = (points[:, 0] >= x_bounds[0]) & (points[:, 0] <= x_bounds[1])
        y_valid = (points[:, 1] >= y_bounds[0]) & (points[:, 1] <= y_bounds[1])
        valid_mask = x_valid & y_valid
        
        # Additional check for reasonable numerical ranges
        x_finite = np.isfinite(points[:, 0]) & (np.abs(points[:, 0]) > 1e-15)
        y_finite = np.isfinite(points[:, 1]) & (np.abs(points[:, 1]) > 1e-15)
        finite_mask = x_finite & y_finite
        
        # Combine all filters
        final_mask = valid_mask & finite_mask
        filtered_points = points[final_mask]
        
        if len(filtered_points) < 3:
            return None
            
        # Check for numerical stability
        x_range = np.ptp(filtered_points[:, 0])
        y_range = np.ptp(filtered_points[:, 1])
        
        if x_range < 1e-10 or y_range < 1e-10 or x_range > 1e15 or y_range > 1e15:
            return None
            
        return filtered_points

    def _plot_boundary_shape(self, filtered_points: np.ndarray, ax: Axes, alpha_shape: bool = True, 
                           alpha: float = 0.01, **kwargs) -> None:
        """Helper method to plot boundary shapes."""
        facecolor = kwargs.pop('facecolor', None)
        alpha_fill = kwargs.pop('alpha_fill', 0.3)
        hatch = kwargs.pop('hatch', None)
        linewidth = kwargs.get('linewidth', 3)

        try:
            if alpha_shape:
                shape = alphashape.alphashape(filtered_points, alpha)
                if isinstance(shape, Polygon):
                    x, y = shape.exterior.xy
                    if facecolor:
                        ax.fill(x, y, facecolor=facecolor, alpha=alpha_fill, hatch=hatch, **kwargs)
                    if linewidth > 0:
                        ax.plot(x, y, **kwargs)
                else:
                    # Fallback to convex hull if alphashape fails
                    hull = ConvexHull(filtered_points)
                    boundary = np.append(hull.vertices, hull.vertices[0])
                    if facecolor:
                        ax.fill(filtered_points[boundary, 0], filtered_points[boundary, 1], 
                               facecolor=facecolor, alpha=alpha_fill, hatch=hatch, **kwargs)
                    if linewidth > 0:
                        ax.plot(filtered_points[boundary, 0], filtered_points[boundary, 1], **kwargs)
            else:
                hull = ConvexHull(filtered_points)
                boundary = np.append(hull.vertices, hull.vertices[0])
                if facecolor:
                    ax.fill(filtered_points[boundary, 0], filtered_points[boundary, 1], 
                           facecolor=facecolor, alpha=alpha_fill, hatch=hatch, **kwargs)
                if linewidth > 0:
                    ax.plot(filtered_points[boundary, 0], filtered_points[boundary, 1], **kwargs)
        except Exception as e:
            print(f"Warning: Could not compute boundary shape: {str(e)}")

    def plot_boundary(self, data: pd.DataFrame, xcol: str, ycol: str, ax: Axes, 
                     alpha_shape: bool = True, alpha: float = 0.01, **kwargs) -> None:
        """
        Plot the boundary around a group of 2D points.

        Parameters:
            data (pd.DataFrame): Subset of points.
            xcol, ycol (str): Column names for X and Y.
            ax (matplotlib axis): Axis to plot on.
            alpha_shape (bool): Whether to use concave alphashape (True) or convex hull (False).
            alpha (float): Alpha parameter for alphashape. Smaller values = tighter fit.
            **kwargs: Extra plotting args including facecolor, alpha_fill, and hatch for filled regions.
        """
        points = data[[xcol, ycol]].dropna().values
        points = np.asarray(points, dtype=float)

        if len(points) < 3:
            return

        filtered_points = self._validate_and_filter_points(points, xcol, ycol)
        if filtered_points is None:
            return

        self._plot_boundary_shape(filtered_points, ax, alpha_shape, alpha, **kwargs)

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
        """Helper method to plot exoplanet overlay with detection method colors."""
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
        """Helper method to setup detectability plot with common elements."""
        L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        
        im = ax.pcolormesh(L_edges, D_edges, region, 
                          cmap=LinearSegmentedColormap.from_list('pink', ['#f7cac9', '#88b04b']), 
                          shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_title(title)
        self._plot_reference_lines(ax, L_vals, is_au=False)
        return im

    def plot_boundaries(self) -> None:
        """Plot stellar temperature vs semi-major axis with boundaries only."""
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

        # Masks and data subsets
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
        
        # Compute boundaries
        boundary_cache = {}
        fig, ax = plt.subplots(figsize=(6, 6))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]
            
        for category, data in data_subsets.items():
            if not isinstance(data, pd.DataFrame):
                continue
            valid_data = data[[x_var, y_var]].dropna()
            if len(valid_data) < 3:
                continue
            points = np.asarray(valid_data.values, dtype=float)
            filtered_points = self._validate_and_filter_points(points, x_var, y_var)
            if filtered_points is None:
                continue
            try:
                hull = ConvexHull(filtered_points)
                boundary = np.append(hull.vertices, hull.vertices[0])
                boundary_points = filtered_points[boundary]
                boundary_cache[category] = boundary_points
            except Exception as e:
                print(f"Warning: Could not compute boundary for {x_var} vs {y_var} ({category}): {str(e)}")
                continue
                
        for category, color, label in boundary_configs:
            if category in boundary_cache:
                boundary_points = boundary_cache[category]
                linewidth = 3 if color == 'green' else 0
                alpha_fill = 0.3 if color in ['red', 'gold', 'blue'] else None
                facecolor = color if color in ['red', 'gold', 'blue'] else None
                if facecolor:
                    ax.fill(boundary_points[:, 0], boundary_points[:, 1], facecolor=facecolor, alpha=alpha_fill, color=color, linewidth=linewidth, label=label)
                elif linewidth > 0:
                    ax.plot(boundary_points[:, 0], boundary_points[:, 1], color=color, linewidth=linewidth, label=label)
                    
        ax.set_xlabel(self._get_axis_label(x_var))
        ax.set_ylabel(self._get_axis_label(y_var))
        ax.set_title(f'{self._get_axis_label(x_var)} vs {self._get_axis_label(y_var)}')
        
        # Set axis limits
        x_data = self.df[x_var]
        if isinstance(x_data, pd.Series):
            x_data = x_data.dropna()
        if len(x_data) > 0:
            x_min = _safe_float(x_data.min())
            x_max = _safe_float(x_data.max())
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                ax.set_xlim(x_min * 0.9, x_max * 1.1)
                
        y_data = self.df[y_var]
        if isinstance(y_data, pd.Series):
            y_data = y_data.dropna()
        if len(y_data) > 0:
            y_min = _safe_float(y_data.min())
            y_max = _safe_float(y_data.max())
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                ax.set_ylim(y_min * 0.9, y_max * 1.1)
                
        legend_elements = [
            Patch(facecolor='red', alpha=0.3, label='Flux Rejected'),
            Patch(facecolor='gold', alpha=0.3, label='Exozodi Rejected'),
            Patch(facecolor='blue', alpha=0.3, label='IWA Rejected'),
            Patch(facecolor='green', alpha=1.0, label='Detected')
        ]
        ax.legend(handles=legend_elements, loc='upper left', fontsize=14)
        plt.tight_layout()
        self._save_plot(fig, 'panel_temp_s_vs_semimajor_p')

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

    def plot_luminosity_distance(self) -> None:
        """Plot detectability for habitable planet radius range with exoplanet overlay."""
        # --- Panel 1: L vs D, region coloring ---
        L_vals = np.logspace(np.log10(self.xlim_min), np.log10(self.xlim_max), const.L_GRID_SIZE)
        D_vals = np.linspace(self.ylim_min, self.ylim_max, const.D_GRID_SIZE)
        L_grid, D_grid = np.meshgrid(L_vals, D_vals)
        a_hz_m = np.sqrt(L_grid) * const.au_to_m
        distance_m = D_grid * const.pc_to_m
        theta_arcsec = (a_hz_m / distance_m) * const.rad_to_arcsec
        Rp_small = const.R_earth_min_habitable * const.R_earth
        Rp_large = const.R_earth_max_habitable * const.R_earth
        flux_ratio_small = const.A_g_earth * (Rp_small / a_hz_m) ** 2 * const.Phi_alpha
        flux_ratio_large = const.A_g_earth * (Rp_large / a_hz_m) ** 2 * const.Phi_alpha
        detect_small = (flux_ratio_small >= self.best_flux_limit) & (theta_arcsec >= self.theta_limit_rad * const.rad_to_arcsec)
        detect_large = (flux_ratio_large >= self.best_flux_limit) & (theta_arcsec >= self.theta_limit_rad * const.rad_to_arcsec)
        region = np.zeros_like(L_grid, dtype=int)
        region[detect_small | detect_large] = 1

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(10, 6))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]
        im = self._setup_detectability_plot(ax, L_vals, D_vals, region, 
                                           f'Detectable Radius Range ({const.R_earth_min_habitable}–{const.R_earth_max_habitable} R⊕)')
        ax.set_ylim(self.ylim_min, self.ylim_max)
        ax.set_xlim(self.xlim_min, self.xlim_max)

        # --- Flux threshold boundary (vertical black line) ---
        L_flux_boundary = (const.A_g_earth * const.R_earth**2 * const.Phi_alpha) / (self.best_flux_limit * const.au_to_m**2)
        ax.axvline(x=float(L_flux_boundary), color='black', linestyle=':', linewidth=2, label='Flux Limit Boundary (1.0 R⊕)')
        
        # --- Pink fill for region outside flux limit (too faint to detect) ---
        ax.fill_betweenx([0, self.ylim_max], L_flux_boundary, self.xlim_max, 
                        color='pink', alpha=0.3, label='Too faint to detect')

        # --- Angular separation threshold boundary (black dashed curve) ---
        L_star_vals = L_vals
        D_theta_boundary = (np.sqrt(L_star_vals) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        mask = D_theta_boundary <= self.ylim_max
        ax.plot(L_star_vals[mask], D_theta_boundary[mask], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)')

        # --- Shaded box for M dwarfs observable region ---
        # Use the black dashed curve (angular separation limit) as the top boundary
        # The observable region is below the black curve and within the M dwarf luminosity range
        L_m_dwarf_range = np.linspace(0, const.L_m_dwarf_max, 100)
        D_theta_boundary_mdwarf = (np.sqrt(L_m_dwarf_range) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        
        # Only fill where the boundary is within plot limits
        mask = (D_theta_boundary_mdwarf <= self.ylim_max) & (D_theta_boundary_mdwarf >= 0)
        if np.any(mask):
            ax.fill_between(L_m_dwarf_range[mask], 0, D_theta_boundary_mdwarf[mask], 
                           color='darkgreen', alpha=1, label='M dwarfs observable by HWO')

        # --- Exoplanet overlay ---
        plotted_methods = self._plot_exoplanet_overlay(ax, region_lum=(self.xlim_min, self.xlim_max), region_dist=(self.ylim_min, self.ylim_max))

        # --- Legend (after exoplanet overlay so detection methods are included) ---
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', label=f'M dwarf Region (L = {const.L_m_dwarf_min}, {const.L_m_dwarf_max})'),
            Line2D([0], [0], color='gold', linestyle='--', label=f'G Star Region (L={const.L_g_star_min}, {const.L_g_star_max})'),
            Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='Flux Limit Boundary (1.0 R⊕)'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)'),
            Patch(facecolor='darkgreen', alpha=1, label='M dwarfs with habitable planets observable by HWO'),
            Patch(facecolor='pink', alpha=0.3, label='Too faint to detect')
        ]
        
        # Add exoplanet detection method colors to legend
        for method in plotted_methods:
            legend_elements.append(
                Line2D([0], [0], marker='o', color=self.DETECTION_COLORS[method], markersize=8, 
                       linestyle='', label=f'R<{const.R_earth_max_habitable}R⊕ found by {method}')
            )
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

        # --- Colorbar ---
        plt.tight_layout()
        self._save_plot(fig, 'distance_luminosity')
