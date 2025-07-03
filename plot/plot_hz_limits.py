import pandas as pd
import numpy as np
from typing import Optional
from scipy.spatial.qhull import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import alphashape
from shapely.geometry import Polygon
from matplotlib.colors import LinearSegmentedColormap
import os

from plot.base_plotter import BasePlotter
from tools import physics_constants as const
from plot.exoplanet_data_utils import load_exoplanet_luminosity_distance

plt.rcParams.update({'font.size': 16})

class PlotHZLimits(BasePlotter):
    """
    Plotter for M-dwarf HZ limits and related plots.
    
    Features:
    1. 3x1 boundary plots for data analysis
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, name: str = 'HWO',
                 nruns: int = 1, star_catalog: str = 'Gaia', **kwargs):
        """Initialize with optional dataframe and parameters."""
        if df is None:
            df = self._create_minimal_dataframe()
        super().__init__(df, nruns, star_catalog, name)
        
        # Cache HWO constants to avoid repeated instantiation
        self.hwo_best = const.HWOConstants('best')
        self.best_flux_limit = float(self.hwo_best.min_planet_flux_star_ratio)
        self.iwa_limit = float(self.hwo_best.iwa)
        self.max_z = float(self.hwo_best.max_z)
        self.theta_limit_rad = self.iwa_limit * const.arcsec_to_radians

    def plot_all(self) -> None:
        """Generate all M-dwarf HZ limit plots."""
        if not self._validate_data():
            return
            
        # 3x1 boundary plots
        self.plot_boundaries()
        self.plot_detectability_panel()
        self.plot_detectability_panel_au_vs_distance()

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
        # Filter out values that are too close to zero or too large
        x_finite = np.isfinite(points[:, 0]) & (np.abs(points[:, 0]) > 1e-15)
        y_finite = np.isfinite(points[:, 1]) & (np.abs(points[:, 1]) > 1e-15)
        finite_mask = x_finite & y_finite
        
        # Combine all filters
        final_mask = valid_mask & finite_mask
        filtered_points = points[final_mask]
        
        if len(filtered_points) < 3:
            return None
            
        # Check for numerical stability - ensure reasonable coordinate ranges
        x_range = np.ptp(filtered_points[:, 0])
        y_range = np.ptp(filtered_points[:, 1])
        
        # If ranges are too small or too large, skip
        if x_range < 1e-10 or y_range < 1e-10:
            return None
        if x_range > 1e15 or y_range > 1e15:
            return None
            
        return filtered_points

    def plot_boundary(self, data, xcol, ycol, ax, alpha_shape=True, alpha=0.01, **kwargs):
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

        if len(points) < 3:
            return  # Need at least 3 points to make a boundary

        # Validate and filter points to remove extreme values
        filtered_points = self._validate_and_filter_points(points, xcol, ycol)
        if filtered_points is None:
            return  # Not enough valid points after filtering

        # Extract fill parameters
        facecolor = kwargs.pop('facecolor', None)
        alpha_fill = kwargs.pop('alpha_fill', 0.3)
        hatch = kwargs.pop('hatch', None)
        linewidth = kwargs.get('linewidth', 3)  # Default linewidth

        try:
            if alpha_shape:
                shape = alphashape.alphashape(filtered_points, alpha)
                if isinstance(shape, Polygon):
                    x, y = shape.exterior.xy
                    # Plot filled region if facecolor is specified
                    if facecolor:
                        ax.fill(x, y, facecolor=facecolor, alpha=alpha_fill, hatch=hatch, **kwargs)
                    # Plot boundary line only if linewidth > 0
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
            # Catch any Qhull or other geometric errors and skip this boundary
            print(f"Warning: Could not compute boundary for {xcol} vs {ycol}: {str(e)}")
            return

    def _create_minimal_dataframe(self) -> pd.DataFrame:
        """Create a minimal dataframe for the plotter to work with."""
        return pd.DataFrame({'dummy': [1]})

    def _get_panel_detection_mask(self, df_panel: pd.DataFrame) -> pd.Series:
        """Get detection mask for a specific panel."""
        mask_best, _ = self._get_detection_masks()
        # Convert to pandas Series for proper indexing
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

    def plot_boundaries(self) -> None:
        """Plot a single panel: stellar temperature vs semi-major axis with boundaries only (no density)."""
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

    def _compute_fraction_detectable(self, AU_grid, D_grid, A_g, Phi, L_check):
        """Compute the fraction of L values for which a planet is detectable at each AU and D (vectorized)."""
        # Vectorized calculation
        a_m = AU_grid[..., None] * const.au_to_m   # shape (N, M, 1)
        d_m = D_grid[..., None] * const.pc_to_m   # shape (N, M, 1)
        theta_arcsec = (a_m / d_m) * const.rad_to_arcsec
        flux_ratio = A_g * (const.R_earth / a_m) ** 2 * Phi
        # L_check is shape (K,), broadcast to (N, M, K)
        detectable = (flux_ratio >= const.HWOConstants('best').min_planet_flux_star_ratio) & (theta_arcsec >= const.HWOConstants('best').iwa)
        fraction_detectable = detectable.sum(axis=2) / len(L_check)
        return fraction_detectable

    def _plot_reference_lines(self, ax, xvals, is_au=False):
        """Plot reference lines for M-dwarf and G-star regions."""
        for L, color in zip([0.08, 0.6, 1.5], ['red', 'gold', 'gold']):
            val = np.sqrt(L) if is_au else L
            ax.axvline(val, color=color, linestyle='--', linewidth=2)

    def plot_detectability_panel(self):
        """Plot detectability for 0.5–2.6 R⊕ and 'none' as a two-panel figure, overlaying exoplanet data in the zoomed panel."""
        # --- Constants ---
        AU_m = const.au_to_m 
        A_g = getattr(const, 'A_g_earth', 0.2)
        Phi = getattr(const, 'Phi_alpha', 1.0)
        flux_threshold = const.HWOConstants('best').min_planet_flux_star_ratio
        theta_limit_arcsec = const.HWOConstants('best').iwa

        # --- Panel 1: L vs D, region coloring ---
        L_vals = np.logspace(-2, 1, 300)
        D_vals = np.linspace(4, 20, 300)
        L_grid, D_grid = np.meshgrid(L_vals, D_vals)
        a_hz_m = np.sqrt(L_grid) * AU_m
        distance_m = D_grid * const.pc_to_m
        theta_arcsec = (a_hz_m / distance_m) * const.rad_to_arcsec
        Rp_small = 0.5 * const.R_earth
        Rp_large = 2.6 * const.R_earth
        flux_ratio_small = A_g * (Rp_small / a_hz_m) ** 2 * Phi
        flux_ratio_large = A_g * (Rp_large / a_hz_m) ** 2 * Phi
        detect_small = (flux_ratio_small >= flux_threshold) & (theta_arcsec >= theta_limit_arcsec)
        detect_large = (flux_ratio_large >= flux_threshold) & (theta_arcsec >= theta_limit_arcsec)
        region = np.zeros_like(L_grid, dtype=int)
        region[detect_small | detect_large] = 1

        # --- Plotting: Two panels ---
        fig, ax = plt.subplots(figsize=(10, 6))
        L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        # Single panel
        im = ax.pcolormesh(L_edges, D_edges, region, cmap=LinearSegmentedColormap.from_list('pinkgreen', ['#f7cac9', '#88b04b']), shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_ylim(4, 20)
        ax.set_xlim(L_vals[0], L_vals[-1])
        ax.set_title('Detectable Radius Range (0.5–2.6 R⊕)')
        self._plot_reference_lines(ax, L_vals, is_au=False)

        # --- Restore flux threshold boundary (vertical purple line) ---
        try:
            AU_m_val = float(AU_m)
        except Exception:
            AU_m_val = 1.496e11
        try:
            rad2arcsec_val = float(rad2arcsec)
        except Exception:
            rad2arcsec_val = 206265
        try:
            theta_limit_arcsec_val = float(theta_limit_arcsec)
        except Exception:
            theta_limit_arcsec_val = 0.0206
        try:
            pc_m_val = float(pc_m)
        except Exception:
            pc_m_val = 3.086e16
        A_g_val = A_g
        Phi_val = Phi
        flux_threshold_val = flux_threshold
        L_flux_boundary = (A_g_val * const.R_earth**2 * Phi_val) / (flux_threshold_val * AU_m_val**2)
        ax.axvline(x=L_flux_boundary, color='purple', linestyle=':', linewidth=2, label='Flux Limit Boundary (1.0 R⊕)')

        # --- Restore angular separation threshold boundary (black dashed curve) ---
        L_star_vals = L_vals
        D_theta_boundary = (np.sqrt(L_star_vals) * AU_m_val * rad2arcsec_val) / (theta_limit_arcsec_val * pc_m_val)
        mask = D_theta_boundary <= 20
        ax.plot(L_star_vals[mask], D_theta_boundary[mask], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)')

        # --- (Optional) Fitted line (if previously present) ---
        # Example: Fit a line to the boundary (customize as needed)
        # from numpy.polynomial.polynomial import Polynomial
        # fit_mask = (L_star_vals > 0.01) & (L_star_vals < 0.4)
        # p = Polynomial.fit(np.log10(L_star_vals[fit_mask]), np.log10(D_theta_boundary[fit_mask]), 1)
        # ax.plot(L_star_vals[fit_mask], 10**p(np.log10(L_star_vals[fit_mask])), color='blue', linestyle='-', linewidth=2, label='Fitted Line')

        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', label='M-dwarf Region (L = 0.001, L=0.08)'),
            Line2D([0], [0], color='gold', linestyle='--', label='G Star Region (L=0.6, 1.5)'),
            Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Flux Limit Boundary (1.0 R⊕)'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=14)

        # --- Exoplanet overlay using utility function (left panel) ---
        exo_df_left = load_exoplanet_luminosity_distance(region_lum=(0.01, 10), region_dist=(4, 20), return_names=True)
        if exo_df_left is not None and not exo_df_left.empty:
            ax.scatter(exo_df_left['Luminosity'], exo_df_left['Distance'], color='black', s=8, alpha=0.5, label='2025 Exoplanet Hosts')
            if 'Planet Name' in exo_df_left.columns:
                for _, row in exo_df_left.iterrows():
                    ax.text(row['Luminosity'], row['Distance']+0.15, str(row['Planet Name']), fontsize=7, ha='center', va='bottom', rotation=30)

        # --- Colorbar ---
        cbar = fig.colorbar(im, ax=ax, location='right', shrink=0.8, label='Fraction of L detectable (0.08–1e3)')
        plt.tight_layout()
        self._save_plot(fig, 'detectability_panel_twopanel')

    def plot_detectability_panel_au_vs_distance(self):
        """Plot detectability as AU vs distance, with background gradient showing fraction of L detectable (single panel)."""
        # --- Grid ---
        AU_vals = np.linspace(0.01, 1e3, 1000)  # AU
        D_vals = np.linspace(4, 20, 1000)  # Distance [pc]
        AU_grid, D_grid = np.meshgrid(AU_vals, D_vals)
        # --- Constants ---
        AU_m = const.au_to_m
        pc_m = 3.086e16
        rad2arcsec = 206265
        A_g = getattr(const, 'A_g_earth', 0.2)
        Phi = getattr(const, 'Phi_alpha', 1.0)
        flux_threshold = const.HWOConstants('best').min_planet_flux_star_ratio
        theta_limit_arcsec = const.HWOConstants('best').iwa
        L_vals = np.logspace(np.log10(0.08), 3, 100)
        fraction_detectable = np.zeros_like(AU_grid, dtype=float)
        for idx in range(AU_grid.shape[0]):
            for jdx in range(AU_grid.shape[1]):
                a_au = AU_grid[idx, jdx]
                d_pc = D_grid[idx, jdx]
                a_m = a_au * AU_m
                d_m = d_pc * pc_m
                theta_arcsec = (a_m / d_m) * rad2arcsec
                count = 0
                for L in L_vals:
                    flux_ratio = A_g * (const.R_earth / a_m) ** 2 * Phi
                    if (flux_ratio >= flux_threshold) and (theta_arcsec >= theta_limit_arcsec):
                        count += 1
                fraction_detectable[idx, jdx] = count / len(L_vals)
        cmap = LinearSegmentedColormap.from_list('redyellowgreen', ['#f44336', '#fff176', '#4caf50'])
        fig, ax = plt.subplots(figsize=(8, 6))
        AU_edges = np.linspace(AU_vals[0], AU_vals[-1], AU_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        im = ax.pcolormesh(AU_edges, D_edges, fraction_detectable, cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Semi-major Axis [AU]')
        ax.set_ylabel('Distance [pc]')
        ax.set_ylim(4, 20)
        ax.set_xlim(0.01, 1e3)
        ax.set_title('Detectability by Stellar Luminosity (1 R⊕, L ≥ 0.08 L☉)')
        for L, color in zip([0.08, 0.6, 1.5], ['red', 'gold', 'gold']):
            ax.axvline(np.sqrt(L), color=color, linestyle='--', linewidth=2)
        cbar = fig.colorbar(im, ax=ax, label='Fraction of L detectable (0.08–1e3)')
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', label='M-dwarf Region (a=√0.08)'),
            Line2D([0], [0], color='gold', linestyle='--', label='G Star Region (a=√0.6, √1.5)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=14)
        plt.tight_layout()
        self._save_plot(fig, 'detectability_panel_au_vs_distance')

    def plot_density_bins(self, data, xcol, ycol, ax, kind='hist2d', cmap='Greys', gridsize=50, **kwargs):
        """Plot a density map (hexbin or hist2d) for the given data and axis."""
        points = data[[xcol, ycol]].dropna().values
        if len(points) == 0:
            return

        if kind == 'hexbin':
            # C=None means color based on count
            plot_obj = ax.hexbin(points[:, 0], points[:, 1], gridsize=gridsize, cmap=cmap, **kwargs)
        elif kind == 'hist2d':
            plot_obj = ax.hist2d(points[:, 0], points[:, 1], bins=gridsize, cmap=cmap, **kwargs)
        else:
            raise ValueError("kind must be 'hexbin' or 'hist2d'")
        return plot_obj
