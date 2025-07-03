import pandas as pd
import numpy as np
from typing import Optional
from scipy.spatial.qhull import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import alphashape
from shapely.geometry import Polygon

from plot.base_plotter import BasePlotter
from tools import physics_constants as const

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

    def plot_detectability_panel(self):
        """Plot detectability for 0.5–2.6 R⊕ and 'none' as a single figure, using pcolormesh for log axes."""

        # --- Grid ---
        L_vals = np.logspace(-2, 1, 1000)  # Stellar Luminosity [L☉]
        D_vals = np.linspace(4, 15, 1000)  # Distance [pc]
        L_grid, D_grid = np.meshgrid(L_vals, D_vals)

        # --- Constants ---
        R_earth_m = const.R_earth if hasattr(const, 'R_earth') else 6.371e6
        AU_m = const.au_to_m if hasattr(const, 'au_to_m') else 1.496e11
        pc_m = 3.086e16
        rad2arcsec = 206265
        A_g = getattr(const, 'A_g_earth', 0.2)
        Phi = getattr(const, 'Phi_alpha', 1.0)
        flux_threshold = getattr(self, 'best_flux_limit', 2.5e-11)
        theta_limit_arcsec = getattr(self, 'theta_limit_rad', 0.0206) * rad2arcsec if hasattr(self, 'theta_limit_rad') else 0.0206

        # --- Derived HZ orbit (a = sqrt(L) * AU) ---
        a_hz_m = np.sqrt(L_grid) * AU_m
        distance_m = D_grid * pc_m
        theta_arcsec = (a_hz_m / distance_m) * rad2arcsec

        # --- Planet flux ratio for 0.5 R_earth and 2.6 R_earth ---
        Rp_small = 0.5 * R_earth_m
        Rp_large = 2.6 * R_earth_m
        flux_ratio_small = A_g * (Rp_small / a_hz_m) ** 2 * Phi
        flux_ratio_large = A_g * (Rp_large / a_hz_m) ** 2 * Phi

        # --- Check detectability for each radius ---
        detect_small = (flux_ratio_small >= flux_threshold) & (theta_arcsec >= theta_limit_arcsec)
        detect_large = (flux_ratio_large >= flux_threshold) & (theta_arcsec >= theta_limit_arcsec)

        # --- Assign region codes ---
        # 1 = detectable (0.5–2.6 R⊕)
        # 0 = not detectable
        region = np.zeros_like(L_grid, dtype=int)
        region[detect_small | detect_large] = 1

        # --- Custom colormap for the 2 categories ---
        from matplotlib.colors import ListedColormap
        cmap = ListedColormap(['#f7cac9', '#88b04b'])  # pink, green

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(8, 6))
        if isinstance(ax, np.ndarray):
            ax = ax.flat[0]
        L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        im = ax.pcolormesh(L_edges, D_edges, region, cmap=cmap, shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_ylim(4, 15)
        ax.set_title('Detectable Radius Range (0.5–2.6 R⊕)\n% of Full Range Detectable')

        # --- Reference lines for M-dwarf and G-star regions ---
        ax.axvline(0.08, color='red', linestyle='--', linewidth=2)
        ax.axvline(0.6, color='gold', linestyle='--', linewidth=2)
        ax.axvline(1.5, color='gold', linestyle='--', linewidth=2)

        # --- Add flux threshold boundary (vertical purple line) ---
        A_g_val = A_g
        Rp_one_earth_m = R_earth_m
        Phi_val = Phi
        flux_threshold_val = flux_threshold
        try:
            AU_m_val = float(AU_m)
        except Exception:
            AU_m_val = 1.496e11
        L_flux_boundary = (A_g_val * Rp_one_earth_m**2 * Phi_val) / (flux_threshold_val * AU_m_val**2)
        ax.axvline(x=L_flux_boundary, color='purple', linestyle=':', linewidth=2,
                   label=f'Flux Limit Boundary (1.0 R⊕)')

        # --- Add angular separation threshold boundary (black dashed curve) ---
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
        L_star_vals = L_vals
        D_theta_boundary = (np.sqrt(L_star_vals) * AU_m_val * rad2arcsec_val) / \
                           (theta_limit_arcsec_val * pc_m_val)
        # Only plot where D_theta_boundary <= 15
        mask = D_theta_boundary <= 15
        ax.plot(L_star_vals[mask], D_theta_boundary[mask], color='black', linestyle='--', linewidth=2,
                label=f'HWO Angular Sep. Limit (1.0 R⊕)')
        # Annotate the equation on the plot with actual values
        eqn = (
            r"$D = \frac{{\sqrt{{L}} \times {}\,\mathrm{{m}} \times {}}}{{{}\,\mathrm{{arcsec}} \times {}\,\mathrm{{m}}}}$"
            .format(
                f'{AU_m_val:.2e}',
                f'{rad2arcsec_val:.1f}',
                f'{theta_limit_arcsec_val:.4f}',
                f'{pc_m_val:.2e}'
            )
        )
        ax.text(0.05, 0.85, eqn, transform=ax.transAxes, fontsize=14, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # --- Legend construction ---
        legend_elements = [
            Patch(facecolor='#f7cac9', label='None'),
            Patch(facecolor='#88b04b', label='0.5–2.6 R⊕'),
            Line2D([0], [0], color='red', linestyle='--', label='M-dwarf Region'),
            Line2D([0], [0], color='gold', linestyle='--', label='G Star Region'),
            plt.Line2D([0], [0], color='purple', linestyle=':', linewidth=2, label='Flux Limit Boundary (1.0 R⊕)'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit (1.0 R⊕)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=14)

        plt.tight_layout()
        self._save_plot(fig, 'detectability_panel')

    def plot_detectability_panel_au_vs_distance(self):
        """Plot detectability as AU vs distance, with two panels: (1) ylim up to 20, (2) ylim up to 14 and xlim is M-dwarf region."""
        # --- Grid ---
        AU_vals = np.linspace(0.01, 1e3, 1000)  # AU
        D_vals = np.linspace(4, 20, 1000)  # Distance [pc] (max 20 for both panels)
        AU_grid, D_grid = np.meshgrid(AU_vals, D_vals)

        # --- Constants ---
        R_earth_m = const.R_earth if hasattr(const, 'R_earth') else 6.371e6
        AU_m = const.au_to_m if hasattr(const, 'au_to_m') else 1.496e11
        pc_m = 3.086e16
        rad2arcsec = 206265
        A_g = getattr(const, 'A_g_earth', 0.2)
        Phi = getattr(const, 'Phi_alpha', 1.0)
        flux_threshold = getattr(self, 'best_flux_limit', 2.5e-11)
        theta_limit_arcsec = getattr(self, 'theta_limit_rad', 0.0206) * rad2arcsec if hasattr(self, 'theta_limit_rad') else 0.0206

        # --- For each AU and D, compute fraction of L in [0.08, 1e3] that is detectable ---
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
                    flux_ratio = A_g * (R_earth_m / a_m) ** 2 * Phi
                    if (flux_ratio >= flux_threshold) and (theta_arcsec >= theta_limit_arcsec):
                        count += 1
                fraction_detectable[idx, jdx] = count / len(L_vals)

        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        # Red (0) -> Yellow (0.5) -> Green (1)
        cmap = LinearSegmentedColormap.from_list('redyellowgreen', ['#f44336', '#fff176', '#4caf50'])

        # --- Plotting: Two panels ---
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        AU_edges = np.linspace(AU_vals[0], AU_vals[-1], AU_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)

        # Panel 1: Full AU, D up to 20
        im0 = axs[0].pcolormesh(AU_edges, D_edges, fraction_detectable, cmap=cmap, shading='auto', vmin=0, vmax=1)
        axs[0].set_xscale('log')
        axs[0].set_xlabel('Semi-major Axis [AU]')
        axs[0].set_ylabel('Distance [pc]')
        axs[0].set_ylim(4, 20)
        axs[0].set_xlim(0.01, 1e3)
        axs[0].set_title('All Stars (up to 20 pc)')
        for L, color in zip([0.08, 0.6, 1.5], ['red', 'gold', 'gold']):
            axs[0].axvline(np.sqrt(L), color=color, linestyle='--', linewidth=2)

        # Panel 2: M-dwarf region, D up to 14
        im1 = axs[1].pcolormesh(AU_edges, D_edges, fraction_detectable, cmap=cmap, shading='auto', vmin=0, vmax=1)
        axs[1].set_xscale('log')
        axs[1].set_xlabel('Semi-major Axis [AU]')
        axs[1].set_ylim(4, 14)
        axs[1].set_xlim(0.01, 0.4)
        axs[1].set_title('M-dwarf Region (up to 14 pc)')
        for L, color in zip([0.08, 0.6, 1.5], ['red', 'gold', 'gold']):
            axs[1].axvline(np.sqrt(L), color=color, linestyle='--', linewidth=2)

        # --- Colorbar ---
        cbar = fig.colorbar(im0, ax=axs, location='right', shrink=0.8, label='Fraction of L detectable (0.08–1e3)')

        # --- Legend ---
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linestyle='--', label='M-dwarf Region (a=√0.08)'),
            Line2D([0], [0], color='gold', linestyle='--', label='G Star Region (a=√0.6, √1.5)')
        ]
        axs[0].legend(handles=legend_elements, loc='lower right', fontsize=14)

        plt.tight_layout()
        self._save_plot(fig, 'detectability_panel_au_vs_distance_twopanel')

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
