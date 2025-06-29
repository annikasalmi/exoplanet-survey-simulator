import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from typing import Optional
from plot.base_plotter import BasePlotter
# For boundary plotting
from scipy.spatial.qhull import ConvexHull
# For smoother, non-convex boundaries
import alphashape
from shapely.geometry import Polygon

from tools import physics_constants as const

class PlotHZLimits(BasePlotter):
    """
    Plotter for M-dwarf HZ limits and related plots.
    
    Features:
    1. IWA vs flux ratio plots with rejection reasons
    2. Temperature vs semi-major axis plots with rejection reasons
    3. Temperature vs semi-major axis boundaries plot
    4. Earth analog detectability plots
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, name: str = 'HWO',
                 nruns: int = 1, star_catalog: str = 'Gaia', **kwargs):
        """Initialize with optional dataframe and parameters."""
        if df is None:
            df = self._create_minimal_dataframe()
        super().__init__(df, nruns, star_catalog, name)
        
        # Cache HWO constants to avoid repeated instantiation
        self.hwo_best = const.HWOConstants('best')
        self.hwo_worst = const.HWOConstants('worst')
        self.best_flux_limit = float(self.hwo_best.min_planet_flux_star_ratio)
        self.iwa_limit = float(self.hwo_best.iwa)
        self.max_z = float(self.hwo_best.max_z)
        self.theta_limit_rad = self.iwa_limit * const.arcsec_to_radians

    def plot_all(self) -> None:
        """Generate all M-dwarf HZ limit plots."""
        if not self._validate_data():
            return
            
        self.plot_temperature_vs_au_with_boundaries()
        self.plot_earth_analog_detectability()
        self.plot_4x4_panels_with_boundaries()
        print("M-dwarf HZ limits plots generated!")

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

    def _validate_required_columns(self, df: pd.DataFrame, required_cols: list) -> bool:
        """Validate that required columns exist in dataframe."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing required columns: {missing_cols}")
            return False
        return True

    def _get_flux_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to flux ratio."""
        return df['flux_ratio_value_best'] < self.best_flux_limit

    def _get_iwa_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to IWA."""
        return df['maxangsep'] < self.iwa_limit

    def _get_exozodi_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to exozodi."""
        return df['z'] > self.max_z

    def _compute_luminosity(self, T_eff: np.ndarray, R_star: np.ndarray) -> np.ndarray:
        """
        Compute stellar luminosity using the Stefan-Boltzmann law.

        Parameters:
        - T_eff : np.ndarray
            Effective temperature of the star(s) in Kelvin.
        - R_star : np.ndarray
            Radius of the star(s) in solar radii.

        Returns:
        - Luminosity in solar luminosities.
        """
        return R_star ** 2 * (T_eff / 5780) ** 4

    def plot_temperature_vs_au_with_boundaries(self) -> None:
        """Plot single plot showing stellar temperature vs planetary semi-major axis with all boundaries overlaid."""
        if self.df.empty:
            print("Warning: No data available for temperature vs semi-major axis plot")
            return
        
        required_cols = ['radius_p', 'temp_s', 'semimajor_p']
        if not self._validate_required_columns(self.df, required_cols):
            print("Warning: Missing required columns for temperature vs semi-major axis plot")
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Get all masks at once to avoid repeated calculations
        detected_mask = self._get_panel_detection_mask(self.df)
        flux_rejected = self._get_flux_rejection_mask(self.df)
        iwa_rejected = self._get_iwa_rejection_mask(self.df)
        exozodi_rejected = self._get_exozodi_rejection_mask(self.df)
        non_detected_mask = ~detected_mask
        
        # Plot boundaries in order (back to front)
        boundary_configs = [
            (flux_rejected, 'red', 'Flux Rejected'),
            (exozodi_rejected, 'gold', 'Exozodi Rejected'),
            (iwa_rejected, 'blue', 'IWA Rejected'),
            (detected_mask, 'green', 'Detected')
        ]
        
        for mask, color, label in boundary_configs:
            if mask.any():
                data = self.df[mask]
                linewidth = 3 if color in ['gray', 'green'] else 0
                alpha_fill = 0.3 if color in ['red', 'gold', 'blue'] else None
                facecolor = color if color in ['red', 'gold', 'blue'] else None
                
                self.plot_boundary(data, 'temp_s', 'semimajor_p', ax, 
                                 alpha=0.01, color=color, linewidth=linewidth, 
                                 linestyle='-', alpha_fill=alpha_fill, 
                                 facecolor=facecolor, label=label)
        
        self._setup_plot_style(ax, 'Stellar Temperature (K)', 'Semi-major Axis (AU)', 
                              'Planet Boundaries by Detection/Rejection Type for HWO')
        ax.grid(True, alpha=0.4)
        ax.set_xlim(3300, 8000)
        ax.set_ylim(0, 1.5)
        
        plt.tight_layout()
        self._save_plot(fig, 'temperature_vs_au_boundaries_overlaid')

    def plot_earth_analog_detectability(self) -> None:
        """Plot detectability of Earth-like planets around different stellar types with detectability percentage gradient."""
        # Grid of stellar luminosity and distance
        L_star_vals = np.logspace(-3, 1, 300)  # Extended to 10 L☉
        distance_vals = np.linspace(1, 50, 300)
        R_planet_vals = np.linspace(0.5, 1.5, 100)

        L_grid, D_grid = np.meshgrid(L_star_vals, distance_vals)
        
        # Vectorized calculation for much better performance
        detectability_percentage = self._calculate_detectability_vectorized(
            L_star_vals, distance_vals, R_planet_vals, L_grid, D_grid
        )

        # Create single plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Plot the gradient background using the detectability percentage
        im = ax.contourf(L_grid, D_grid, detectability_percentage, levels=50, cmap='RdYlGn')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Detectability Across Planet Radii of 0.5-1.5 R⊕)')
        
        # Add contour lines for specific percentage values
        percentage_contours = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        contour_levels = [p for p in percentage_contours if p <= np.max(detectability_percentage)]
        if contour_levels:
            cs = ax.contour(L_grid, D_grid, detectability_percentage, levels=contour_levels, colors='white', alpha=0.7, linewidths=1)
            ax.clabel(cs, inline=True, fontsize=8, fmt='%.0f%%')
        
        ax.set_xscale('log')
        self._setup_plot_style(ax, 'Stellar Luminosity [L☉]', 'Distance [pc]', 
                              'Planet Detectability Percentage\n(0.5-1.5 R⊕ planets, flux ratio & IWA OK)')
        
        # Set plot limits
        ax.set_xlim(0.01, 5.0)  
        ax.set_ylim(1, 20)
        
        # M-dwarf region box (0.001 to 0.08 L☉, 1 to 20 pc)
        mdwarf_box = Rectangle((0.001, 1), 0.079, 19, 
                              linewidth=2, edgecolor='red', facecolor='none', 
                              linestyle='--', label='M-dwarf Region')
        ax.add_patch(mdwarf_box)
        
        # G star region box (0.4 to 1.5 L☉, 1 to 20 pc)
        g_star_box = Rectangle((0.4, 1), 1.1, 19, 
                              linewidth=2, edgecolor='yellow', facecolor='none', 
                              linestyle='--', label='G Star Region')
        ax.add_patch(g_star_box)
        
        # Add legend for the regions
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        self._save_plot(fig, 'earth_analog_detectability_percentage')

    def _calculate_detectability_vectorized(self, L_star_vals, distance_vals, R_planet_vals, L_grid, D_grid):
        """Vectorized calculation of detectability percentage for much better performance."""
        # Pre-calculate constants
        a_hz_vals = np.sqrt(L_star_vals) * const.au_to_m
        distance_m_vals = distance_vals * 3.086e16
        
        # Create 3D arrays for vectorized computation
        L_3d, D_3d, R_3d = np.meshgrid(L_star_vals, distance_vals, R_planet_vals, indexing='ij')
        
        # Vectorized calculations
        a_hz_3d = np.sqrt(L_3d) * const.au_to_m
        distance_m_3d = D_3d * 3.086e16
        theta_3d = a_hz_3d / distance_m_3d
        
        # Vectorized contrast calculation
        contrast_3d = const.A_g_earth * (R_3d * const.R_earth / a_hz_3d) ** 2 * const.Phi_alpha
        
        # Vectorized detectability check
        detectable_3d = (contrast_3d >= self.best_flux_limit) & (theta_3d >= self.theta_limit_rad)
        
        # Sum along planet radius axis to get percentage
        detectability_percentage = np.mean(detectable_3d, axis=2)
        
        return detectability_percentage.T  # Transpose to match original shape

    def plot_4x4_panels_with_boundaries(self) -> None:
        """Plot 4x4 panels showing all combinations of semimajor_p, p_orb, radius_p, temp, temp_s, mass_s, flux_p with boundaries."""
        if self.df.empty:
            print("Warning: No data available for 4x4 panel plot")
            return
        
        # Define all the variables we want to plot
        variables = ['semimajor_p', 'p_orb', 'radius_p', 'temp', 'temp_s', 'mass_s', 'flux_p']
        
        # Check which variables are available in the dataframe
        available_vars = [var for var in variables if var in self.df.columns]
        if len(available_vars) < 2:
            print(f"Warning: Need at least 2 variables, but only found: {available_vars}")
            return
        
        print("Pre-computing boundaries for speed optimization...")
        
        # Pre-compute all masks and data subsets once
        detected_mask = self._get_panel_detection_mask(self.df)
        flux_rejected = self._get_flux_rejection_mask(self.df)
        iwa_rejected = self._get_iwa_rejection_mask(self.df)
        exozodi_rejected = self._get_exozodi_rejection_mask(self.df)
        
        # Pre-filter data for each category to avoid repeated filtering
        data_subsets = {}
        for name, mask in [('detected', detected_mask), ('flux_rejected', flux_rejected), 
                          ('iwa_rejected', iwa_rejected), ('exozodi_rejected', exozodi_rejected)]:
            if mask.any():
                data_subsets[name] = self.df[mask]
        
        # Pre-compute boundaries for all variable combinations and categories
        boundary_cache = {}
        from itertools import combinations
        var_combinations = list(combinations(available_vars, 2))
        
        for x_var, y_var in var_combinations[:16]:  # Only first 16 combinations
            for category, data in data_subsets.items():
                # Skip if not enough data points
                valid_data = data[[x_var, y_var]].dropna()
                if len(valid_data) < 3:
                    continue
                
                points = valid_data.values
                
                # Use the same validation and filtering as plot_boundary
                filtered_points = self._validate_and_filter_points(points, x_var, y_var)
                if filtered_points is None:
                    continue
                
                try:
                    # Use convex hull instead of alpha shape for speed
                    hull = ConvexHull(filtered_points)
                    boundary = np.append(hull.vertices, hull.vertices[0])
                    boundary_points = filtered_points[boundary]
                    
                    key = (x_var, y_var, category)
                    boundary_cache[key] = boundary_points
                except Exception as e:
                    # Skip if convex hull fails
                    print(f"Warning: Could not compute boundary for {x_var} vs {y_var} ({category}): {str(e)}")
                    continue
        
        print(f"Computed {len(boundary_cache)} boundaries, creating plot...")
        
        # Create 4x4 subplot grid
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        axes = axes.flatten()
        
        # Define boundary configurations
        boundary_configs = [
            ('flux_rejected', 'red', 'Flux Rejected'),
            ('exozodi_rejected', 'gold', 'Exozodi Rejected'),
            ('iwa_rejected', 'blue', 'IWA Rejected'),
            ('detected', 'green', 'Detected')
        ]
        
        # Plot each combination
        for i, (x_var, y_var) in enumerate(var_combinations):
            if i >= 16:  # Only plot first 16 combinations (4x4 grid)
                break
                
            ax = axes[i]
            
            # Plot boundaries for this panel using cached results
            for category, color, label in boundary_configs:
                key = (x_var, y_var, category)
                if key in boundary_cache:
                    boundary_points = boundary_cache[key]
                    
                    # Determine plotting parameters
                    linewidth = 3 if color in ['gray', 'green'] else 0
                    alpha_fill = 0.3 if color in ['red', 'gold', 'blue'] else None
                    facecolor = color if color in ['red', 'gold', 'blue'] else None
                    
                    # Plot the boundary
                    if facecolor:
                        ax.fill(boundary_points[:, 0], boundary_points[:, 1], 
                               facecolor=facecolor, alpha=alpha_fill, 
                               color=color, linewidth=linewidth, label=label)
                    elif linewidth > 0:
                        ax.plot(boundary_points[:, 0], boundary_points[:, 1], 
                               color=color, linewidth=linewidth, label=label)
            
            # Set labels and title
            ax.set_xlabel(self._get_axis_label(x_var))
            ax.set_ylabel(self._get_axis_label(y_var))
            ax.set_title(f'{self._get_axis_label(x_var)} vs {self._get_axis_label(y_var)}')
            ax.grid(True, alpha=0.4)
            
            # Set reasonable axis limits based on data
            if not self.df[x_var].isna().all():
                x_min = self.df[x_var].min()
                x_max = self.df[x_var].max()
                # Check for reasonable bounds
                if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                    ax.set_xlim(x_min * 0.9, x_max * 1.1)
            if not self.df[y_var].isna().all():
                y_min = self.df[y_var].min()
                y_max = self.df[y_var].max()
                # Check for reasonable bounds
                if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                    ax.set_ylim(y_min * 0.9, y_max * 1.1)
        
        # Hide unused subplots
        for i in range(len(var_combinations), 16):
            axes[i].set_visible(False)
        
        # Add legend to the first subplot
        if len(var_combinations) > 0 and len(var_combinations) <= 16:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        self._save_plot(fig, '4x4_panels_boundaries')
        print("4x4 panel plot completed!")
    
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