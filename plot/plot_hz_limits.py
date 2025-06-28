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
        print("M-dwarf HZ limits plots generated!")

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

        # Extract fill parameters
        facecolor = kwargs.pop('facecolor', None)
        alpha_fill = kwargs.pop('alpha_fill', 0.3)
        hatch = kwargs.pop('hatch', None)
        linewidth = kwargs.get('linewidth', 3)  # Default linewidth

        if alpha_shape:
            shape = alphashape.alphashape(points, alpha)
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
                hull = ConvexHull(points)
                boundary = np.append(hull.vertices, hull.vertices[0])
                if facecolor:
                    ax.fill(points[boundary, 0], points[boundary, 1], 
                           facecolor=facecolor, alpha=alpha_fill, hatch=hatch, **kwargs)
                if linewidth > 0:
                    ax.plot(points[boundary, 0], points[boundary, 1], **kwargs)
        else:
            hull = ConvexHull(points)
            boundary = np.append(hull.vertices, hull.vertices[0])
            if facecolor:
                ax.fill(points[boundary, 0], points[boundary, 1], 
                       facecolor=facecolor, alpha=alpha_fill, hatch=hatch, **kwargs)
            if linewidth > 0:
                ax.plot(points[boundary, 0], points[boundary, 1], **kwargs)

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