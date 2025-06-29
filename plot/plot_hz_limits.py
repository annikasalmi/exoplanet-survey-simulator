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
        self.plot_3x1_panels_with_boundaries()
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

    def _get_flux_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to flux ratio."""
        return df['flux_ratio_value_best'] < self.best_flux_limit

    def _get_iwa_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to IWA."""
        return df['maxangsep'] < self.iwa_limit

    def _get_exozodi_rejection_mask(self, df: pd.DataFrame) -> pd.Series:
        """Get mask for planets rejected due to exozodi."""
        return df['z'] > self.max_z

    def plot_3x1_panels_with_boundaries(self) -> None:
        """Plot 3x1 panels showing specific combinations with semi-major axis on y-axis and improved legend placement."""
        if self.df.empty:
            print("Warning: No data available for 3x1 panel plot")
            return
        
        # Define the specific variable combinations we want to plot (semi-major axis on y-axis)
        var_combinations = [
            ('p_orb', 'semimajor_p'),      # Orbital period vs semi-major axis
            ('temp_s', 'semimajor_p'),     # Stellar temperature vs semi-major axis  
            ('radius_p', 'semimajor_p')    # Planet radius vs semi-major axis
        ]
        
        # Check which variables are available in the dataframe
        available_vars = []
        for x_var, y_var in var_combinations:
            if x_var in self.df.columns and y_var in self.df.columns:
                available_vars.append((x_var, y_var))
        
        if len(available_vars) < 1:
            print(f"Warning: Need at least 1 variable combination, but only found: {available_vars}")
            return
        
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
        
        for x_var, y_var in available_vars:
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
        
        # Create 3x1 subplot grid
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Define boundary configurations
        boundary_configs = [
            ('flux_rejected', 'red', 'Flux Rejected'),
            ('exozodi_rejected', 'gold', 'Exozodi Rejected'),
            ('iwa_rejected', 'blue', 'IWA Rejected'),
            ('detected', 'green', 'Detected')
        ]
        
        # Plot each combination
        for i, (x_var, y_var) in enumerate(available_vars):
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
            x_data = self.df[x_var].dropna()
            if len(x_data) > 0:
                x_min = float(x_data.min())
                x_max = float(x_data.max())
                # Check for reasonable bounds
                if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
                    ax.set_xlim(x_min * 0.9, x_max * 1.1)
            
            y_data = self.df[y_var].dropna()
            if len(y_data) > 0:
                y_min = float(y_data.min())
                y_max = float(y_data.max())
                # Check for reasonable bounds
                if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                    ax.set_ylim(y_min * 0.9, y_max * 1.1)
        
        # Add legend in a more logical place - centered at the bottom
        if len(available_vars) > 0:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.3, label='Flux Rejected'),
                Patch(facecolor='gold', alpha=0.3, label='Exozodi Rejected'),
                Patch(facecolor='blue', alpha=0.3, label='IWA Rejected'),
                Patch(facecolor='green', alpha=1.0, label='Detected')
            ]
            # Place legend in the upper left of the leftmost plot
            axes[0].legend(handles=legend_elements, loc='upper left', 
                          frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        self._save_plot(fig, '3x1_panels_boundaries')
        print("3x1 panel plot completed!")
    
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