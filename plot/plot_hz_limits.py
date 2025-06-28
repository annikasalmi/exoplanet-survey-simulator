import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from plot.base_plotter import BasePlotter
from tools import physics_constants as const

class PlotHZLimits(BasePlotter):
    """
    Plotter for:
    1. Flux ratio vs. distance for a habitable M-dwarf exoplanet, with HWO flux ratio limits.
    2. IWA vs. distance, with HWO IWA limits and HZ overlay.
    """
    
    def __init__(self, df=None, name='HWO', nruns=1, star_catalog='Gaia', **kwargs):
        """Initialize with optional dataframe and parameters."""
        # Create a minimal dataframe if none provided
        if df is None:
            df = self._create_minimal_dataframe()
        
        super().__init__(df, nruns, star_catalog, name)

    def _create_minimal_dataframe(self):
        """Create a minimal dataframe for the plotter to work with."""
        import pandas as pd
        return pd.DataFrame({'dummy': [1]})  # Minimal dataframe

    def plot_all(self):
        """Generate all M-dwarf HZ limit plots using dataframe distances."""
        # Use distances from dataframe if available, otherwise use default range
        if 'distance' in self.df.columns:
            distances_pc = self.df['distance'].values
            # Filter to reasonable range and remove duplicates
            mask = (distances_pc >= 2) & (distances_pc <= 20)
            distances_pc = np.unique(distances_pc[mask])
            if len(distances_pc) == 0:
                distances_pc = np.linspace(2, 20, 200)
        else:
            distances_pc = np.linspace(2, 20, 200)
        
        self.plot_mdwarf_hz_limits(distances_pc)
        self.plot_iwa_vs_a_by_star_type()
        self.plot_radius_vs_flux_by_star_type()
        self.plot_temperature_vs_distance()
        print(f"M-dwarf HZ limits plots generated using {len(distances_pc)} distance points!")

    def plot_mdwarf_hz_limits(self, distances_pc: Optional[np.ndarray] = None) -> None:
        """Plot M-dwarf HZ limits: flux ratio vs distance and IWA vs distance."""
        if distances_pc is None:
            distances_pc = np.linspace(2, 20, 200)
        
        distances_m = distances_pc * const.pc_to_m
        params = self._setup_mdwarf_parameters()
        
        # Calculate flux ratios
        F_p = self._planck_function(params['lambda_obs'], params['T_planet'])
        F_s = self._planck_function(params['lambda_obs'], params['T_star'])
        flux_ratio_surface = (params['R_planet']**2 * F_p) / (params['R_star']**2 * F_s)
        flux_ratio = flux_ratio_surface * np.ones_like(distances_pc)
        
        # Get HWO limits for both best and worst cases
        hwo_best = const.HWOConstants('best')
        hwo_worst = const.HWOConstants('worst')
        IWA_best_rad = hwo_best.iwa  # in radians
        IWA_worst_rad = hwo_worst.iwa  # in radians
        IWA_best_AU = IWA_best_rad * distances_m / const.au_to_m
        IWA_worst_AU = IWA_worst_rad * distances_m / const.au_to_m
    
        self._plot_iwa_vs_a(distances_pc, IWA_best_AU, IWA_worst_AU, params['hz_au'])

    def _plot_iwa_vs_a(self, distances_pc: np.ndarray, IWA_best_AU: np.ndarray, 
                              IWA_worst_AU: np.ndarray, hz_au: float) -> None:
        """Plot IWA vs distance and HZ location for M-dwarf."""
        fig, mpl_ax = plt.subplots(figsize=(7, 5))
        
        # Convert distances from pc to AU for x-axis
        distances_au = distances_pc * const.pc_to_m / const.au_to_m
        
        # Add scatter plots of dataframe habitable zone data if available
        if not self.df.empty and 'hz_in' in self.df.columns and 'hz_out' in self.df.columns and 'maxangsep' in self.df.columns and 'semimajor_p' in self.df.columns:
            # Get detection mask
            mask_best, _ = self._get_detection_masks()
            
            # Create masks for habitable zone planets
            hz_mask = (self.df['semimajor_p'] >= self.df['hz_in']) & (self.df['semimajor_p'] <= self.df['hz_out'])
            non_hz_mask = ~hz_mask
            
            # Plot undetected planets in gray (all planets)
            mpl_ax.scatter(self.df['semimajor_p'], self.df['maxangsep'], 
                          color='gray', alpha=0.3, s=10, label='All planets')
            
            # Plot detected habitable zone planets in blue
            detected_hz_mask = hz_mask & mask_best
            if detected_hz_mask.any():
                mpl_ax.scatter(self.df.loc[detected_hz_mask, 'semimajor_p'], self.df.loc[detected_hz_mask, 'maxangsep'], 
                              color='blue', alpha=0.8, s=20, label='Detected planets in HZ')
            
            # Plot detected non-habitable zone planets in yellow
            detected_non_hz_mask = non_hz_mask & mask_best
            if detected_non_hz_mask.any():
                mpl_ax.scatter(self.df.loc[detected_non_hz_mask, 'semimajor_p'], self.df.loc[detected_non_hz_mask, 'maxangsep'], 
                              color='yellow', alpha=0.8, s=20, label='Detected planets outside HZ')
            
            # Add HWO IWA limits as horizontal lines
            hwo_best = const.HWOConstants('best')
            hwo_worst = const.HWOConstants('worst')
            iwa_best_arcsec = hwo_best.iwa  # IWA is already in arcseconds
            iwa_worst_arcsec = hwo_worst.iwa  # IWA is already in arcseconds
            
            # Plot IWA lines across the full x-axis range
            mpl_ax.axhline(y=iwa_best_arcsec, color='green', linestyle='--', alpha=0.7, linewidth=2, label='HWO Best IWA')
            mpl_ax.axhline(y=iwa_worst_arcsec, color='red', linestyle='--', alpha=0.7, linewidth=2, label='HWO Worst IWA')
            
            # Set plot limits based on habitable zone data
            hz_min = self.df['hz_in'].min()
            hz_max = self.df['hz_out'].max()
            angsep_min = self.df['maxangsep'].min()
            angsep_max = self.df['maxangsep'].max()
            
            # Add some padding to the limits
            hz_padding = (hz_max - hz_min) * 0.1
            angsep_padding = (angsep_max - angsep_min) * 0.1
            
            # Limit x-axis to 1.5 AU for detected planets across all stellar types
            x_limit = min(1.5, hz_max + hz_padding)
            
            # Ensure y-axis limits include the IWA values
            y_min = min(angsep_min - angsep_padding, iwa_best_arcsec, iwa_worst_arcsec)
            y_max = max(angsep_max + angsep_padding, iwa_best_arcsec, iwa_worst_arcsec)
            
            mpl_ax.set_xlim(hz_min - hz_padding, x_limit)
            mpl_ax.set_ylim(y_min, y_max)
        else:
            # Fallback to original lines if no habitable zone data
            mpl_ax.plot(distances_au, IWA_best_AU, label='HWO Best IWA (AU)', color='green')
            mpl_ax.plot(distances_au, IWA_worst_AU, label='HWO Worst IWA (AU)', color='red')
            mpl_ax.axhline(y=hz_au, color='blue', linestyle=':', label='Habitable Zone (HZ)')
        
        mpl_ax.set_xlabel('Planet semi-major axis (AU)')
        mpl_ax.set_ylabel('Maximum Angular Separation (arcsec)')
        mpl_ax.set_title('AU vs. Angular Separation for all planets')
        mpl_ax.grid(True)
        mpl_ax.legend()
        
        self._save_plot(fig, 'iwa_vs_a')

    def plot_iwa_vs_a_by_star_type(self):
        """Plot AU vs Angular Separation with panels for different star types."""
        if self.df.empty or 'stype' not in self.df.columns:
            print("Warning: No star type data available for panel plot")
            return
        
        # Create masks for different star types
        m_stars_mask = self.df['stype'] == 'M'
        gk_stars_mask = self.df['stype'].isin(['G', 'K'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot M-type stars
        self._plot_panel(ax1, self.df[m_stars_mask], 'M-type Stars')
        
        # Plot G/K-type stars
        self._plot_panel(ax2, self.df[gk_stars_mask], 'G/K-type Stars')
        
        plt.tight_layout()
        self._save_plot(fig, 'iwa_vs_a_by_star_type')

    def _plot_panel(self, ax, df_panel, title):
        """Plot a single panel for the given star type data."""
        if df_panel.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Check if required columns exist
        required_cols = ['hz_in', 'hz_out', 'maxangsep', 'semimajor_p']
        if not all(col in df_panel.columns for col in required_cols):
            ax.text(0.5, 0.5, 'Missing required data columns', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Get detection mask for this panel
        mask_best, _ = self._get_detection_masks()
        panel_detection_mask = mask_best[df_panel.index]
        
        # Create masks for habitable zone planets
        hz_mask = (df_panel['semimajor_p'] >= df_panel['hz_in']) & (df_panel['semimajor_p'] <= df_panel['hz_out'])
        non_hz_mask = ~hz_mask
        
        # Plot all planets in gray (background)
        ax.scatter(df_panel['semimajor_p'], df_panel['maxangsep'], 
                  color='gray', alpha=0.3, s=10, label='All planets')
        
        # Plot detected habitable zone planets in blue
        detected_hz_mask = hz_mask & panel_detection_mask
        if detected_hz_mask.any():
            ax.scatter(df_panel.loc[detected_hz_mask, 'semimajor_p'], df_panel.loc[detected_hz_mask, 'maxangsep'], 
                      color='blue', alpha=0.8, s=20, label='Detected planets in HZ')
        
        # Plot detected non-habitable zone planets in yellow
        detected_non_hz_mask = non_hz_mask & panel_detection_mask
        if detected_non_hz_mask.any():
            ax.scatter(df_panel.loc[detected_non_hz_mask, 'semimajor_p'], df_panel.loc[detected_non_hz_mask, 'maxangsep'], 
                      color='yellow', alpha=0.8, s=20, label='Detected planets outside HZ')
        
        # Add HWO IWA limits
        hwo_best = const.HWOConstants('best')
        hwo_worst = const.HWOConstants('worst')
        iwa_best_arcsec = hwo_best.iwa  # IWA is already in arcseconds
        iwa_worst_arcsec = hwo_worst.iwa  # IWA is already in arcseconds
        
        # Get the x-axis range for plotting IWA lines
        x_min = df_panel['semimajor_p'].min()
        x_max = df_panel['semimajor_p'].max()
        
        # Plot IWA lines across the full x-axis range
        ax.axhline(y=iwa_best_arcsec, color='green', linestyle='--', alpha=0.7, linewidth=2, label='HWO Best IWA')
        ax.axhline(y=iwa_worst_arcsec, color='red', linestyle='--', alpha=0.7, linewidth=2, label='HWO Worst IWA')
        
        # Set plot limits based on panel data
        if not df_panel.empty:
            hz_min = df_panel['hz_in'].min()
            hz_max = df_panel['hz_out'].max()
            angsep_min = df_panel['maxangsep'].min()
            angsep_max = df_panel['maxangsep'].max()
            
            # Add some padding to the limits
            hz_padding = (hz_max - hz_min) * 0.1
            angsep_padding = (angsep_max - angsep_min) * 0.1
            
            # Limit x-axis to 1.5 AU for G/K type stars and detected planets
            if 'G/K' in title or 'G/K-type' in title:
                x_limit = 1.5
            else:
                x_limit = hz_max + hz_padding
            
            # Ensure y-axis limits include the IWA values
            y_min = min(angsep_min - angsep_padding, iwa_best_arcsec, iwa_worst_arcsec)
            y_max = max(angsep_max + angsep_padding, iwa_best_arcsec, iwa_worst_arcsec)
            
            ax.set_xlim(hz_min - hz_padding, x_limit)
            ax.set_ylim(y_min, y_max)
        
        ax.set_xlabel('Planet semi-major axis (AU)')
        ax.set_ylabel('Maximum Angular Separation (arcsec)')
        ax.set_title(f'{title} - AU vs. Angular Separation')
        ax.grid(True)
        ax.legend()

    def plot_radius_vs_flux_by_star_type(self):
        """Plot Planet Radius vs Flux Ratio with panels for different star types."""
        if self.df.empty or 'stype' not in self.df.columns:
            print("Warning: No star type data available for radius vs flux plot")
            return
        
        # Create masks for different star types
        m_stars_mask = self.df['stype'] == 'M'
        gk_stars_mask = self.df['stype'].isin(['G', 'K'])
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot M-type stars
        self._plot_radius_flux_panel(ax1, self.df[m_stars_mask], 'M-type Stars')
        
        # Plot G/K-type stars
        self._plot_radius_flux_panel(ax2, self.df[gk_stars_mask], 'G/K-type Stars')
        
        plt.tight_layout()
        self._save_plot(fig, 'radius_vs_flux_by_star_type')

    def _plot_radius_flux_panel(self, ax, df_panel, title):
        """Plot a single panel for radius vs flux ratio."""
        if df_panel.empty:
            ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Check if required columns exist
        required_cols = ['radius_p', 'flux_ratio_value_best']
        if not all(col in df_panel.columns for col in required_cols):
            ax.text(0.5, 0.5, 'Missing required data columns', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return
        
        # Get detection mask for this panel
        mask_best, _ = self._get_detection_masks()
        panel_detection_mask = mask_best[df_panel.index]
        
        # Create masks for detected and non-detected planets
        detected_mask = panel_detection_mask
        non_detected_mask = ~panel_detection_mask
        
        # Plot non-detected planets in gray (background)
        if non_detected_mask.any():
            ax.scatter(df_panel.loc[non_detected_mask, 'radius_p'], 
                      df_panel.loc[non_detected_mask, 'flux_ratio_value_best'], 
                      color='gray', alpha=0.3, s=10, label='Non-detected planets')
        
        # For non-detected planets, determine the rejection method based on individual pass/fail columns
        if non_detected_mask.any():
            # Debug: print available columns
            print(f"Available columns in panel: {list(df_panel.columns)}")
            print(f"Non-detected planets: {non_detected_mask.sum()}")
            
            # Check for flux ratio rejection (red, alpha=0.5)
            if 'flux_pass_best' in df_panel.columns:
                print(f"flux_pass_best data type: {df_panel['flux_pass_best'].dtype}")
                print(f"flux_pass_best unique values: {df_panel['flux_pass_best'].unique()}")
                flux_rejected = non_detected_mask & ~df_panel['flux_pass_best'].astype(bool)
                print(f"Flux rejected planets: {flux_rejected.sum()}")
                if flux_rejected.any():
                    ax.scatter(df_panel.loc[flux_rejected, 'radius_p'], 
                              df_panel.loc[flux_rejected, 'flux_ratio_value_best'], 
                              color='red', alpha=0.5, s=20, marker='o', 
                              label='Flux ratio rejected')
            else:
                print("flux_pass_best column not found!")
            
            # Check for IWA rejection (yellow, alpha=0.8)
            if 'iwa_pass_best' in df_panel.columns:
                print(f"iwa_pass_best data type: {df_panel['iwa_pass_best'].dtype}")
                print(f"iwa_pass_best unique values: {df_panel['iwa_pass_best'].unique()}")
                iwa_rejected = non_detected_mask & ~df_panel['iwa_pass_best'].astype(bool)
                print(f"IWA rejected planets: {iwa_rejected.sum()}")
                if iwa_rejected.any():
                    ax.scatter(df_panel.loc[iwa_rejected, 'radius_p'], 
                              df_panel.loc[iwa_rejected, 'flux_ratio_value_best'], 
                              color='yellow', alpha=0.8, s=20, marker='o', 
                              label='IWA rejected')
            else:
                print("iwa_pass_best column not found!")
            
            # Check for exozodi rejection (blue, alpha=0.5)
            if 'z_pass_best' in df_panel.columns:
                print(f"z_pass_best data type: {df_panel['z_pass_best'].dtype}")
                print(f"z_pass_best unique values: {df_panel['z_pass_best'].unique()}")
                exozodi_rejected = non_detected_mask & ~df_panel['z_pass_best'].astype(bool)
                print(f"Exozodi rejected planets: {exozodi_rejected.sum()}")
                if exozodi_rejected.any():
                    ax.scatter(df_panel.loc[exozodi_rejected, 'radius_p'], 
                              df_panel.loc[exozodi_rejected, 'flux_ratio_value_best'], 
                              color='blue', alpha=0.5, s=20, marker='o', 
                              label='Exozodi rejected')
            else:
                print("z_pass_best column not found!")
        
        # Create masks for habitable zone planets
        hz_mask = (df_panel['semimajor_p'] >= df_panel['hz_in']) & (df_panel['semimajor_p'] <= df_panel['hz_out'])
        non_hz_mask = ~hz_mask
        
        # Plot detected habitable zone planets in green
        detected_hz_mask = hz_mask & detected_mask
        if detected_hz_mask.any():
            ax.scatter(df_panel.loc[detected_hz_mask, 'radius_p'], 
                      df_panel.loc[detected_hz_mask, 'flux_ratio_value_best'], 
                      color='green', alpha=0.8, s=20, label='Detected planets in HZ')
        
        # Plot detected non-habitable zone planets in green (same color for all detected)
        detected_non_hz_mask = non_hz_mask & detected_mask
        if detected_non_hz_mask.any():
            ax.scatter(df_panel.loc[detected_non_hz_mask, 'radius_p'], 
                      df_panel.loc[detected_non_hz_mask, 'flux_ratio_value_best'], 
                      color='green', alpha=0.8, s=20, marker='o', label='Detected planets outside HZ')
        
        
        # Add HWO flux ratio limits
        hwo_best = const.HWOConstants('best')
        hwo_worst = const.HWOConstants('worst')
        best_flux_limit = float(hwo_best.min_planet_flux_star_ratio)
        worst_flux_limit = float(hwo_worst.min_planet_flux_star_ratio)
        
        ax.axhline(y=best_flux_limit, color='green', linestyle='--', alpha=0.7, label='HWO Best Flux Limit')
        ax.axhline(y=worst_flux_limit, color='red', linestyle='--', alpha=0.7, label='HWO Worst Flux Limit')
        
        # Set plot limits based on panel data
        if not df_panel.empty:
            radius_min = df_panel['radius_p'].min()
            radius_max = df_panel['radius_p'].max()
            
            # Add some padding to the radius limits
            radius_padding = (radius_max - radius_min) * 0.1
            
            ax.set_xlim(radius_min - radius_padding, radius_max + radius_padding)
            # Let matplotlib automatically set y-axis limits for log scale
        
        ax.set_xlabel('Planet Radius (R_earth)')
        ax.set_ylabel('Flux Ratio (planet/star)')
        ax.set_title(f'{title} - Radius vs. Flux Ratio')
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()

    def plot_temperature_vs_distance(self):
        """Plot stellar temperature vs distance with planets colored by rejection method."""
        if self.df.empty:
            print("Warning: No data available for temperature vs distance plot")
            return
        
        # Check for required columns
        required_cols = ['temp_s', 'distance_s', 'detected_best']
        if not all(col in self.df.columns for col in required_cols):
            print("Warning: Missing required columns for temperature vs distance plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get detection mask using the proper detection logic
        mask_best, _ = self._get_detection_masks()
        
        # Separate detected and non-detected planets
        detected_mask = mask_best
        non_detected_mask = ~mask_best
        
        # Plot non-detected planets in gray
        if non_detected_mask.any():
            ax.scatter(self.df.loc[non_detected_mask, 'temp_s'], 
                      self.df.loc[non_detected_mask, 'distance_s'],
                      color='gray', alpha=0.6, s=30, marker='o', 
                      label='Non-detected planets')
        
        # For non-detected planets, determine the rejection method based on individual pass/fail columns
        if non_detected_mask.any():
            # Check for flux ratio rejection (red, alpha=0.5)
            if 'flux_pass_best' in self.df.columns:
                flux_rejected = non_detected_mask & ~self.df['flux_pass_best']
                if flux_rejected.any():
                    ax.scatter(self.df.loc[flux_rejected, 'temp_s'], 
                              self.df.loc[flux_rejected, 'distance_s'],
                              color='red', alpha=0.5, s=30, marker='o', 
                              label='Flux ratio rejected')
            
            # Check for IWA rejection (yellow, alpha=0.8)
            if 'iwa_pass_best' in self.df.columns:
                iwa_rejected = non_detected_mask & ~self.df['iwa_pass_best']
                if iwa_rejected.any():
                    ax.scatter(self.df.loc[iwa_rejected, 'temp_s'], 
                              self.df.loc[iwa_rejected, 'distance_s'],
                              color='yellow', alpha=0.8, s=30, marker='o', 
                              label='IWA rejected')
            
            # Check for exozodi rejection (blue, alpha=0.5)
            if 'z_pass_best' in self.df.columns:
                exozodi_rejected = non_detected_mask & ~self.df['z_pass_best']
                if exozodi_rejected.any():
                    ax.scatter(self.df.loc[exozodi_rejected, 'temp_s'], 
                              self.df.loc[exozodi_rejected, 'distance_s'],
                              color='blue', alpha=0.5, s=30, marker='o', 
                              label='Exozodi rejected')
        
        # Plot detected planets (those that passed all tests) in green
        if detected_mask.any():
            ax.scatter(self.df.loc[detected_mask, 'temp_s'], 
                      self.df.loc[detected_mask, 'distance_s'],
                      color='green', alpha=0.8, s=30, 
                      label='Detected planets')
        
        ax.set_xlabel('Stellar Temperature (K)')
        ax.set_ylabel('Distance from Star (pc)')
        ax.set_title(f'Stellar Temperature vs Distance for {self.name} ({self.nruns} runs)\nStar Catalog: {self.star_catalog}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable axis limits
        if not self.df.empty:
            temp_min = self.df['temp_s'].min()
            temp_max = self.df['temp_s'].max()
            dist_min = self.df['distance_s'].min()
            dist_max = self.df['distance_s'].max()
            
            # Add some padding
            temp_padding = (temp_max - temp_min) * 0.05
            dist_padding = (dist_max - dist_min) * 0.05
            
            ax.set_xlim(temp_min - temp_padding, temp_max + temp_padding)
            ax.set_ylim(dist_min - dist_padding, dist_max + dist_padding)
        
        self._save_plot(fig, 'temperature_vs_distance')

    def _get_detection_masks(self):
        """Get detection masks for best and worst cases."""
        # Use the detected_best column which already includes all conditions from hwo_data.py
        # This includes: iwa_condition & flux_condition & min_photons_rate_condition & z_condition
        if 'detected_best' in self.df.columns:
            mask_best = self.df['detected_best'].values
        else:
            mask_best = np.zeros(len(self.df), dtype=bool)
            
        if 'detected_worst' in self.df.columns:
            mask_worst = self.df['detected_worst'].values
        else:
            mask_worst = np.zeros(len(self.df), dtype=bool)
            
        return mask_best, mask_worst