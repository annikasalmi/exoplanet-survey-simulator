import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tools import physics_constants as const
from tools.plotting_constants import DETECTION_COLORS

class SimpleHZPlotter:
    """Simplified habitable zone limits plotter."""
    
    def __init__(self, xlim_min=0.01, xlim_max=15.0, ylim_min=1.0, ylim_max=35.0):
        self.xlim_min = xlim_min
        self.xlim_max = xlim_max
        self.ylim_min = ylim_min
        self.ylim_max = ylim_max
        
        # HWO parameters
        self.hwo = const.HWOConstants('best')
        self.flux_limit = float(self.hwo.min_planet_flux_star_ratio)
        self.iwa_limit = float(self.hwo.iwa)
        self.theta_limit_rad = self.iwa_limit * const.arcsec_to_radians

    def plot_luminosity_distance(self):
        """Plot detectability for habitable planets."""
        # Create grid
        L_vals = np.logspace(np.log10(self.xlim_min), np.log10(self.xlim_max), 100)
        D_vals = np.linspace(self.ylim_min, self.ylim_max, 100)
        L_grid, D_grid = np.meshgrid(L_vals, D_vals)
        
        # Calculate habitable zone distances (basic approximation)
        a_hz_m = np.sqrt(L_grid) * const.au_to_m
        distance_m = D_grid * const.pc_to_m
        theta_arcsec = (a_hz_m / distance_m) * const.rad_to_arcsec
        
        # Calculate flux ratios
        Rp_small = const.R_earth_min_habitable * const.R_earth
        Rp_large = const.R_earth_max_habitable * const.R_earth
        T_star = (L_grid * const.temp_sun**4)**0.25
        T_planet = const.T_earth
        
        flux_ratio_small = (T_planet * Rp_small**2) / (T_star * const.R_sun**2) / (distance_m / const.pc_to_m) ** 2
        flux_ratio_large = (T_planet * Rp_large**2) / (T_star * const.R_sun**2) / (distance_m / const.pc_to_m) ** 2
        
        # Determine detectable regions
        detect_small = (flux_ratio_small >= self.flux_limit) & (theta_arcsec >= self.theta_limit_rad * const.rad_to_arcsec)
        detect_large = (flux_ratio_large >= self.flux_limit) & (theta_arcsec >= self.theta_limit_rad * const.rad_to_arcsec)
        region = np.zeros_like(L_grid, dtype=int)
        region[detect_small | detect_large] = 1

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add pink fill for too faint region
        ax.fill_betweenx([0, self.ylim_max], self.xlim_min, 0.1, 
                        color='pink', alpha=0.3, label='Too faint to detect')
        
        # Setup detectability plot
        L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
        D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)
        
        im = ax.pcolormesh(L_edges, D_edges, region, 
                           cmap=plt.cm.RdYlGn, shading='auto', vmin=0, vmax=1)
        ax.set_xscale('log')
        ax.set_xlabel('Stellar Luminosity [L☉]')
        ax.set_ylabel('Distance [pc]')
        ax.set_title(f'Detectable Radius Range ({const.R_earth_min_habitable}–{const.R_earth_max_habitable} R⊕)')
        ax.set_ylim(self.ylim_min, self.ylim_max)
        ax.set_xlim(self.xlim_min, 2.0)
        
        # Add angular separation threshold boundary
        D_theta_boundary = (np.sqrt(L_vals) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        mask = D_theta_boundary <= self.ylim_max
        ax.plot(L_vals[mask], D_theta_boundary[mask], color='black', linestyle='--', linewidth=2, 
               label='HWO Angular Sep. Limit')

        # Add reference lines
        for L, color in zip([const.L_m_dwarf_max, const.L_g_star_min, const.L_g_star_max], ['red', 'gold', 'gold']):
            ax.axvline(L, color=color, linestyle='--', linewidth=2)

        # Add M dwarfs observable region
        L_m_dwarf_range = np.linspace(0, const.L_m_dwarf_max, 100)
        D_theta_boundary_mdwarf = (np.sqrt(L_m_dwarf_range) * const.au_to_m * const.rad_to_arcsec) / (self.theta_limit_rad * const.rad_to_arcsec * const.pc_to_m)
        mask = (D_theta_boundary_mdwarf <= self.ylim_max) & (D_theta_boundary_mdwarf >= 0)
        if np.any(mask):
            ax.fill_between(L_m_dwarf_range[mask], 0, D_theta_boundary_mdwarf[mask], 
                           color='darkgreen', alpha=1, label='M dwarfs observable by HWO')

        # Add specific planet labels
        specific_planets = {
            'Proxima Cen b': (0.0016, 1.3),
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
            Line2D([0], [0], color='red', linestyle='--', label=f'M dwarf Region'),
            Line2D([0], [0], color='gold', linestyle='--', label=f'G Star Region'),
            Line2D([0], [0], color='black', linestyle='--', linewidth=2, label='HWO Angular Sep. Limit'),
            Patch(facecolor='darkgreen', alpha=1, label='M dwarfs observable by HWO'),
            Patch(facecolor='pink', alpha=0.3, label='Too faint to detect')
        ]
        
        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)
        plt.tight_layout()
        plt.savefig('distance_luminosity_simple.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

if __name__ == "__main__":
    plotter = SimpleHZPlotter()
    plotter.plot_luminosity_distance()
    print("Plot saved as distance_luminosity_simple.png") 