import numpy as np
import matplotlib.pyplot as plt
from tools import physics_constants as const

class PlotMdwarfHZLimits:
    """
    Plotter for:
    1. Flux ratio vs. distance for a habitable M-dwarf exoplanet, with HWO flux ratio limits.
    2. IWA vs. distance, with HWO IWA limits and HZ overlay.
    """
    def __init__(self, df=None, name='HWO', nruns=1, star_catalog='Gaia', **kwargs):
        self.df = df
        self.name = name
        self.nruns = nruns
        self.star_catalog = star_catalog
        # M-dwarf and planet parameters
        self.T_star = 3200  # K (typical M-dwarf)
        self.R_star = 0.2 * const.R_sun  # 0.2 solar radii
        self.T_planet = 288  # K (Earth-like)
        self.R_planet = const.R_earth
        # Wavelength for flux ratio (mid-IR, e.g., 10 micron)
        self.lambda_obs = 10e-6  # m
        # Habitable zone distance (approx, in AU)
        self.hz_au = 0.1  # for M-dwarf, can adjust as needed

        # Distance array (in pc)
        self.distances_pc = np.linspace(2, 20, 200)
        self.distances_m = self.distances_pc * const.pc_to_m

    def planck(self, wavelength, T):
        return (2 * const.h * const.c**2) / (wavelength**5 * (np.exp((const.h * const.c) / (wavelength * const.k * T)) - 1))

    def plot_all(self):
        # --- 1. Flux ratio vs. distance ---
        F_p = self.planck(self.lambda_obs, self.T_planet)
        F_s = self.planck(self.lambda_obs, self.T_star)
        flux_ratio_surface = (self.R_planet**2 * F_p) / (self.R_star**2 * F_s)
        flux_ratio = flux_ratio_surface * np.ones_like(self.distances_pc)

        # HWO flux ratio limits
        hwo_best = const.HWOConstants('best')
        best_flux_limit = hwo_best.min_planet_flux_star_ratio

        plt.figure(figsize=(7,5))
        plt.plot(self.distances_pc, flux_ratio, label='Flux Ratio (planet/star) at 10 μm')
        plt.axhline(y=float(best_flux_limit), color='green', linestyle='--', label='HWO Best Flux Ratio Limit')
        plt.xlabel('Distance to System (pc)')
        plt.ylabel('Flux Ratio (planet/star)')
        plt.title('Flux Ratio vs. Distance for Habitable M-dwarf Exoplanet')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('flux_ratio_vs_distance_mdwarf.png')
        plt.close()

        # --- 2. IWA vs. distance, and HZ location ---
        # Use HWO IWA for best scenario
        IWA_best_rad = hwo_best.iwa  # in radians
        IWA_best_AU = IWA_best_rad * self.distances_m / const.au_to_m

        plt.figure(figsize=(7,5))
        plt.plot(self.distances_pc, IWA_best_AU, label='HWO Best IWA (AU)', color='green')
        plt.axhline(y=float(self.hz_au), color='blue', linestyle=':', label='Habitable Zone (HZ)')
        plt.xlabel('Distance to System (pc)')
        plt.ylabel('Projected IWA (AU)')
        plt.title('IWA vs. Distance and Habitable Zone for M-dwarf')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('iwa_vs_distance_mdwarf.png')
        plt.close()