import numpy as np

h = 6.62607015e-34    # Planck's constant (J·s)
c = 2.99792458e8      # Speed of light (m/s)
k = 1.380649e-23      # Boltzmann constant (J/K)
sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
R_earth = 6.371e6        # Earth's radius in meters
pc_to_m = 3.0857e16      # 1 parsec in meters
temp_sun = 5772         # temp of Sun in Kelvin
R_sun = 6.957e8         # Solar radius in meters
au_to_m = 1.496e11      # 1 au in meters
A_g_earth = 0.3         # Albedo of Earth
Phi_alpha = 0.5         # Phase function at quadrature
arcsec_to_radians = np.pi / (180 * 3600)  # 1 arcsecond in radians
rad_to_arcsec = 206265

# HWO specific
class HWOConstants():
    def __init__(self, scenario='best'):
        self.scenario = scenario
        if scenario not in ['best', 'worst']:
            raise ValueError("Only 'best' and 'worst' case HWO scenarios are implemented.")
        if self.scenario == 'best':
            self.iwa = 20.6e-3
            self.min_planet_flux_star_ratio = 2.5e-11
            self.min_wavelength_hwo = 200e-9  # Minimum wavelength for HWO (m)
            self.max_wavelength_hwo = 2500e-9  # Maximum wavelength for HWO (m)
            self.min_photons = 1 # Minimum photons for detection (photons/hour/micron)
            self.max_z = 10 # Max z value for detection (zodis)
        elif self.scenario == 'worst':
            self.iwa = 124e-3
            self.min_planet_flux_star_ratio = 10e-10
            self.min_wavelength_hwo = 300e-9  # Minimum wavelength for HWO (m)
            self.max_wavelength_hwo = 2400e-9  # Maximum wavelength for HWO (m)
            self.min_photons = 50 # Minimum photons for detection
            self.max_z = 1 # Max z value for detection (zodis)



# Distance limits for exoplanet analysis
D_min_pc = 4.0         # Minimum distance (pc)
D_max_pc = 20.0        # Maximum distance (pc)


# Habitable planet radius limits
R_earth_min_habitable = 0.5  # Minimum habitable planet radius (Earth radii)
R_earth_max_habitable = 2.6  # Maximum habitable planet radius (Earth radii)

# Example habitable planet parameters
T_earth = 288  # Earth-like planet temperature (K)
R_earth_example = 1.0  # Example habitable planet radius (Earth radii)

# Stellar luminosity limits for M dwarf and G-star regions
L_m_dwarf_min = 0.001  # Minimum M dwarf luminosity (L☉)
L_m_dwarf_max = 0.08   # Maximum M dwarf luminosity (L☉)
L_g_star_min = 0.6     # Minimum G-star luminosity (L☉)
L_g_star_max = 1.5     # Maximum G-star luminosity (L☉)

# Grid resolution parameters
L_GRID_SIZE = 300      # Grid size for luminosity calculations
D_GRID_SIZE = 300      # Grid size for distance calculations
AU_GRID_SIZE = 1000    # Grid size for AU calculations