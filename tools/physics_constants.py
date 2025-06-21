h = 6.62607015e-34    # Planck's constant (J·s)
c = 2.99792458e8      # Speed of light (m/s)
k = 1.380649e-23      # Boltzmann constant (J/K)
sigma = 5.670374419e-8  # Stefan-Boltzmann constant (W/m^2/K^4)
R_earth = 6.371e6        # Earth's radius in meters
pc_to_m = 3.0857e16      # 1 parsec in meters
temp_sun = 5772         # temp of Sun in Kelvin

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
        elif self.scenario == 'worst':
            self.iwa = 124e-3
            self.min_planet_flux_star_ratio = 10e-10
            self.min_wavelength_hwo = 300e-9  # Minimum wavelength for HWO (m)
            self.max_wavelength_hwo = 2400e-9  # Maximum wavelength for HWO (m)
            self.min_photons = 50 # Minimum photons for detection
