import pandas as pd
import numpy as np

import tools.physics_constants as const
from tools.physics_constants import HWOConstants as HWO
from lifesim.core.data import Data
from typing import Union


# TODO: automatically add data storage for all
class HWOData():
    """
    The data class is the central storage class for catalogs, options, parameters and data. Any
    data used in simulations should be stored in this class. Via the bus, access to the data class
    is given to all modules.

    Attributes
    ----------
    inst : dict
        Data used for simulation of the instrument.
    catalog : pd.DataFrame
        Catalog containing all exoplanets in the sample.
    single : dict
        Data used for the spectral simulation of single exoplanets.
    other : dict
        Data storage for any other pertinent data.
    options : Options
        Location of the Options class. All options and free parameters used in a LIFEsim simulation
        must be stored here.
    """
    def __init__(self, data: Union[Data, pd.DataFrame]):
        if isinstance(data, Data):
            self.catalog = data.catalog
        elif isinstance(data, pd.DataFrame):
            self.catalog = data
        else:
            raise TypeError('Needs to be a pd.DataFrame or type Data object for data.')
        
        # Validate that required columns exist
        self._validate_catalog()

    def _validate_catalog(self) -> None:
        """Validate that the catalog has all required columns."""
        required_columns = ['temp_p', 'temp_s', 'radius_p', 'radius_s', 'distance_s', 'maxangsep']
        missing_columns = [col for col in required_columns if col not in self.catalog.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in catalog: {missing_columns}")

    def blackbody_flux(self, wavelength_m, temperature_K):
        """
        Calculate the blackbody spectral radiance at a given wavelength and temperature.
        
        Parameters:
            wavelength_m: Wavelength in meters (float or array)
            temperature_K: Temperature in Kelvin (float)
            
        Returns:
            Spectral radiance in W·sr⁻¹·m⁻³
        """
        wavelength_m = np.asarray(wavelength_m)
        temperature_K = np.asarray(temperature_K)
        
        exponent = (const.h * const.c) / (wavelength_m * const.k * temperature_K)
        numerator = 2 * const.h * const.c**2
        denominator = (wavelength_m**5) * (np.exp(exponent) - 1)
        return numerator / denominator

    def calc_planet_flux(self, case: str):
        flux_planet_a = self.blackbody_flux(HWO(case).min_wavelength_hwo, self.catalog.temp_p.values)
        flux_planet_b = self.blackbody_flux(HWO(case).max_wavelength_hwo, self.catalog.temp_p.values)

        if case == 'best':
            # Best case: use minimum flux (easier to detect)
            planet_flux = np.minimum(flux_planet_a, flux_planet_b)
        elif case == 'worst':
            # Worst case: use maximum flux (harder to detect)
            planet_flux = np.maximum(flux_planet_a, flux_planet_b)
        else:
            raise ValueError("case must be 'best' or 'worst'")

        return planet_flux
    
    def bolometric_flux(self, T, d_pc=10, R=const.R_earth):
        """
        Calculate bolometric flux of a planet at distance d.

        Parameters:
            T (float): Planet temperature in Kelvin
            R (float): Planet radius in meters (default: Earth)
            d_pc (float): Distance to planet in parsecs (default: 10 pc)

        Returns:
            float: Flux in W/m^2 at Earth
        """
        d = d_pc * const.pc_to_m  # convert distance to meters
        flux = const.sigma * T**4 * (R**2 / d**2)
        return flux

    def calc_flux_ratio(self, case: str):
        hwo = HWO(case)
        flux_planet_a = self.blackbody_flux(hwo.min_wavelength_hwo, self.catalog.temp_p.values)
        flux_planet_b = self.blackbody_flux(hwo.max_wavelength_hwo, self.catalog.temp_p.values)

        flux_star_a = self.blackbody_flux(hwo.min_wavelength_hwo, self.catalog.temp_s.values)
        flux_star_b = self.blackbody_flux(hwo.max_wavelength_hwo, self.catalog.temp_s.values)

        flux_ratio_a = self.catalog.radius_p.values**2 * flux_planet_a / (self.catalog.radius_s.values**2 * flux_star_a)
        flux_ratio_b = self.catalog.radius_p.values**2 * flux_planet_b / (self.catalog.radius_s.values**2 * flux_star_b)

        if case == 'best':
            flux_ratio = np.maximum(flux_ratio_a, flux_ratio_b)
        elif case == 'worst':
            flux_ratio = np.minimum(flux_ratio_a, flux_ratio_b)
        else:
            raise ValueError("case must be 'best' or 'worst'")

        return flux_ratio
    
    def photon_energy(self, wavelength):
        return const.h * const.c / wavelength  # in joules

    def photon_rate_per_hour_per_micron(self, flux_w_m2, wavelength_m):
        """
        Calculate photon rate in photons/hour/μm given:
        - flux in W/m² (Watts per square meter)
        - wavelength in meters

        Assumes collection over 1 m².

        Returns: photon rate in photons/hour/μm
        """
        # Constants
        h = 6.62607015e-34  # Planck's constant (J·s)
        c = 2.99792458e8    # Speed of light (m/s)

        # Convert wavelength to meters (if it's not already)
        wavelength_m = np.asarray(wavelength_m)

        # Energy per photon (Joules)
        energy_per_photon = h * c / wavelength_m

        # Photon rate (photons/sec/m²)
        photons_per_second_per_m2 = flux_w_m2 / energy_per_photon

        # Convert to photons/hour/m²
        photons_per_hour_per_m2 = photons_per_second_per_m2 * 3600

        # Convert to photons/hour/μm (assuming 1 μm bandwidth)
        # This is a spectral density, so we multiply by wavelength bandwidth
        wavelength_um = wavelength_m * 1e6  # convert m to μm
        photons_per_hour_per_um = photons_per_hour_per_m2 * wavelength_um

        return photons_per_hour_per_um
    
    def calc_photons(self, case: str):
        # Calculate photon rates for both wavelength limits using spectral flux density
        # Use blackbody flux (spectral) instead of bolometric flux
        
        # Get spectral flux density at the specific wavelengths
        spectral_flux_a = self.blackbody_flux(HWO(case).min_wavelength_hwo, self.catalog.temp_p.values)
        spectral_flux_b = self.blackbody_flux(HWO(case).max_wavelength_hwo, self.catalog.temp_p.values)
        
        # Convert to flux at Earth (accounting for distance and planet size)
        # spectral_flux is in W·sr⁻¹·m⁻³, we need W/m²
        # For a planet: flux_at_earth = spectral_flux * (planet_radius² / distance²) * π
        flux_at_earth_a = spectral_flux_a * (self.catalog.radius_p.values**2 / (self.catalog.distance_s.values * const.pc_to_m)**2) * np.pi
        flux_at_earth_b = spectral_flux_b * (self.catalog.radius_p.values**2 / (self.catalog.distance_s.values * const.pc_to_m)**2) * np.pi
        
        # Calculate photon rates
        photon_rate_a = self.photon_rate_per_hour_per_micron(flux_at_earth_a, HWO(case).min_wavelength_hwo)
        photon_rate_b = self.photon_rate_per_hour_per_micron(flux_at_earth_b, HWO(case).max_wavelength_hwo)

        # Return the appropriate photon rate based on case
        if case == 'best':
            # Best case: use minimum photon rate (easier to detect)
            photons = np.minimum(photon_rate_a, photon_rate_b)
        elif case == 'worst': 
            # Worst case: use maximum photon rate (harder to detect)
            photons = np.maximum(photon_rate_a, photon_rate_b)
        else:
            raise ValueError("case must be 'best' or 'worst'")
        return photons
    
    def calc_iwa_constraint(self):
        iwa_constraint = self.catalog.maxangsep
        return iwa_constraint
    
    def determine_detectable(self):
        # Evaluate individual constraints
        cases = ['best', 'worst']

        for c in cases:
            iwa_condition = self.calc_iwa_constraint() >= HWO(c).iwa
            flux_condition = self.calc_flux_ratio(c) >= HWO(c).min_planet_flux_star_ratio
            min_photon_rate_condition = self.calc_photons(c) <= HWO(c).min_photons

            # Store individual condition results (boolean)
            self.catalog['iwa_pass_' + c] = iwa_condition
            self.catalog['flux_pass_' + c] = flux_condition  # Changed from flux_ratio_ to flux_pass_
            self.catalog['min_photons_pass_' + c] = min_photon_rate_condition
            
            # Store actual values (separate columns)
            self.catalog['flux_ratio_value_' + c] = self.calc_flux_ratio(c)
            self.catalog['photon_rate_value_' + c] = self.calc_photons(c)

            # Total combined detection condition
            total_condition = iwa_condition & flux_condition & min_photon_rate_condition
            self.catalog['detected_' + c] = total_condition
        return self.catalog
