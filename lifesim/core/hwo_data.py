import pandas as pd
import numpy as np

import tools.constants as const
from lifesim.core.data import Data


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
    def __init__(self, data):#: Data | pd.DataFrame):
        if type(data) == Data:
            self.catalog=Data.catalog
        elif type(data) == pd.DataFrame:
            self.catalog = data
        else:
            raise TypeError('Needs to be a pd.DataFrame or type Data object for data.')
        self.flux_ratio = self.calc_flux()
        self.iwa_constraint = self.calc_iwa_constraint()

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

    def calc_planet_flux(self):

        flux_planet_a = self.blackbody_flux(const.min_wavelength_hwo, self.catalog.temp_p.values)
        flux_planet_b = self.blackbody_flux(const.max_wavelength_hwo, self.catalog.temp_p.values)

        planet_flux = np.maximum(flux_planet_a, flux_planet_b)
        return planet_flux

    def calc_flux(self):
        flux_planet_a = self.blackbody_flux(const.min_wavelength_hwo, self.catalog.temp_p.values)
        flux_planet_b = self.blackbody_flux(const.max_wavelength_hwo, self.catalog.temp_p.values)

        flux_star_a = self.blackbody_flux(const.min_wavelength_hwo, self.catalog.temp_s.values)
        flux_star_b = self.blackbody_flux(const.max_wavelength_hwo, self.catalog.temp_s.values)

        flux_ratio_a = self.catalog.radius_p.values**2 * flux_planet_a / (self.catalog.radius_s.values**2 * flux_star_a)
        flux_ratio_b = self.catalog.radius_p.values**2 * flux_planet_b / (self.catalog.radius_s.values**2 * flux_star_b)

        flux_ratio = np.maximum(flux_ratio_a, flux_ratio_b)

        return flux_ratio
    
    def calc_iwa_constraint(self):
        iwa_constraint = self.catalog.angsep
        return iwa_constraint
    
    def determine_detectable(self):
        iwa_condition = self.iwa_constraint >= const.iwa
        flux_condition = self.flux_ratio >= const.planet_flux_star_ratio
        min_flux_condition = self.calc_planet_flux() >= const.min_flux
        total_condition = iwa_condition & flux_condition & min_flux_condition
        self.catalog['hwo_detectable'] = total_condition
        return self.catalog.hwo_detectable
