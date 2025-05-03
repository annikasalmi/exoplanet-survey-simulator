import sys
import warnings

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic

from lifesim.util.options import Options
from lifesim.util.habitable import single_habitable_zone
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
        self.IWA = 124e-6
        self.planet_flux_star_ratio = 10e-10
        self.flux_ratio = self.calc_flux()
        self.iwa_constraint = self.calc_iwa_constraint()

    def calc_flux(self):
        flux_ratio = (self.catalog.radius_p.values / self.catalog.radius_s.values)**2 * \
                            self.catalog.temp_p.values / self.catalog.temp_s.values
        return flux_ratio
    
    def calc_iwa_constraint(self):
        iwa_constraint = self.catalog.sep_p.values /self.catalog.distance_s
        return iwa_constraint
    
    def determine_detectable(self):
        iwa_condition = self.iwa_constraint >= self.IWA
        flux_condition = self.flux_ratio >= self.flux_ratio
        total_condition = iwa_condition & flux_condition
        self.catalog['hwo_detectable'] = total_condition
        return self.catalog.hwo_detectable
