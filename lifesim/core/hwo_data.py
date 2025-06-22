import pandas as pd
import numpy as np

import tools.physics_constants as const
from tools.physics_constants import HWOConstants as HWO
from lifesim.core.data import Data
from typing import Union
from PPop.ExozodiModels.Ertel2020 import ExozodiModel


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

    def calc_exozodi_surface_brightness_constraint(self, case: str, exozodi_scenario: str = 'baseline'):
        """
        Calculate exozodi surface brightness constraint using the criterion:
        L_zodi(θ) ≥ C_inst · L_⋆, where C_inst = min_planet_flux_star_ratio from HWOConstants
        
        Parameters
        ----------
        case : str
            'best' or 'worst' case scenario
        exozodi_scenario : str
            'baseline', 'pessimistic', or 'optimistic' exozodi scenario
            
        Returns
        -------
        is_rejected : array
            Boolean array indicating which planets are rejected due to exozodi surface brightness
        surface_brightness_ratios : array
            Ratio of exozodi surface brightness to (contrast_limit * star_surface_brightness)
        """
        # Initialize exozodi model
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        exozodi_model = ExozodiModel(exozodi_scenario, rng)
        
        # Get the instrument contrast limit for this case
        instrument_contrast_limit = HWO(case).min_planet_flux_star_ratio
        
        # Calculate angular separations for each planet
        angular_separations = self.catalog['maxangsep'].values  # arcseconds
        
        # Calculate surface brightness constraint for each planet
        is_rejected = []
        surface_brightness_ratios = []
        
        for i in range(len(self.catalog)):
            # Get planet properties
            star_temp = self.catalog.iloc[i]['temp_s']
            star_radius = self.catalog.iloc[i]['radius_s']
            distance = self.catalog.iloc[i]['distance_s']
            angular_sep = angular_separations[i]
            
            # Check surface brightness criterion
            rejected, exozodi_sb, star_sb, _ = exozodi_model.checkExozodiSurfaceBrightnessCriterion(
                star_temp, star_radius, distance, angular_sep, instrument_contrast_limit
            )
            
            is_rejected.append(rejected)
            
            # Calculate ratio for analysis
            if star_sb > 0:
                ratio = exozodi_sb / (instrument_contrast_limit * star_sb)
            else:
                ratio = np.inf
            surface_brightness_ratios.append(ratio)
        
        return np.array(is_rejected), np.array(surface_brightness_ratios)

    def calc_exozodi_constraint(self, case: str, exozodi_scenario: str = 'baseline'):
        """
        Calculate exozodi flux constraint for each planet.
        
        Parameters
        ----------
        case : str
            'best' or 'worst' case scenario
        exozodi_scenario : str
            'baseline', 'pessimistic', or 'optimistic' exozodi scenario
            
        Returns
        -------
        exozodi_flux_ratios : array
            Ratio of exozodi flux to planet flux for each planet
        """
        # Initialize exozodi model
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        exozodi_model = ExozodiModel(exozodi_scenario, rng)
        
        # Calculate planet flux in HWO band
        planet_flux = self.calc_planet_flux(case)
        
        # Calculate exozodi flux for each planet
        exozodi_fluxes = []
        for i in range(len(self.catalog)):
            # Get planet properties
            star_temp = self.catalog.iloc[i]['temp_s']
            star_radius = self.catalog.iloc[i]['radius_s']
            distance = self.catalog.iloc[i]['distance_s']
            
            # Estimate planet semi-major axis (this would ideally come from the catalog)
            # For now, use a reasonable estimate based on planet temperature
            planet_temp = self.catalog.iloc[i]['temp_p']
            # Simple estimate: a = sqrt(L_star / (4πσT^4)) where T is planet temperature
            star_luminosity = 4 * np.pi * (star_radius * const.R_sun)**2 * const.sigma * star_temp**4
            planet_semi_major_axis = np.sqrt(star_luminosity / (4 * np.pi * const.sigma * planet_temp**4)) / const.au_to_m
            
            # Calculate exozodi flux at planet distance
            exozodi_flux, _ = exozodi_model.getExozodiFluxAtPlanetDistance(
                star_temp, star_radius, distance, planet_semi_major_axis
            )
            exozodi_fluxes.append(exozodi_flux)
        
        exozodi_fluxes = np.array(exozodi_fluxes)
        
        # Calculate flux ratio (planet flux / exozodi flux)
        # We want planet flux > exozodi flux for detection
        flux_ratios = planet_flux / exozodi_fluxes
        
        return flux_ratios
    
    def calc_exozodi_constraint_simple(self, case: str, exozodi_level: float = 1.0):
        """
        Simplified exozodi constraint using a fixed exozodi level.
        
        Parameters
        ----------
        case : str
            'best' or 'worst' case scenario
        exozodi_level : float
            Exozodi level (1.0 = solar system level)
            
        Returns
        -------
        exozodi_flux_ratios : array
            Ratio of planet flux to exozodi flux
        """
        # Calculate planet flux in HWO band
        planet_flux = self.calc_planet_flux(case)
        
        # For simplified approach, assume exozodi flux is proportional to stellar flux
        # and scales with exozodi level
        stellar_flux = self.blackbody_flux(HWO(case).min_wavelength_hwo, self.catalog.temp_s.values)
        
        # Exozodi flux is approximately exozodi_level * stellar_flux * (some scaling factor)
        # The scaling factor accounts for the fact that exozodi emission is in IR
        # and we're observing in the HWO band
        ir_scaling_factor = 0.1  # Rough estimate: ~10% of stellar flux in IR
        exozodi_flux = exozodi_level * stellar_flux * ir_scaling_factor
        
        # Calculate flux ratio (planet flux / exozodi flux)
        flux_ratios = planet_flux / exozodi_flux
        
        return flux_ratios
    
    def determine_detectable(self, use_exozodi_constraint: bool = True, exozodi_scenario: str = 'baseline',
                           use_surface_brightness_criterion: bool = True, ignore_exozodi_rejections: bool = False):
        """
        Determine which planets are detectable based on all constraints.
        
        Parameters
        ----------
        use_exozodi_constraint : bool
            Whether to include exozodi constraint in detection logic
        exozodi_scenario : str
            Exozodi scenario to use ('baseline', 'pessimistic', 'optimistic')
        use_surface_brightness_criterion : bool
            Whether to use the new surface brightness criterion (L_zodi(θ) ≥ C_inst · L_⋆)
            If False, uses the old flux ratio approach
        ignore_exozodi_rejections : bool
            If True, exozodi rejections are calculated but not applied to final detection
            (useful for analyzing other rejection criteria)
        """
        # Evaluate individual constraints
        cases = ['best', 'worst']

        for c in cases:
            iwa_condition = self.calc_iwa_constraint() >= HWO(c).iwa
            flux_condition = self.calc_flux_ratio(c) >= HWO(c).min_planet_flux_star_ratio
            min_photon_rate_condition = self.calc_photons(c) <= HWO(c).min_photons

            # Store individual condition results (boolean)
            self.catalog['iwa_pass_' + c] = iwa_condition
            self.catalog['flux_pass_' + c] = flux_condition
            self.catalog['min_photons_pass_' + c] = min_photon_rate_condition
            
            # Store actual values (separate columns)
            self.catalog['flux_ratio_value_' + c] = self.calc_flux_ratio(c)
            self.catalog['photon_rate_value_' + c] = self.calc_photons(c)

            # Calculate exozodi constraint if requested
            if use_exozodi_constraint:
                if use_surface_brightness_criterion:
                    # Use new surface brightness criterion
                    exozodi_rejected, surface_brightness_ratios = self.calc_exozodi_surface_brightness_constraint(
                        c, exozodi_scenario
                    )
                    # Planet is rejected if exozodi surface brightness exceeds the criterion
                    exozodi_condition = ~exozodi_rejected  # Invert because we want planets that pass
                    self.catalog['exozodi_pass_' + c] = exozodi_condition
                    self.catalog['exozodi_surface_brightness_ratio_' + c] = surface_brightness_ratios
                    
                    # Store the rejection reason for analysis
                    self.catalog['exozodi_surface_brightness_rejected_' + c] = exozodi_rejected
                else:
                    # Use old flux ratio approach
                    exozodi_flux_ratios = self.calc_exozodi_constraint(c, exozodi_scenario)
                    # Planet flux should be greater than exozodi flux (ratio > 1)
                    exozodi_condition = exozodi_flux_ratios > 1.0
                    self.catalog['exozodi_pass_' + c] = exozodi_condition
                    self.catalog['exozodi_flux_ratio_' + c] = exozodi_flux_ratios
                
                # Apply exozodi constraint to final detection only if not ignored
                if ignore_exozodi_rejections:
                    # Total combined detection condition excluding exozodi
                    total_condition = iwa_condition & flux_condition & min_photon_rate_condition
                else:
                    # Total combined detection condition including exozodi
                    total_condition = (iwa_condition & flux_condition & 
                                     min_photon_rate_condition & exozodi_condition)
            else:
                # Total combined detection condition without exozodi
                total_condition = iwa_condition & flux_condition & min_photon_rate_condition
            
            self.catalog['detected_' + c] = total_condition
        
        return self.catalog
