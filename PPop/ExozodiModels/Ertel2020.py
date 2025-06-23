"""
# =============================================================================
# P-POP
# A Monte-Carlo tool to simulate exoplanet populations
# =============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.stats as stats
from scipy.integrate import quad

from tools.paths import PPOP_DIR
from tools.physics_constants import h, c, k, sigma, R_earth, pc_to_m, R_sun, temp_sun


# =============================================================================
# ERTEL2020
# =============================================================================

class ExozodiModel():
    """
    https://ui.adsabs.harvard.edu/abs/2020AJ....159..177E/abstract
    """
    
    def __init__(self,
                 Scenario,rng):
        """
        Parameters
        ----------
        Scenario: 'baseline', 'pessimistic', 'optimistic'
            Scenario for exozodi level.
        """
        self.rng = rng
        
        # Model parameters.
        if (Scenario == 'baseline'):
            self.ExozodiData = np.load(os.path.join(PPOP_DIR,'ExozodiModels/ExozodiNominal.npy'))
        elif (Scenario == 'pessimistic'):
            self.ExozodiData = np.load('ExozodiModels/ExozodiPessimistic.npy')
        elif (Scenario == 'optimistic'):
            self.ExozodiData = np.load('ExozodiModels/ExozodiOptimistic.npy')
        else:
            print('--> WARNING: '+str(Scenario)+' is an unknown scenario')
            Scenario = 'baseline'
            self.ExozodiData = np.load('ExozodiModels/ExozodiNominal.npy')
        print('--> Using scenario '+str(Scenario))
        
        self.LogExozodiKDE = stats.gaussian_kde(np.log(self.ExozodiData[0]))
        
        # HWO wavelength range (meters)
        self.hwo_min_wavelength = 200e-9  # 200 nm
        self.hwo_max_wavelength = 2500e-9  # 2.5 μm
        
        pass
   
    def getExozodiLevel(self):
        """
        Returns
        -------
        z: float
            Exozodiacal dust level of drawn system.
        """
        
        z = np.exp(self.LogExozodiKDE.resample(1)[0][0])
        
        return z
    
    def getExozodiFluxAtPlanetAU(self, star_temp_K, star_radius_Rsun, distance_pc, exozodi_level=None, planet_semi_major_axis_au=None):
        """
        Calculate the exozodi flux that would be observed by HWO at a given planet semi-major axis (AU).
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii (already in correct units)
        distance_pc : float
            Distance to star in parsecs
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
        planet_semi_major_axis_au : float, optional
            Planet's semi-major axis in AU. If None, uses reference distance (10 AU)
        
        Returns
        -------
        exozodi_flux_hwo : float
            Exozodi flux in the HWO wavelength range (W/m²)
        exozodi_level : float
            The exozodi level used for calculation
        """
        if exozodi_level is None:
            exozodi_level = self.getExozodiLevel()
        
        # Convert star properties to SI units
        star_radius_m = star_radius_Rsun * R_sun
        distance_m = distance_pc * pc_to_m
        
        # Calculate star's bolometric luminosity
        star_luminosity = 4 * np.pi * star_radius_m**2 * sigma * star_temp_K**4
        
        # Calculate the fraction of exozodi emission in HWO band
        exozodi_temp = self._calculate_exozodi_temperature_at_planet_au(star_temp_K, star_radius_Rsun, distance_pc, planet_semi_major_axis_au)
        
        # Calculate blackbody flux in HWO band for exozodi
        exozodi_flux_hwo = self._blackbody_flux_in_band(
            exozodi_temp, 
            self.hwo_min_wavelength, 
            self.hwo_max_wavelength
        )
        
        # Calculate total exozodi flux at Earth
        star_flux_at_earth = star_luminosity / (4 * np.pi * distance_m**2)
        total_exozodi_flux = exozodi_level * star_flux_at_earth
        
        # Scale to HWO band
        exozodi_flux_hwo_at_earth = total_exozodi_flux * exozodi_flux_hwo
        
        return exozodi_flux_hwo_at_earth, exozodi_level
    
    def _blackbody_flux_in_band(self, temperature_K, min_wavelength_m, max_wavelength_m):
        """
        Numerically integrate Planck's law to find the fraction of blackbody
        emission within a wavelength band.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin
        min_wavelength_m : float
            Minimum wavelength in meters
        max_wavelength_m : float
            Maximum wavelength in meters
            
        Returns
        -------
        fraction : float
            Fraction of total blackbody emission in the specified band
        """
        def planck_lambda(wavelength, T):
            """Spectral radiance per unit wavelength (W·sr⁻¹·m⁻³)"""
            a = 2.0 * h * c**2
            b = h * c / (wavelength * k * T)
            
            # Handle overflow for very small wavelengths or high temperatures
            if np.any(b > 700):  # exp(700) is close to overflow
                return np.zeros_like(wavelength)
            
            return a / (wavelength**5 * (np.exp(b) - 1.0))
        
        # Total blackbody flux integrated over all wavelengths (Stefan-Boltzmann law)
        # Total flux per steradian: σT⁴/π
        total_flux_per_sr = sigma * temperature_K**4 / np.pi

        # Numerically integrate the Planck function over the band
        # This gives us flux per steradian in the band
        band_flux_per_sr, _ = quad(planck_lambda, min_wavelength_m, max_wavelength_m, args=(temperature_K,))

        # Fraction of flux in the band
        fraction = band_flux_per_sr / total_flux_per_sr
        return fraction
    
    def getExozodiFluxAtPlanetDistance(self, star_temp_K, star_radius_Rsun, distance_pc, 
                                      planet_semi_major_axis_au, exozodi_level=None):
        """
        Calculate the exozodi flux at a specific planet distance, accounting for radial dependence.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii (already in correct units)
        distance_pc : float
            Distance to star in parsecs
        planet_semi_major_axis_au : float
            Planet's semi-major axis in AU
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
            
        Returns
        -------
        exozodi_flux_at_planet : float
            Exozodi flux at the planet's distance in HWO band (W/m²)
        exozodi_level : float
            The exozodi level used for calculation
        """
        if exozodi_level is None:
            exozodi_level = self.getExozodiLevel()
        
        # Convert star properties to SI units
        star_radius_m = star_radius_Rsun * R_sun
        distance_m = distance_pc * pc_to_m
        
        # Calculate star's bolometric luminosity
        star_luminosity = 4 * np.pi * star_radius_m**2 * sigma * star_temp_K**4
        
        # Calculate the fraction of exozodi emission in HWO band
        exozodi_temp = self._calculate_exozodi_temperature_at_planet_au(star_temp_K, star_radius_Rsun, distance_pc, 
                                                          planet_semi_major_axis_au)
        
        # Calculate blackbody flux in HWO band for exozodi
        exozodi_flux_hwo = self._blackbody_flux_in_band(
            exozodi_temp,
            self.hwo_min_wavelength,
            self.hwo_max_wavelength
        )
        
        # Calculate total exozodi flux at Earth (system-wide)
        star_flux_at_earth = star_luminosity / (4 * np.pi * distance_m**2)
        total_exozodi_flux = exozodi_level * star_flux_at_earth
        
        # Apply radial dependence: exozodi brightness decreases with distance from star
        # This follows a power law: brightness ∝ r^(-alpha) where alpha is typically 1-2
        # For exozodi, alpha ≈ 1.5 is a reasonable approximation
        alpha = 1.5  # Radial power law index for exozodi brightness
        
        # Reference distance (typically where exozodi level is measured, ~10 AU)
        reference_distance_au = 10.0
        
        # Calculate radial scaling factor
        if planet_semi_major_axis_au > 0:
            radial_scaling = (reference_distance_au / planet_semi_major_axis_au) ** alpha
        else:
            radial_scaling = 1.0  # Avoid division by zero
        
        # Calculate exozodi flux at the planet's distance
        exozodi_flux_at_planet = total_exozodi_flux * exozodi_flux_hwo * radial_scaling
        
        return exozodi_flux_at_planet, exozodi_level
    
    def getExozodiFluxRatioAtPlanetDistance(self, star_temp_K, star_radius_Rsun, distance_pc,
                                           planet_radius_Rearth, planet_temp_K, planet_semi_major_axis_au,
                                           exozodi_level=None):
        """
        Calculate the ratio of exozodi flux to planet flux at the planet's orbital distance.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distance_pc : float
            Distance to star in parsecs
        planet_radius_Rearth : float
            Planet radius in Earth radii
        planet_temp_K : float
            Planet temperature in Kelvin
        planet_semi_major_axis_au : float
            Planet's semi-major axis in AU
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
            
        Returns
        -------
        flux_ratio : float
            Ratio of exozodi flux to planet flux at planet's distance
        exozodi_flux : float
            Exozodi flux at planet's distance in HWO band
        planet_flux : float
            Planet flux in HWO band
        """
        # Get exozodi flux at planet's distance
        exozodi_flux, exozodi_level = self.getExozodiFluxAtPlanetDistance(
            star_temp_K, star_radius_Rsun, distance_pc, planet_semi_major_axis_au, exozodi_level
        )
        
        # Calculate planet flux in HWO band
        planet_flux = self._planet_flux_in_hwo_band(
            planet_radius_Rearth, planet_temp_K, distance_pc
        )
        
        # Calculate ratio
        flux_ratio = exozodi_flux / planet_flux if planet_flux > 0 else np.inf
        
        return flux_ratio, exozodi_flux, planet_flux
    
    def getExozodiRadialProfile(self, star_temp_K, star_radius_Rsun, distance_pc, 
                               exozodi_level=None, distances_au=None):
        """
        Calculate exozodi flux as a function of distance from the star.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distance_pc : float
            Distance to star in parsecs
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
        distances_au : array-like, optional
            Distances from star in AU to calculate flux at
            
        Returns
        -------
        distances_au : array
            Distances from star in AU
        exozodi_fluxes : array
            Exozodi flux at each distance in HWO band (W/m²)
        """
        if exozodi_level is None:
            exozodi_level = self.getExozodiLevel()
        if distances_au is None:
            distances_au = np.logspace(-1, 2, 100)  # 0.1 to 100 AU
        
        # Calculate exozodi flux at each distance
        exozodi_fluxes = []
        for dist in distances_au:
            exozodi_flux, _ = self.getExozodiFluxAtPlanetAU(
                star_temp_K, star_radius_Rsun, distance_pc, exozodi_level, planet_semi_major_axis_au=dist
            )
            exozodi_fluxes.append(exozodi_flux)
        
        return np.array(distances_au), np.array(exozodi_fluxes)
    
    def getOptimalPlanetDistance(self, star_temp_K, star_radius_Rsun, distance_pc,
                                planet_radius_Rearth, planet_temp_K, exozodi_level=None,
                                max_distance_au=50.0):
        """
        Find the optimal distance for planet detection (minimum exozodi/planet flux ratio).
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distance_pc : float
            Distance to star in parsecs
        planet_radius_Rearth : float
            Planet radius in Earth radii
        planet_temp_K : float
            Planet temperature in Kelvin
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
        max_distance_au : float
            Maximum distance to search in AU
            
        Returns
        -------
        optimal_distance_au : float
            Distance with minimum exozodi/planet flux ratio
        min_flux_ratio : float
            Minimum flux ratio achieved
        """
        # Create distance grid
        distances_au = np.logspace(-1, np.log10(max_distance_au), 1000)
        
        # Calculate flux ratios at each distance
        flux_ratios = []
        for dist in distances_au:
            ratio, _, _ = self.getExozodiFluxRatioAtPlanetDistance(
                star_temp_K, star_radius_Rsun, distance_pc,
                planet_radius_Rearth, planet_temp_K, dist, exozodi_level
            )
            flux_ratios.append(ratio)
        
        flux_ratios = np.array(flux_ratios)
        
        # Find minimum (excluding infinite values)
        valid_indices = np.isfinite(flux_ratios)
        if np.any(valid_indices):
            min_idx = np.argmin(flux_ratios[valid_indices])
            optimal_distance_au = distances_au[valid_indices][min_idx]
            min_flux_ratio = flux_ratios[valid_indices][min_idx]
        else:
            optimal_distance_au = 1.0  # Default to 1 AU
            min_flux_ratio = np.inf
        
        return optimal_distance_au, min_flux_ratio
    
    def _planet_flux_in_hwo_band(self, planet_radius_Rearth, planet_temp_K, distance_pc):
        """
        Calculate planet flux in HWO wavelength band.
        
        Parameters
        ----------
        planet_radius_Rearth : float
            Planet radius in Earth radii
        planet_temp_K : float
            Planet temperature in Kelvin
        distance_pc : float
            Distance to planet in parsecs
            
        Returns
        -------
        planet_flux : float
            Planet flux in HWO band (W/m²)
        """
        # Convert to SI units
        planet_radius_m = planet_radius_Rearth * R_earth
        distance_m = distance_pc * pc_to_m
        
        # Calculate total planet luminosity
        planet_luminosity = 4 * np.pi * planet_radius_m**2 * sigma * planet_temp_K**4
        
        # Calculate fraction in HWO band
        fraction_in_band = self._blackbody_flux_in_band(
            planet_temp_K, 
            self.hwo_min_wavelength, 
            self.hwo_max_wavelength
        )
        
        # Calculate flux at Earth
        planet_flux = planet_luminosity * fraction_in_band / (4 * np.pi * distance_m**2)
        
        return planet_flux
    
    def SummaryPlot(self,
                    Ntest=100000,
                    FigDir=None,
                    block=True):
        """
        Parameters
        ----------
        Ntest: int
            Number of test draws for summary plot.
        FigDir: str
            Directory to which summary plots are saved.
        block: bool
            If True, blocks plots when showing.
        """
        
        Ntest = Ntest//10
        z = []
        for i in range(Ntest):
            z += [self.getExozodiLevel()]
        z = np.array(z)
        
        print('--> Ertel2020:\n%.2f/%.2f median/mean system exozodiacal dust level' % (np.median(z), np.mean(z)))
        
        Colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        f, ax = plt.subplots(1, 1)
        Weight = 1./len(z)
        y, x, _ = ax.hist(z, bins=np.logspace(-5, 5, 50), weights=np.ones_like(z)*Weight, color=Colors[0], alpha=0.5, label='KDE')
        Weight = 1./len(self.ExozodiData[0])
        ax.hist(self.ExozodiData[0], bins=np.logspace(-5, 5, 50), weights=np.ones_like(self.ExozodiData[0])*Weight, color=Colors[1], alpha=0.5, label='Original')
        temp = np.linspace(-5, 5, 50)
        step = (temp[1]-temp[0])*np.log(10.)
        ax.plot(x, self.LogExozodiKDE.pdf(np.log(x))*step, color=Colors[0])
        ax.set_xscale('log')
        ax.grid(axis='y')
        ax.set_xlabel('System exozodiacal dust level')
        ax.set_ylabel('Fraction')
        ax.legend()
        plt.suptitle('Ertel2020')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if (FigDir is not None):
            plt.savefig(FigDir+'ExozodiModel.pdf')
        plt.show(block=block)
        plt.close()
        
        pass

    def _calculate_exozodi_temperature_at_planet_au(self, star_temp_K, star_radius_Rsun, distance_pc, planet_semi_major_axis_au=None):
        """
        Calculate exozodi dust temperature based on radiative equilibrium with the star.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distance_pc : float
            Distance to star in parsecs
        planet_semi_major_axis_au : float, optional
            Distance from star in AU where exozodi temperature is calculated
            If None, uses a reference distance of 10 AU
        
        Returns
        -------
        exozodi_temp : float
            Exozodi dust temperature in Kelvin
        """
        if planet_semi_major_axis_au is None:
            planet_semi_major_axis_au = 10.0
        
        # Convert to SI units
        star_radius_m = star_radius_Rsun * R_sun
        distance_au_to_m = planet_semi_major_axis_au * 1.496e11  # AU to meters
        
        # Calculate stellar flux at the exozodi distance
        # F = L_star / (4πr²) = σT_star⁴ * (R_star/r)²
        stellar_flux_at_exozodi = sigma * star_temp_K**4 * (star_radius_m / distance_au_to_m)**2
        
        # For dust in radiative equilibrium, the dust temperature is related to the stellar flux
        # Assuming the dust absorbs and emits as a blackbody, and is in radiative equilibrium:
        # σT_dust⁴ = A * F_star / 4, where A is the absorption efficiency
        # For typical dust grains, A ≈ 0.5-1.0 in the visible/IR
        # We'll use A = 0.7 as a reasonable average value
        
        absorption_efficiency = 0.7  # Typical value for dust grains
        
        # Calculate dust temperature from radiative equilibrium
        # T_dust⁴ = A * F_star / (4σ)
        exozodi_temp = (absorption_efficiency * stellar_flux_at_exozodi / (4 * sigma))**(1/4)
        
        # Add some physical constraints
        # Dust temperature should be reasonable (not too hot or too cold)
        exozodi_temp = np.clip(exozodi_temp, 50, 1000)  # Reasonable range for exozodi dust
        
        return exozodi_temp
    
    def getExozodiTemperatureProfileAtPlanetAU(self, star_temp_K, star_radius_Rsun, distances_au=None):
        """
        Calculate exozodi temperature as a function of distance from the star.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distances_au : array-like, optional
            Distances from star in AU to calculate temperature at
            
        Returns
        -------
        distances_au : array
            Distances from star in AU
        exozodi_temperatures : array
            Exozodi temperature at each distance in Kelvin
        """
        if distances_au is None:
            # Default distance range
            distances_au = np.logspace(-1, 2, 100)  # 0.1 to 100 AU
        
        # Calculate temperature at each distance
        exozodi_temperatures = []
        for dist in distances_au:
            temp = self._calculate_exozodi_temperature_at_planet_au(star_temp_K, star_radius_Rsun, 10.0, dist)
            exozodi_temperatures.append(temp)
        
        return np.array(distances_au), np.array(exozodi_temperatures)

    def getExozodiSurfaceBrightness(self, star_temp_K, star_radius_Rsun, distance_pc, 
                                   angular_separation_arcsec, exozodi_level=None):
        """
        Calculate exozodi surface brightness at a given angular separation.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distance_pc : float
            Distance to star in parsecs
        angular_separation_arcsec : float
            Angular separation from star in arcseconds
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
            
        Returns
        -------
        surface_brightness : float
            Exozodi surface brightness at the given angular separation (W/m²/arcsec²)
        exozodi_level : float
            The exozodi level used for calculation
        """
        if exozodi_level is None:
            exozodi_level = self.getExozodiLevel()
        
        # Convert angular separation to physical distance
        angular_separation_rad = angular_separation_arcsec * np.pi / (180 * 3600)  # Convert arcsec to radians
        physical_distance_au = angular_separation_rad * distance_pc * 206265  # Convert to AU
        
        # Calculate star luminosity in solar units
        star_luminosity_solar = (star_radius_Rsun**2) * (star_temp_K / temp_sun)**4
        
        # Kennedy+2015 parameters for radial profile
        alpha = 0.34
        r_in = 0.034422617777777775 * np.sqrt(star_luminosity_solar)  # Inner radius in AU
        r_0 = np.sqrt(star_luminosity_solar)  # Reference radius in AU
        sigma_zero = 7.11889e-8  # Sigma_{m,0} from Kennedy+2015
        
        # Check if the angular separation is within the exozodi disk
        if physical_distance_au < r_in:
            return 0.0, exozodi_level
        
        # Calculate temperature at this distance (Kennedy+2015 Eq. 2)
        temp_dust = 278.3 * (star_luminosity_solar**0.25) / np.sqrt(physical_distance_au)
        
        # Calculate surface density (Kennedy+2015 Eq. 3)
        sigma = sigma_zero * exozodi_level * (physical_distance_au / r_0)**(-alpha)
        
        # Calculate blackbody emission in HWO band
        fraction_in_band = self._blackbody_flux_in_band(
            temp_dust, 
            self.hwo_min_wavelength, 
            self.hwo_max_wavelength
        )
        
        # Calculate surface brightness
        # Surface brightness = sigma * blackbody_emission * fraction_in_band
        # Convert to W/m²/arcsec²
        # Note: sigma is in kg/m², we need to convert to proper units
        # The surface brightness should be proportional to the dust emission
        # and inversely proportional to the angular area
        
        # Calculate the dust emission per unit area
        dust_emission_per_area = sigma * sigma * temp_dust**4 * fraction_in_band
        
        # Convert to surface brightness at Earth
        # Surface brightness = emission_per_area / (distance² * angular_area)
        distance_m = distance_pc * pc_to_m
        angular_area_arcsec2 = np.pi * (angular_separation_arcsec * 4.848e-6)**2  # Convert arcsec² to sr
        
        surface_brightness = dust_emission_per_area / (distance_m**2 * angular_area_arcsec2)
        
        return surface_brightness, exozodi_level
    
    def checkExozodiSurfaceBrightnessCriterion(self, star_temp_K, star_radius_Rsun, distance_pc,
                                              angular_separation_arcsec, instrument_contrast_limit,
                                              exozodi_level=None):
        """
        Check if exozodi surface brightness exceeds the instrument contrast limit.
        
        Parameters
        ----------
        star_temp_K : float
            Star effective temperature in Kelvin
        star_radius_Rsun : float
            Star radius in solar radii
        distance_pc : float
            Distance to star in parsecs
        angular_separation_arcsec : float
            Angular separation from star in arcseconds
        instrument_contrast_limit : float
            Instrument contrast limit at the given angular separation
        exozodi_level : float, optional
            Exozodi level (if None, will draw from model)
            
        Returns
        -------
        is_rejected : bool
            True if planet should be rejected due to exozodi surface brightness
        surface_brightness : float
            Exozodi surface brightness at the given angular separation
        star_brightness : float
            Star brightness for comparison
        exozodi_level : float
            The exozodi level used for calculation
        """
        if exozodi_level is None:
            exozodi_level = self.getExozodiLevel()
        
        # Calculate exozodi surface brightness
        exozodi_surface_brightness, exozodi_level = self.getExozodiSurfaceBrightness(
            star_temp_K, star_radius_Rsun, distance_pc, angular_separation_arcsec, exozodi_level
        )
        
        # Calculate star brightness in HWO band
        star_luminosity = 4 * np.pi * (star_radius_Rsun * R_sun)**2 * sigma * star_temp_K**4
        star_flux_at_earth = star_luminosity / (4 * np.pi * (distance_pc * pc_to_m)**2)
        
        # Calculate fraction of star emission in HWO band
        star_fraction_in_band = self._blackbody_flux_in_band(
            star_temp_K, 
            self.hwo_min_wavelength, 
            self.hwo_max_wavelength
        )
        
        # Star brightness in HWO band
        star_brightness_hwo = star_flux_at_earth * star_fraction_in_band
        
        # Convert to surface brightness (assuming point source)
        # For a point source, surface brightness = flux / (angular_area)
        # We'll use a reference angular area of 1 arcsec²
        star_surface_brightness = star_brightness_hwo / (np.pi * (1.0 * 4.848e-6)**2)  # W/m²/arcsec²
        
        # Check the criterion: L_zodi(θ) ≥ C_inst · L_⋆
        is_rejected = exozodi_surface_brightness >= (instrument_contrast_limit * star_surface_brightness)
        
        return is_rejected, exozodi_surface_brightness, star_surface_brightness, exozodi_level
