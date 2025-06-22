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
    
    def getExozodiFlux(self, star_temp_K, star_radius_Rsun, distance_pc, exozodi_level=None):
        """
        Calculate the exozodi flux that would be observed by HWO.
        
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
        
        # Exozodi flux is proportional to star luminosity and exozodi level
        # The exozodi level z represents the ratio of exozodi to stellar flux
        # at a reference wavelength (typically ~10 μm)
        
        # For HWO wavelength range, we need to account for the spectral distribution
        # Exozodi emission peaks in the infrared, so we use a scaling factor
        # This is a simplified model - could be refined with more detailed spectral modeling
        
        # Calculate the fraction of exozodi emission in HWO band
        # Exozodi emission is roughly blackbody-like with temperature calculated from radiative equilibrium
        exozodi_temp = self._calculate_exozodi_temperature(star_temp_K, star_radius_Rsun, distance_pc)
        
        # Calculate blackbody flux in HWO band for exozodi
        exozodi_flux_hwo = self._blackbody_flux_in_band(
            exozodi_temp, 
            self.hwo_min_wavelength, 
            self.hwo_max_wavelength
        )
        
        # Calculate total exozodi flux at Earth
        # exozodi_level * star_flux gives the total exozodi flux
        # We scale by the fraction in HWO band
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
        exozodi_temp = self._calculate_exozodi_temperature(star_temp_K, star_radius_Rsun, distance_pc, 
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
            # Default distance range
            distances_au = np.logspace(-1, 2, 100)  # 0.1 to 100 AU
        
        # Calculate system-wide exozodi flux
        exozodi_flux_system, _ = self.getExozodiFlux(
            star_temp_K, star_radius_Rsun, distance_pc, exozodi_level
        )
        
        # Apply radial dependence
        alpha = 1.5  # Radial power law index
        reference_distance_au = 10.0
        
        # Calculate radial scaling for each distance
        radial_scaling = (reference_distance_au / distances_au) ** alpha
        
        # Calculate exozodi flux at each distance
        exozodi_fluxes = exozodi_flux_system * radial_scaling
        
        return distances_au, exozodi_fluxes
    
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

    def _calculate_exozodi_temperature(self, star_temp_K, star_radius_Rsun, distance_pc, 
                                      planet_semi_major_axis_au=None):
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
            # Use reference distance where exozodi level is typically measured
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
    
    def getExozodiTemperatureProfile(self, star_temp_K, star_radius_Rsun, distances_au=None):
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
            temp = self._calculate_exozodi_temperature(star_temp_K, star_radius_Rsun, 10.0, dist)
            exozodi_temperatures.append(temp)
        
        return np.array(distances_au), np.array(exozodi_temperatures)
