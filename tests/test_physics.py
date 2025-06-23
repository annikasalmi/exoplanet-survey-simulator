"""
Unit tests for physics calculations in the mdwarf-habitability project.
"""

import unittest
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy import constants

# Import the modules to test
from PPop.ExozodiModels.Ertel2020 import ExozodiModel
from lifesim.core.hwo_data import HWOData
from tools.physics_constants import h, c, k, sigma, R_earth, pc_to_m, R_sun


class TestBlackbodyPhysics(unittest.TestCase):
    """Test blackbody radiation calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.exozodi_model = ExozodiModel('baseline', self.rng)
        
    def test_planck_integration_full_range(self):
        """Test that Planck integration over full wavelength range gives 1.0."""
        temp = 300  # K
        min_wl = 1e-9  # 1 nm
        max_wl = 1e-3  # 1 mm (very wide range)
        
        fraction = self.exozodi_model._blackbody_flux_in_band(temp, min_wl, max_wl)
        
        # Should be very close to 1.0 for full wavelength range
        self.assertAlmostEqual(fraction, 1.0, places=2)
        
    def test_planck_integration_hwo_band(self):
        """Test Planck integration for HWO wavelength band."""
        temp = 300  # K
        min_wl = 200e-9  # 200 nm
        max_wl = 2500e-9  # 2.5 μm
        
        fraction = self.exozodi_model._blackbody_flux_in_band(temp, min_wl, max_wl)
        
        # Should be between 0 and 1
        self.assertGreater(fraction, 0)
        self.assertLess(fraction, 1)
        
    def test_planck_temperature_dependence(self):
        """Test that Planck integration shows correct temperature dependence."""
        min_wl = 200e-9  # 200 nm
        max_wl = 2500e-9  # 2.5 μm
        
        temps = [100, 300, 500, 1000]  # K
        fractions = []
        
        for temp in temps:
            fraction = self.exozodi_model._blackbody_flux_in_band(temp, min_wl, max_wl)
            fractions.append(fraction)
        
        # Higher temperatures should have higher fractions in HWO band
        # (since peak shifts to shorter wavelengths)
        for i in range(1, len(fractions)):
            self.assertGreater(fractions[i], fractions[i-1])
            
    def test_wien_displacement_law(self):
        """Test that Planck integration respects Wien's displacement law."""
        temp = 300  # K
        wien_peak = 2.898e-3 / temp  # Wien's displacement law in meters
        
        # Test bands around the peak
        bands = [
            (wien_peak * 0.1, wien_peak * 0.5),  # Below peak
            (wien_peak * 0.5, wien_peak * 2.0),  # Around peak
            (wien_peak * 2.0, wien_peak * 10.0), # Above peak
        ]
        
        fractions = []
        for min_wl, max_wl in bands:
            fraction = self.exozodi_model._blackbody_flux_in_band(temp, min_wl, max_wl)
            fractions.append(fraction)
        
        # Band around peak should have highest fraction
        self.assertGreater(fractions[1], fractions[0])  # Around peak > below peak
        self.assertGreater(fractions[1], fractions[2])  # Around peak > above peak
        
    def test_planck_numerical_stability(self):
        """Test numerical stability of Planck integration."""
        temp = 300  # K
        min_wl = 200e-9  # 200 nm
        max_wl = 2500e-9  # 2.5 μm
        
        # Run multiple times to check stability
        fractions = []
        for _ in range(10):
            fraction = self.exozodi_model._blackbody_flux_in_band(temp, min_wl, max_wl)
            fractions.append(fraction)
        
        # All results should be identical (within numerical precision)
        for i in range(1, len(fractions)):
            self.assertAlmostEqual(fractions[i], fractions[0], places=10)


class TestExozodiTemperature(unittest.TestCase):
    """Test exozodi temperature calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.exozodi_model = ExozodiModel('baseline', self.rng)
        
    def test_exozodi_temperature_radiative_equilibrium(self):
        """Test that exozodi temperature follows radiative equilibrium."""
        star_temp = 5778  # Solar temperature
        star_radius = 1.0  # Solar radius
        distance_pc = 10.0  # 10 parsecs
        
        # Test at different distances
        distances = [1, 5, 10, 20, 50]  # AU
        temperatures = []
        
        for dist in distances:
            temp = self.exozodi_model._calculate_exozodi_temperature(
                star_temp, star_radius, distance_pc, dist
            )
            temperatures.append(temp)
        
        # Temperature should decrease with distance (T ∝ r^(-1/2))
        for i in range(1, len(temperatures)):
            self.assertGreater(temperatures[i-1], temperatures[i])
            
    def test_exozodi_temperature_stellar_dependence(self):
        """Test that exozodi temperature depends on stellar properties."""
        distance_pc = 10.0
        distance_au = 10.0
        
        # Test different star types
        star_types = [
            (3000, 0.3),  # M-dwarf
            (4500, 0.7),  # K-dwarf
            (5778, 1.0),  # G-dwarf (Sun)
            (7000, 1.3),  # F-dwarf
            (9000, 1.8),  # A-dwarf
        ]
        
        temperatures = []
        for star_temp, star_radius in star_types:
            temp = self.exozodi_model._calculate_exozodi_temperature(
                star_temp, star_radius, distance_pc, distance_au
            )
            temperatures.append(temp)
        
        # Hotter stars should produce higher exozodi temperatures
        for i in range(1, len(temperatures)):
            self.assertGreater(temperatures[i], temperatures[i-1])
            
    def test_exozodi_temperature_bounds(self):
        """Test that exozodi temperature stays within reasonable bounds."""
        star_temp = 5778
        star_radius = 1.0
        distance_pc = 10.0
        
        # Test extreme distances
        distances = [0.1, 1, 10, 100, 1000]  # AU
        
        for dist in distances:
            temp = self.exozodi_model._calculate_exozodi_temperature(
                star_temp, star_radius, distance_pc, dist
            )
            
            # Should be within reasonable bounds
            self.assertGreater(temp, 50)   # Not too cold
            self.assertLess(temp, 1000)    # Not too hot
            
    def test_exozodi_temperature_profile(self):
        """Test exozodi temperature profile calculation."""
        star_temp = 5778
        star_radius = 1.0
        
        distances, temperatures = self.exozodi_model.getExozodiTemperatureProfileAtPlanetAU(
            star_temp, star_radius
        )
        
        # Should return arrays of same length
        self.assertEqual(len(distances), len(temperatures))
        
        # Temperatures should decrease with distance
        for i in range(1, len(temperatures)):
            self.assertGreater(temperatures[i-1], temperatures[i])


class TestExozodiFlux(unittest.TestCase):
    """Test exozodi flux calculations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rng = np.random.default_rng(42)
        self.exozodi_model = ExozodiModel('baseline', self.rng)
        
    def test_exozodi_flux_distance_dependence(self):
        """Test that exozodi flux decreases with distance from star."""
        star_temp = 5778
        star_radius = 1.0
        distance_pc = 10.0
        exozodi_level = 1.0
        
        distances = [1, 5, 10, 20, 50]  # AU
        fluxes = []
        
        for dist in distances:
            flux, _ = self.exozodi_model.getExozodiFluxAtPlanetDistance(
                star_temp, star_radius, distance_pc, dist, exozodi_level
            )
            fluxes.append(flux)
        
        # Flux should decrease with distance
        for i in range(1, len(fluxes)):
            self.assertGreater(fluxes[i-1], fluxes[i])
            
    def test_exozodi_flux_level_dependence(self):
        """Test that exozodi flux scales with exozodi level."""
        star_temp = 5778
        star_radius = 1.0
        distance_pc = 10.0
        planet_distance_au = 10.0
        
        levels = [0.1, 1.0, 10.0]
        fluxes = []
        
        for level in levels:
            flux, _ = self.exozodi_model.getExozodiFluxAtPlanetDistance(
                star_temp, star_radius, distance_pc, planet_distance_au, level
            )
            fluxes.append(flux)
        
        # Flux should scale linearly with exozodi level
        self.assertAlmostEqual(fluxes[1] / fluxes[0], 10.0, places=1)
        self.assertAlmostEqual(fluxes[2] / fluxes[1], 10.0, places=1)
        
    def test_exozodi_flux_ratio_calculation(self):
        """Test exozodi to planet flux ratio calculation."""
        star_temp = 5778
        star_radius = 1.0
        distance_pc = 10.0
        planet_radius = 1.0  # Earth radius
        planet_temp = 300  # K
        planet_distance_au = 1.0  # 1 AU
        
        ratio, exozodi_flux, planet_flux = self.exozodi_model.getExozodiFluxRatioAtPlanetDistance(
            star_temp, star_radius, distance_pc, planet_radius, planet_temp, 
            planet_distance_au
        )
        
        # Ratio should be positive
        self.assertGreater(ratio, 0)
        
        # Should match manual calculation
        expected_ratio = exozodi_flux / planet_flux
        self.assertAlmostEqual(ratio, expected_ratio, places=10)


class TestHWODataPhysics(unittest.TestCase):
    """Test physics calculations in HWOData class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample catalog
        np.random.seed(42)
        n_planets = 100
        
        data = {
            'temp_p': np.random.uniform(200, 600, n_planets),
            'temp_s': np.random.uniform(3000, 7000, n_planets),
            'radius_p': np.random.uniform(0.5, 2.0, n_planets),
            'radius_s': np.random.uniform(0.3, 1.5, n_planets),
            'distance_s': np.random.uniform(5, 50, n_planets),
            'maxangsep': np.random.uniform(0.1, 2.0, n_planets),
        }
        
        self.catalog = pd.DataFrame(data)
        self.hwo_data = HWOData(self.catalog)
        
    def test_blackbody_flux_calculation(self):
        """Test blackbody flux calculation."""
        wavelength = 500e-9  # 500 nm
        temperature = 300  # K
        
        flux = self.hwo_data.blackbody_flux(wavelength, temperature)
        
        # Should be positive
        self.assertGreater(flux, 0)
        
        # Should match theoretical value
        theoretical_flux = (2 * h * c**2) / (wavelength**5 * (np.exp(h * c / (wavelength * k * temperature)) - 1))
        self.assertAlmostEqual(flux, theoretical_flux, places=10)
        
    def test_planet_flux_calculation(self):
        """Test planet flux calculation."""
        # Test both cases
        for case in ['best', 'worst']:
            flux = self.hwo_data.calc_planet_flux(case)
            
            # Should return array of correct length
            self.assertEqual(len(flux), len(self.catalog))
            
            # All fluxes should be positive
            self.assertTrue(np.all(flux > 0))
            
    def test_flux_ratio_calculation(self):
        """Test flux ratio calculation."""
        # Test both cases
        for case in ['best', 'worst']:
            ratio = self.hwo_data.calc_flux_ratio(case)
            
            # Should return array of correct length
            self.assertEqual(len(ratio), len(self.catalog))
            
            # All ratios should be positive
            self.assertTrue(np.all(ratio > 0))
            
    def test_photon_calculation(self):
        """Test photon rate calculation."""
        # Test both cases
        for case in ['best', 'worst']:
            photons = self.hwo_data.calc_photons(case)
            
            # Should return array of correct length
            self.assertEqual(len(photons), len(self.catalog))
            
            # All photon rates should be positive
            self.assertTrue(np.all(photons > 0))
            
    def test_exozodi_constraint_calculation(self):
        """Test exozodi constraint calculation."""
        # Test both cases
        for case in ['best', 'worst']:
            # Use the new surface brightness criterion
            rejected, ratios = self.hwo_data.calc_exozodi_surface_brightness_constraint(case, 'baseline')
            
            # Should return array of correct length
            self.assertEqual(len(ratios), len(self.catalog))
            
            # All ratios should be finite
            self.assertTrue(np.all(np.isfinite(ratios)))
            
            # Rejected should be boolean array
            self.assertEqual(len(rejected), len(self.catalog))
            self.assertTrue(np.all(np.isfinite(rejected) | (rejected == True) | (rejected == False)))


class TestPhysicsConstants(unittest.TestCase):
    """Test physics constants."""
    
    def test_planck_constant(self):
        """Test Planck constant value."""
        self.assertAlmostEqual(h, constants.h, places=10)
        
    def test_speed_of_light(self):
        """Test speed of light value."""
        self.assertAlmostEqual(c, constants.c, places=10)
        
    def test_boltzmann_constant(self):
        """Test Boltzmann constant value."""
        self.assertAlmostEqual(k, constants.k, places=10)
        
    def test_stefan_boltzmann_constant(self):
        """Test Stefan-Boltzmann constant value."""
        self.assertAlmostEqual(sigma, constants.sigma, places=10)
        
    def test_earth_radius(self):
        """Test Earth radius value."""
        self.assertAlmostEqual(R_earth, 6.371e6, places=3)  # meters
        
    def test_solar_radius(self):
        """Test solar radius value."""
        self.assertAlmostEqual(R_sun, 6.957e8, places=3)  # meters


if __name__ == '__main__':
    unittest.main() 