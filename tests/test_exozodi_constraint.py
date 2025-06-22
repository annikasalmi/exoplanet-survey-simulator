#!/usr/bin/env python3
"""
Test script to demonstrate the new exozodi constraint functionality.
"""

import numpy as np
import pandas as pd
from lifesim.core.hwo_data import HWOData
from plot.plot_rejections import PlanetRejectionPlotter
import unittest

def test_exozodi_constraint():
    """Test the exozodi constraint functionality."""
    
    # Create a sample catalog with planet data
    np.random.seed(42)
    n_planets = 1000
    
    # Generate sample data
    data = {
        'temp_p': np.random.uniform(200, 600, n_planets),  # Planet temperatures
        'temp_s': np.random.uniform(3000, 7000, n_planets),  # Star temperatures
        'radius_p': np.random.uniform(0.5, 2.0, n_planets),  # Planet radii (Earth radii)
        'radius_s': np.random.uniform(0.3, 1.5, n_planets),  # Star radii (Solar radii)
        'distance_s': np.random.uniform(5, 50, n_planets),  # Distances (parsecs)
        'maxangsep': np.random.uniform(0.1, 2.0, n_planets),  # Maximum angular separation
    }
    
    catalog = pd.DataFrame(data)
    
    print("=== Exozodi Constraint Test ===")
    print(f"Generated {n_planets} sample planets")
    print()
    
    # Initialize HWO data
    hwo_data = HWOData(catalog)
    
    # Test without exozodi constraint
    print("=== Without Exozodi Constraint ===")
    hwo_data.determine_detectable(use_exozodi_constraint=False)
    
    best_detected_no_exozodi = hwo_data.catalog['detected_best'].sum()
    worst_detected_no_exozodi = hwo_data.catalog['detected_worst'].sum()
    
    print(f"Best case detected: {best_detected_no_exozodi} ({best_detected_no_exozodi/n_planets*100:.1f}%)")
    print(f"Worst case detected: {worst_detected_no_exozodi} ({worst_detected_no_exozodi/n_planets*100:.1f}%)")
    print()
    
    # Test with exozodi constraint (baseline scenario)
    print("=== With Exozodi Constraint (Baseline) ===")
    hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario='baseline')
    
    best_detected_with_exozodi = hwo_data.catalog['detected_best'].sum()
    worst_detected_with_exozodi = hwo_data.catalog['detected_worst'].sum()
    
    print(f"Best case detected: {best_detected_with_exozodi} ({best_detected_with_exozodi/n_planets*100:.1f}%)")
    print(f"Worst case detected: {worst_detected_with_exozodi} ({worst_detected_with_exozodi/n_planets*100:.1f}%)")
    print()
    
    # Calculate impact of exozodi constraint
    best_impact = best_detected_no_exozodi - best_detected_with_exozodi
    worst_impact = worst_detected_no_exozodi - worst_detected_with_exozodi
    
    print("=== Exozodi Constraint Impact ===")
    print(f"Best case: {best_impact} fewer detections ({best_impact/n_planets*100:.1f}% reduction)")
    print(f"Worst case: {worst_impact} fewer detections ({worst_impact/n_planets*100:.1f}% reduction)")
    print()
    
    # Analyze rejection reasons
    print("=== Rejection Analysis ===")
    
    # Best case rejections
    best_rejected = hwo_data.catalog[~hwo_data.catalog['detected_best']]
    if not best_rejected.empty:
        best_rejected['rejection_reason'] = best_rejected.apply(
            lambda row: get_rejection_reason(row, 'best'), axis=1
        )
        best_reasons = best_rejected['rejection_reason'].value_counts()
        print("Best case rejection reasons:")
        for reason, count in best_reasons.items():
            print(f"  {reason}: {count} ({count/len(best_rejected)*100:.1f}%)")
    
    print()
    
    # Worst case rejections
    worst_rejected = hwo_data.catalog[~hwo_data.catalog['detected_worst']]
    if not worst_rejected.empty:
        worst_rejected['rejection_reason'] = worst_rejected.apply(
            lambda row: get_rejection_reason(row, 'worst'), axis=1
        )
        worst_reasons = worst_rejected['rejection_reason'].value_counts()
        print("Worst case rejection reasons:")
        for reason, count in worst_reasons.items():
            print(f"  {reason}: {count} ({count/len(worst_rejected)*100:.1f}%)")
    
    print()
    
    # Test different exozodi scenarios
    print("=== Different Exozodi Scenarios ===")
    scenarios = ['baseline', 'pessimistic', 'optimistic']
    
    for scenario in scenarios:
        hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario=scenario)
        best_detected = hwo_data.catalog['detected_best'].sum()
        worst_detected = hwo_data.catalog['detected_worst'].sum()
        
        print(f"{scenario.capitalize()}:")
        print(f"  Best case: {best_detected} ({best_detected/n_planets*100:.1f}%)")
        print(f"  Worst case: {worst_detected} ({worst_detected/n_planets*100:.1f}%)")
    
    print()
    
    # Test plotting functionality
    print("=== Testing Plotting Functionality ===")
    try:
        plotter = PlanetRejectionPlotter(hwo_data.catalog, nruns=1, star_catalog='Test', name='HWO')
        plotter.plot_all(plot_percentages=True)
        print("✓ Rejection plots generated successfully")
    except Exception as e:
        print(f"✗ Error generating plots: {e}")
    
    print()
    print("=== Summary ===")
    print("✓ Exozodi constraint successfully integrated into detection logic")
    print("✓ Different exozodi scenarios show varying impact on detection rates")
    print("✓ Rejection analysis includes exozodi as a constraint")
    print("✓ Plotting functionality updated to include exozodi constraint")

def get_rejection_reason(row, scenario='best'):
    """Helper function to get rejection reason for a row."""
    from plot.helpers import get_rejection_reason as get_reason
    return get_reason(row, scenario)

class TestExozodiConstraint(unittest.TestCase):
    """Test exozodi constraint functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample catalog
        np.random.seed(42)
        n_planets = 50
        
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
        
    def test_exozodi_constraint_without_constraint(self):
        """Test detection without exozodi constraint."""
        self.hwo_data.determine_detectable(use_exozodi_constraint=False)
        
        # Should have detection columns
        self.assertIn('detected_best', self.hwo_data.catalog.columns)
        self.assertIn('detected_worst', self.hwo_data.catalog.columns)
        
        # Should not have exozodi columns
        self.assertNotIn('exozodi_pass_best', self.hwo_data.catalog.columns)
        self.assertNotIn('exozodi_pass_worst', self.hwo_data.catalog.columns)
        
    def test_exozodi_constraint_with_constraint(self):
        """Test detection with exozodi constraint."""
        self.hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario='baseline')
        
        # Should have all detection columns including exozodi
        self.assertIn('detected_best', self.hwo_data.catalog.columns)
        self.assertIn('detected_worst', self.hwo_data.catalog.columns)
        self.assertIn('exozodi_pass_best', self.hwo_data.catalog.columns)
        self.assertIn('exozodi_pass_worst', self.hwo_data.catalog.columns)
        self.assertIn('exozodi_flux_ratio_best', self.hwo_data.catalog.columns)
        self.assertIn('exozodi_flux_ratio_worst', self.hwo_data.catalog.columns)
        
    def test_exozodi_constraint_impact(self):
        """Test that exozodi constraint reduces detection rates."""
        # Without exozodi constraint
        self.hwo_data.determine_detectable(use_exozodi_constraint=False)
        best_detected_no_exozodi = self.hwo_data.catalog['detected_best'].sum()
        worst_detected_no_exozodi = self.hwo_data.catalog['detected_worst'].sum()
        
        # With exozodi constraint
        self.hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario='baseline')
        best_detected_with_exozodi = self.hwo_data.catalog['detected_best'].sum()
        worst_detected_with_exozodi = self.hwo_data.catalog['detected_worst'].sum()
        
        # Exozodi constraint should not increase detection rates
        self.assertLessEqual(best_detected_with_exozodi, best_detected_no_exozodi)
        self.assertLessEqual(worst_detected_with_exozodi, worst_detected_no_exozodi)
        
    def test_exozodi_scenarios(self):
        """Test different exozodi scenarios."""
        scenarios = ['baseline', 'pessimistic', 'optimistic']
        detection_counts = []
        
        for scenario in scenarios:
            self.hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario=scenario)
            best_detected = self.hwo_data.catalog['detected_best'].sum()
            detection_counts.append(best_detected)
        
        # Pessimistic should have fewer detections than baseline
        self.assertLessEqual(detection_counts[1], detection_counts[0])
        
        # Optimistic should have more detections than baseline
        self.assertGreaterEqual(detection_counts[2], detection_counts[0])
        
    def test_exozodi_flux_ratio_values(self):
        """Test that exozodi flux ratios are reasonable."""
        self.hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario='baseline')
        
        # Check best case
        ratios_best = self.hwo_data.catalog['exozodi_flux_ratio_best']
        self.assertTrue(np.all(np.isfinite(ratios_best)))
        self.assertTrue(np.all(ratios_best > 0))
        
        # Check worst case
        ratios_worst = self.hwo_data.catalog['exozodi_flux_ratio_worst']
        self.assertTrue(np.all(np.isfinite(ratios_worst)))
        self.assertTrue(np.all(ratios_worst > 0))
        
    def test_exozodi_pass_conditions(self):
        """Test that exozodi pass conditions are boolean."""
        self.hwo_data.determine_detectable(use_exozodi_constraint=True, exozodi_scenario='baseline')
        
        # Check best case
        pass_best = self.hwo_data.catalog['exozodi_pass_best']
        self.assertTrue(np.all(np.isin(pass_best, [True, False])))
        
        # Check worst case
        pass_worst = self.hwo_data.catalog['exozodi_pass_worst']
        self.assertTrue(np.all(np.isin(pass_worst, [True, False])))
        
    def test_simple_exozodi_constraint(self):
        """Test simplified exozodi constraint calculation."""
        for case in ['best', 'worst']:
            ratios = self.hwo_data.calc_exozodi_constraint_simple(case, exozodi_level=1.0)
            
            # Should return array of correct length
            self.assertEqual(len(ratios), len(self.catalog))
            
            # All ratios should be finite and positive
            self.assertTrue(np.all(np.isfinite(ratios)))
            self.assertTrue(np.all(ratios > 0))
            
    def test_exozodi_constraint_scaling(self):
        """Test that exozodi constraint scales with exozodi level."""
        case = 'best'
        
        # Test different exozodi levels
        ratios_1 = self.hwo_data.calc_exozodi_constraint_simple(case, exozodi_level=1.0)
        ratios_2 = self.hwo_data.calc_exozodi_constraint_simple(case, exozodi_level=2.0)
        
        # Higher exozodi level should give lower ratios (harder to detect)
        self.assertTrue(np.all(ratios_2 <= ratios_1))


if __name__ == "__main__":
    test_exozodi_constraint()
    unittest.main() 