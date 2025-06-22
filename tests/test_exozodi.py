#!/usr/bin/env python3
"""
Simple test script for exozodi functionality.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from lifesim.core.hwo_data import HWOData


def test_exozodi_basic():
    """Test basic exozodi functionality."""
    print("Testing basic exozodi functionality...")
    
    # Create simple test data
    test_data = pd.DataFrame({
        'temp_p': [300, 350, 400],
        'temp_s': [5000, 5500, 6000],
        'radius_p': [1.0, 1.5, 2.0],
        'radius_s': [1.0, 1.0, 1.0],
        'distance_s': [10, 20, 30],
        'maxangsep': [0.1, 0.2, 0.3],
    })
    
    # Initialize HWO data
    hwo_data = HWOData(test_data)
    
    # Test exozodi level calculation
    try:
        exozodi_levels = hwo_data.calc_exozodi_constraint('baseline')
        print(f"✓ Exozodi levels calculated: {exozodi_levels}")
        print(f"  Range: {exozodi_levels.min():.2e} to {exozodi_levels.max():.2e}")
    except Exception as e:
        print(f"✗ Error calculating exozodi levels: {e}")
        return False
    
    # Test SNR impact calculation
    try:
        snr_impact = hwo_data.calc_exozodi_snr_impact('best', exozodi_levels)
        print(f"✓ SNR impact calculated: {snr_impact}")
        print(f"  Range: {snr_impact.min():.3f} to {snr_impact.max():.3f}")
    except Exception as e:
        print(f"✗ Error calculating SNR impact: {e}")
        return False
    
    # Test detection threshold calculation
    try:
        threshold = hwo_data.calc_exozodi_detection_threshold('best', exozodi_levels)
        print(f"✓ Detection threshold calculated: {threshold}")
        print(f"  Range: {threshold.min():.2e} to {threshold.max():.2e}")
    except Exception as e:
        print(f"✗ Error calculating detection threshold: {e}")
        return False
    
    return True


def test_exozodi_detection():
    """Test exozodi-inclusive detection analysis."""
    print("\nTesting exozodi-inclusive detection analysis...")
    
    # Create test data
    test_data = pd.DataFrame({
        'temp_p': [300, 350, 400, 250, 450],
        'temp_s': [5000, 5500, 6000, 4000, 7000],
        'radius_p': [1.0, 1.5, 2.0, 0.8, 2.5],
        'radius_s': [1.0, 1.0, 1.0, 0.8, 1.2],
        'distance_s': [10, 20, 30, 15, 40],
        'maxangsep': [0.1, 0.2, 0.3, 0.05, 0.4],
    })
    
    # Initialize HWO data
    hwo_data = HWOData(test_data)
    
    try:
        # Test detection without exozodi
        catalog_no_exozodi = hwo_data.determine_detectable()
        print(f"✓ Detection without exozodi: {catalog_no_exozodi['detected_best'].sum()}/{len(catalog_no_exozodi)} planets detected")
        
        # Test detection with exozodi
        catalog_with_exozodi = hwo_data.determine_detectable_with_exozodi('baseline')
        print(f"✓ Detection with exozodi: {catalog_with_exozodi['detected_with_exozodi_best'].sum()}/{len(catalog_with_exozodi)} planets detected")
        
        # Check if exozodi levels were added
        if 'exozodi_level' in catalog_with_exozodi.columns:
            print(f"✓ Exozodi levels added to catalog")
        else:
            print("✗ Exozodi levels not found in catalog")
            return False
            
        # Check if lost planets were calculated
        if 'lost_to_exozodi_best' in catalog_with_exozodi.columns:
            lost_planets = catalog_with_exozodi['lost_to_exozodi_best'].sum()
            print(f"✓ Planets lost to exozodi: {lost_planets}")
        else:
            print("✗ Lost planets calculation not found")
            return False
            
    except Exception as e:
        print(f"✗ Error in detection analysis: {e}")
        return False
    
    return True


def test_exozodi_scenarios():
    """Test different exozodi scenarios."""
    print("\nTesting different exozodi scenarios...")
    
    # Create test data
    test_data = pd.DataFrame({
        'temp_p': [300, 350, 400],
        'temp_s': [5000, 5500, 6000],
        'radius_p': [1.0, 1.5, 2.0],
        'radius_s': [1.0, 1.0, 1.0],
        'distance_s': [10, 20, 30],
        'maxangsep': [0.1, 0.2, 0.3],
    })
    
    scenarios = ['baseline', 'pessimistic', 'optimistic']
    
    for scenario in scenarios:
        try:
            hwo_data = HWOData(test_data.copy())
            exozodi_levels = hwo_data.calc_exozodi_constraint(scenario)
            print(f"✓ {scenario.capitalize()} scenario: {exozodi_levels.mean():.2e} (mean)")
        except Exception as e:
            print(f"✗ Error with {scenario} scenario: {e}")
            return False
    
    return True


def main():
    """Run all tests."""
    print("Exozodi Functionality Tests")
    print("=" * 40)
    
    tests = [
        test_exozodi_basic,
        test_exozodi_detection,
        test_exozodi_scenarios,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed.")
    
    return passed == total


if __name__ == "__main__":
    main() 