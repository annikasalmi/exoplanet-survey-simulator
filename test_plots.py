#!/usr/bin/env python3
"""
Simple test script to verify that plotting functions work and display plots on screen.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_test_data():
    """Create some test data for plotting."""
    np.random.seed(42)
    n_planets = 1000
    
    # Create test DataFrame
    data = {
        'run': np.repeat(range(5), n_planets // 5),
        'stype': np.random.choice(['M', 'K', 'G', 'F'], n_planets),
        'radius_p': np.random.uniform(0.5, 6.0, n_planets),
        'distance_s': np.random.uniform(1, 15, n_planets),
        'temp_p': np.random.uniform(200, 500, n_planets),
        'habitable': np.random.choice([True, False], n_planets, p=[0.3, 0.7]),
        'maxangsep': np.random.uniform(0.01, 0.1, n_planets),
        'flux_ratio_value_best': np.random.uniform(1e-10, 1e-6, n_planets),
        'flux_ratio_value_worst': np.random.uniform(1e-11, 1e-7, n_planets),
        'photon_rate_value_best': np.random.uniform(1e3, 1e6, n_planets),
        'photon_rate_value_worst': np.random.uniform(1e2, 1e5, n_planets),
        'z': np.random.uniform(0.1, 10, n_planets),
        'p_orb': np.random.uniform(0.5, 500, n_planets),
        'mass_p': np.random.uniform(0.1, 10, n_planets),
    }
    
    df = pd.DataFrame(data)
    
    # Add detection flags (simplified)
    df['detected_best'] = np.random.choice([True, False], n_planets, p=[0.4, 0.6])
    df['detected_worst'] = np.random.choice([True, False], n_planets, p=[0.2, 0.8])
    
    # Add pass/fail columns
    df['flux_pass_best'] = df['flux_ratio_value_best'] > 1e-8
    df['flux_pass_worst'] = df['flux_ratio_value_worst'] > 1e-9
    df['min_photons_pass_best'] = df['photon_rate_value_best'] > 1e4
    df['min_photons_pass_worst'] = df['photon_rate_value_worst'] > 1e3
    df['iwa_pass_best'] = df['maxangsep'] > 0.02
    df['iwa_pass_worst'] = df['maxangsep'] > 0.03
    df['z_pass_best'] = df['z'] <= 5
    df['z_pass_worst'] = df['z'] <= 2
    
    return df

def test_plots():
    """Test the plotting functions."""
    print("Creating test data...")
    df = create_test_data()
    
    print("Testing plot_by_type...")
    from plot.plot_by_type import PlotPlanetType
    plotter = PlotPlanetType(df=df, nruns=5, star_catalog='Test', name='HWO')
    
    # Test individual plots
    print("Testing plot_by_star...")
    plotter.plot_by_star()
    
    print("Testing plot_by_planet...")
    plotter.plot_by_planet()
    
    print("Testing plot_distances...")
    plotter.plot_distances()
    
    print("Testing plot_detections...")
    from plot.plot_detections import PlanetDetectionPlotter
    detection_plotter = PlanetDetectionPlotter(df=df, nruns=5, star_catalog='Test', name='HWO')
    detection_plotter.plot_efficiency_multipanel()
    
    print("Testing plot_rejections...")
    from plot.plot_rejections import PlanetRejectionPlotter
    rejection_plotter = PlanetRejectionPlotter(df=df, nruns=5, star_catalog='Test', name='HWO')
    rejection_plotter.plot_failures_percentages()
    rejection_plotter.plot_failures_histogram()
    
    print("Testing M-dwarf HZ limits...")
    from plot.plot_mdwarf_hz_limits import PlotMdwarfHZLimits
    mdwarf_plotter = PlotMdwarfHZLimits(df=df, nruns=5, star_catalog='Test', name='HWO')
    mdwarf_plotter.plot_all()
    
    print("All plots completed! You should have seen multiple plot windows appear.")
    print("Press Enter to close all plots...")
    input()

if __name__ == "__main__":
    test_plots() 