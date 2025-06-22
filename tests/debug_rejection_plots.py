#!/usr/bin/env python3
"""
Debug script to check why flux_ratio and min_detected plots are blank in the failure_multipanel plot.
"""

import pandas as pd
import numpy as np
from lifesim.core.hwo_data import HWOData
from plot.plot_rejections import PlanetRejectionPlotter

def create_sample_data():
    """Create sample data to test the rejection plotting."""
    # Create sample planet data
    np.random.seed(42)
    n_planets = 1000
    
    data = {
        'temp_p': np.random.uniform(200, 400, n_planets),  # Planet temperatures
        'temp_s': np.random.uniform(3000, 7000, n_planets),  # Star temperatures
        'radius_p': np.random.uniform(0.5, 2.0, n_planets),  # Planet radii (Earth radii)
        'radius_s': np.random.uniform(0.5, 2.0, n_planets),  # Star radii (Solar radii)
        'distance_s': np.random.uniform(1, 20, n_planets),  # Distance (pc)
        'maxangsep': np.random.uniform(0.01, 0.1, n_planets),  # Angular separation (arcsec)
        'run': np.random.randint(0, 3, n_planets),  # Run number
        'stype': np.random.choice(['M', 'K', 'G', 'F'], n_planets),  # Star type
        'radius_bin': np.random.choice(['0.5-1.0', '1.0-1.5', '1.5-2.0'], n_planets),  # Radius bin
    }
    
    df = pd.DataFrame(data)
    return df

def main():
    print("Creating sample data...")
    df = create_sample_data()
    
    print("Initializing HWO data...")
    hwo_data = HWOData(df)
    
    print("Determining detectable planets...")
    result_df = hwo_data.determine_detectable()
    
    print(f"DataFrame shape: {result_df.shape}")
    print(f"Columns: {list(result_df.columns)}")
    
    # Check for key columns
    key_columns = [
        'detected_best', 'detected_worst',
        'flux_ratio_value_best', 'flux_ratio_value_worst',
        'photon_rate_value_best', 'photon_rate_value_worst',
        'flux_pass_best', 'flux_pass_worst',
        'min_photons_pass_best', 'min_photons_pass_worst',
        'iwa_pass_best', 'iwa_pass_worst'
    ]
    
    print("\nChecking key columns:")
    for col in key_columns:
        exists = col in result_df.columns
        print(f"  {col}: {'✓' if exists else '✗'}")
        if exists:
            print(f"    - Non-null values: {result_df[col].notna().sum()}")
            if result_df[col].dtype == bool:
                print(f"    - True values: {result_df[col].sum()}")
            else:
                print(f"    - Range: {result_df[col].min():.2e} to {result_df[col].max():.2e}")
    
    # Check detection statistics
    print(f"\nDetection statistics:")
    print(f"  Total planets: {len(result_df)}")
    print(f"  Detected (best): {result_df['detected_best'].sum()}")
    print(f"  Detected (worst): {result_df['detected_worst'].sum()}")
    print(f"  Not detected (best): {(~result_df['detected_best']).sum()}")
    print(f"  Not detected (worst): {(~result_df['detected_worst']).sum()}")
    
    # Test rejection plotting
    print("\nTesting rejection plotting...")
    try:
        plotter = PlanetRejectionPlotter(result_df, nruns=3, star_catalog='Test', name='HWO')
        plotter.plot_failures_histogram()
        print("✓ Histogram plotting completed")
    except Exception as e:
        print(f"✗ Error in histogram plotting: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 