#!/usr/bin/env python3
"""
Quick script to check temperature range in data files.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def check_temp_range(filename):
    """Check temperature range in a data file."""
    print(f"\n=== Checking {filename} ===")
    
    try:
        # First check what keys are available
        with pd.HDFStore(filename, 'r') as store:
            print(f"Available keys: {store.keys()}")
            
        # Try to read the first available key
        with pd.HDFStore(filename, 'r') as store:
            keys = store.keys()
            if keys:
                key = keys[0]  # Use the first available key
                print(f"Using key: {key}")
                df = store[key]
                
                print(f"Data shape: {df.shape}")
                print(f"Temperature range: {df['temp_p'].min():.2f} - {df['temp_p'].max():.2f} K")
                print(f"Radius range: {df['radius_p'].min():.2f} - {df['radius_p'].max():.2f} R⊕")
                
                # Check temperature distribution
                temp_bins = np.linspace(100, 400, 31)
                temp_hist, _ = np.histogram(df['temp_p'], bins=temp_bins)
                print(f"Temperature distribution:")
                for i, count in enumerate(temp_hist):
                    if count > 0:
                        temp_range = f"{temp_bins[i]:.0f}-{temp_bins[i+1]:.0f}K"
                        print(f"  {temp_range}: {count} planets")
                
                return df
            else:
                print("No keys found in HDF5 file")
                return None
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

if __name__ == "__main__":
    # Check a few data files
    data_files = [
        "run/lifesim/data/Gaia/test_runs_495_catalog.hdf5",
        "run/lifesim/data/Gaia/test_runs_463_catalog.hdf5",
        "run/lifesim/data/Gaia/test_runs_479_catalog.hdf5"
    ]
    
    for filename in data_files:
        df = check_temp_range(filename)
        if df is not None:
            break  # Just check the first successful one 