#!/usr/bin/env python3
"""
Debug script to investigate white spots in the temperature/radius plot.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from plot.helpers import get_detection_masks

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def debug_white_spots(filename):
    """Debug what's causing white spots in the temperature/radius plot."""
    print(f"\n=== Debugging white spots in {filename} ===")
    
    try:
        # Read the data
        with pd.HDFStore(filename, 'r') as store:
            df = store['/catalog']
        
        print(f"Data shape: {df.shape}")
        
        # Define temperature and radius ranges (same as in plot)
        temp_range = (125, 500)
        radius_range = (0, 15)
        temp_bins = np.linspace(temp_range[0], temp_range[1], 40)
        radius_bins = np.linspace(radius_range[0], radius_range[1], 30)
        
        # Calculate total counts in each 2D bin
        total_counts, _, _ = np.histogram2d(df['temp_p'], df['radius_p'], bins=[temp_bins, radius_bins])
        
        # Get detection masks (using the same logic as get_detection_masks)
        mask_best, mask_worst = get_detection_masks(df, 'HWO')  # Assuming HWO for this test
        
        # Calculate detected counts
        detected_counts, _, _ = np.histogram2d(df[mask_best]['temp_p'], df[mask_best]['radius_p'], 
                                              bins=[temp_bins, radius_bins])
        
        # Create mask for zero detection bins
        zero_detection_mask = (total_counts > 0) & (detected_counts == 0)
        
        print(f"Total bins: {total_counts.size}")
        print(f"Bins with planets: {np.sum(total_counts > 0)}")
        print(f"Bins with detected planets: {np.sum(detected_counts > 0)}")
        print(f"Zero detection bins: {np.sum(zero_detection_mask)}")
        print(f"Empty bins (no planets): {np.sum(total_counts == 0)}")
        
        # Check for any edge cases
        print(f"\nEdge cases:")
        print(f"Bins with total > 0 but detected = 0: {np.sum(zero_detection_mask)}")
        print(f"Bins with total = 0 but detected > 0: {np.sum((total_counts == 0) & (detected_counts > 0))}")
        
        # Find specific white spots (bins that should be colored but aren't)
        if np.sum(zero_detection_mask) > 0:
            print(f"\nSample zero detection bins (should be dark red):")
            zero_indices = np.where(zero_detection_mask)
            for i in range(min(5, len(zero_indices[0]))):
                temp_idx, radius_idx = zero_indices[0][i], zero_indices[1][i]
                temp_val = temp_bins[temp_idx]
                radius_val = radius_bins[radius_idx]
                total_val = total_counts[temp_idx, radius_idx]
                detected_val = detected_counts[temp_idx, radius_idx]
                print(f"  Bin ({temp_idx}, {radius_idx}): temp={temp_val:.1f}K, radius={radius_val:.1f}R⊕, total={total_val}, detected={detected_val}")
        
        # Check if there are any NaN or inf values
        print(f"\nData quality check:")
        print(f"NaN in total_counts: {np.sum(np.isnan(total_counts))}")
        print(f"NaN in detected_counts: {np.sum(np.isnan(detected_counts))}")
        print(f"Inf in total_counts: {np.sum(np.isinf(total_counts))}")
        print(f"Inf in detected_counts: {np.sum(np.isinf(detected_counts))}")
        
        return df
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

if __name__ == "__main__":
    # Check a data file
    data_file = "run/lifesim/data/Gaia/test_runs_495_catalog.hdf5"
    df = debug_white_spots(data_file) 