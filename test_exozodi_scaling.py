"""
Test exozodi constraint without scaling factor.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from lifesim.core.hwo_data import HWOData


def test_exozodi_constraint():
    """Test exozodi constraint without scaling factor."""
    
    # Load data
    file_path = 'run/hwo/data/Gaia/hwo_catalog_0.csv'
    df = pd.read_csv(file_path)
    print(f"✅ Loaded data from: {file_path}")
    print(f"   Shape: {df.shape}")
    
    # Create HWOData object
    hwo_data = HWOData(df)
    
    # Test different exozodi scenarios
    scenarios = ['baseline', 'pessimistic', 'optimistic']
    
    print("\n" + "="*60)
    print("TESTING EXOZODI SCENARIOS (NO SCALING FACTOR)")
    print("="*60)
    
    for scenario in scenarios:
        print(f"\nTesting scenario: {scenario}")
        
        try:
            # Run detection with this scenario
            result = hwo_data.determine_detectable(
                use_exozodi_constraint=True,
                exozodi_scenario=scenario,
                ignore_exozodi_rejections=False
            )
            
            # Count rejections
            if 'exozodi_surface_brightness_rejected_best' in result.columns:
                rejected_best = result['exozodi_surface_brightness_rejected_best'].sum()
                rejected_worst = result['exozodi_surface_brightness_rejected_worst'].sum()
                total = len(result)
                
                print(f"  Best case rejected: {rejected_best}/{total} ({rejected_best/total*100:.1f}%)")
                print(f"  Worst case rejected: {rejected_worst}/{total} ({rejected_worst/total*100:.1f}%)")
                
                # Show some ratio statistics
                ratios_best = result['exozodi_surface_brightness_ratio_best']
                ratios_worst = result['exozodi_surface_brightness_ratio_worst']
                print(f"  Best case ratios - Min: {ratios_best.min():.2e}, Max: {ratios_best.max():.2e}")
                print(f"  Worst case ratios - Min: {ratios_worst.min():.2e}, Max: {ratios_worst.max():.2e}")
            else:
                print(f"  No rejection column found")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("1. If no planets are rejected even without scaling, the exozodi model may be producing very low values")
    print("2. The instrument contrast limit might be very high, making the criterion easy to pass")
    print("3. Consider checking the exozodi model parameters or using a different approach")


if __name__ == "__main__":
    test_exozodi_constraint() 