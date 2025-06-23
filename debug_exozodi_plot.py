"""
Debug script to investigate why the exozodi plot is empty.
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from lifesim.core.hwo_data import HWOData


def debug_exozodi_data(df):
    """Debug exozodi data to understand why plots are empty."""
    
    print("="*60)
    print("EXOZODI DATA DEBUGGING")
    print("="*60)
    
    # Check if exozodi columns exist
    exozodi_columns = [
        'exozodi_surface_brightness_ratio_best',
        'exozodi_surface_brightness_ratio_worst',
        'exozodi_surface_brightness_rejected_best',
        'exozodi_surface_brightness_rejected_worst',
        'exozodi_pass_best',
        'exozodi_pass_worst'
    ]
    
    print("\n1. Checking exozodi columns:")
    for col in exozodi_columns:
        if col in df.columns:
            print(f"  ✅ {col}: exists")
        else:
            print(f"  ❌ {col}: missing")
    
    # Check if any exozodi columns exist
    existing_exozodi_cols = [col for col in exozodi_columns if col in df.columns]
    if not existing_exozodi_cols:
        print("\n❌ No exozodi columns found! This means exozodi constraint was not applied.")
        print("   Make sure to run determine_detectable() with use_exozodi_constraint=True")
        return
    
    print(f"\n✅ Found {len(existing_exozodi_cols)} exozodi columns")
    
    # Analyze surface brightness ratios
    if 'exozodi_surface_brightness_ratio_best' in df.columns:
        ratios_best = df['exozodi_surface_brightness_ratio_best']
        print(f"\n2. Surface brightness ratios (best case):")
        print(f"   Min: {ratios_best.min():.2e}")
        print(f"   Max: {ratios_best.max():.2e}")
        print(f"   Mean: {ratios_best.mean():.2e}")
        print(f"   Median: {ratios_best.median():.2e}")
        print(f"   NaN values: {ratios_best.isna().sum()}")
        print(f"   Infinite values: {np.isinf(ratios_best).sum()}")
        
        # Check if any ratios are above threshold (1.0)
        above_threshold = (ratios_best >= 1.0).sum()
        print(f"   Ratios >= 1.0 (should be rejected): {above_threshold}/{len(ratios_best)} ({above_threshold/len(ratios_best)*100:.1f}%)")
    
    if 'exozodi_surface_brightness_ratio_worst' in df.columns:
        ratios_worst = df['exozodi_surface_brightness_ratio_worst']
        print(f"\n3. Surface brightness ratios (worst case):")
        print(f"   Min: {ratios_worst.min():.2e}")
        print(f"   Max: {ratios_worst.max():.2e}")
        print(f"   Mean: {ratios_worst.mean():.2e}")
        print(f"   Median: {ratios_worst.median():.2e}")
        print(f"   NaN values: {ratios_worst.isna().sum()}")
        print(f"   Infinite values: {np.isinf(ratios_worst).sum()}")
        
        # Check if any ratios are above threshold (1.0)
        above_threshold = (ratios_worst >= 1.0).sum()
        print(f"   Ratios >= 1.0 (should be rejected): {above_threshold}/{len(ratios_worst)} ({above_threshold/len(ratios_worst)*100:.1f}%)")
    
    # Check rejection flags
    if 'exozodi_surface_brightness_rejected_best' in df.columns:
        rejected_best = df['exozodi_surface_brightness_rejected_best']
        print(f"\n4. Rejection flags (best case):")
        print(f"   True (rejected): {rejected_best.sum()}/{len(rejected_best)} ({rejected_best.sum()/len(rejected_best)*100:.1f}%)")
        print(f"   False (not rejected): {(~rejected_best).sum()}/{len(rejected_best)} ({(~rejected_best).sum()/len(rejected_best)*100:.1f}%)")
    
    if 'exozodi_surface_brightness_rejected_worst' in df.columns:
        rejected_worst = df['exozodi_surface_brightness_rejected_worst']
        print(f"\n5. Rejection flags (worst case):")
        print(f"   True (rejected): {rejected_worst.sum()}/{len(rejected_worst)} ({rejected_worst.sum()/len(rejected_worst)*100:.1f}%)")
        print(f"   False (not rejected): {(~rejected_worst).sum()}/{len(rejected_worst)} ({(~rejected_worst).sum()/len(rejected_worst)*100:.1f}%)")
    
    # Check pass flags
    if 'exozodi_pass_best' in df.columns:
        pass_best = df['exozodi_pass_best']
        print(f"\n6. Pass flags (best case):")
        print(f"   True (pass): {pass_best.sum()}/{len(pass_best)} ({pass_best.sum()/len(pass_best)*100:.1f}%)")
        print(f"   False (fail): {(~pass_best).sum()}/{len(pass_best)} ({(~pass_best).sum()/len(pass_best)*100:.1f}%)")
    
    if 'exozodi_pass_worst' in df.columns:
        pass_worst = df['exozodi_pass_worst']
        print(f"\n7. Pass flags (worst case):")
        print(f"   True (pass): {pass_worst.sum()}/{len(pass_worst)} ({pass_worst.sum()/len(pass_worst)*100:.1f}%)")
        print(f"   False (fail): {(~pass_worst).sum()}/{len(pass_worst)} ({(~pass_worst).sum()/len(pass_worst)*100:.1f}%)")
    
    # Check overall detection
    if 'detected_best' in df.columns:
        detected_best = df['detected_best']
        print(f"\n8. Overall detection (best case):")
        print(f"   Detected: {detected_best.sum()}/{len(detected_best)} ({detected_best.sum()/len(detected_best)*100:.1f}%)")
        print(f"   Not detected: {(~detected_best).sum()}/{len(detected_best)} ({(~detected_best).sum()/len(detected_best)*100:.1f}%)")
    
    if 'detected_worst' in df.columns:
        detected_worst = df['detected_worst']
        print(f"\n9. Overall detection (worst case):")
        print(f"   Detected: {detected_worst.sum()}/{len(detected_worst)} ({detected_worst.sum()/len(detected_worst)*100:.1f}%)")
        print(f"   Not detected: {(~detected_worst).sum()}/{len(detected_worst)} ({(~detected_worst).sum()/len(detected_worst)*100:.1f}%)")
    
    # Check scaling factor M
    print(f"\n10. Scaling factor analysis:")
    print(f"    The exozodi fluxes are multiplied by M=0.1 (10% of original)")
    print(f"    This means exozodi surface brightness ratios are reduced by 90%")
    print(f"    If original ratios were mostly < 10, they would now be < 1.0")
    print(f"    This could explain why no planets are being rejected!")
    
    # Sample some data
    print(f"\n11. Sample data (first 5 planets):")
    if 'exozodi_surface_brightness_ratio_best' in df.columns:
        print(f"    Surface brightness ratios (best): {df['exozodi_surface_brightness_ratio_best'].head().values}")
    if 'exozodi_surface_brightness_rejected_best' in df.columns:
        print(f"    Rejected (best): {df['exozodi_surface_brightness_rejected_best'].head().values}")
    if 'exozodi_pass_best' in df.columns:
        print(f"    Pass (best): {df['exozodi_pass_best'].head().values}")


def test_different_scaling_factors(df):
    """Test how different scaling factors affect exozodi rejection."""
    
    print("\n" + "="*60)
    print("TESTING DIFFERENT SCALING FACTORS")
    print("="*60)
    
    # Create HWOData object
    hwo_data = HWOData(df)
    
    scaling_factors = [0.01, 0.1, 1.0, 10.0]
    
    for M in scaling_factors:
        print(f"\nTesting M = {M} ({M*100:.0f}% of original flux):")
        
        try:
            # Run detection with this scaling factor
            result = hwo_data.determine_detectable(
                use_exozodi_constraint=True,
                exozodi_scenario='baseline',
                M=M,
                ignore_exozodi_rejections=False
            )
            
            # Count rejections
            if 'exozodi_surface_brightness_rejected_best' in result.columns:
                rejected = result['exozodi_surface_brightness_rejected_best'].sum()
                total = len(result)
                print(f"  Rejected: {rejected}/{total} ({rejected/total*100:.1f}%)")
            else:
                print(f"  No rejection column found")
                
        except Exception as e:
            print(f"  Error: {e}")


def main():
    """Main debugging function."""
    
    # Load a specific data file that we know exists
    file_path = 'run/hwo/data/Gaia/hwo_catalog_495.csv'
    
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded data from: {file_path}")
        print(f"   Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Could not find data file: {file_path}")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Debug the data
    debug_exozodi_data(df)
    
    # Test different scaling factors
    test_different_scaling_factors(df)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("1. If no planets are rejected, try increasing the scaling factor M")
    print("2. Check if exozodi constraint is actually being applied")
    print("3. Verify that the exozodi model is working correctly")
    print("4. Consider using M=1.0 to see the full exozodi effect")


if __name__ == "__main__":
    main() 