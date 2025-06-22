#!/usr/bin/env python3
"""
Test script for the new exozodi surface brightness criterion.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PPop.ExozodiModels.Ertel2020 import ExozodiModel
from lifesim.core.hwo_data import HWOData


def create_test_catalog(n_planets=100):
    """Create a test catalog with realistic planet and star properties."""
    np.random.seed(42)
    
    # Generate diverse star properties
    star_temps = np.random.uniform(3000, 7000, n_planets)  # K
    star_radii = np.random.uniform(0.5, 2.0, n_planets)    # Solar radii
    distances = np.random.uniform(5, 50, n_planets)        # pc
    
    # Generate planet properties
    planet_radii = np.random.uniform(0.5, 2.5, n_planets)  # Earth radii
    planet_temps = np.random.uniform(200, 500, n_planets)  # K
    
    # Calculate angular separations (simplified)
    # Assume planets are at habitable zone distances
    habitable_zone_au = np.sqrt(star_temps / 5778)  # Simplified HZ calculation
    angular_separations = habitable_zone_au / distances  # arcseconds
    
    # Create catalog
    catalog = pd.DataFrame({
        'temp_s': star_temps,
        'radius_s': star_radii,
        'distance_s': distances,
        'temp_p': planet_temps,
        'radius_p': planet_radii,
        'maxangsep': angular_separations
    })
    
    return catalog


def test_surface_brightness_criterion():
    """Test the new surface brightness criterion."""
    print("Testing exozodi surface brightness criterion...")
    
    # Create test catalog
    catalog = create_test_catalog(50)
    print(f"Created test catalog with {len(catalog)} planets")
    
    # Create HWO data object
    hwo_data = HWOData(catalog)
    
    # Test both old and new approaches
    print("\n1. Testing old flux ratio approach...")
    hwo_data_old = HWOData(catalog.copy())
    result_old = hwo_data_old.determine_detectable(
        use_exozodi_constraint=True, 
        exozodi_scenario='baseline',
        use_surface_brightness_criterion=False
    )
    
    best_detected_old = result_old['detected_best'].sum()
    worst_detected_old = result_old['detected_worst'].sum()
    print(f"   Best case detected: {best_detected_old}")
    print(f"   Worst case detected: {worst_detected_old}")
    
    print("\n2. Testing new surface brightness approach...")
    result_new = hwo_data.determine_detectable(
        use_exozodi_constraint=True, 
        exozodi_scenario='baseline',
        use_surface_brightness_criterion=True,
        ignore_exozodi_rejections=True
    )
    
    best_detected_new = result_new['detected_best'].sum()
    worst_detected_new = result_new['detected_worst'].sum()
    print(f"   Best case detected: {best_detected_new}")
    print(f"   Worst case detected: {worst_detected_new}")
    
    # Compare results
    print(f"\n3. Comparison:")
    print(f"   Best case difference: {best_detected_new - best_detected_old}")
    print(f"   Worst case difference: {worst_detected_new - worst_detected_old}")
    
    # Analyze surface brightness ratios
    if 'exozodi_surface_brightness_ratio_best' in result_new.columns:
        ratios_best = result_new['exozodi_surface_brightness_ratio_best']
        ratios_worst = result_new['exozodi_surface_brightness_ratio_worst']
        
        print(f"\n4. Surface brightness ratio statistics:")
        print(f"   Best case - Mean: {np.mean(ratios_best):.2e}, Median: {np.median(ratios_best):.2e}")
        print(f"   Worst case - Mean: {np.mean(ratios_worst):.2e}, Median: {np.median(ratios_worst):.2e}")
        
        # Count rejections
        rejected_best = result_new['exozodi_surface_brightness_rejected_best'].sum()
        rejected_worst = result_new['exozodi_surface_brightness_rejected_worst'].sum()
        print(f"   Planets rejected by surface brightness criterion:")
        print(f"     Best case: {rejected_best} ({rejected_best/len(catalog)*100:.1f}%)")
        print(f"     Worst case: {rejected_worst} ({rejected_worst/len(catalog)*100:.1f}%)")
    
    return result_new


def test_exozodi_model_directly():
    """Test the ExozodiModel surface brightness methods directly."""
    print("\n" + "="*50)
    print("Testing ExozodiModel surface brightness methods directly...")
    
    # Create exozodi model
    rng = np.random.default_rng(42)
    exozodi_model = ExozodiModel('baseline', rng)
    
    # Test parameters
    star_temp = 5778  # Solar temperature
    star_radius = 1.0  # Solar radius
    distance = 10.0    # 10 pc
    angular_separation = 0.1  # 0.1 arcsec
    contrast_limit = 1e-6  # More realistic contrast limit
    
    print(f"Test parameters:")
    print(f"  Star temperature: {star_temp} K")
    print(f"  Star radius: {star_radius} R_sun")
    print(f"  Distance: {distance} pc")
    print(f"  Angular separation: {angular_separation} arcsec")
    print(f"  Contrast limit: {contrast_limit}")
    
    # Test surface brightness calculation
    surface_brightness, exozodi_level = exozodi_model.getExozodiSurfaceBrightness(
        star_temp, star_radius, distance, angular_separation
    )
    print(f"\nSurface brightness calculation:")
    print(f"  Exozodi level: {exozodi_level:.2e}")
    print(f"  Surface brightness: {surface_brightness:.2e} W/m²/arcsec²")
    
    # Test criterion check
    is_rejected, exozodi_sb, star_sb, _ = exozodi_model.checkExozodiSurfaceBrightnessCriterion(
        star_temp, star_radius, distance, angular_separation, contrast_limit
    )
    print(f"\nCriterion check:")
    print(f"  Exozodi surface brightness: {exozodi_sb:.2e} W/m²/arcsec²")
    print(f"  Star surface brightness: {star_sb:.2e} W/m²/arcsec²")
    print(f"  Contrast limit × star brightness: {contrast_limit * star_sb:.2e} W/m²/arcsec²")
    print(f"  Is rejected: {is_rejected}")
    
    # Test with different angular separations
    print(f"\nTesting different angular separations:")
    separations = [0.01, 0.1, 1.0, 10.0]  # arcsec
    for sep in separations:
        rejected, exozodi_sb, star_sb, _ = exozodi_model.checkExozodiSurfaceBrightnessCriterion(
            star_temp, star_radius, distance, sep, contrast_limit
        )
        print(f"  {sep:5.2f} arcsec: rejected={rejected}, exozodi_sb={exozodi_sb:.2e}")


def create_comparison_plot(result_df):
    """Create a plot comparing the old and new approaches."""
    print("\nCreating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exozodi Constraint Comparison: Old vs New Approach', fontsize=16)
    
    # Plot 1: Detection rates
    ax1 = axes[0, 0]
    cases = ['Best', 'Worst']
    old_detected = [result_df['detected_best'].sum(), result_df['detected_worst'].sum()]
    new_detected = [result_df['detected_best'].sum(), result_df['detected_worst'].sum()]
    
    x = np.arange(len(cases))
    width = 0.35
    
    ax1.bar(x - width/2, old_detected, width, label='Old Approach (Flux Ratio)', alpha=0.7)
    ax1.bar(x + width/2, new_detected, width, label='New Approach (Surface Brightness)', alpha=0.7)
    ax1.set_xlabel('Case')
    ax1.set_ylabel('Number of Detected Planets')
    ax1.set_title('Detection Rates Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cases)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Surface brightness ratios distribution
    ax2 = axes[0, 1]
    if 'exozodi_surface_brightness_ratio_best' in result_df.columns:
        ratios_best = result_df['exozodi_surface_brightness_ratio_best']
        ratios_worst = result_df['exozodi_surface_brightness_ratio_worst']
        
        ax2.hist(ratios_best, bins=20, alpha=0.7, label='Best Case', density=True)
        ax2.hist(ratios_worst, bins=20, alpha=0.7, label='Worst Case', density=True)
        ax2.axvline(x=1.0, color='red', linestyle='--', label='Rejection Threshold')
        ax2.set_xscale('log')
        ax2.set_xlabel('Exozodi Surface Brightness Ratio')
        ax2.set_ylabel('Density')
        ax2.set_title('Surface Brightness Ratio Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Angular separation vs surface brightness ratio
    ax3 = axes[1, 0]
    if 'exozodi_surface_brightness_ratio_best' in result_df.columns:
        scatter = ax3.scatter(result_df['maxangsep'], result_df['exozodi_surface_brightness_ratio_best'],
                             c=result_df['distance_s'], cmap='viridis', alpha=0.6, s=50)
        ax3.set_yscale('log')
        ax3.set_xlabel('Angular Separation (arcsec)')
        ax3.set_ylabel('Surface Brightness Ratio')
        ax3.set_title('Angular Separation vs Surface Brightness Ratio')
        ax3.axhline(y=1.0, color='red', linestyle='--', label='Rejection Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='Distance (pc)')
    
    # Plot 4: Star temperature vs surface brightness ratio
    ax4 = axes[1, 1]
    if 'exozodi_surface_brightness_ratio_best' in result_df.columns:
        scatter = ax4.scatter(result_df['temp_s'], result_df['exozodi_surface_brightness_ratio_best'],
                             c=result_df['radius_s'], cmap='plasma', alpha=0.6, s=50)
        ax4.set_yscale('log')
        ax4.set_xlabel('Star Temperature (K)')
        ax4.set_ylabel('Surface Brightness Ratio')
        ax4.set_title('Star Temperature vs Surface Brightness Ratio')
        ax4.axhline(y=1.0, color='red', linestyle='--', label='Rejection Threshold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Star Radius (R_sun)')
    
    plt.tight_layout()
    plt.savefig('surface_brightness_criterion_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def main():
    """Run the complete test."""
    print("Exozodi Surface Brightness Criterion Test")
    print("=" * 50)
    
    # Test the new criterion
    result_df = test_surface_brightness_criterion()
    
    # Test the model directly
    test_exozodi_model_directly()
    
    # Create comparison plot
    create_comparison_plot(result_df)
    
    print(f"\nTest results saved to 'surface_brightness_criterion_comparison.png'")
    print("\n🎉 Surface brightness criterion test complete!")


if __name__ == "__main__":
    main() 