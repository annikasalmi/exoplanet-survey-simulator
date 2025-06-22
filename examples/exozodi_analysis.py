#!/usr/bin/env python3
"""
Example script demonstrating exozodi analysis for HWO.

This script shows how to:
1. Load planet data
2. Calculate detection probabilities with and without exozodi
3. Analyze the impact of exozodi on planet detection
4. Compare different exozodi scenarios
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to the path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from lifesim.core.hwo_data import HWOData
from lifesim.core.data import Data


def create_sample_data(n_planets=1000):
    """Create sample planet data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic sample data
    data = {
        'temp_p': np.random.uniform(200, 400, n_planets),  # Planet temperatures
        'temp_s': np.random.uniform(3000, 7000, n_planets),  # Star temperatures
        'radius_p': np.random.uniform(0.5, 2.0, n_planets),  # Planet radii (Earth units)
        'radius_s': np.random.uniform(0.5, 1.5, n_planets),  # Star radii (Solar units)
        'distance_s': np.random.uniform(5, 50, n_planets),  # Distances (pc)
        'maxangsep': np.random.uniform(0.01, 0.5, n_planets),  # Angular separation (arcsec)
    }
    
    return pd.DataFrame(data)


def analyze_exozodi_impact():
    """Analyze the impact of exozodi on planet detection."""
    
    # Create sample data
    print("Creating sample planet data...")
    sample_df = create_sample_data(1000)
    
    # Initialize HWO data
    hwo_data = HWOData(sample_df)
    
    # Analyze without exozodi
    print("\nAnalyzing detection without exozodi...")
    catalog_no_exozodi = hwo_data.determine_detectable()
    
    # Analyze with different exozodi scenarios
    scenarios = ['baseline', 'pessimistic', 'optimistic']
    results = {}
    
    for scenario in scenarios:
        print(f"\nAnalyzing detection with {scenario} exozodi scenario...")
        hwo_data_scenario = HWOData(sample_df.copy())
        catalog_with_exozodi = hwo_data_scenario.determine_detectable_with_exozodi(scenario)
        results[scenario] = catalog_with_exozodi
    
    # Print summary statistics
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    
    for case in ['best', 'worst']:
        print(f"\n{case.upper()} CASE SCENARIO:")
        print("-" * 40)
        
        # Without exozodi
        detected_no_exozodi = catalog_no_exozodi[f'detected_{case}'].sum()
        total_planets = len(catalog_no_exozodi)
        
        print(f"Without exozodi: {detected_no_exozodi}/{total_planets} planets detected ({detected_no_exozodi/total_planets*100:.1f}%)")
        
        # With exozodi
        for scenario in scenarios:
            catalog = results[scenario]
            detected_with_exozodi = catalog[f'detected_with_exozodi_{case}'].sum()
            planets_lost = catalog[f'lost_to_exozodi_{case}'].sum()
            
            print(f"With {scenario} exozodi: {detected_with_exozodi}/{total_planets} planets detected ({detected_with_exozodi/total_planets*100:.1f}%)")
            print(f"  Planets lost to exozodi: {planets_lost} ({planets_lost/detected_no_exozodi*100:.1f}% of originally detectable)")
    
    return catalog_no_exozodi, results


def plot_exozodi_impact(catalog_no_exozodi, results):
    """Create plots showing the impact of exozodi."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Impact of Exozodi on HWO Planet Detection', fontsize=16)
    
    # Plot 1: Detection rates by scenario
    ax1 = axes[0, 0]
    scenarios = ['baseline', 'pessimistic', 'optimistic']
    cases = ['best', 'worst']
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    for i, case in enumerate(cases):
        detection_rates = []
        for scenario in scenarios:
            catalog = results[scenario]
            detected = catalog[f'detected_with_exozodi_{case}'].sum()
            total = len(catalog)
            detection_rates.append(detected / total * 100)
        
        ax1.bar(x + i*width, detection_rates, width, label=f'{case.capitalize()} case')
    
    ax1.set_xlabel('Exozodi Scenario')
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Planet Detection Rates')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels([s.capitalize() for s in scenarios])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Exozodi levels distribution
    ax2 = axes[0, 1]
    for scenario in scenarios:
        catalog = results[scenario]
        ax2.hist(catalog['exozodi_level'], bins=30, alpha=0.7, label=scenario.capitalize(), density=True)
    
    ax2.set_xlabel('Exozodi Level')
    ax2.set_ylabel('Density')
    ax2.set_title('Exozodi Level Distribution')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Planets lost to exozodi
    ax3 = axes[1, 0]
    for i, case in enumerate(cases):
        planets_lost = []
        for scenario in scenarios:
            catalog = results[scenario]
            lost = catalog[f'lost_to_exozodi_{case}'].sum()
            planets_lost.append(lost)
        
        ax3.bar(x + i*width, planets_lost, width, label=f'{case.capitalize()} case')
    
    ax3.set_xlabel('Exozodi Scenario')
    ax3.set_ylabel('Number of Planets Lost')
    ax3.set_title('Planets Lost to Exozodi')
    ax3.set_xticks(x + width/2)
    ax3.set_xticklabels([s.capitalize() for s in scenarios])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flux ratio vs exozodi level
    ax4 = axes[1, 1]
    catalog = results['baseline']  # Use baseline for this plot
    
    # Plot detected vs non-detected planets
    detected_best = catalog['detected_with_exozodi_best']
    detected_worst = catalog['detected_with_exozodi_worst']
    
    ax4.scatter(catalog.loc[~detected_best, 'exozodi_level'], 
                catalog.loc[~detected_best, 'flux_ratio_value_best'], 
                alpha=0.6, label='Not detected (best case)', s=20)
    ax4.scatter(catalog.loc[detected_best, 'exozodi_level'], 
                catalog.loc[detected_best, 'flux_ratio_value_best'], 
                alpha=0.6, label='Detected (best case)', s=20)
    
    ax4.set_xlabel('Exozodi Level')
    ax4.set_ylabel('Planet/Star Flux Ratio')
    ax4.set_title('Detection vs Exozodi Level')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exozodi_impact_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the exozodi analysis."""
    print("HWO Exozodi Impact Analysis")
    print("=" * 40)
    
    # Run the analysis
    catalog_no_exozodi, results = analyze_exozodi_impact()
    
    # Create plots
    print("\nCreating plots...")
    plot_exozodi_impact(catalog_no_exozodi, results)
    
    print("\nAnalysis complete! Check 'exozodi_impact_analysis.png' for plots.")


if __name__ == "__main__":
    main() 