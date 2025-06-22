#!/usr/bin/env python3
"""
Comprehensive test script for exozodi analysis with visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import P-POP components
from PPop.ExozodiModels.Ertel2020 import ExozodiModel
from PPop.Star import Star


def generate_test_data(n_systems=100):
    """Generate realistic test data for exozodi analysis."""
    print(f"Generating {n_systems} test systems...")
    
    np.random.seed(42)  # For reproducible results
    
    # Generate diverse star properties
    star_data = []
    for i in range(n_systems):
        # Random star properties (realistic ranges)
        star_temp = np.random.uniform(3000, 7000)  # K
        star_radius = np.random.uniform(0.5, 2.0)   # Solar radii
        distance = np.random.uniform(5, 50)         # pc
        
        star = Star(
            Name=f"Star_{i}",  # Add required Name parameter
            Teff=star_temp,
            Rad=star_radius,
            Mass=np.random.uniform(0.5, 2.0),
            Dist=distance,
            Stype='G2V',  # Simplified
            RA=0.0,
            Dec=0.0
        )
        star_data.append(star)
    
    # Generate planet properties for each system
    planet_data = []
    for i, star in enumerate(star_data):
        n_planets = np.random.randint(1, 4)  # 1-3 planets per system
        
        for j in range(n_planets):
            # Random planet properties
            planet_radius = np.random.uniform(0.5, 2.5)  # Earth radii
            planet_temp = np.random.uniform(200, 500)    # K
            
            planet_data.append({
                'system_id': i,
                'star_temp': star.Teff,
                'star_radius': star.Rad,
                'distance': star.Dist,
                'planet_radius': planet_radius,
                'planet_temp': planet_temp
            })
    
    return star_data, planet_data


def calculate_exozodi_effects(star_data, planet_data, exozodi_scenario='baseline'):
    """Calculate exozodi effects for all systems."""
    print(f"Calculating exozodi effects for {exozodi_scenario} scenario...")
    
    # Create exozodi model
    exozodi_model = ExozodiModel(exozodi_scenario, np.random.default_rng(42))
    
    results = []
    
    for planet in planet_data:
        # Get exozodi level for this system
        exozodi_level = exozodi_model.getExozodiLevel()
        
        # Calculate exozodi flux
        exozodi_flux, _ = exozodi_model.getExozodiFlux(
            star_temp_K=planet['star_temp'],
            star_radius_Rsun=planet['star_radius'],
            distance_pc=planet['distance'],
            exozodi_level=exozodi_level
        )
        
        # Calculate planet flux
        planet_flux = exozodi_model._planet_flux_in_hwo_band(
            planet_radius_Rearth=planet['planet_radius'],
            planet_temp_K=planet['planet_temp'],
            distance_pc=planet['distance']
        )
        
        # Calculate star flux in HWO band (simplified)
        star_flux = exozodi_model._blackbody_flux_in_band(
            planet['star_temp'],
            exozodi_model.hwo_min_wavelength,
            exozodi_model.hwo_max_wavelength
        ) * 4 * np.pi * (planet['star_radius'] * 6.957e8)**2 * 5.670374419e-8 * planet['star_temp']**4 / (4 * np.pi * (planet['distance'] * 3.0857e16)**2)
        
        # Calculate ratios
        exozodi_planet_ratio = exozodi_flux / planet_flux if planet_flux > 0 else np.inf
        exozodi_star_ratio = exozodi_flux / star_flux if star_flux > 0 else np.inf
        
        results.append({
            **planet,
            'exozodi_level': exozodi_level,
            'exozodi_flux': exozodi_flux,
            'planet_flux': planet_flux,
            'star_flux': star_flux,
            'exozodi_planet_ratio': exozodi_planet_ratio,
            'exozodi_star_ratio': exozodi_star_ratio
        })
    
    return pd.DataFrame(results)


def create_exozodi_plots(results_df):
    """Create comprehensive plots showing exozodi effects."""
    print("Creating exozodi analysis plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exozodi Impact Analysis for HWO', fontsize=16, fontweight='bold')
    
    # Plot 1: Exozodi level distribution
    ax1 = axes[0, 0]
    ax1.hist(results_df['exozodi_level'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xscale('log')
    ax1.set_xlabel('Exozodi Level')
    ax1.set_ylabel('Number of Systems')
    ax1.set_title('Exozodi Level Distribution')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Exozodi flux distribution
    ax2 = axes[0, 1]
    ax2.hist(results_df['exozodi_flux'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xscale('log')
    ax2.set_xlabel('Exozodi Flux in HWO Band (W/m²)')
    ax2.set_ylabel('Number of Systems')
    ax2.set_title('Exozodi Flux Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Exozodi/Planet flux ratio
    ax3 = axes[0, 2]
    # Filter out infinite values
    valid_ratios = results_df[results_df['exozodi_planet_ratio'] != np.inf]['exozodi_planet_ratio']
    ax3.hist(valid_ratios, bins=30, alpha=0.7, color='salmon', edgecolor='black')
    ax3.axvline(x=1.0, color='red', linestyle='--', label='Exozodi = Planet')
    ax3.set_xscale('log')
    ax3.set_xlabel('Exozodi/Planet Flux Ratio')
    ax3.set_ylabel('Number of Systems')
    ax3.set_title('Exozodi vs Planet Flux Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Exozodi/Star flux ratio
    ax4 = axes[1, 0]
    valid_star_ratios = results_df[results_df['exozodi_star_ratio'] != np.inf]['exozodi_star_ratio']
    ax4.hist(valid_star_ratios, bins=30, alpha=0.7, color='gold', edgecolor='black')
    ax4.set_xscale('log')
    ax4.set_xlabel('Exozodi/Star Flux Ratio')
    ax4.set_ylabel('Number of Systems')
    ax4.set_title('Exozodi vs Star Flux Ratio')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Scatter plot - Planet temp vs Exozodi/Planet ratio
    ax5 = axes[1, 1]
    valid_data = results_df[results_df['exozodi_planet_ratio'] != np.inf]
    scatter = ax5.scatter(valid_data['planet_temp'], valid_data['exozodi_planet_ratio'], 
                         c=valid_data['planet_radius'], cmap='viridis', alpha=0.6, s=50)
    ax5.set_yscale('log')
    ax5.set_xlabel('Planet Temperature (K)')
    ax5.set_ylabel('Exozodi/Planet Flux Ratio')
    ax5.set_title('Planet Temperature vs Exozodi Impact')
    ax5.axhline(y=1.0, color='red', linestyle='--', label='Exozodi = Planet')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax5, label='Planet Radius (R⊕)')
    
    # Plot 6: Scatter plot - Distance vs Exozodi/Planet ratio
    ax6 = axes[1, 2]
    scatter = ax6.scatter(valid_data['distance'], valid_data['exozodi_planet_ratio'], 
                         c=valid_data['star_temp'], cmap='plasma', alpha=0.6, s=50)
    ax6.set_yscale('log')
    ax6.set_xlabel('Distance (pc)')
    ax6.set_ylabel('Exozodi/Planet Flux Ratio')
    ax6.set_title('Distance vs Exozodi Impact')
    ax6.axhline(y=1.0, color='red', linestyle='--', label='Exozodi = Planet')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax6, label='Star Temperature (K)')
    
    plt.tight_layout()
    plt.savefig('exozodi_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def create_comparison_plot(results_df):
    """Create a focused comparison plot of exozodi ratios."""
    print("Creating comparison plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Exozodi Flux Ratio Comparison', fontsize=16, fontweight='bold')
    
    # Filter out infinite values
    valid_data = results_df[
        (results_df['exozodi_planet_ratio'] != np.inf) & 
        (results_df['exozodi_star_ratio'] != np.inf)
    ]
    
    # Plot 1: Histogram comparison
    ax1.hist(valid_data['exozodi_planet_ratio'], bins=30, alpha=0.7, 
             label='Exozodi/Planet', color='red', edgecolor='black')
    ax1.hist(valid_data['exozodi_star_ratio'], bins=30, alpha=0.7, 
             label='Exozodi/Star', color='blue', edgecolor='black')
    ax1.set_xscale('log')
    ax1.set_xlabel('Flux Ratio')
    ax1.set_ylabel('Number of Systems')
    ax1.set_title('Distribution of Exozodi Flux Ratios')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot comparison
    ax2.scatter(valid_data['exozodi_star_ratio'], valid_data['exozodi_planet_ratio'], 
               alpha=0.6, s=50, c=valid_data['distance'], cmap='viridis')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Exozodi/Star Flux Ratio')
    ax2.set_ylabel('Exozodi/Planet Flux Ratio')
    ax2.set_title('Exozodi/Star vs Exozodi/Planet Ratios')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Exozodi = Planet')
    ax2.axvline(x=1.0, color='blue', linestyle='--', label='Exozodi = Star')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.colorbar(ax2.collections[0], ax=ax2, label='Distance (pc)')
    
    plt.tight_layout()
    plt.savefig('exozodi_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def print_summary_statistics(results_df):
    """Print summary statistics of the analysis."""
    print("\n" + "="*60)
    print("EXOZODI ANALYSIS SUMMARY")
    print("="*60)
    
    total_systems = len(results_df)
    valid_planet_ratios = results_df[results_df['exozodi_planet_ratio'] != np.inf]
    valid_star_ratios = results_df[results_df['exozodi_star_ratio'] != np.inf]
    
    print(f"\nTotal systems analyzed: {total_systems}")
    print(f"Valid planet ratios: {len(valid_planet_ratios)}")
    print(f"Valid star ratios: {len(valid_star_ratios)}")
    
    # Exozodi level statistics
    print(f"\nExozodi Level Statistics:")
    print(f"  Mean: {results_df['exozodi_level'].mean():.2e}")
    print(f"  Median: {results_df['exozodi_level'].median():.2e}")
    print(f"  Range: {results_df['exozodi_level'].min():.2e} to {results_df['exozodi_level'].max():.2e}")
    
    # Planet ratio statistics
    if len(valid_planet_ratios) > 0:
        print(f"\nExozodi/Planet Flux Ratio Statistics:")
        print(f"  Mean: {valid_planet_ratios['exozodi_planet_ratio'].mean():.2f}")
        print(f"  Median: {valid_planet_ratios['exozodi_planet_ratio'].median():.2f}")
        print(f"  Systems where exozodi > planet: {len(valid_planet_ratios[valid_planet_ratios['exozodi_planet_ratio'] > 1.0])}")
        print(f"  Systems where exozodi < 0.1× planet: {len(valid_planet_ratios[valid_planet_ratios['exozodi_planet_ratio'] < 0.1])}")
    
    # Star ratio statistics
    if len(valid_star_ratios) > 0:
        print(f"\nExozodi/Star Flux Ratio Statistics:")
        print(f"  Mean: {valid_star_ratios['exozodi_star_ratio'].mean():.2e}")
        print(f"  Median: {valid_star_ratios['exozodi_star_ratio'].median():.2e}")
        print(f"  Systems where exozodi > 0.01× star: {len(valid_star_ratios[valid_star_ratios['exozodi_star_ratio'] > 0.01])}")
    
    # Impact assessment
    if len(valid_planet_ratios) > 0:
        hidden_planets = len(valid_planet_ratios[valid_planet_ratios['exozodi_planet_ratio'] > 1.0])
        hidden_percentage = (hidden_planets / len(valid_planet_ratios)) * 100
        print(f"\nImpact Assessment:")
        print(f"  Planets potentially hidden by exozodi: {hidden_planets} ({hidden_percentage:.1f}%)")
        print(f"  Planets with low exozodi impact (<0.1×): {len(valid_planet_ratios[valid_planet_ratios['exozodi_planet_ratio'] < 0.1])}")


def main():
    """Run the complete exozodi analysis."""
    print("Exozodi Analysis Test")
    print("=" * 40)
    
    # Generate test data
    star_data, planet_data = generate_test_data(100)
    
    # Calculate exozodi effects
    results_df = calculate_exozodi_effects(star_data, planet_data, 'baseline')
    
    # Create plots
    create_exozodi_plots(results_df)
    create_comparison_plot(results_df)
    
    # Print summary statistics
    print_summary_statistics(results_df)
    
    # Save results to CSV
    results_df.to_csv('exozodi_test_results.csv', index=False)
    print(f"\nResults saved to 'exozodi_test_results.csv'")
    print(f"Plots saved as 'exozodi_analysis_results.png' and 'exozodi_ratio_comparison.png'")
    
    print("\n🎉 Analysis complete! Check the generated files and plots.")


if __name__ == "__main__":
    main() 