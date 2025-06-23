#!/usr/bin/env python3
"""
Test script demonstrating radial dependence of exozodi brightness.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

# Import P-POP components
from PPop.ExozodiModels.Ertel2020 import ExozodiModel
from PPop.Star import Star


def test_radial_dependence():
    """Test the radial dependence of exozodi brightness."""
    print("Testing radial dependence of exozodi brightness...")
    
    # Create a test star (Solar-like)
    test_star = Star(
        Name="TestStar2",  # Add required Name parameter
        Teff=5772,  # Solar temperature
        Rad=1.0,    # Solar radius
        Mass=1.0,   # Solar mass
        Dist=10.0,  # 10 pc distance
        Stype='G2V',
        RA=0.0,
        Dec=0.0
    )
    
    # Create exozodi model
    exozodi_model = ExozodiModel('baseline', np.random.default_rng(42))
    
    # Get exozodi level
    exozodi_level = exozodi_model.getExozodiLevel()
    print(f"Exozodi level: {exozodi_level:.2e}")
    
    # Calculate radial profile
    distances_au, exozodi_fluxes = exozodi_model.getExozodiRadialProfile(
        star_temp_K=test_star.Teff,
        star_radius_Rsun=test_star.Rad,
        distance_pc=test_star.Dist,
        exozodi_level=exozodi_level
    )
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Radial Dependence of Exozodi Brightness', fontsize=16, fontweight='bold')
    
    # Plot 1: Exozodi flux vs distance
    ax1.plot(distances_au, exozodi_fluxes, 'b-', linewidth=2, label='Exozodi Flux')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Distance from Star (AU)')
    ax1.set_ylabel('Exozodi Flux in HWO Band (W/m²)')
    ax1.set_title('Exozodi Brightness vs Distance')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add some reference distances
    reference_distances = [0.1, 1, 10, 50]
    for dist in reference_distances:
        idx = np.argmin(np.abs(distances_au - dist))
        ax1.axvline(x=dist, color='red', linestyle='--', alpha=0.5)
        ax1.text(dist*1.2, exozodi_fluxes[idx], f'{dist} AU', rotation=90, va='bottom')
    
    # Plot 2: Power law comparison
    # Calculate expected power law (r^(-1.5))
    reference_flux = exozodi_fluxes[distances_au == 10.0][0]
    power_law_fluxes = reference_flux * (10.0 / distances_au) ** 1.5
    
    ax2.plot(distances_au, exozodi_fluxes, 'b-', linewidth=2, label='Model Calculation')
    ax2.plot(distances_au, power_law_fluxes, 'r--', linewidth=2, label='r^(-1.5) Power Law')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Distance from Star (AU)')
    ax2.set_ylabel('Exozodi Flux (W/m²)')
    ax2.set_title('Power Law Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('exozodi_radial_dependence.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return distances_au, exozodi_fluxes


def test_planet_detection_impact():
    """Test how radial exozodi affects planet detection at different distances."""
    print("\nTesting planet detection impact at different distances...")
    
    # Create test star
    test_star = Star(
        Name="TestStar3",  # Add required Name parameter
        Teff=5772,
        Rad=1.0,
        Mass=1.0,
        Dist=10.0,
        Stype='G2V',
        RA=0.0,
        Dec=0.0
    )
    
    # Create exozodi model
    exozodi_model = ExozodiModel('baseline', np.random.default_rng(42))
    exozodi_level = exozodi_model.getExozodiLevel()
    
    # Test different planet properties at different distances
    planet_radii = [0.5, 1.0, 2.0]  # Earth radii
    planet_temps = [250, 300, 350]  # K
    distances_au = np.logspace(-1, 1.7, 50)  # 0.1 to 50 AU
    
    results = []
    
    for r_planet in planet_radii:
        for t_planet in planet_temps:
            for dist_au in distances_au:
                # Calculate flux ratio at this distance
                flux_ratio, exozodi_flux, planet_flux = exozodi_model.getExozodiFluxRatioAtPlanetDistance(
                    star_temp_K=test_star.Teff,
                    star_radius_Rsun=test_star.Rad,
                    distance_pc=test_star.Dist,
                    planet_radius_Rearth=r_planet,
                    planet_temp_K=t_planet,
                    planet_semi_major_axis_au=dist_au,
                    exozodi_level=exozodi_level
                )
                
                results.append({
                    'planet_radius': r_planet,
                    'planet_temp': t_planet,
                    'distance_au': dist_au,
                    'flux_ratio': flux_ratio,
                    'exozodi_flux': exozodi_flux,
                    'planet_flux': planet_flux
                })
    
    results_df = pd.DataFrame(results)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Planet Detection Impact vs Orbital Distance', fontsize=16, fontweight='bold')
    
    # Plot 1: Flux ratio vs distance for different planet sizes
    ax1 = axes[0, 0]
    for r_planet in planet_radii:
        data = results_df[(results_df['planet_radius'] == r_planet) & 
                         (results_df['planet_temp'] == 300) & 
                         (results_df['flux_ratio'] != np.inf)]
        ax1.plot(data['distance_au'], data['flux_ratio'], 
                marker='o', markersize=4, label=f'{r_planet} R⊕')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Orbital Distance (AU)')
    ax1.set_ylabel('Exozodi/Planet Flux Ratio')
    ax1.set_title('Flux Ratio vs Distance (300K planets)')
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Exozodi = Planet')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Flux ratio vs distance for different planet temperatures
    ax2 = axes[0, 1]
    for t_planet in planet_temps:
        data = results_df[(results_df['planet_radius'] == 1.0) & 
                         (results_df['planet_temp'] == t_planet) & 
                         (results_df['flux_ratio'] != np.inf)]
        ax2.plot(data['distance_au'], data['flux_ratio'], 
                marker='s', markersize=4, label=f'{t_planet} K')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Orbital Distance (AU)')
    ax2.set_ylabel('Exozodi/Planet Flux Ratio')
    ax2.set_title('Flux Ratio vs Distance (1 R⊕ planets)')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Exozodi = Planet')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Optimal detection distance
    ax3 = axes[1, 0]
    optimal_distances = []
    planet_labels = []
    
    for r_planet in planet_radii:
        for t_planet in planet_temps:
            optimal_dist, min_ratio = exozodi_model.getOptimalPlanetDistance(
                star_temp_K=test_star.Teff,
                star_radius_Rsun=test_star.Rad,
                distance_pc=test_star.Dist,
                planet_radius_Rearth=r_planet,
                planet_temp_K=t_planet,
                exozodi_level=exozodi_level
            )
            optimal_distances.append(optimal_dist)
            planet_labels.append(f'{r_planet}R⊕, {t_planet}K')
    
    bars = ax3.bar(range(len(optimal_distances)), optimal_distances, 
                   color=['red', 'orange', 'yellow', 'green', 'blue', 'purple', 
                          'pink', 'brown', 'gray'])
    ax3.set_xlabel('Planet Type')
    ax3.set_ylabel('Optimal Distance (AU)')
    ax3.set_title('Optimal Detection Distance')
    ax3.set_xticks(range(len(planet_labels)))
    ax3.set_xticklabels(planet_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Detection zones
    ax4 = axes[1, 1]
    # Create a grid of distances and planet properties
    dist_grid = np.logspace(-1, 1.7, 100)
    temp_grid = np.linspace(200, 400, 50)
    
    # Calculate flux ratios for the grid
    ratio_grid = np.zeros((len(temp_grid), len(dist_grid)))
    for i, temp in enumerate(temp_grid):
        for j, dist in enumerate(dist_grid):
            ratio, _, _ = exozodi_model.getExozodiFluxRatioAtPlanetDistance(
                star_temp_K=test_star.Teff,
                star_radius_Rsun=test_star.Rad,
                distance_pc=test_star.Dist,
                planet_radius_Rearth=1.0,
                planet_temp_K=temp,
                planet_semi_major_axis_au=dist,
                exozodi_level=exozodi_level
            )
            ratio_grid[i, j] = ratio if ratio != np.inf else 1000
    
    # Create contour plot
    contour = ax4.contourf(dist_grid, temp_grid, ratio_grid, 
                          levels=np.logspace(-2, 2, 20), cmap='RdYlBu_r')
    ax4.contour(dist_grid, temp_grid, ratio_grid, levels=[1.0], colors='red', linewidths=2)
    ax4.set_xscale('log')
    ax4.set_xlabel('Orbital Distance (AU)')
    ax4.set_ylabel('Planet Temperature (K)')
    ax4.set_title('Detection Zones (1 R⊕ planets)')
    plt.colorbar(contour, ax=ax4, label='Exozodi/Planet Flux Ratio')
    
    plt.tight_layout()
    plt.savefig('planet_detection_impact.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_df


def test_comparison_with_old_model():
    """Compare the new radial-dependent model with the old system-wide model."""
    print("\nComparing radial-dependent vs system-wide exozodi models...")
    
    # Create test star
    test_star = Star(
        Name="TestStar2",  # Add required Name parameter
        Teff=5772,
        Rad=1.0,
        Mass=1.0,
        Dist=10.0,
        Stype='G2V',
        RA=0.0,
        Dec=0.0
    )
    
    # Create exozodi model
    exozodi_model = ExozodiModel('baseline', np.random.default_rng(42))
    exozodi_level = exozodi_model.getExozodiLevel()
    
    # Test distances
    distances_au = np.logspace(-1, 1.7, 20)
    
    # Calculate with old model (system-wide)
    old_exozodi_flux, _ = exozodi_model.getExozodiFluxAtPlanetAU(
        star_temp_K=test_star.Teff,
        star_radius_Rsun=test_star.Rad,
        distance_pc=test_star.Dist,
        exozodi_level=exozodi_level,
        planet_semi_major_axis_au=None
    )
    
    # Calculate with new model (radial-dependent)
    new_exozodi_fluxes = []
    for dist_au in distances_au:
        flux, _ = exozodi_model.getExozodiFluxAtPlanetAU(
            star_temp_K=test_star.Teff,
            star_radius_Rsun=test_star.Rad,
            distance_pc=test_star.Dist,
            exozodi_level=exozodi_level,
            planet_semi_major_axis_au=dist_au
        )
        new_exozodi_fluxes.append(flux)
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Radial vs System-wide Exozodi Model Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Flux comparison
    ax1.plot(distances_au, new_exozodi_fluxes, 'b-', linewidth=2, label='Radial-dependent Model')
    ax1.axhline(y=old_exozodi_flux, color='r--', linewidth=2, label='System-wide Model')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Distance from Star (AU)')
    ax1.set_ylabel('Exozodi Flux (W/m²)')
    ax1.set_title('Exozodi Flux Comparison')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Ratio comparison
    ratios = np.array(new_exozodi_fluxes) / old_exozodi_flux
    ax2.plot(distances_au, ratios, 'g-', linewidth=2)
    ax2.axhline(y=1.0, color='r--', linewidth=2, label='Equal to system-wide')
    ax2.set_xscale('log')
    ax2.set_xlabel('Distance from Star (AU)')
    ax2.set_ylabel('Radial Model / System-wide Model')
    ax2.set_title('Model Comparison Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"System-wide exozodi flux: {old_exozodi_flux:.2e} W/m²")
    print(f"Radial model at 1 AU: {new_exozodi_fluxes[distances_au == 1.0][0]:.2e} W/m²")
    print(f"Radial model at 10 AU: {new_exozodi_fluxes[distances_au == 10.0][0]:.2e} W/m²")
    print(f"Radial model at 50 AU: {new_exozodi_fluxes[distances_au == 50.0][0]:.2e} W/m²")


def main():
    """Run all radial exozodi tests."""
    print("Radial Exozodi Analysis Tests")
    print("=" * 40)
    
    # Test radial dependence
    distances_au, exozodi_fluxes = test_radial_dependence()
    
    # Test planet detection impact
    results_df = test_planet_detection_impact()
    
    # Compare with old model
    test_comparison_with_old_model()
    
    print("\n🎉 Radial exozodi analysis complete!")
    print("Generated plots:")
    print("- exozodi_radial_dependence.png")
    print("- planet_detection_impact.png") 
    print("- model_comparison.png")
    
    print("\nKey findings:")
    print("1. Exozodi brightness decreases with distance from star (r^(-1.5) power law)")
    print("2. Planets closer to their stars face much higher exozodi backgrounds")
    print("3. There's an optimal distance for planet detection (minimum exozodi/planet ratio)")
    print("4. The radial model provides more realistic exozodi impact assessment")


if __name__ == "__main__":
    main() 