#!/usr/bin/env python3
"""
Test script to demonstrate the new exozodi temperature calculation
based on radiative equilibrium instead of fixed 300K.
"""

import numpy as np
import matplotlib.pyplot as plt
from PPop.ExozodiModels.Ertel2020 import ExozodiModel

def test_exozodi_temperature_calculation():
    """Test the new exozodi temperature calculation."""
    
    # Initialize the exozodi model
    rng = np.random.default_rng(42)
    exozodi_model = ExozodiModel('baseline', rng)
    
    # Test parameters
    star_temp_K = 5778  # Solar temperature
    star_radius_Rsun = 1.0  # Solar radius
    distance_pc = 10.0  # 10 parsecs
    
    print("=== Exozodi Temperature Calculation Test ===")
    print(f"Star temperature: {star_temp_K} K")
    print(f"Star radius: {star_radius_Rsun} R_sun")
    print(f"Distance: {distance_pc} pc")
    print()
    
    # Test temperature at different distances
    distances_au = [1, 5, 10, 20, 50]
    
    print("Distance (AU) | Calculated Temp (K) | Old Fixed Temp (K)")
    print("-" * 55)
    
    for dist in distances_au:
        calculated_temp = exozodi_model._calculate_exozodi_temperature(
            star_temp_K, star_radius_Rsun, distance_pc, dist
        )
        print(f"{dist:11.1f} | {calculated_temp:16.1f} | {300:16.1f}")
    
    print()
    
    # Test temperature profile
    print("=== Temperature Profile ===")
    distances_au, temperatures = exozodi_model.getExozodiTemperatureProfileAtPlanetAU(
        star_temp_K, star_radius_Rsun
    )
    
    print(f"Temperature range: {np.min(temperatures):.1f} - {np.max(temperatures):.1f} K")
    print(f"Temperature at 1 AU: {temperatures[distances_au >= 1.0][0]:.1f} K")
    print(f"Temperature at 10 AU: {temperatures[distances_au >= 10.0][0]:.1f} K")
    
    # Create a plot
    plt.figure(figsize=(10, 6))
    
    # Plot temperature profile
    plt.subplot(1, 2, 1)
    plt.loglog(distances_au, temperatures, 'b-', linewidth=2, label='Calculated')
    plt.axhline(y=300.0, color='r', linestyle='--', label='Fixed (300K)')
    plt.xlabel('Distance from Star (AU)')
    plt.ylabel('Exozodi Temperature (K)')
    plt.title('Exozodi Temperature vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot temperature ratio (calculated/fixed)
    plt.subplot(1, 2, 2)
    temp_ratio = temperatures / 300.0
    plt.loglog(distances_au, temp_ratio, 'g-', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='Fixed temp')
    plt.xlabel('Distance from Star (AU)')
    plt.ylabel('Temperature Ratio (Calculated/Fixed)')
    plt.title('Temperature Ratio vs Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exozodi_temperature_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test with different star types
    print("\n=== Different Star Types ===")
    star_types = [
        ('M-dwarf', 3000, 0.3),
        ('K-dwarf', 4500, 0.7),
        ('G-dwarf (Sun)', 5778, 1.0),
        ('F-dwarf', 7000, 1.3),
        ('A-dwarf', 9000, 1.8)
    ]
    
    print("Star Type    | Star Temp (K) | Radius (R_sun) | Exozodi Temp at 10 AU (K)")
    print("-" * 75)
    
    for star_type, temp, radius in star_types:
        exozodi_temp = exozodi_model._calculate_exozodi_temperature(
            temp, radius, distance_pc, 10.0
        )
        print(f"{star_type:11s} | {temp:13.0f} | {radius:13.1f} | {exozodi_temp:25.1f}")
    
    print("\n=== Key Improvements ===")
    print("1. Temperature now varies with distance from star (T ∝ r^(-1/2))")
    print("2. Temperature depends on stellar properties (luminosity)")
    print("3. More physically realistic than fixed 300K")
    print("4. Accounts for radiative equilibrium with the star")
    print("5. Includes reasonable temperature bounds (50-1000K)")

if __name__ == "__main__":
    test_exozodi_temperature_calculation() 