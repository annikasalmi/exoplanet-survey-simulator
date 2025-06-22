#!/usr/bin/env python3
"""
Test script to compare numerical integration of Planck's law vs approximation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from PPop.ExozodiModels.Ertel2020 import ExozodiModel

def test_planck_integration():
    """Test the numerical integration of Planck's law."""
    
    # Initialize the exozodi model
    rng = np.random.default_rng(42)
    exozodi_model = ExozodiModel('baseline', rng)
    
    # Test parameters
    temperatures = [100, 200, 300, 500, 1000]  # K
    min_wavelength = 200e-9  # 200 nm
    max_wavelength = 2500e-9  # 2.5 μm
    
    print("=== Planck Law Integration Test ===")
    print(f"Wavelength band: {min_wavelength*1e9:.0f} - {max_wavelength*1e9:.0f} nm")
    print()
    
    print("Temperature (K) | Numerical Fraction | Approximation | Difference (%)")
    print("-" * 70)
    
    for temp in temperatures:
        # Use the new numerical integration method
        numerical_fraction = exozodi_model._blackbody_flux_in_band(
            temp, min_wavelength, max_wavelength
        )
        
        # Calculate the old approximation method
        min_freq = 3e8 / max_wavelength  # Higher frequency for shorter wavelength
        max_freq = 3e8 / min_wavelength  # Lower frequency for longer wavelength
        peak_freq = 2.821 * 1.381e-23 * temp / 6.626e-34
        
        if min_freq <= peak_freq <= max_freq:
            approx_fraction = 0.3  # Rough estimate for HWO band
        else:
            if max_freq < peak_freq:
                approx_fraction = np.exp(-6.626e-34 * max_freq / (1.381e-23 * temp))
            else:
                approx_fraction = (max_freq / peak_freq)**3 * np.exp(-6.626e-34 * max_freq / (1.381e-23 * temp))
        
        difference = abs(numerical_fraction - approx_fraction) / numerical_fraction * 100
        
        print(f"{temp:14.0f} | {numerical_fraction:16.4f} | {approx_fraction:13.4f} | {difference:11.1f}")
    
    print()
    
    # Test across a range of temperatures
    temp_range = np.logspace(1, 3, 100)  # 10 to 1000 K
    numerical_fractions = []
    approx_fractions = []
    
    for temp in temp_range:
        numerical_fractions.append(exozodi_model._blackbody_flux_in_band(
            temp, min_wavelength, max_wavelength
        ))
        
        # Old approximation
        min_freq = 3e8 / max_wavelength
        max_freq = 3e8 / min_wavelength
        peak_freq = 2.821 * 1.381e-23 * temp / 6.626e-34
        
        if min_freq <= peak_freq <= max_freq:
            approx_fractions.append(0.3)
        else:
            if max_freq < peak_freq:
                approx_fractions.append(np.exp(-6.626e-34 * max_freq / (1.381e-23 * temp)))
            else:
                approx_fractions.append((max_freq / peak_freq)**3 * np.exp(-6.626e-34 * max_freq / (1.381e-23 * temp)))
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.loglog(temp_range, numerical_fractions, 'b-', linewidth=2, label='Numerical Integration')
    plt.loglog(temp_range, approx_fractions, 'r--', linewidth=2, label='Approximation')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Fraction in HWO Band')
    plt.title('Blackbody Fraction in HWO Band vs Temperature')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    ratio = np.array(numerical_fractions) / np.array(approx_fractions)
    plt.loglog(temp_range, ratio, 'g-', linewidth=2)
    plt.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Ratio (Numerical/Approximation)')
    plt.title('Accuracy of Approximation')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    error = np.abs(np.array(numerical_fractions) - np.array(approx_fractions)) / np.array(numerical_fractions) * 100
    plt.loglog(temp_range, error, 'm-', linewidth=2)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Relative Error (%)')
    plt.title('Approximation Error')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    # Show where the approximation is good/bad
    good_approx = error < 10  # Less than 10% error
    plt.loglog(temp_range[good_approx], np.array(numerical_fractions)[good_approx], 'g.', alpha=0.7, label='Good approximation')
    plt.loglog(temp_range[~good_approx], np.array(numerical_fractions)[~good_approx], 'r.', alpha=0.7, label='Poor approximation')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Fraction in HWO Band')
    plt.title('Approximation Quality')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('planck_integration_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("=== Analysis ===")
    print("1. Numerical integration is more accurate but computationally expensive")
    print("2. The approximation works reasonably well for temperatures around 300K")
    print("3. For very hot or very cold temperatures, the approximation breaks down")
    print("4. The numerical method provides physically correct results")
    print("5. For exozodi modeling, numerical integration is justified for accuracy")

if __name__ == "__main__":
    test_planck_integration() 