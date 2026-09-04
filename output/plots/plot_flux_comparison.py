import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.paths import LIFESIM_OUTER_DIR
import tools.physics_constants as const

def blackbody_flux(wavelength_m, temperature_K):
    """
    Calculate the blackbody spectral radiance at a given wavelength and temperature.
    
    Parameters:
        wavelength_m: Wavelength in meters (float or array)
        temperature_K: Temperature in Kelvin (float)
        
    Returns:
        Spectral radiance in W·sr⁻¹·m⁻³
    """
    wavelength_m = np.asarray(wavelength_m)
    temperature_K = np.asarray(temperature_K)
    
    exponent = (const.h * const.c) / (wavelength_m * const.k * temperature_K)
    numerator = 2 * const.h * const.c**2
    denominator = (wavelength_m**5) * (np.exp(exponent) - 1)
    return numerator / denominator

def plot_planetary_vs_stellar_flux():
    """
    Create a log-log plot of planetary flux vs stellar flux for exoplanets.
    Planetary flux = Rp^2 * B(Tp)
    Stellar flux = Rs^2 * B(Ts)
    """
    # Load exoplanets data
    exo_path = os.path.join(LIFESIM_OUTER_DIR, 'exoplanets_2026.csv')
    df = pd.read_csv(exo_path)
    
    # Filter for valid data
    df = df[
        (df['pl_rade'].notnull()) & 
        (df['pl_eqt'].notnull()) & 
        (df['st_rad'].notnull()) & 
        (df['st_teff'].notnull()) &
        (df['pl_name'].notnull())
    ]
    df = df.reset_index(drop=True)
    
    print(f"Loaded {len(df)} exoplanets with valid data")
    
    # Use HWO wavelength range (same as HWOData)
    min_wavelength = 200e-9  # 200 nm (best case)
    max_wavelength = 2500e-9  # 2500 nm (best case)
    
    # Calculate fluxes at both wavelengths (same as HWOData)
    # Planetary flux: Rp^2 * B(Tp)
    planet_temp = df['pl_eqt'].values
    planet_radius = df['pl_rade'].values * const.R_earth  # Convert to meters
    planet_blackbody_a = blackbody_flux(min_wavelength, planet_temp)
    planet_blackbody_b = blackbody_flux(max_wavelength, planet_temp)
    planetary_flux_a = planet_radius**2 * planet_blackbody_a
    planetary_flux_b = planet_radius**2 * planet_blackbody_b
    
    # Stellar flux: Rs^2 * B(Ts)
    stellar_temp = df['st_teff'].values
    stellar_radius = df['st_rad'].values * const.R_sun  # Convert to meters
    stellar_blackbody_a = blackbody_flux(min_wavelength, stellar_temp)
    stellar_blackbody_b = blackbody_flux(max_wavelength, stellar_temp)
    stellar_flux_a = stellar_radius**2 * stellar_blackbody_a
    stellar_flux_b = stellar_radius**2 * stellar_blackbody_b
    
    # Calculate flux ratios at both wavelengths (same as HWOData)
    flux_ratio_a = planetary_flux_a / stellar_flux_a
    flux_ratio_b = planetary_flux_b / stellar_flux_b
    
    # Use maximum for best case detection (same as HWOData)
    flux_ratios = np.maximum(flux_ratio_a, flux_ratio_b)
    
    # Calculate statistics
    threshold = 2.5e-11
    below_threshold = np.sum(flux_ratios < threshold)
    percentage = (below_threshold / len(flux_ratios)) * 100
    print(f"Percentage of exoplanets with flux ratio < {threshold:.1e}: {percentage:.2f}% ({below_threshold}/{len(flux_ratios)})")
    
    # Use the maximum flux ratios for plotting (best case)
    planetary_flux = np.maximum(planetary_flux_a, planetary_flux_b)
    stellar_flux = np.maximum(stellar_flux_a, stellar_flux_b)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Plot all planets as gray dots
    plt.scatter(stellar_flux, planetary_flux, alpha=0.6, s=30, c='gray', edgecolors='none')
    
    # Add labels for some interesting planets
    interesting_planets = [
        'Proxima Cen b', 'TOI-700 d', 'TRAPPIST-1 b', 'TRAPPIST-1 c', 'TRAPPIST-1 d',
        'TRAPPIST-1 e', 'TRAPPIST-1 f', 'TRAPPIST-1 g', 'GJ 1132 b', 'LHS 1140 b',
        'Kepler-186 f', 'Kepler-442 b', 'Kepler-62 f', 'Kepler-296 e', 'Kepler-296 f'
    ]
    
    for planet_name in interesting_planets:
        mask = df['pl_name'] == planet_name
        if mask.any():
            idx = mask.idxmax()
            plt.annotate(planet_name, 
                        (stellar_flux[idx], planetary_flux[idx]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add line for constant flux ratio (planetary_flux = 2.5e-11 * stellar_flux)
    flux_ratio = 2.5*10**(-11)
    min_stellar = stellar_flux.min()
    max_stellar = stellar_flux.max()
    x_vals = np.logspace(np.log10(min_stellar), np.log10(max_stellar), 100)
    y_vals = flux_ratio * x_vals
    plt.plot(x_vals, y_vals, 'k--', alpha=0.7, label='Flux ratio = 2.5e-11')
    
    # Add line for flux ratio = 10e-5
    flux_ratio_2 = 10**(-5)
    y_vals_2 = flux_ratio_2 * x_vals
    plt.plot(x_vals, y_vals_2, 'r--', alpha=0.7, label='Flux ratio = 10e-5')
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Labels and title
    plt.xlabel('Stellar Flux [W·sr⁻¹·m⁻³]')
    plt.ylabel('Planetary Flux [W·sr⁻¹·m⁻³]')
    plt.title('Planetary vs Stellar Flux at HWO wavelengths\n(Rp²×B(Tp) vs Rs²×B(Ts))')
    plt.legend()
    
    # Save the plot
    output_dir = os.path.join(LIFESIM_OUTER_DIR, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'planetary_vs_stellar_flux.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {os.path.join(output_dir, 'planetary_vs_stellar_flux.png')}")

def plot_flux_ratio_vs_wavelength():
    """
    Create a plot showing flux ratio vs wavelength for an Earth-like planet around a Sun-like star.
    """
    # Define Earth-like planet and Sun-like star parameters
    T_earth = 288  # K (Earth's equilibrium temperature)
    T_sun = 5778   # K (Sun's effective temperature)
    R_earth = const.R_earth  # m
    R_sun = const.R_sun      # m
    
    # Create wavelength range (200nm to 28.5μm to cover all missions)
    wavelengths = np.logspace(np.log10(200e-9), np.log10(28.5e-6), 1000)  # 200nm to 28.5μm
    
    # Calculate blackbody fluxes
    planet_blackbody = blackbody_flux(wavelengths, T_earth)
    stellar_blackbody = blackbody_flux(wavelengths, T_sun)
    
    # Calculate fluxes
    planetary_flux = R_earth**2 * planet_blackbody
    stellar_flux = R_sun**2 * stellar_blackbody
    
    # Calculate flux ratio
    flux_ratio = planetary_flux / stellar_flux
    
    # Add M dwarf system
    T_mdwarf = 3000  # K (typical M dwarf temperature)
    R_mdwarf = 0.3 * const.R_sun  # m (typical M dwarf radius)
    R_planet_mdwarf = 0.2 * const.R_earth  # m (smaller planet around M dwarf)
    
    # Calculate fluxes for M dwarf system
    planet_blackbody_mdwarf = blackbody_flux(wavelengths, T_earth)  # Same planet temperature
    stellar_blackbody_mdwarf = blackbody_flux(wavelengths, T_mdwarf)
    
    planetary_flux_mdwarf = R_planet_mdwarf**2 * planet_blackbody_mdwarf
    stellar_flux_mdwarf = R_mdwarf**2 * stellar_blackbody_mdwarf
    
    # Calculate flux ratio for M dwarf system
    flux_ratio_mdwarf = planetary_flux_mdwarf / stellar_flux_mdwarf
    
    # Create the plot
    plt.figure(figsize=(8, 5))
    
    # Add first 100 planets from exoplanet CSV
    print("Loading exoplanet data for wavelength analysis...")
    exo_path = os.path.join(LIFESIM_OUTER_DIR, 'exoplanets_2026.csv')
    df_exo = pd.read_csv(exo_path)
    
    # Filter for valid data (first 100 planets)
    df_exo = df_exo[
        (df_exo['pl_rade'].notnull()) & 
        (df_exo['pl_eqt'].notnull()) & 
        (df_exo['st_rad'].notnull()) & 
        (df_exo['st_teff'].notnull())
    ].head(100)
    
    print(f"Calculating flux ratios for {len(df_exo)} exoplanets...")
    
    # Calculate flux ratios for each planet at each wavelength
    for i, row in df_exo.iterrows():
        T_planet = row['pl_eqt']
        T_star = row['st_teff']
        R_planet = row['pl_rade'] * const.R_earth
        R_star = row['st_rad'] * const.R_sun
        
        # Calculate blackbody fluxes
        planet_bb = blackbody_flux(wavelengths, T_planet)
        stellar_bb = blackbody_flux(wavelengths, T_star)
        
        # Calculate fluxes
        planet_flux = R_planet**2 * planet_bb
        star_flux = R_star**2 * stellar_bb
        
        # Calculate flux ratio
        flux_ratio_planet = planet_flux / star_flux
        
        # Plot as thin, transparent line
        plt.plot(wavelengths * 1e6, flux_ratio_planet, 'gray', alpha=0.1, linewidth=0.5)
    
    # Plot flux ratio vs wavelength
    plt.plot(wavelengths * 1e6, flux_ratio, 'b-', label='Earth-like planet / Sun-like star')
    plt.plot(wavelengths * 1e6, flux_ratio_mdwarf, 'r-', label='Earth-like planet / M dwarf')
    
    # Add HWO threshold line
    threshold = 2.5e-11
    plt.axhline(y=threshold, color='k', linestyle='--', alpha=0.7, label=f'HWO threshold = {threshold:.1e}')
    
    # Add HWO wavelength range lines
    plt.axvline(x=0.2, color='green', linestyle='-', alpha=0.8, label='HWO (0.2-2.5 μm)')
    plt.axvline(x=2.5, color='green', linestyle='-', alpha=0.8)
    
    # Add LIFE wavelength range lines
    plt.axvline(x=4.0, color='blue', linestyle='-', alpha=0.8, label='LIFE (4.0-18.5 μm)')
    plt.axvline(x=18.5, color='blue', linestyle='-', alpha=0.8)
    
    # Add JWST wavelength range lines
    plt.axvline(x=0.6, color='orange', linestyle='-', alpha=0.8, label='JWST (0.6-28.5 μm)')
    plt.axvline(x=28.5, color='orange', linestyle='-', alpha=0.8)
    
    # Set log scales
    plt.xscale('log')
    plt.yscale('log')
    
    # Set axis limits to show full wavelength range
    plt.xlim(0, 33)
    
    # Labels and title
    plt.xlabel('Wavelength [μm]')
    plt.ylabel('Flux Ratio [Planetary/Stellar]')
    plt.title('Flux Ratio vs Wavelength')
    plt.legend()
    
    # Save the plot
    output_dir = os.path.join(LIFESIM_OUTER_DIR, 'plots')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'flux_ratio_vs_wavelength.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Flux ratio plot saved to {os.path.join(output_dir, 'flux_ratio_vs_wavelength.png')}")

if __name__ == "__main__":
    plot_planetary_vs_stellar_flux()
    # plot_flux_ratio_vs_wavelength() 