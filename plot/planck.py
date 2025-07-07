import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools import physics_constants as const
from tools.paths import PLOTS_DIR
from lifesim.util.habitable import single_habitable_zone

plt.rcParams.update({'font.size': 16})

def planck(wavelength_m, temperature):
    """Calculate spectral radiance using Planck's Law."""
    h, c, k = const.h, const.c, const.k
    exponent = h * c / (wavelength_m * k * temperature)
    exp_term = np.where(exponent > 100, np.inf, np.exp(exponent))
    B_lambda = (2 * h * c**2) / (wavelength_m**5) / (exp_term - 1)
    return np.where(np.isinf(exp_term), 0, B_lambda)

def lambert_phase(alpha_rad):
    """Lambertian phase function."""
    alpha_rad = np.clip(alpha_rad, 0, np.pi)
    return (np.sin(alpha_rad) + (np.pi - alpha_rad) * np.cos(alpha_rad)) / np.pi

def calculate_system_fluxes(T_star, T_planet, R_star, R_planet, D, wavelength_m, Ag, alpha_rad):
    """Calculate fluxes for a star-planet system."""
    # Convert to meters
    R_star_m = R_star * const.R_sun
    R_planet_m = R_planet * const.R_earth
    D_m = D * const.au_to_m
    
    # Calculate fluxes
    flux_star = planck(wavelength_m, T_star) * (R_star_m / D_m)**2
    flux_planet = planck(wavelength_m, T_planet) * (R_planet_m / D_m)**2
    
    # Reflected light
    phase = lambert_phase(alpha_rad)
    reflected_flux = flux_star * Ag * (R_planet_m / D_m)**2 * phase
    
    # Total planet flux and contrast
    total_planet_flux = flux_planet + reflected_flux
    # Avoid division by zero or very small values
    contrast = np.where(flux_star > 1e-50, total_planet_flux / flux_star, 0)
    
    return flux_star, flux_planet, reflected_flux, contrast

def plot_absorption_features(ax, wavelength_um, fluxes):
    """Plot atmospheric absorption features as gray lines with different linestyles."""
    features = {
        'H₂O': [0.94, 1.13, 1.38, 1.87],
        'CO₂': [2.0],
        'CH₄': [1.66, 2.3],
        'O₂': [0.688, 0.760, 1.06, 1.27],
        'NH₃': [1.5, 2.0, 2.2]
    }
    gray_shades = ['#888888', '#AAAAAA', '#666666', '#BBBBBB', '#444444']
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]

    for i, (molecule, wavelengths) in enumerate(features.items()):
        color = gray_shades[i % len(gray_shades)]
        linestyle = linestyles[i % len(linestyles)]
        for wavelength in wavelengths:
            if 0.2 <= wavelength <= 2.5:
                idx = np.argmin(np.abs(wavelength_um - wavelength))
                max_flux = max(fluxes[idx] for fluxes in fluxes) + 1000
                ax.axvline(x=wavelength, color=color, alpha=0.7, linestyle=linestyle, linewidth=1)
                # Special handling for 2.0 μm overlap
                if wavelength == 2.0 and molecule == 'CO₂':
                    ax.text(wavelength + 0.01, max_flux * 1.7, molecule, rotation=90, fontsize=14,
                            color=color, ha='left', va='bottom')
                elif wavelength == 2.0 and molecule == 'NH₃':
                    ax.text(wavelength - 0.01, max_flux * 1.3, molecule, rotation=90, fontsize=14,
                            color=color, ha='right', va='bottom')
                else:
                    # Offset label if another molecule is already at this wavelength
                    y_offset = 1.5
                    ax.text(wavelength, max_flux * y_offset, molecule, rotation=90, fontsize=14,
                            color=color, ha='right', va='bottom')



def main():
    """Main analysis and plotting function."""
    # Parameters
    wavelength_um = np.linspace(0.01, 10, 500)
    wavelength_m = wavelength_um * 1e-6
    
    # System parameters
    params = {
        'mdwarf': {'T_star': 3000, 'R_star': 0.13},  # Correct radius for 3000K M dwarf
        'sun': {'T_star': const.temp_sun, 'R_star': 1.0}
    }
    
    # Calculate habitable zone for M dwarf
    _, _, _, _, _, hz_center = single_habitable_zone(
        model='MS', temp_s=params['mdwarf']['T_star'], radius_s=params['mdwarf']['R_star']
    )
    
    # Use habitable zone center for M dwarf, 1 AU for Sun-like
    T_planet, R_planet, Ag, alpha_rad = const.T_earth, const.R_earth_example, const.A_g_earth, np.pi/4
    D_mdwarf = hz_center  # Use habitable zone center for M dwarf
    D_sun = 1.0  # Use 1 AU for Sun-like star
    
    # Calculate fluxes for both systems
    results = {}
    # M dwarf system with habitable zone distance
    results['mdwarf'] = calculate_system_fluxes(
        params['mdwarf']['T_star'], T_planet, params['mdwarf']['R_star'], 
        R_planet, D_mdwarf, wavelength_m, Ag, alpha_rad
    )
    # Sun-like system with 1 AU distance
    results['sun'] = calculate_system_fluxes(
        params['sun']['T_star'], T_planet, params['sun']['R_star'], 
        R_planet, D_sun, wavelength_m, Ag, alpha_rad
    )
    
    # Extract results
    _, flux_planet_mdwarf, reflected_mdwarf, _ = results['mdwarf']
    _, flux_planet_sun, reflected_sun, _ = results['sun']
    
    # Create single plot for both systems
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if isinstance(ax, np.ndarray):
        ax = ax.flat[0]
    
    # Plot M dwarf system
    ax.plot(wavelength_um, flux_planet_mdwarf, label='M dwarf planet emission (288K)', lw=2, color='darkgreen')
    ax.plot(wavelength_um, reflected_mdwarf, label='M dwarf reflected starlight', lw=2, color='red')
    
    # Plot Sun-like system
    ax.plot(wavelength_um, flux_planet_sun, label='Sun-like planet emission (288K)', lw=2, color='lightgreen')
    ax.plot(wavelength_um, reflected_sun, label='Sun-like reflected starlight', lw=2, color='orange')
    
    # Setup plot
    ax.set_ylabel('Spectral Radiance\n(W·m⁻³·sr⁻¹)')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_yscale('log')
    ax.set_xlim(0, 3)
    
    # Calculate proper y limits to include all data
    all_fluxes = [flux_planet_mdwarf, reflected_mdwarf, flux_planet_mdwarf + reflected_mdwarf,
                  flux_planet_sun, reflected_sun, flux_planet_sun + reflected_sun]
    min_flux = min([np.min(flux) for flux in all_fluxes if np.any(flux > 0)])
    max_flux = max([np.max(flux) for flux in all_fluxes])
    # Ensure positive limits for log scale and make room for reflected light
    min_flux = max(min_flux * 0.01, 1e-30)  # Lower minimum to see reflected light
    max_flux = max(max_flux * 10e4, 1e-20)    # Adjust maximum
    ax.set_ylim(min_flux, max_flux)
    
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add absorption features
    plot_absorption_features(ax, wavelength_um, all_fluxes)
    # Calculate contrast using approximation: T_p * R_p^2 / (T_star * R_star^2)
    T_planet_K = const.T_earth  # Planet temperature in Kelvin
    R_planet_earth = const.R_earth_example  # Planet radius in Earth radii
    
    # Convert to SI units using physics constants
    T_planet_SI = T_planet_K
    R_planet_SI = R_planet_earth * const.R_earth
    R_star_mdwarf_SI = params['mdwarf']['R_star'] * const.R_sun
    R_star_sun_SI = params['sun']['R_star'] * const.R_sun
    
    # Calculate contrast ratios
    contrast_mdwarf_approx = (T_planet_SI * R_planet_SI**2) / (params['mdwarf']['T_star'] * R_star_mdwarf_SI**2)
    contrast_sun_approx = (T_planet_SI * R_planet_SI**2) / (params['sun']['T_star'] * R_star_sun_SI**2)
    
    # # Add Fp/F* text boxes using the approximation (styled like legend)
    # ax.text(0.02, 0.12, f'M dwarf Fp/F*: {contrast_mdwarf_approx:.2e}', 
    #         transform=ax.transAxes, fontsize=14, verticalalignment='bottom',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
    # ax.text(0.02, 0.04, f'Sun-like Fp/F*: {contrast_sun_approx:.2e}', 
    #         transform=ax.transAxes, fontsize=14, verticalalignment='bottom',
    #         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=0.5))
    
    fig.suptitle('Planet Spectral Characteristics: M dwarf vs Sun-like Systems', fontsize=16, y=0.95)
    
    # Add system info as proper subtitle
    # system_info = f'M dwarf: T={params["mdwarf"]["T_star"]}K, R={params["mdwarf"]["R_star"]}R☉ at {D_mdwarf:.4f} AU | Sun-like: T={const.temp_sun}K, R={params["sun"]["R_star"]}R☉ at {D_sun:.1f} AU'
    # fig.text(0.5, 0.92, system_info, fontsize=14, horizontalalignment='center')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.95)
    plt.savefig(os.path.join(PLOTS_DIR, 'other_useful', 'planet_spectra_separate.png'), dpi=150, bbox_inches='tight')

if __name__ == "__main__":
    main() 
    