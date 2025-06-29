import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tools import physics_constants as const
from tools.paths import PLOTS_DIR

def planck(wavelength_m, temperature):
    """Calculate spectral radiance using Planck's Law."""
    h, c, k = const.h, const.c, const.k
    exponent = h * c / (wavelength_m * k * temperature)
    exp_term = np.where(exponent > 700, np.inf, np.exp(exponent))
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
    contrast = total_planet_flux / flux_star
    
    return flux_star, flux_planet, reflected_flux, contrast

def plot_absorption_features(ax, wavelength_um, fluxes):
    """Plot atmospheric absorption features."""
    features = {
        'H₂O': [0.94, 1.13, 1.38, 1.87],
        'CO₂': [2.0],
        'CH₄': [1.66, 2.3],
        'O₂': [0.688, 0.760, 1.06, 1.27],
        'NH₃': [1.5, 2.0, 2.2]
    }
    colors = ['purple', 'green', 'orange', 'brown', 'cyan']
    
    for i, (molecule, wavelengths) in enumerate(features.items()):
        for wavelength in wavelengths:
            if 0.2 <= wavelength <= 2.5:
                idx = np.argmin(np.abs(wavelength_um - wavelength))
                max_flux = max(fluxes[idx] for fluxes in fluxes)
                ax.axvline(x=wavelength, color=colors[i % len(colors)], alpha=0.7, linestyle='--', linewidth=1)
                ax.text(wavelength, max_flux * 1.5, molecule, rotation=90, fontsize=8, 
                       color=colors[i % len(colors)], ha='right', va='bottom')

def main():
    """Main analysis and plotting function."""
    # Parameters
    wavelength_um = np.linspace(0.01, 10, 500)
    wavelength_m = wavelength_um * 1e-6
    
    # System parameters
    params = {
        'mdwarf': {'T_star': 3000, 'R_star': 0.2},
        'sun': {'T_star': 5778, 'R_star': 1.0}
    }
    T_planet, R_planet, D, Ag, alpha_rad = 288, 1, 1, 0.3, np.pi/4
    
    # Calculate fluxes for both systems
    results = {}
    for system, p in params.items():
        results[system] = calculate_system_fluxes(
            p['T_star'], T_planet, p['R_star'], R_planet, D, wavelength_m, Ag, alpha_rad
        )
    
    # Extract results
    flux_planet_mdwarf, reflected_mdwarf, contrast_mdwarf = results['mdwarf'][1:4]
    reflected_sun, contrast_sun = results['sun'][2:4]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot spectral radiance
    ax.plot(wavelength_um, flux_planet_mdwarf, label='Planet (M-dwarf)', lw=2, color='green')
    ax.plot(wavelength_um, reflected_mdwarf, label='Reflected (M-dwarf)', lw=2, color='red')
    ax.plot(wavelength_um, reflected_sun, label='Reflected (Sun-like)', lw=2, color='gold')
    
    # Setup plot
    ax.set_ylabel('Spectral Radiance\n(W·m⁻³·sr⁻¹)')
    ax.set_xlabel('Wavelength (µm)')
    ax.set_yscale('log')
    ax.set_xlim(0, 3)
    ax.set_ylim(1e-30, 10)
    ax.set_title('Planet Emission + Reflected Starlight')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add HWO box and contrast info
    HWO_box = Rectangle((0.2, 1e-50), 2.3, 20 - 1e-10, linewidth=2, edgecolor='black',
                        facecolor='none', linestyle='--', label='HWO observable')
    ax.add_patch(HWO_box)
    
    contrast_info = f'Planet/Star Flux Ratios:\nM-dwarf: {np.max(contrast_mdwarf):.2e}\nSun-like: {np.max(contrast_sun):.2e}'
    ax.text(0.02, 0.98, contrast_info, transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add absorption features
    plot_absorption_features(ax, wavelength_um, [flux_planet_mdwarf, reflected_mdwarf, reflected_sun])
    
    fig.suptitle('Planet Blackbody Emission + Reflected Starlight\nM-dwarf vs Sun-like Systems', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'other_useful', 'flux_ratios.png'))

if __name__ == "__main__":
    main() 
    