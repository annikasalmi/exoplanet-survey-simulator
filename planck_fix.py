import numpy as np
import matplotlib.pyplot as plt
from tools import physics_constants as const

def planck(wavelength_m, temperature):
    """
    Calculate spectral radiance (W·sr⁻¹·m⁻³) using Planck's Law.
    
    Parameters:
    wavelength_m : array-like
        Wavelength in meters
    temperature : float
        Temperature in Kelvin
    
    Returns:
    array-like
        Spectral radiance in W·sr⁻¹·m⁻³
    """
    # Planck's constants from physics_constants
    h = const.h  # Planck's constant (J·s)
    c = const.c  # Speed of light (m/s)
    k = const.k  # Boltzmann constant (J/K)
    
    # Planck's Law: B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    # Avoid division by zero and overflow
    exponent = h * c / (wavelength_m * k * temperature)
    
    # Handle overflow for very small wavelengths
    exp_term = np.where(exponent > 700, np.inf, np.exp(exponent))
    
    # Calculate Planck function
    B_lambda = (2 * h * c**2) / (wavelength_m**5) / (exp_term - 1)
    
    # Handle cases where exp_term is inf (wavelength too small)
    B_lambda = np.where(np.isinf(exp_term), 0, B_lambda)
    
    return B_lambda

def lambert_phase(alpha_rad):
    """Lambertian phase function"""
    alpha_rad = np.clip(alpha_rad, 0, np.pi)
    return (np.sin(alpha_rad) + (np.pi - alpha_rad) * np.cos(alpha_rad)) / np.pi

def reflected_flux(F_star_at_earth, Ag, Rp, a, alpha_rad):
    """Reflected flux from planet, as observed at Earth (W/m²/µm)"""
    phase = lambert_phase(alpha_rad)
    return F_star_at_earth * Ag * (Rp / a)**2 * phase

# Define parameters
wavelength_um = np.linspace(0.2, 2.5, 1000)  # microns (0.2 to 2.5 µm)
wavelength_m = wavelength_um * 1e-6         # convert to meters

T_star = 3000       # M-dwarf
T_planet = 288      # Earth-like

# Approximate radii and distance (arbitrary units for flux scaling)
R_star = 0.2        # Solar radii
R_planet = 1        # Earth radii
D = 1               # AU (set to 1 for relative scaling)

# Convert to meters
R_star_m = R_star * const.R_sun
R_planet_m = R_planet * const.R_earth
D_m = D * const.au_to_m

# Calculate blackbody radiance
B_star = planck(wavelength_m, T_star)
B_planet = planck(wavelength_m, T_planet)

# Scale by (R²/D²) for relative observed flux
flux_star = B_star * (R_star_m / D_m)**2
flux_planet = B_planet * (R_planet_m / D_m)**2

# Calculate reflected light from star
# Parameters for reflected light calculation
Ag = 0.3  # Geometric albedo (Earth-like)
alpha_rad = np.pi/4  # Phase angle (45 degrees, quarter phase)
a = D_m  # Semi-major axis (same as distance for circular orbit)

# Calculate reflected flux from star light (this replaces the star flux)
reflected_flux_star = reflected_flux(flux_star, Ag, R_planet_m, a, alpha_rad)

# Calculate contrast ratio (reflected star light / original star light)
contrast_ratio = reflected_flux_star / flux_star

# Parameters for Sun-like star and Earth
T_sun = 5778       # Sun-like star (G-type)
T_earth = 288      # Earth-like planet
R_sun = 1.0        # Solar radii
R_earth_planet = 1 # Earth radii
D_earth = 1        # AU

# Convert to meters
R_sun_m = R_sun * const.R_sun
R_earth_planet_m = R_earth_planet * const.R_earth
D_earth_m = D_earth * const.au_to_m

# Calculate blackbody radiance for Sun-Earth system
B_sun = planck(wavelength_m, T_sun)
B_earth_planet = planck(wavelength_m, T_earth)

# Scale by (R²/D²) for relative observed flux
flux_sun = B_sun * (R_sun_m / D_earth_m)**2
flux_earth_planet = B_earth_planet * (R_earth_planet_m / D_earth_m)**2

# Calculate reflected light from star for Sun-like system
# Parameters for reflected light calculation
Ag = 0.3  # Geometric albedo (Earth-like)
alpha_rad = np.pi/4  # Phase angle (45 degrees, quarter phase)
a = D_earth_m  # Semi-major axis (same as distance for circular orbit)

# Calculate reflected flux from star light (this replaces the star flux)
reflected_flux_sun = reflected_flux(flux_sun, Ag, R_earth_planet_m, a, alpha_rad)

# Calculate contrast ratio (reflected star light / planet emission)
contrast_ratio_earth = reflected_flux_sun / flux_earth_planet

# Create combined plot with both systems
plt.figure(figsize=(12, 8))

# Plot all components on the same axes
plt.plot(wavelength_um, flux_planet, label=f'Habitable planet blackbody', lw=2, color='green')
plt.plot(wavelength_um, reflected_flux_star, label=f'Reflected Starlight (M-dwarf)', lw=2, color='red')
plt.plot(wavelength_um, reflected_flux_sun, label=f'Reflected Starlight (Sun-like)', lw=2, color='gold', linestyle='--')

plt.xlabel('Wavelength (µm)')
plt.ylabel('Flux (arbitrary units)')
plt.title('Planet Blackbody Emission + Reflected Starlight\nM-dwarf vs Sun-like Systems')
plt.legend()
plt.yscale('log')
plt.xlim(0.2, 2.5)
plt.ylim(1e-10, 10)
plt.grid(True, alpha=0.3)

# Add absorption lines for major atmospheric molecules
absorption_features = {
    'H₂O': [0.94, 1.13, 1.38, 1.87],  # Water vapor bands in our range
    'CO₂': [2.0],  # Key bands
    'CH₄': [1.66, 2.3],  # Methane bands
    'O₂': [0.688, 0.760, 1.06, 1.27],  # Oxygen bands
    'NH₃': [1.5, 2.0, 2.2],  # Ammonia bands in our range
}

# Colors for different molecules
colors = ['purple', 'green', 'orange', 'brown', 'cyan', 'pink', 'gray', 'olive', 'navy', 'maroon', 'teal', 'coral']

# Add absorption features
for i, (molecule, wavelengths) in enumerate(absorption_features.items()):
    for wavelength in wavelengths:
        if 0.2 <= wavelength <= 2.5:  # Only plot features in our range
            idx = np.argmin(np.abs(wavelength_um - wavelength))
            max_flux = max(flux_planet[idx], reflected_flux_star[idx], reflected_flux_sun[idx])
            
            plt.axvline(x=wavelength, color=colors[i % len(colors)], alpha=0.7, linestyle='--', linewidth=1)
            plt.text(wavelength, max_flux * 1.5, molecule, rotation=90, fontsize=8, 
                    color=colors[i % len(colors)], ha='right', va='bottom')

# Add overall title
plt.suptitle('Planet Blackbody Emission + Reflected Starlight\nM-dwarf vs Sun-like Systems', fontsize=14, y=0.98)

# Add legend for absorption features
plt.figtext(0.02, 0.98, 'Dashed lines: Atmospheric absorption features', 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print some key values
print(f"Peak wavelength for planet (M-dwarf system): {wavelength_um[np.argmax(flux_planet)]:.2f} µm")
print(f"Peak wavelength for planet (Sun-like system): {wavelength_um[np.argmax(flux_earth_planet)]:.2f} µm")
print(f"Ratio of reflected starlight to planet emission (M-dwarf): {np.max(contrast_ratio):.2e}")
print(f"Ratio of reflected starlight to planet emission (Sun-like): {np.max(contrast_ratio_earth):.2e}")
print(f"Ratio (M-dwarf/Sun-like): {np.max(contrast_ratio)/np.max(contrast_ratio_earth):.2f}")
print(f"Phase angle: {alpha_rad*180/np.pi:.1f}° (quarter phase)")
print(f"Geometric albedo: {Ag}")
print(f"Note: Green = planet blackbody emission, Blue = reflected starlight from planet")

# Print absorption features in our range
print("\nMajor absorption features in 0.2-2.5 µm range:")
for molecule, wavelengths in absorption_features.items():
    in_range = [w for w in wavelengths if 0.2 <= w <= 2.5]
    if in_range:
        print(f"{molecule}: {in_range} µm") 