import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

# --- Sample data for illustration purposes ---
L_vals = np.logspace(-2, 1, 1000)  # Stellar Luminosity [L☉]
D_vals = np.linspace(4, 15, 1000)    # Distance [pc]
L_grid, D_grid = np.meshgrid(L_vals, D_vals)

# --- Synthetic detectability range data ---
# Simulate % of range detectable (0: none, 0.5: partial, 1: full)
detectability = np.ones_like(L_grid)

# --- Constants ---
R_earth_m = 6.371e6
AU_m = 1.496e11
pc_m = 3.086e16
rad2arcsec = 206265

A_g = 0.2
Phi = 1.0
flux_threshold = 2.5e-11
theta_limit_arcsec = 0.0206  # 30 mas

# --- Derived HZ orbit (a = sqrt(L) * AU) ---
a_hz_m = np.sqrt(L_grid) * AU_m
distance_m = D_grid * pc_m
theta_arcsec = (a_hz_m / distance_m) * rad2arcsec

# --- Planet flux ratios for two radii ---
Rp_small = 0.5 * R_earth_m
Rp_large = 2.6 * R_earth_m

flux_ratio_small = A_g * (Rp_small / a_hz_m) ** 2 * Phi
flux_ratio_large = A_g * (Rp_large / a_hz_m) ** 2 * Phi

# --- Check detectability for each radius ---
detect_small = (flux_ratio_small >= flux_threshold) & (theta_arcsec >= theta_limit_arcsec)
detect_large = (flux_ratio_large >= flux_threshold) & (theta_arcsec >= theta_limit_arcsec)

# --- Assign region codes ---
# 2 = fully detectable (small detectable)
# 1 = partially detectable (only large detectable)
# 0 = not detectable
detectability = np.zeros_like(L_grid, dtype=int)
detectability[detect_large] = 1
detectability[detect_small] = 2  # overrides partial if both are True

# --- Convert percent range detectable to 3 categories ---
# 0 = None, 1 = Partial (1.1–2.6 R⊕), 2 = Full (0.5–2.6 R⊕)
region = detectability

# --- Custom colormap for the 3 categories ---
cmap = ListedColormap(['#f7cac9', '#f4b183', '#88b04b'])  # pink, orange, green

# --- Plotting ---
fig, ax = plt.subplots(figsize=(12, 8))

L_edges = np.logspace(np.log10(L_vals[0]), np.log10(L_vals[-1]), L_vals.size + 1)
D_edges = np.linspace(D_vals[0], D_vals[-1], D_vals.size + 1)

im = ax.pcolormesh(L_edges, D_edges, region, cmap=cmap, shading='auto', vmin=0, vmax=2)

ax.set_xscale('log')
ax.set_xlabel('Stellar Luminosity [L☉]')
ax.set_ylabel('Distance [pc]')
ax.set_title('Detectable Radius Range (0.5–2.6 R⊕)\n% of Full Range Detectable')

# --- Reference lines for M-dwarf and G-star regions ---
ax.axvline(0.08, color='red', linestyle='--', linewidth=2)
ax.axvline(0.6, color='gold', linestyle='--', linewidth=2)
ax.axvline(1.5, color='gold', linestyle='--', linewidth=2)

# --- Legend construction ---
legend_elements = [
    Patch(facecolor='#f7cac9', label='None'),
    Patch(facecolor='#f4b183', label='1.1–2.6 R⊕'),
    Patch(facecolor='#88b04b', label='0.5–2.6 R⊕'),
    Line2D([0], [0], color='red', linestyle='--', label='M-dwarf Region'),
    Line2D([0], [0], color='gold', linestyle='--', label='G Star Region')
]
ax.legend(handles=legend_elements, loc='lower right')

# --- Colorbar ---
cbar = fig.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5])
cbar.ax.set_yticklabels(['None', '1.1–2.6 R⊕', '0.5–2.6 R⊕'])
cbar.set_label("Detectability Regions")

# Plot it
ax.plot(L_fit, D_fit, color='black', linestyle='--', linewidth=2, label='Boundary Fit')

plt.tight_layout()
plt.show()
