"""
Debug script to visualize exozodi surface brightness levels and ratios.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Path to your catalog (update if needed)
CATALOG_PATH = 'run/hwo/data/Gaia/hwo_catalog_0.csv'

# Load data
if not os.path.exists(CATALOG_PATH):
    raise FileNotFoundError(f"File not found: {CATALOG_PATH}")
df = pd.read_csv(CATALOG_PATH)

# Check for required columns
required_cols = [
    'exozodi_surface_brightness_ratio_best',
    'exozodi_surface_brightness_ratio_worst',
]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Try to find raw exozodi and star surface brightness columns if present
possible_sb_cols = [
    'exozodi_surface_brightness_best',
    'star_surface_brightness_best',
    'exozodi_surface_brightness_worst',
    'star_surface_brightness_worst',
]

has_raw_sb = all(col in df.columns for col in possible_sb_cols[:2])

plt.figure(figsize=(14, 6))

# 1. Histogram of exozodi surface brightness ratio (best)
plt.subplot(1, 2, 1)
ratios = df['exozodi_surface_brightness_ratio_best']
plt.hist(ratios, bins=np.logspace(-10, 2, 60), color='skyblue', edgecolor='black', alpha=0.8)
plt.axvline(1.0, color='red', linestyle='--', label='Rejection threshold (1.0)')
plt.xscale('log')
plt.xlabel('Exozodi Surface Brightness Ratio (Best)')
plt.ylabel('Number of Planets')
plt.title('Distribution of Exozodi Surface Brightness Ratios (Best)')
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.5)

# 2. Histogram of raw exozodi surface brightness (if available)
plt.subplot(1, 2, 2)
if has_raw_sb:
    exozodi_sb = df['exozodi_surface_brightness_best']
    plt.hist(exozodi_sb, bins=np.logspace(np.log10(exozodi_sb[exozodi_sb>0].min()), np.log10(exozodi_sb.max()), 60), color='orange', edgecolor='black', alpha=0.8)
    plt.xscale('log')
    plt.xlabel('Exozodi Surface Brightness (Best)')
    plt.title('Raw Exozodi Surface Brightness (Best)')
    plt.grid(True, which='both', ls='--', alpha=0.5)
else:
    plt.text(0.5, 0.5, 'Raw exozodi surface brightness not found in file', ha='center', va='center', fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.suptitle('Exozodi Surface Brightness Debugger', fontsize=16, y=1.04)
plt.show() 