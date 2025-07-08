import pandas as pd
import matplotlib.pyplot as plt
import os

# Determine the correct path to the CSV
csv_path = "exoplanets_all_2025.csv"
if not os.path.exists(csv_path):
    csv_path = "../exoplanets_all_2025.csv"

df = pd.read_csv(csv_path)

# Filter for planets with R < 2.6 R_earth and stellar temperature < 4000K
filtered = df[(df['pl_rade'] < 2.6) & (df['st_teff'] < 4000)]

# Drop rows with missing values in required columns (subset must be a list)
filtered = filtered.dropna(subset=["pl_rade", "st_teff", "disc_year"])

# Find the coolest star in the dataset
coolest_star = filtered.loc[filtered['st_teff'].idxmin()]
print(f"Coolest star: {coolest_star['pl_name']} with T_eff = {coolest_star['st_teff']:.1f}K")

# Find LP 890-9 b radius
lp_890_9_b = filtered[filtered['pl_name'] == 'LP 890-9 b']
if not lp_890_9_b.empty:
    radius = lp_890_9_b.iloc[0]['pl_rade']
    print(f"LP 890-9 b radius: {radius:.3f} R⊕")
else:
    print("LP 890-9 b not found in the filtered dataset")

plt.figure(figsize=(15, 6))
scatter = plt.scatter(filtered['st_teff'], filtered['disc_year'], 
                     c=filtered['pl_rade'], alpha=0.7, s=50)

# Add planet names as labels with better positioning
for idx, row in filtered.iterrows():
    # Alternate label positions to reduce overlap with larger offsets
    offset_x = 8 if idx % 2 == 0 else -8
    offset_y = 8 if idx % 3 == 0 else -8
    plt.annotate(row['pl_name'], 
                (row['st_teff'], row['disc_year']), 
                xytext=(offset_x, offset_y), textcoords='offset points',
                fontsize=6, alpha=0.8, ha='center', va='center', rotation=30)
    
    # Add rectangle around TOI 700 planets
    if 'TOI-700' in str(row['pl_name']):
        rect = plt.Rectangle((row['st_teff'] - 25, row['disc_year'] - 0.2), 
                           width=50, height=0.4, 
                           fill=False, color='black', linewidth=2)
        plt.gca().add_patch(rect)

# Add legend for the red rectangle
from matplotlib.patches import Rectangle
legend_elements = [Rectangle((0, 0), 1, 1, fill=False, color='black', linewidth=2, label='TOI-700 planets')]
plt.legend(handles=legend_elements, loc='lower right')

plt.xlabel('Stellar Effective Temperature (K)', fontsize=16)
plt.ylabel('Discovery Year', fontsize=16)
plt.title('Exoplanets (R < 2.6 R$_\oplus$) around Stars with T$_{eff}$ < 4000K', fontsize=16)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Planet Radius (R$_\oplus$)', rotation=270, labelpad=15)

plt.tight_layout()
plt.show() 