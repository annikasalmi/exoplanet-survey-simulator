#!/usr/bin/env python3

import os

from tools.exoplanet_catalog import load_and_filter_exoplanets
from telescopes.hwo.detection_model import HWOData

# Load and process data
print("Loading exoplanet data...")
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
df = load_and_filter_exoplanets(
    os.path.join(REPO_ROOT, 'data/exoplanet_csv', 'exoplanets_2026.csv'))

print(f"\nBefore HWO processing:")
print(f"  Total planets: {len(df)}")
print(f"  detected_best column exists: {'detected_best' in df.columns}")
print(f"  detected_worst column exists: {'detected_worst' in df.columns}")

# Process through HWO
print(f"\nProcessing through HWO...")
hwo_data = HWOData(df)
hwo_data.determine_detectable()
df_processed = hwo_data.catalog

print(f"\nAfter HWO processing:")
print(f"  Total planets: {len(df_processed)}")
print(f"  detected_best column exists: {'detected_best' in df_processed.columns}")
print(f"  detected_worst column exists: {'detected_worst' in df_processed.columns}")

if 'detected_best' in df_processed.columns:
    print("Unique values in detected_best:", df_processed['detected_best'].unique())
    print("Value counts for detected_best:", df_processed['detected_best'].value_counts())
    print("Sample detected_best values:", df_processed['detected_best'].head(10).tolist())
else:
    print("  ERROR: detected_best column not found!") 