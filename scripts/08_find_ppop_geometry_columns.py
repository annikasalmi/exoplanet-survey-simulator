import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"

df = pd.read_csv(csv_path)

keywords = [
    "inc", "incl", "i_", "ecc", "omega", "arg",
    "radius", "mass", "semi", "period", "mag", "phot", "lum"
]

for key in keywords:
    matches = [c for c in df.columns if key.lower() in c.lower()]
    if matches:
        print(f"\nColumns matching '{key}':")
        for c in matches:
            print("  ", c)

print("\nAll columns:")
print(df.columns.tolist())