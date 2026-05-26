import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT/"mdwarf-habitability" / "run" / "hwo" / "data" / "Gaia" / "hwo_catalog_0.csv"
print("Reading:", csv_path)

df = pd.read_csv(csv_path)

print("\nNumber of planets:", len(df))
print("\nColumns:")
print(df.columns.tolist())

important_cols = [
    "radius_p",
    "p_orb",
    "semimajor_p",
    "mass_p",
    "temp_p",
    "radius_s",
    "mass_s",
    "temp_s",
    "distance_s",
    "stype",
    "habitable",
    "detected_best",
    "detected_worst",
]

existing_cols = [c for c in important_cols if c in df.columns]

print("\nFirst 10 planets:")
print(df[existing_cols].head(10))

print("\nPlanet radius summary:")
print(df["radius_p"].describe())

print("\nOrbital period summary:")
print(df["p_orb"].describe())

if "detected_best" in df.columns:
    print("\nDetected best counts:")
    print(df["detected_best"].value_counts(dropna=False))

if "detected_worst" in df.columns:
    print("\nDetected worst counts:")
    print(df["detected_worst"].value_counts(dropna=False))

if "habitable" in df.columns:
    print("\nHabitable counts:")
    print(df["habitable"].value_counts(dropna=False))