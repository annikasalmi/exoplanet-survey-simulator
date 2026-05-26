import numpy as np
from run.kepler.run_Kepler import main

df = main(
    parallel=False,
    nruns=np.arange(1),
    star_catalog="Gaia",
    run_anew=True,
)

print("\nKepler smoke test worked.")
print("Number of planets:", len(df))

important_cols = [
    "radius_p",
    "p_orb",
    "semimajor_p",
    "radius_s",
    "transit_probability",
    "transit_depth",
    "n_transits_kepler",
    "snr_kepler",
    "detected_best",
    "detected_worst",
]

existing_cols = [c for c in important_cols if c in df.columns]

print("\nFirst 10 useful rows:")
print(df[existing_cols].head(10))

print("\nDetection counts:")
print(df["detected_best"].value_counts(dropna=False))

print("\nTransit probability summary:")
print(df["transit_probability"].describe())

print("\nTransit depth summary:")
print(df["transit_depth"].describe())

print("\nSNR summary:")
print(df["snr_kepler"].describe())