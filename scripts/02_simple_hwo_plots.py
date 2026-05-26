import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "run" / "hwo" / "data" / "Gaia" / "hwo_catalog_0.csv"
out_dir = ROOT / "my_outputs" / "first_hwo_plots"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

print("Loaded planets:", len(df))

# Plot 1: Radius vs period
plt.figure(figsize=(8, 6))
plt.scatter(df["p_orb"], df["radius_p"], s=8, alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Orbital period, p_orb [days]")
plt.ylabel("Planet radius, radius_p [Earth radii]")
plt.title("All injected planets: radius vs period")
plt.tight_layout()
plt.savefig(out_dir / "01_all_radius_vs_period.png", dpi=200)
plt.close()

# Plot 2: Radius vs incoming flux, if flux column exists
flux_col = None
for candidate in ["flux_p", "Fp", "fp"]:
    if candidate in df.columns:
        flux_col = candidate
        break

if flux_col is not None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[flux_col], df["radius_p"], s=8, alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(f"Incoming/planet flux column: {flux_col}")
    plt.ylabel("Planet radius, radius_p [Earth radii]")
    plt.title("All injected planets: radius vs flux")
    plt.tight_layout()
    plt.savefig(out_dir / "02_all_radius_vs_flux.png", dpi=200)
    plt.close()

# Plot 3: Detected vs missed, if detected_best exists
if "detected_best" in df.columns:
    detected = df[df["detected_best"] == True]
    missed = df[df["detected_best"] == False]

    plt.figure(figsize=(8, 6))
    plt.scatter(missed["p_orb"], missed["radius_p"], s=8, alpha=0.35, label="Missed")
    plt.scatter(detected["p_orb"], detected["radius_p"], s=12, alpha=0.8, label="Detected")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Orbital period, p_orb [days]")
    plt.ylabel("Planet radius, radius_p [Earth radii]")
    plt.title("HWO detected vs missed: radius vs period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "03_hwo_detected_vs_missed_radius_period.png", dpi=200)
    plt.close()

print("Saved plots to:", out_dir)