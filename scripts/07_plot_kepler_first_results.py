import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"

out_dir = ROOT / "my_outputs" / "week2_kepler"
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

print("Loaded Kepler planets:", len(df))

# Plot 1: all planets, radius vs period
plt.figure(figsize=(8, 6))
plt.scatter(df["p_orb"], df["radius_p"], s=8, alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Orbital period [days]")
plt.ylabel("Planet radius [Earth radii]")
plt.title("Kepler input population: radius vs period")
plt.tight_layout()
plt.savefig(out_dir / "01_kepler_all_radius_vs_period.png", dpi=200)
plt.close()

# Plot 2: detected vs missed
detected = df[df["detected_best"] == True]
missed = df[df["detected_best"] == False]

plt.figure(figsize=(8, 6))
plt.scatter(missed["p_orb"], missed["radius_p"], s=8, alpha=0.35, label="Missed")
plt.scatter(detected["p_orb"], detected["radius_p"], s=12, alpha=0.8, label="Detected")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Orbital period [days]")
plt.ylabel("Planet radius [Earth radii]")
plt.title("First toy Kepler model: detected vs missed")
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / "02_kepler_detected_vs_missed_radius_period.png", dpi=200)
plt.close()

# Plot 3: SNR vs radius
plt.figure(figsize=(8, 6))
plt.scatter(df["radius_p"], df["snr_kepler"], s=8, alpha=0.5)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Planet radius [Earth radii]")
plt.ylabel("Toy Kepler SNR")
plt.title("Toy Kepler SNR vs planet radius")
plt.tight_layout()
plt.savefig(out_dir / "03_kepler_snr_vs_radius.png", dpi=200)
plt.close()

# Plot 4: transit probability vs period
plt.figure(figsize=(8, 6))
plt.scatter(df["p_orb"], df["transit_probability"], s=8, alpha=0.5)
plt.xscale("log")
plt.xlabel("Orbital period [days]")
plt.ylabel("Transit probability")
plt.title("Transit probability vs orbital period")
plt.tight_layout()
plt.savefig(out_dir / "04_kepler_transit_probability_vs_period.png", dpi=200)
plt.close()

print("Saved plots to:", out_dir)