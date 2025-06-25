import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from run.hwo.hwo_run_multiple import main as main_hwo
from run.run_sim import run_sim

# Set up parameters as in run_sim's __main__
PARALLEL = True
STAR_CATALOG = 'Gaia'
NRUNS = np.arange(500)

if __name__ == '__main__':
    df = run_sim(func=main_hwo, parallel=True, star_catalog=STAR_CATALOG, nruns=NRUNS,run_anew=False, name = 'hwo', plot=False)

    if 'z' not in df.columns:
        raise ValueError("Column 'z' not found in the DataFrame.")

    # z = df['z'].dropna().to_numpy()

    print(f"\n'z' column statistics:")
    print(f"  Count: {len(z)}")
    print(f"  Min: {np.min(z):.3g}")
    print(f"  Max: {np.max(z):.3g}")
    print(f"  Mean: {np.mean(z):.3g}")
    print(f"  Median: {np.median(z):.3g}")
    print(f"  Std: {np.std(z):.3g}")

    print(f"  First 10 values: {z[:10]}")

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(z, bins=50, alpha=0.7, color='green', edgecolor='black')
    plt.title("Histogram of 'z' column")
    plt.xlabel('z')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(__file__), 'z_hist.png')
    plt.savefig(out_png)
    print(f"  Histogram saved to: {out_png}")
    plt.close() 