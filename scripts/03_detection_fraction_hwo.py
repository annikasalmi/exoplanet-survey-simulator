import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "run" / "hwo" / "data" / "Gaia" / "hwo_catalog_0.csv"

df = pd.read_csv(csv_path)

if "detected_best" not in df.columns:
    raise ValueError("No detected_best column found. Check HWO output.")

# Radius bins
df["radius_bin_simple"] = pd.cut(
    df["radius_p"],
    bins=[0, 1.5, 3.0, 6.0, 100],
    labels=["Earth-size <1.5", "Sub-Neptune 1.5-3", "Neptune-ish 3-6", "Giant >6"],
)

# Period bins
df["period_bin_simple"] = pd.cut(
    df["p_orb"],
    bins=[0, 10, 50, 200, 10000],
    labels=["short <10d", "medium 10-50d", "long 50-200d", "very long >200d"],
)

summary = (
    df.groupby(["radius_bin_simple", "period_bin_simple"], observed=False)
    .agg(
        injected=("detected_best", "size"),
        detected=("detected_best", "sum"),
    )
    .reset_index()
)

summary["detection_fraction"] = summary["detected"] / summary["injected"]

print(summary)

out_path = ROOT / "my_outputs" / "hwo_detection_fraction_table.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(out_path, index=False)

print("Saved:", out_path)