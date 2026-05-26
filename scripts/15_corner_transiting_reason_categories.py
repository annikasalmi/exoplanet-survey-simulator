from pathlib import Path
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import corner
except ImportError as e:
    raise ImportError(
        "Missing package 'corner'. Install it with: pip install corner"
    ) from e


# ============================================================
# Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "run").exists():
    ROOT = Path.cwd()

DATA_GLOB = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_*.csv"

OUT_DIR = ROOT / "my_outputs" / "15_w2_kepler_reason_corner"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Use the same 5 variables as before.
PLOT_COLS = [
    "log10_radius_p",
    "log10_radius_s",
    "log10_p_orb",
    "log10_flux_p",
    "kepler_mag",
]

PLOT_LABELS = [
    r"$\log_{10}(R_p)$",
    r"$\log_{10}(R_\star)$",
    r"$\log_{10}(P_{\rm orb})$",
    r"$\log_{10}(F_p)$",
    r"$Kp$",
]

# To keep plots fast.
MAX_BACKGROUND_POINTS = 30000

# Raw counts on the diagonal are easier to interpret.
NORMALIZE_HISTOGRAMS = False

# Category colors
CATEGORY_COLORS = {
    "too_few_transits": "tab:blue",
    "host_star_too_faint": "tab:red",
    "too_shallow_or_low_mes": "tab:purple",
    "detected": "tab:orange",
}


# ============================================================
# Helpers
# ============================================================

def to_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.lower()
        .str.strip()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def first_existing_column(df, candidates, name_for_error):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"Could not find a column for {name_for_error}. Tried: {candidates}"
    )


def load_catalogues():
    files = sorted(glob.glob(str(DATA_GLOB)))
    if not files:
        raise FileNotFoundError(f"No files found at: {DATA_GLOB}")

    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df["source_file"] = Path(file).name
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(files)} file(s)")
    print(f"Total planets before cleaning: {len(df)}")
    return df


def prepare_columns(df):
    required = [
        "radius_p",
        "radius_s",
        "p_orb",
        "flux_p",
        "kepler_mag_used",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required physical columns: {missing}")

    # Resolve detector columns robustly.
    transit_col = first_existing_column(
        df,
        ["transiting_geometric"],
        "transiting flag",
    )
    enough_col = first_existing_column(
        df,
        ["kepler_enough_transits"],
        "enough transits flag",
    )
    bright_col = first_existing_column(
        df,
        ["bright_enough_kepler", "kepler_star_bright_enough"],
        "bright-enough flag",
    )
    depth_col = first_existing_column(
        df,
        ["kepler_depth_good", "kepler_depth_pass"],
        "depth/MES flag",
    )
    detected_col = first_existing_column(
        df,
        ["detected"],
        "detected flag",
    )

    # Keep only positive values for log quantities.
    for col in ["radius_p", "radius_s", "p_orb", "flux_p"]:
        df = df[df[col] > 0].copy()

    # Plot columns
    df["log10_radius_p"] = np.log10(df["radius_p"])
    df["log10_radius_s"] = np.log10(df["radius_s"])
    df["log10_p_orb"] = np.log10(df["p_orb"])
    df["log10_flux_p"] = np.log10(df["flux_p"])
    df["kepler_mag"] = df["kepler_mag_used"]

    # Standardized booleans
    df["is_transiting"] = to_bool(df[transit_col])
    df["has_enough_transits"] = to_bool(df[enough_col])
    df["star_bright_enough"] = to_bool(df[bright_col])
    df["passes_depth_or_mes"] = to_bool(df[depth_col])
    df["is_detected"] = to_bool(df[detected_col])

    # Remove rows with NaN/infs in plotting columns
    df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=PLOT_COLS)
        .copy()
    )

    print(f"Total planets after cleaning: {len(df)}")

    return df


def assign_transiting_reason_categories(df):
    """
    Make mutually exclusive categories among ONLY transiting planets.

    New category logic:

      1) detected
      2) host star too faint
      3) long period / low repeat signal
      4) too shallow / low MES
      5) uncategorized_transiting_fail

    Important:
    'long_period_low_repeat' is not a strict pipeline-failure reason.
    It is a scientific regime bucket that helps us study Kepler's bias
    against long-period planets.
    """
    df = df.copy()

    df["reason_category"] = "not_transiting"

    transiting = df["is_transiting"]
    not_detected = transiting & (~df["is_detected"])

    # Start with safety bucket
    df.loc[transiting, "reason_category"] = "uncategorized_transiting_fail"

    # Detected planets
    df.loc[
        transiting & df["is_detected"],
        "reason_category"
    ] = "detected"

    # Host star too faint
    df.loc[
        not_detected
        & (~df["star_bright_enough"]),
        "reason_category"
    ] = "host_star_too_faint"

    # ------------------------------------------------------------
    # New scientific category:
    # long-period / low-repeat transiting planets
    # ------------------------------------------------------------
    usable_for_repeat = (
        not_detected
        & df["star_bright_enough"]
        & df["p_orb"].notna()
        & df["n_transits_keplerish"].notna()
    )

    repeat_df = df[usable_for_repeat].copy()

    if len(repeat_df) > 0:
        # Long-period threshold:
        # top 25% longest-period transiting non-detected planets.
        period_threshold = repeat_df["p_orb"].quantile(0.75)

        # Low-repeat threshold:
        # bottom 25% in number of transits.
        ntransit_threshold = repeat_df["n_transits_keplerish"].quantile(0.25)

        print("\nLong-period / low-repeat thresholds:")
        print(f"p_orb >= {period_threshold:.3f} days")
        print(f"n_transits_keplerish <= {ntransit_threshold:.3f}")

        long_period_low_repeat = (
            usable_for_repeat
            & (
                (df["p_orb"] >= period_threshold)
                | (df["n_transits_keplerish"] <= ntransit_threshold)
            )
        )

        df.loc[
            long_period_low_repeat,
            "reason_category"
        ] = "long_period_low_repeat"

    # Too shallow / low MES
    # Everything that is still not detected, not faint, and not already
    # long-period/low-repeat goes here if it fails the depth/MES condition.
    df.loc[
        not_detected
        & df["star_bright_enough"]
        & (df["reason_category"] == "uncategorized_transiting_fail")
        & (~df["passes_depth_or_mes"]),
        "reason_category"
    ] = "too_shallow_or_low_mes"

    return df


def get_plot_ranges(df, cols):
    ranges = []
    for c in cols:
        x = df[c].to_numpy()
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        pad = 0.03 * (xmax - xmin) if xmax > xmin else 0.1
        ranges.append((xmin - pad, xmax + pad))
    return ranges


def make_one_corner_plot(background_df, highlight_df, title, color, out_path, ranges):
    if len(background_df) > MAX_BACKGROUND_POINTS:
        background_df = background_df.sample(MAX_BACKGROUND_POINTS, random_state=42)

    hist_kwargs = {
        "density": NORMALIZE_HISTOGRAMS,
        "histtype": "step",
        "linewidth": 1.2,
    }

    fig = corner.corner(
        background_df[PLOT_COLS].to_numpy(),
        labels=PLOT_LABELS,
        range=ranges,
        color="0.7",
        bins=30,
        plot_datapoints=True,
        plot_density=False,
        plot_contours=False,
        fill_contours=False,
        show_titles=False,
        hist_kwargs=hist_kwargs,
    )

    if len(highlight_df) > 0:
        corner.corner(
            highlight_df[PLOT_COLS].to_numpy(),
            labels=PLOT_LABELS,
            range=ranges,
            fig=fig,
            color=color,
            bins=30,
            plot_datapoints=True,
            plot_density=False,
            plot_contours=False,
            fill_contours=False,
            show_titles=False,
            hist_kwargs=hist_kwargs,
        )

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    df = load_catalogues()
    df = prepare_columns(df)
    df = assign_transiting_reason_categories(df)

    # Only use transiting planets as the gray background
    transiting_df = df[df["is_transiting"]].copy()

    print("\nCounts among ALL planets:")
    print(df["reason_category"].value_counts())

    print("\nCounts among TRANSITING planets only:")
    print(transiting_df["reason_category"].value_counts())

    # Warn if anything landed in the safety bucket
    safety_count = (transiting_df["reason_category"] == "uncategorized_transiting_fail").sum()
    if safety_count > 0:
        print(
            f"\nWarning: {safety_count} transiting planets fell into "
            f"'uncategorized_transiting_fail'."
        )

    ranges = get_plot_ranges(transiting_df, PLOT_COLS)

    categories_to_plot = [
        "too_few_transits",
        "host_star_too_faint",
        "too_shallow_or_low_mes",
        "detected",
    ]

    pretty_names = {
        "too_few_transits": "Transiting but Too Few Transits",
        "host_star_too_faint": "Transiting but Host Star Too Faint",
        "too_shallow_or_low_mes": "Transiting but Too Shallow / Low MES",
        "detected": "Detected",
    }

    for cat in categories_to_plot:
        sub = transiting_df[transiting_df["reason_category"] == cat].copy()

        title = (
            f"Kepler Detectability Corner Plot\n"
            f"Gray = all transiting planets, "
            f"{CATEGORY_COLORS[cat]} = {pretty_names[cat]}\n"
            f"N_transiting = {len(transiting_df)}, N_category = {len(sub)}"
        )

        out_path = OUT_DIR / f"corner_{cat}.png"

        make_one_corner_plot(
            background_df=transiting_df,
            highlight_df=sub,
            title=title,
            color=CATEGORY_COLORS[cat],
            out_path=out_path,
            ranges=ranges,
        )

    # Optional: save the categorized table
    out_csv = OUT_DIR / "kepler_catalog_with_reason_categories.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved categorized catalogue to: {out_csv}")


if __name__ == "__main__":
    main()