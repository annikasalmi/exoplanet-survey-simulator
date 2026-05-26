from pathlib import Path
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

OUT_DIR = ROOT / "my_outputs" / "w2_kepler_rm_plots_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose "best" or "worst" for brightness/depth logic
MODE = "best"

# Focus window to match your comparison plots
APPLY_FOCUS_WINDOW = True
MASS_MIN = 0.0
MASS_MAX = 12.0
RADIUS_MIN = 0.5
RADIUS_MAX = 2.2

# Optional: use insolation as marker size
USE_INSOLATION_SIZE = True

# Save the plotted dataframe with rocky-curve diagnostics
SAVE_PLOT_DATA_WITH_DIAGNOSTICS = True


# ============================================================
# Helpers: loading data
# ============================================================

def load_all_kepler_catalogs(data_dir):
    pattern = os.path.join(data_dir, "kepler_catalog_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No Kepler catalog files found in:\n{data_dir}\n\n"
        )

    dfs = []

    for f in files:
        df_i = pd.read_csv(f)

        try:
            run_number = int(Path(f).stem.split("_")[-1])
        except ValueError:
            run_number = len(dfs)

        df_i["run"] = run_number
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(files)} Kepler catalog files.")
    print(f"Total planets before filtering: {len(df)}")

    return df


def to_bool_series(s):
    if s.dtype == bool:
        return s.fillna(False)

    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        return s.isin(["true", "1", "yes", "y"])

    return s.fillna(0).astype(bool)


def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + str(missing)
            + "\n\nAvailable columns:\n"
            + str(df.columns.tolist())
            + "\n\nFix: regenerate Kepler CSVs after updating kepler_data.py."
        )


def force_numeric_columns(df, cols):
    """Convert important plotting columns to numeric and keep impossible values as NaN."""
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# Helpers: rocky reference curve from ref.ddat
# ============================================================

def load_rocky_reference_curve(ref_path=REF_CURVE_PATH):
    """
    Load the Earth-like rocky mass-radius reference curve.

    ref.ddat columns:
        Column 1: Total mass in Earth masses
        Column 2: Total radius in Earth radii

    The first two columns are the black dashed curve in the original notebook.
    """
    ref_path = Path(ref_path)

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Could not find the rocky reference table at:\n{ref_path}\n\n"
            "You said you moved it to:\n"
            "run/kepler/reference_curves/ref.ddat\n\n"
            "Check that this path exists relative to your project root."
        )

    ref = np.loadtxt(ref_path, comments="#")

    if ref.ndim == 1:
        ref = ref.reshape(1, -1)

    if ref.shape[1] < 2:
        raise ValueError(
            f"Reference table must have at least two columns: mass and radius.\n"
            f"Current shape: {ref.shape}"
        )

    m_ref = ref[:, 0].astype(float)
    r_ref = ref[:, 1].astype(float)

    good = (
        np.isfinite(m_ref)
        & np.isfinite(r_ref)
        & (m_ref > 0)
        & (r_ref > 0)
    )

    m_ref = m_ref[good]
    r_ref = r_ref[good]

    order = np.argsort(m_ref)
    m_ref = m_ref[order]
    r_ref = r_ref[order]

    if len(m_ref) < 2:
        raise ValueError("Reference curve needs at least two valid mass-radius points.")

    return m_ref, r_ref


def rocky_radius_from_mass(m_planet, m_ref=None, r_ref=None):
    """
    Interpolate the rocky reference radius at a planet mass.

    Returns NaN outside the reference-table mass range instead of extrapolating,
    because extrapolation would invent a curve not present in ref.ddat.
    """
    if m_ref is None or r_ref is None:
        m_ref, r_ref = load_rocky_reference_curve()

    m_planet = np.asarray(m_planet, dtype=float)

    return np.interp(
        m_planet,
        m_ref,
        r_ref,
        left=np.nan,
        right=np.nan,
    )


def add_rocky_diagnostics(df, ref_path=REF_CURVE_PATH):
    """
    Add scientifically useful columns comparing each planet to the rocky curve.

    rocky_radius_ref:
        Expected radius from ref.ddat at the planet's mass.

    radius_ratio_to_rocky:
        R_planet / R_rocky.
        Around 1 means close to the rocky curve.
        Above 1 means puffier/larger than rocky expectation.

    radius_excess_to_rocky:
        R_planet - R_rocky in Earth radii.
    """
    df = df.copy()

    m_ref, r_ref = load_rocky_reference_curve(ref_path)
    df["rocky_radius_ref"] = rocky_radius_from_mass(df["mass_p"], m_ref, r_ref)
    df["radius_ratio_to_rocky"] = df["radius_p"] / df["rocky_radius_ref"]
    df["radius_excess_to_rocky"] = df["radius_p"] - df["rocky_radius_ref"]

    return df


def rocky_curve(ax, ref_path=REF_CURVE_PATH):
    """Draw the original ref.ddat curve, not the old approximation R = M^0.27."""
    m_ref, r_ref = load_rocky_reference_curve(ref_path)

    if APPLY_FOCUS_WINDOW:
        m_line_min = max(MASS_MIN, np.nanmin(m_ref))
        m_line_max = min(MASS_MAX, np.nanmax(m_ref))
    else:
        m_line_min = np.nanmin(m_ref)
        m_line_max = np.nanmax(m_ref)

    if m_line_max <= m_line_min:
        # This should not happen with your current focus window, but prevents a crash.
        return

    # Smooth line made by interpolation through the real table points.
    m_line = np.linspace(m_line_min, m_line_max, 500)
    r_line = rocky_radius_from_mass(m_line, m_ref, r_ref)

    ax.plot(
        m_line,
        r_line,
        "--",
        color="black",
        linewidth=1.7,
        alpha=0.9,
        zorder=3,
    )


# ============================================================
# Helpers: plotting
# ============================================================

def setup_axis(ax):
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=12)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=12)

    if APPLY_FOCUS_WINDOW:
        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_ylim(RADIUS_MIN, RADIUS_MAX)

    ax.grid(True, alpha=0.28)


def get_marker_sizes(df_sub):
    if not USE_INSOLATION_SIZE or "flux_p" not in df_sub.columns:
        return 32

    flux = pd.to_numeric(df_sub["flux_p"], errors="coerce")
    log_flux = np.log10(flux.clip(lower=1e-12))
    log_flux = np.asarray(log_flux, dtype=float)

    finite = np.isfinite(log_flux)

    if finite.sum() == 0:
        return np.full(len(df_sub), 36)

    # Normalize safely using the finite values only.
    lo = np.nanpercentile(log_flux[finite], 5)
    hi = np.nanpercentile(log_flux[finite], 95)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
        return np.full(len(df_sub), 36)

    norm = (log_flux - lo) / (hi - lo)
    norm = np.clip(norm, 0, 1)
    norm[~finite] = 0.5

    # Marker sizes: 22 to 80
    return 22 + 58 * norm


def plot_detected_vs_not(ax, df_sub, title, detected_col):
    if len(df_sub) == 0:
        ax.text(
            0.5,
            0.5,
            "No planets\nin this category",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.set_title(f"{title}\nN = 0", fontsize=13)
        setup_axis(ax)
        rocky_curve(ax)
        return

    detected = to_bool_series(df_sub[detected_col])

    df_missed = df_sub[~detected].copy()
    df_detected = df_sub[detected].copy()

    # Plot missed first, detected second.
    if len(df_missed) > 0:
        ax.scatter(
            df_missed["mass_p"],
            df_missed["radius_p"],
            s=get_marker_sizes(df_missed),
            alpha=0.8,
            color="gray",
            edgecolors="none",
            label="Not detected",
            zorder=1,
        )

    if len(df_detected) > 0:
        ax.scatter(
            df_detected["mass_p"],
            df_detected["radius_p"],
            s=get_marker_sizes(df_detected),
            alpha=0.9,
            color="tab:blue",
            edgecolors="black",
            linewidths=0.35,
            label="Detected",
            zorder=2,
        )

    rocky_curve(ax)
    setup_axis(ax)

    ax.set_title(
        f"{title}\nN = {len(df_sub)} | detected = {len(df_detected)}",
        fontsize=13,
    )


# ============================================================
# Load data
# ============================================================

print("Project root:", ROOT)
print("Data directory:", DATA_DIR)
print("Rocky reference table:", REF_CURVE_PATH)

df = load_all_kepler_catalogs(DATA_DIR)

mode = MODE.lower().strip()

if mode not in ["best", "worst"]:
    raise ValueError("MODE must be 'best' or 'worst'.")

transit_col = "transiting_geometric"
bright_col = f"kepler_star_bright_enough_{mode}"
depth_col = f"kepler_depth_pass_{mode}"

# Use the simple detected column, as you asked.
# If it does not exist, fall back to detected_best/worst.
if "detected" in df.columns:
    detected_col = "detected"
else:
    detected_col = f"detected_{mode}"

required_cols = [
    "mass_p",
    "radius_p",
    transit_col,
    bright_col,
    depth_col,
    detected_col,
]

require_columns(df, required_cols)

df = force_numeric_columns(df, ["mass_p", "radius_p", "flux_p"])
df = add_rocky_diagnostics(df, REF_CURVE_PATH)

m_ref, r_ref = load_rocky_reference_curve(REF_CURVE_PATH)
print("\nLoaded rocky reference curve:")
print(f"Reference points: {len(m_ref)}")
print(f"Mass range: {m_ref.min():.4g} to {m_ref.max():.4g} M_earth")
print(f"Radius range: {r_ref.min():.4g} to {r_ref.max():.4g} R_earth")

print("\nUsing columns:")
print("transit_col  =", transit_col)
print("bright_col   =", bright_col)
print("depth_col    =", depth_col)
print("detected_col =", detected_col)


# ============================================================
# Focus window
# ============================================================

if APPLY_FOCUS_WINDOW:
    focus_mask = (
        (df["mass_p"] > MASS_MIN)
        & (df["mass_p"] <= MASS_MAX)
        & (df["radius_p"] >= RADIUS_MIN)
        & (df["radius_p"] <= RADIUS_MAX)
    )

    df_plot = df[focus_mask].copy()

    print("\nApplied focus window:")
    print(f"Mass: {MASS_MIN} to {MASS_MAX} Earth masses")
    print(f"Radius: {RADIUS_MIN} to {RADIUS_MAX} Earth radii")
    print("Planets after focus filter:", len(df_plot))
else:
    df_plot = df.copy()


# ============================================================
# Masks: the big three
# ============================================================

transiting = to_bool_series(df_plot[transit_col])
bright = to_bool_series(df_plot[bright_col])
depth = to_bool_series(df_plot[depth_col])
detected = to_bool_series(df_plot[detected_col])

mask_transiting = transiting
mask_transiting_bright = transiting & bright
mask_transiting_bright_depth = transiting & bright & depth

print("\nBig-three category counts:")
print("Transiting:", int(mask_transiting.sum()))
print("Transiting + bright enough:", int(mask_transiting_bright.sum()))
print(
    "Transiting + bright enough + depth enough:",
    int(mask_transiting_bright_depth.sum()),
)
print("Detected:", int(detected.sum()))

# Sanity check
should_detect = mask_transiting_bright_depth
mismatch = should_detect != detected

print("\nSanity check:")
print("Rows where detected != transiting & bright & depth:", int(mismatch.sum()))

if mismatch.sum() > 0:
    mismatch_path = OUT_DIR / f"big_three_detection_mismatches_{mode}.csv"
    df_plot[mismatch].to_csv(mismatch_path, index=False)
    print("Saved mismatch rows to:")
    print(mismatch_path)


# ============================================================
# Plot
# ============================================================

panels = [
    ("1. Transiting planets", df_plot[mask_transiting].copy()),
    ("2. Transiting + bright enough", df_plot[mask_transiting_bright].copy()),
    (
        "3. Transiting + bright + depth enough",
        df_plot[mask_transiting_bright_depth].copy(),
    ),
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.8),
    constrained_layout=True,
)

for ax, (title, df_sub) in zip(axes, panels):
    plot_detected_vs_not(
        ax=ax,
        df_sub=df_sub,
        title=title,
        detected_col=detected_col,
    )

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Not detected",
        markerfacecolor="gray",
        markersize=9,
        alpha=0.8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Detected by Kepler",
        markerfacecolor="tab:blue",
        markeredgecolor="black",
        markersize=9,
    ),
    Line2D(
        [0],
        [0],
        linestyle="--",
        color="black",
        label="Earth-like rocky curve from ref.ddat",
    ),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=11,
    frameon=False,
)

title = (
    "Kepler big-three detection filter on the Radius–Mass plane\n"
    "Black dashed line = ref.ddat Earth-like rocky curve; "
    "marker size = insolation flux proxy"
    if USE_INSOLATION_SIZE
    else
    "Kepler big-three detection filter on the Radius–Mass plane\n"
    "Black dashed line = ref.ddat Earth-like rocky curve"
)

fig.suptitle(title, fontsize=16)

png_path = OUT_DIR / f"rm_big_three_detected_color_{mode}_with_ref_curve.png"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(png_path)


# ============================================================
# Save counts
# ============================================================

counts_df = pd.DataFrame({
    "category": [
        "transiting",
        "transiting_and_bright",
        "transiting_and_bright_and_depth",
        "detected",
        "mismatch_detected_vs_big_three",
    ],
    "count": [
        int(mask_transiting.sum()),
        int(mask_transiting_bright.sum()),
        int(mask_transiting_bright_depth.sum()),
        int(detected.sum()),
        int(mismatch.sum()),
    ],
})

counts_path = OUT_DIR / f"big_three_detected_color_counts_{mode}_with_ref_curve.csv"
counts_df.to_csv(counts_path, index=False)

print("Saved counts:")
print(counts_path)


# ============================================================
# Save plotted data with rocky-curve diagnostics
# ============================================================

if SAVE_PLOT_DATA_WITH_DIAGNOSTICS:
    diagnostic_path = OUT_DIR / f"rm_big_three_plot_data_{mode}_with_rocky_diagnostics.csv"
    df_plot.to_csv(diagnostic_path, index=False)

    print("Saved plotted data with rocky-curve diagnostics:")
    print(diagnostic_path)
    print("\nNew diagnostic columns:")
    print("  rocky_radius_ref        = interpolated R_rocky from ref.ddat")
    print("  radius_ratio_to_rocky  = R_planet / R_rocky")
    print("  radius_excess_to_rocky = R_planet - R_rocky")
