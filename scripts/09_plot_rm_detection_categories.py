from pathlib import Path
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Settings
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia"
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

OUT_DIR = ROOT / "my_outputs" / "w2_kepler_rm_Insolation_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Choose "best" or "worst" for brightness/depth columns.
DETECTION_MODE = "best"

# Focus on the same physical range as your comparison plots.
APPLY_FOCUS_WINDOW = True
MASS_MIN = 0.0
MASS_MAX = 12.0
RADIUS_MIN = 0.5
RADIUS_MAX = 2.2

# Draw the original rocky/Earth-like mass-radius curve from ref.ddat.
DRAW_ROCKY_CURVE = True

# Use the attached-result style for insolation bins:
#   I < 10, I < 50, I > 50
# The second panel is intentionally cumulative, not 10 <= I < 50.
USE_REFERENCE_STYLE_CUMULATIVE_BINS = True

# Which population should go into the 3-panel insolation figure?
# Options: "detected", "big_three", "all", "transiting"
# "detected" is usually what you want for comparing with observed Kepler-like planets.
INSOLATION_BIN_SOURCE = "detected"

# Marker/visual settings.
POINT_SIZE = 28
POINT_ALPHA = 0.82
CMAP = "plasma"
COLOR_PERCENTILES = (1, 99)
COLORBAR_PER_INSOLATION_PANEL = True

# Save a dataframe with added rocky-curve diagnostics.
SAVE_PLOT_DATA_WITH_DIAGNOSTICS = True


# ============================================================
# Helper functions: loading and validation
# ============================================================

def load_all_kepler_catalogs(data_dir):
    pattern = os.path.join(data_dir, "kepler_catalog_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No Kepler catalog files found here:\n{data_dir}\n\n"
            "Expected files like kepler_catalog_0.csv, kepler_catalog_1.csv, etc."
        )

    dfs = []

    for f in files:
        df_i = pd.read_csv(f)

        # Extract run number from filename.
        name = Path(f).stem
        try:
            run_number = int(name.split("_")[-1])
        except ValueError:
            run_number = len(dfs)

        df_i["run"] = run_number
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(files)} Kepler catalog files.")
    print(f"Total planets before filtering: {len(df)}")

    return df


def to_bool_series(s):
    """Robustly convert a pandas Series to boolean."""
    if s.dtype == bool:
        return s.fillna(False)

    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        return s.isin(["true", "1", "yes", "y", "t"])

    return s.fillna(0).astype(bool)


def require_columns(df, columns):
    missing = [c for c in columns if c not in df.columns]

    if missing:
        raise ValueError(
            "Missing required columns:\n"
            + str(missing)
            + "\n\nAvailable columns:\n"
            + str(df.columns.tolist())
            + "\n\nMost likely fix: regenerate your Kepler CSVs after updating kepler_data.py."
        )


def force_numeric_columns(df, columns):
    """Convert important plotting columns to numeric and keep bad values as NaN."""
    df = df.copy()

    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ============================================================
# Helper functions: rocky reference curve from ref.ddat
# ============================================================

def load_rocky_reference_curve(ref_path=REF_CURVE_PATH):
    """
    Load the Earth-like rocky mass-radius reference curve.

    ref.ddat columns:
        Column 1: total mass in Earth masses
        Column 2: total radius in Earth radii

    This replaces the old toy curve R = M^0.27.
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

    Returns NaN outside the reference-table mass range instead of extrapolating.
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
    Add columns comparing each planet to the rocky curve.

    rocky_radius_ref:
        Expected radius from ref.ddat at this planet's mass.

    radius_ratio_to_rocky:
        R_planet / R_rocky. Around 1 means close to the rocky curve.
        Above 1 means larger/puffier than rocky expectation.

    radius_excess_to_rocky:
        R_planet - R_rocky in Earth radii.
    """
    df = df.copy()

    m_ref, r_ref = load_rocky_reference_curve(ref_path)
    df["rocky_radius_ref"] = rocky_radius_from_mass(df["mass_p"], m_ref, r_ref)
    df["radius_ratio_to_rocky"] = df["radius_p"] / df["rocky_radius_ref"]
    df["radius_excess_to_rocky"] = df["radius_p"] - df["rocky_radius_ref"]

    return df


def add_rocky_curve(ax, ref_path=REF_CURVE_PATH):
    """Draw the real ref.ddat curve as the black dashed reference line."""
    if not DRAW_ROCKY_CURVE:
        return

    m_ref, r_ref = load_rocky_reference_curve(ref_path)

    if APPLY_FOCUS_WINDOW:
        m_line_min = max(MASS_MIN, np.nanmin(m_ref))
        m_line_max = min(MASS_MAX, np.nanmax(m_ref))
    else:
        m_line_min = np.nanmin(m_ref)
        m_line_max = np.nanmax(m_ref)

    if m_line_max <= m_line_min:
        return

    # Smooth curve by linearly interpolating through the original table points.
    m_line = np.linspace(m_line_min, m_line_max, 500)
    r_line = rocky_radius_from_mass(m_line, m_ref, r_ref)

    ax.plot(
        m_line,
        r_line,
        linestyle="--",
        linewidth=1.7,
        color="black",
        alpha=0.92,
        zorder=3,
        label="Earth-like rocky curve from ref.ddat",
    )


# ============================================================
# Helper functions: plotting
# ============================================================

def setup_rm_axis(ax):
    ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=11)
    ax.set_ylabel(r"Radius [$R_\oplus$]", fontsize=11)

    if APPLY_FOCUS_WINDOW:
        ax.set_xlim(MASS_MIN, MASS_MAX)
        ax.set_ylim(RADIUS_MIN, RADIUS_MAX)

    ax.grid(True, alpha=0.28)


def log_flux_values(df_sub):
    flux = pd.to_numeric(df_sub["flux_p"], errors="coerce")
    flux = flux.clip(lower=1e-12)
    return np.log10(flux.to_numpy(dtype=float))


def compute_log_flux_limits(df_sub, fallback=(-1.0, 2.0)):
    if len(df_sub) == 0 or "flux_p" not in df_sub.columns:
        return fallback

    log_flux = log_flux_values(df_sub)
    finite = log_flux[np.isfinite(log_flux)]

    if len(finite) == 0:
        return fallback

    p_lo, p_hi = COLOR_PERCENTILES
    vmin = np.nanpercentile(finite, p_lo)
    vmax = np.nanpercentile(finite, p_hi)

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return fallback

    if np.isclose(vmin, vmax):
        vmin -= 0.1
        vmax += 0.1

    return vmin, vmax


def scatter_rm(ax, df_sub, title, vmin, vmax, subtitle=None):
    """Scatter mass-radius points colored by log10(insolation flux)."""
    setup_rm_axis(ax)
    add_rocky_curve(ax)

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

        ax.set_title(f"{title}\nN = 0", fontsize=12)
        return None

    log_flux = log_flux_values(df_sub)

    sc = ax.scatter(
        df_sub["mass_p"],
        df_sub["radius_p"],
        c=log_flux,
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        edgecolors="none",
        zorder=2,
    )

    if subtitle:
        ax.set_title(f"{title}\n{subtitle}", fontsize=12)
    else:
        ax.set_title(f"{title}\nN = {len(df_sub)}", fontsize=12)

    return sc


def scatter_rm_detected_overlay(ax, df_sub, title, detected_col):
    """
    Detection-category style panel.

    Grey = not detected by the selected current detector.
    Blue = detected.
    This is useful for the big-three progression plot.
    """
    setup_rm_axis(ax)
    add_rocky_curve(ax)

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
        ax.set_title(f"{title}\nN = 0", fontsize=12)
        return

    detected_here = to_bool_series(df_sub[detected_col])
    df_missed = df_sub[~detected_here].copy()
    df_detected = df_sub[detected_here].copy()

    if len(df_missed) > 0:
        ax.scatter(
            df_missed["mass_p"],
            df_missed["radius_p"],
            s=POINT_SIZE,
            alpha=0.62,
            color="gray",
            edgecolors="none",
            label="Not detected",
            zorder=1,
        )

    if len(df_detected) > 0:
        ax.scatter(
            df_detected["mass_p"],
            df_detected["radius_p"],
            s=POINT_SIZE,
            alpha=0.88,
            color="tab:blue",
            edgecolors="black",
            linewidths=0.28,
            label="Detected",
            zorder=2,
        )

    ax.set_title(
        f"{title}\nN = {len(df_sub)} | detected = {len(df_detected)}",
        fontsize=12,
    )


def add_colorbar(fig, sc, ax_or_axes):
    if sc is None:
        return

    cbar = fig.colorbar(
        sc,
        ax=ax_or_axes,
        location="right",
        shrink=0.92,
        pad=0.02,
    )
    cbar.set_label(r"log$_{10}$(Insolation Flux [$I_\oplus$])", fontsize=11)


# ============================================================
# Load data
# ============================================================

print("Project root:", ROOT)
print("Data directory:", DATA_DIR)
print("Rocky reference table:", REF_CURVE_PATH)

df = load_all_kepler_catalogs(DATA_DIR)

mode = DETECTION_MODE.lower().strip()

if mode not in ["best", "worst"]:
    raise ValueError("DETECTION_MODE must be 'best' or 'worst'.")

transit_col = "transiting_geometric"
bright_col = f"kepler_star_bright_enough_{mode}"
depth_col = f"kepler_depth_pass_{mode}"

# Current upgraded detector logic:
# prefer the simple "detected" column if your newer Kepler CSVs have it;
# otherwise fall back to detected_best / detected_worst.
if "detected" in df.columns:
    detected_col = "detected"
else:
    detected_col = f"detected_{mode}"

required_columns = [
    "mass_p",
    "radius_p",
    "flux_p",
    transit_col,
    bright_col,
    depth_col,
    detected_col,
]

require_columns(df, required_columns)

df = force_numeric_columns(df, ["mass_p", "radius_p", "flux_p"])
df = df.dropna(subset=["mass_p", "radius_p", "flux_p"]).copy()
df = add_rocky_diagnostics(df, REF_CURVE_PATH)

m_ref, r_ref = load_rocky_reference_curve(REF_CURVE_PATH)
print("\nLoaded rocky reference curve:")
print(f"Reference points: {len(m_ref)}")
print(f"Mass range: {m_ref.min():.4g} to {m_ref.max():.4g} M_earth")
print(f"Radius range: {r_ref.min():.4g} to {r_ref.max():.4g} R_earth")

print("\nUsing current detector columns:")
print("transit_col  =", transit_col)
print("bright_col   =", bright_col)
print("depth_col    =", depth_col)
print("detected_col =", detected_col)


# ============================================================
# Optional focus filter
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
    print("Planets after filtering:", len(df_plot))

else:
    df_plot = df.copy()
    print("\nNo focus window applied.")


# ============================================================
# Boolean masks from current upgraded detector
# ============================================================

transiting = to_bool_series(df_plot[transit_col])
bright_enough = to_bool_series(df_plot[bright_col])
depth_pass = to_bool_series(df_plot[depth_col])
detected = to_bool_series(df_plot[detected_col])

all_mask = pd.Series(True, index=df_plot.index)
transiting_mask = transiting
transiting_bright_mask = transiting & bright_enough
big_three_mask = transiting & bright_enough & depth_pass

# Missed categories are mutually exclusive under the big-three detector.
missed_non_transiting_mask = ~transiting
missed_too_faint_mask = transiting & (~bright_enough)
missed_depth_too_small_mask = transiting & bright_enough & (~depth_pass)

# This should usually be 0 if detected == transiting & bright & depth.
missed_other_mask = (~detected) & big_three_mask
mismatch_mask = detected != big_three_mask

print("\nDetector/category counts:")
print("All planets:", int(all_mask.sum()))
print("Transiting:", int(transiting_mask.sum()))
print("Transiting + bright enough:", int(transiting_bright_mask.sum()))
print("Transiting + bright + depth enough:", int(big_three_mask.sum()))
print("Detected column:", int(detected.sum()))
print("Missed: non-transiting:", int(missed_non_transiting_mask.sum()))
print("Missed: star too faint:", int(missed_too_faint_mask.sum()))
print("Missed: depth too small:", int(missed_depth_too_small_mask.sum()))
print("Mismatch detected vs big-three:", int(mismatch_mask.sum()))

if mismatch_mask.sum() > 0:
    mismatch_path = OUT_DIR / f"rm_detection_mismatches_{mode}_with_ref_curve.csv"
    df_plot[mismatch_mask].to_csv(mismatch_path, index=False)
    print("\nSaved mismatch rows to:")
    print(mismatch_path)


# ============================================================
# Save counts table
# ============================================================

counts_df = pd.DataFrame({
    "category": [
        "all_planets",
        "transiting",
        "transiting_and_bright",
        "transiting_and_bright_and_depth",
        f"detected_column_{detected_col}",
        "missed_non_transiting",
        "missed_star_too_faint",
        "missed_depth_too_small",
        "missed_other_big_three_but_not_detected",
        "mismatch_detected_vs_big_three",
    ],
    "count": [
        int(all_mask.sum()),
        int(transiting_mask.sum()),
        int(transiting_bright_mask.sum()),
        int(big_three_mask.sum()),
        int(detected.sum()),
        int(missed_non_transiting_mask.sum()),
        int(missed_too_faint_mask.sum()),
        int(missed_depth_too_small_mask.sum()),
        int(missed_other_mask.sum()),
        int(mismatch_mask.sum()),
    ],
})

counts_path = OUT_DIR / f"rm_detection_category_counts_{mode}_with_ref_curve.csv"
counts_df.to_csv(counts_path, index=False)
print("\nSaved counts table:")
print(counts_path)


# ============================================================
# Color scale
# ============================================================

vmin_all, vmax_all = compute_log_flux_limits(df_plot)


# ============================================================
# Figure 1: 2x3 detection category plot, color-coded by insolation
# ============================================================

panel_info = [
    ("All P-Pop planets", df_plot[all_mask]),
    ("Transiting planets", df_plot[transiting_mask]),
    ("Transiting + bright enough", df_plot[transiting_bright_mask]),
    ("Detected / big-three pass", df_plot[big_three_mask]),
    ("Missed: non-transiting", df_plot[missed_non_transiting_mask]),
    ("Missed: star/depth fail", df_plot[missed_too_faint_mask | missed_depth_too_small_mask]),
]

fig, axes = plt.subplots(
    2,
    3,
    figsize=(18, 9.5),
    constrained_layout=True,
)

axes_flat = axes.flatten()
scatter_for_colorbar = None

for ax, (title, df_sub) in zip(axes_flat, panel_info):
    sc = scatter_rm(
        ax=ax,
        df_sub=df_sub,
        title=title,
        vmin=vmin_all,
        vmax=vmax_all,
    )

    if scatter_for_colorbar is None and sc is not None:
        scatter_for_colorbar = sc

add_colorbar(fig, scatter_for_colorbar, axes_flat.tolist())

fig.suptitle(
    "P-Pop → Kepler detection categories on the Radius–Mass plane\n"
    f"{mode} case; black dashed line = ref.ddat rocky curve; color = insolation flux",
    fontsize=16,
)

png_path = OUT_DIR / f"rm_detection_categories_{mode}_with_ref_curve.png"
pdf_path = OUT_DIR / f"rm_detection_categories_{mode}_with_ref_curve.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("\nSaved 2x3 detection-category plot:")
print(png_path)
print(pdf_path)


# ============================================================
# Figure 1b: Big-three progression plot, detected vs not detected
# ============================================================

progression_panels = [
    ("1. Transiting planets", df_plot[transiting_mask].copy()),
    ("2. Transiting + bright enough", df_plot[transiting_bright_mask].copy()),
    ("3. Transiting + bright + depth enough", df_plot[big_three_mask].copy()),
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.8),
    constrained_layout=True,
)

for ax, (title, df_sub) in zip(axes, progression_panels):
    scatter_rm_detected_overlay(
        ax=ax,
        df_sub=df_sub,
        title=title,
        detected_col=detected_col,
    )

# Manual legend for the overlay plot.
handles = [
    plt.Line2D(
        [0], [0], marker="o", color="w", label="Not detected",
        markerfacecolor="gray", markersize=8, alpha=0.65,
    ),
    plt.Line2D(
        [0], [0], marker="o", color="w", label="Detected by Kepler",
        markerfacecolor="tab:blue", markeredgecolor="black", markersize=8,
    ),
    plt.Line2D(
        [0], [0], linestyle="--", color="black",
        label="Earth-like rocky curve from ref.ddat",
    ),
]

fig.legend(
    handles=handles,
    loc="lower center",
    ncol=3,
    fontsize=11,
    frameon=False,
)

fig.suptitle(
    "Kepler big-three detector progression on the Radius–Mass plane\n"
    f"{mode} case; detected column checked against transiting + bright + depth",
    fontsize=16,
)

png_path = OUT_DIR / f"rm_big_three_progression_detected_overlay_{mode}_with_ref_curve.png"
pdf_path = OUT_DIR / f"rm_big_three_progression_detected_overlay_{mode}_with_ref_curve.pdf"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
plt.close(fig)

print("\nSaved big-three detected-overlay plot:")
print(png_path)
print(pdf_path)


# ============================================================
# Figure 2: 1x3 insolation-bin comparison plot, attached-result style
# ============================================================

source = INSOLATION_BIN_SOURCE.lower().strip()

if source == "detected":
    df_bins = df_plot[detected].copy()
    source_label = f"Kepler-detected planets ({detected_col})"
elif source == "big_three":
    df_bins = df_plot[big_three_mask].copy()
    source_label = "Big-three pass planets: transiting + bright + depth"
elif source == "transiting":
    df_bins = df_plot[transiting_mask].copy()
    source_label = "Transiting planets"
elif source == "all":
    df_bins = df_plot.copy()
    source_label = "All P-Pop planets"
else:
    raise ValueError(
        "INSOLATION_BIN_SOURCE must be one of: detected, big_three, transiting, all."
    )

if USE_REFERENCE_STYLE_CUMULATIVE_BINS:
    # Same visual logic as the attached reference figure: I < 10, I < 50, I > 50.
    bin_low = df_bins["flux_p"] < 10
    bin_less_50 = df_bins["flux_p"] < 50
    bin_high = df_bins["flux_p"] > 50

    insolation_panels = [
        (r"I < 10 $I_\oplus$", df_bins[bin_low]),
        (r"I < 50 $I_\oplus$", df_bins[bin_less_50]),
        (r"I > 50 $I_\oplus$", df_bins[bin_high]),
    ]
else:
    # Non-overlapping alternative, useful if you want true bins later.
    bin_low = df_bins["flux_p"] < 10
    bin_mid = (df_bins["flux_p"] >= 10) & (df_bins["flux_p"] < 50)
    bin_high = df_bins["flux_p"] >= 50

    insolation_panels = [
        (r"I < 10 $I_\oplus$", df_bins[bin_low]),
        (r"10 $\leq$ I < 50 $I_\oplus$", df_bins[bin_mid]),
        (r"I $\geq$ 50 $I_\oplus$", df_bins[bin_high]),
    ]

vmin_bins, vmax_bins = compute_log_flux_limits(df_bins, fallback=(vmin_all, vmax_all))

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.8),
    constrained_layout=True,
)

scatter_for_shared_colorbar = None

for ax, (title, df_sub) in zip(axes, insolation_panels):
    subtitle = f"N = {len(df_sub)}"
    sc = scatter_rm(
        ax=ax,
        df_sub=df_sub,
        title=title,
        vmin=vmin_bins,
        vmax=vmax_bins,
        subtitle=subtitle,
    )
    if COLORBAR_PER_INSOLATION_PANEL:
        add_colorbar(fig, sc, ax)
    elif scatter_for_shared_colorbar is None and sc is not None:
        scatter_for_shared_colorbar = sc

if not COLORBAR_PER_INSOLATION_PANEL:
    add_colorbar(fig, scatter_for_shared_colorbar, axes.tolist())

fig.suptitle(
    "Radius–Mass diagram split by insolation flux\n"
    f"{source_label}; black dashed line = ref.ddat rocky curve",
    fontsize=16,
)

png_path = OUT_DIR / f"rm_insolation_bins_{source}_{mode}_reference_style_with_ref_curve.png"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved 1x3 insolation-bin plot:")
print(png_path)


# ============================================================
# Save bin counts and diagnostics
# ============================================================

bin_counts_df = pd.DataFrame({
    "panel": [title for title, _ in insolation_panels],
    "count": [len(df_sub) for _, df_sub in insolation_panels],
    "source": source_label,
})

bin_counts_path = OUT_DIR / f"rm_insolation_bin_counts_{source}_{mode}_with_ref_curve.csv"
bin_counts_df.to_csv(bin_counts_path, index=False)
print("Saved insolation-bin counts:")
print(bin_counts_path)

if SAVE_PLOT_DATA_WITH_DIAGNOSTICS:
    diagnostics_cols = [
        "run",
        "mass_p",
        "radius_p",
        "flux_p",
        "rocky_radius_ref",
        "radius_ratio_to_rocky",
        "radius_excess_to_rocky",
        transit_col,
        bright_col,
        depth_col,
        detected_col,
        "transit_depth_ppm",
        "l_sun",
        "distance_s",
        "star_flux_proxy",
        "approx_mbol",
    ]

    available_diagnostics_cols = [col for col in diagnostics_cols if col in df_plot.columns]

    diagnostics_path = OUT_DIR / f"rm_detection_and_insolation_diagnostics_{mode}_with_ref_curve.csv"
    df_plot[available_diagnostics_cols].to_csv(diagnostics_path, index=False)

    print("\nSaved diagnostics table:")
    print(diagnostics_path)
    print("\nNew rocky-curve diagnostic columns:")
    print("  rocky_radius_ref        = interpolated R_rocky from ref.ddat")
    print("  radius_ratio_to_rocky  = R_planet / R_rocky")
    print("  radius_excess_to_rocky = R_planet - R_rocky")
