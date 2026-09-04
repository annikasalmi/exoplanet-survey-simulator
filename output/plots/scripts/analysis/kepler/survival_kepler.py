"""
13_survival_kepler.py

Same science goal as script 39 but using the Kepler + K2 missions instead of TESS.

WHY KEPLER?
-----------
TESS observes each sky sector for ~27 days, making it nearly blind to planets with
orbital periods > ~13 days (cold, I < 10 I_earth around FGK stars).  Kepler observed
the same field continuously for ~4 years, giving access to orbital periods up to
~600 days and pushing the cold-rocky completeness dramatically higher.

WHAT IS DIFFERENT FROM SCRIPT 39:
  - P-Pop directory : run/kepler/data/Gaia/  (kepler_catalog_*.csv)
  - Detection cols  : transiting_geometric, bright_enough_kepler, detected
  - NASA filter     : disc_facility in {"Kepler", "K2"}
  - Output dir      : output/plots/13_survival_kepler/

WHAT IS THE SAME:
  - Rocky filter modes (all / mass-confirmed / radius-gap)
  - Completeness correction (1/eta weighting)
  - All statistical tests (permutation Q90, permutation count, bootstrap CI, KS)
  - NASA quality filters, rocky threshold, plotting

NOTE ON K2 COMPLETENESS APPROXIMATION:
  K2 used the Kepler spacecraft in ~80-day campaigns after reaction-wheel failure.
  No K2-specific P-Pop is available; K2 planets receive weights from the Kepler
  P-Pop completeness grid.  Because Kepler's 4-year baseline detects cold planets
  (I < 10, P > ~60 d) far more efficiently than an 80-day K2 campaign, the Kepler
  P-Pop overestimates K2 cold completeness → K2 cold-rocky weights are too low →
  conservative bias (under-corrects cold K2 planets).  Net effect is small because
  few K2 planets land in the cold bin, but results should be interpreted accordingly.

Usage:
    python scripts/13_survival_kepler.py
    python scripts/13_survival_kepler.py --rocky-mode radius-gap
    python scripts/13_survival_kepler.py --rocky-mode mass-confirmed --star-types F,G,K,M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -- Project root --------------------------------------------------------------

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "run" / "tess").exists():
            return p
    return start.parents[2]


ROOT = find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "output/plots" / "13_survival_kepler"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Paths ---------------------------------------------------------------------

PPOP_DATA_DIR = ROOT / "run" / "kepler" / "data" / "Gaia_C_F_K_combined"

NASA_FLAGS_CACHE = (
    ROOT / "run" / "kepler" / "data" / "NASA"
    / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)

REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

# Kepler primary mission + K2 (K2 weights approximated via Kepler P-Pop; see docstring).
KEPLER_FACILITIES = {"Kepler", "K2"}

# -- Rocky threshold constants (same as script 36 / script 39) ----------------

LHS1140B_MASS    = 5.60    # M_earth, Cadieux et al. 2024
LHS1140B_RADIUS  = 1.730   # R_earth
ROCKY_RADIUS_GAP = 1.65    # R_earth

# -- Science parameters --------------------------------------------------------

INSOL_EDGES  = np.array([0.1, 10.0, 50.0, 1e5])
INSOL_LABELS = ["I < 10 Iearth", "10 <= I < 50 Iearth", "I >= 50 Iearth"]
INSOL_COLORS = ["#4575b4", "#74add1", "#fdae61"]

COMP_R_EDGES = np.array([0.5, 1.2, 2.0, 3.0, 4.0])
COMP_I_EDGES = np.array([0.1, 2.0, 10.0, 50.0, 500.0, 1e5])

MAX_MASS_REL_UNCERTAINTY   = 0.25
MAX_RADIUS_REL_UNCERTAINTY = 0.08
EXCLUDE_MASS_LIMITS        = True
EXCLUDE_RADIUS_LIMITS      = True
EXCLUDE_CALCULATED_MASSES  = True
REQUIRE_TWO_SIDED_MASS     = True
REQUIRE_TWO_SIDED_RADIUS   = True

MAX_WEIGHT    = 20.0
N_BOOTSTRAP   = 4_000
N_PERMUTATION = 10_000
QUANTILE_TEST = 0.90

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220,
    "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8,
})


# -- Helpers -------------------------------------------------------------------

def infer_stype(teff: pd.Series) -> pd.Series:
    t = pd.to_numeric(teff, errors="coerce")
    s = pd.Series("Unknown", index=t.index, dtype=object)
    s[t >= 7500] = "A"
    s[(t >= 6000) & (t < 7500)] = "F"
    s[(t >= 5200) & (t < 6000)] = "G"
    s[(t >= 3700) & (t < 5200)] = "K"
    s[(t > 0)    & (t < 3700)]  = "M"
    return s


# -- Rocky threshold -----------------------------------------------------------

def load_rocky_curve() -> tuple[np.ndarray, np.ndarray, float]:
    if not REF_CURVE_PATH.exists():
        print(f"WARNING: {REF_CURVE_PATH} not found -- using toy power-law.")
        m = np.linspace(0.05, 30.0, 600)
        r = m ** 0.27
    else:
        ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        m = ref[:, 0].astype(float)
        r = ref[:, 1].astype(float)
        ok = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0)
        m, r = m[ok], r[ok]
        idx = np.argsort(m)
        m, r = m[idx], r[idx]
    r_at_lhs = float(np.interp(LHS1140B_MASS, m, r))
    shift = LHS1140B_RADIUS - r_at_lhs
    print(f"  Rocky curve at LHS 1140 b mass: {r_at_lhs:.4f}  shift={shift:+.4f} R_earth")
    return m, r, shift


def rocky_threshold_at_mass(masses, m_ref, r_ref, shift):
    return np.interp(
        np.asarray(masses, dtype=float), m_ref, r_ref + shift,
        left=np.nan, right=np.nan,
    )


# -- 1. NASA loading (Kepler-only) ---------------------------------------------

def load_nasa(star_types: list[str], rocky_mode: str) -> pd.DataFrame:
    if not NASA_FLAGS_CACHE.exists():
        raise FileNotFoundError(
            f"NASA cache not found:\n  {NASA_FLAGS_CACHE}\n"
            "Run script 36 first."
        )
    raw = pd.read_csv(NASA_FLAGS_CACHE)

    rename = {
        "pl_name":      "planet_name",
        "hostname":     "host_name",
        "pl_bmasse":    "mass_p",
        "pl_bmasseerr1":"mass_err_plus",
        "pl_bmasseerr2":"mass_err_minus",
        "pl_bmasselim": "mass_limit_flag",
        "pl_bmassprov": "mass_provider",
        "pl_rade":      "radius_p",
        "pl_radeerr1":  "radius_err_plus",
        "pl_radeerr2":  "radius_err_minus",
        "pl_radelim":   "radius_limit_flag",
        "pl_insol":     "flux_p",
        "st_teff":      "teff_s",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    # ---- Kepler+K2 filter --------------------------------------------------
    if "disc_facility" in df.columns:
        n_before = len(df)
        df = df[df["disc_facility"].astype(str).str.strip().isin(KEPLER_FACILITIES)].copy()
        print(f"  Kepler+K2 facility filter: {n_before:,} -> {len(df):,} planets")
    else:
        print("  WARNING: disc_facility column not found; cannot filter to Kepler+K2.")

    for col in ["mass_p", "radius_p", "flux_p",
                "mass_err_plus", "mass_err_minus",
                "radius_err_plus", "radius_err_minus",
                "mass_limit_flag", "radius_limit_flag", "teff_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["radius_p", "flux_p"]).copy()
    df = df[(df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()

    if rocky_mode != "radius-gap":
        df = df.dropna(subset=["mass_p"]).copy()
        df = df[df["mass_p"] > 0].copy()
        if EXCLUDE_MASS_LIMITS:
            df = df[~df["mass_limit_flag"].fillna(0).ne(0)].copy()
        if EXCLUDE_CALCULATED_MASSES and "mass_provider" in df.columns:
            calc = df["mass_provider"].astype(str).str.contains(
                "M-R relationship|Calculated", case=False, na=False)
            df = df[~calc].copy()
        if REQUIRE_TWO_SIDED_MASS:
            df = df[df["mass_err_plus"].notna() & df["mass_err_minus"].notna()].copy()
        mass_err = pd.concat(
            [df["mass_err_plus"].abs(), df["mass_err_minus"].abs()], axis=1
        ).max(axis=1)
        df = df[~((mass_err / df["mass_p"]) > MAX_MASS_REL_UNCERTAINTY)].copy()

    if EXCLUDE_RADIUS_LIMITS:
        df = df[~df["radius_limit_flag"].fillna(0).ne(0)].copy()
    if REQUIRE_TWO_SIDED_RADIUS:
        df = df[df["radius_err_plus"].notna() & df["radius_err_minus"].notna()].copy()
    rad_err = pd.concat(
        [df["radius_err_plus"].abs(), df["radius_err_minus"].abs()], axis=1
    ).max(axis=1)
    df = df[~((rad_err / df["radius_p"]) > MAX_RADIUS_REL_UNCERTAINTY)].copy()

    if "stype" in df.columns:
        df["stype_clean"] = df["stype"].astype(str).str.strip().str.upper().str[0]
    else:
        df["stype_clean"] = infer_stype(df.get("teff_s", pd.Series(dtype=float)))
    df = df[df["stype_clean"].isin(star_types)].copy()

    df = df[
        (df["flux_p"]   >= COMP_I_EDGES[0]) & (df["flux_p"]   <= COMP_I_EDGES[-1]) &
        (df["radius_p"] >= COMP_R_EDGES[0]) & (df["radius_p"] <= COMP_R_EDGES[-1])
    ].copy()

    print(f"  NASA quality-filtered ({'/'.join(star_types)}, mode={rocky_mode}): {len(df):,}")
    return df.reset_index(drop=True)


def apply_rocky_filter(df, rocky_mode, m_ref=None, r_ref=None, shift=0.0):
    if rocky_mode == "all":
        return df.copy()
    if rocky_mode == "radius-gap":
        rocky = df[df["radius_p"] <= ROCKY_RADIUS_GAP].copy()
        print(f"  Radius-gap filter (R <= {ROCKY_RADIUS_GAP}): {len(df):,} -> {len(rocky):,}")
        return rocky
    thresh = rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    mask   = (df["radius_p"].to_numpy() <= thresh) & np.isfinite(thresh)
    rocky  = df[mask].copy()
    rocky["rocky_threshold_radius"] = thresh[mask]
    print(f"  Mass-confirmed rocky filter: {len(df):,} -> {len(rocky):,}")
    return rocky


# -- 2. Kepler P-Pop completeness grid ----------------------------------------

def load_kepler_ppop_subset(n_catalogs: int) -> pd.DataFrame:
    files = sorted(PPOP_DATA_DIR.glob("kepler_catalog_*.csv"))[:n_catalogs]
    if not files:
        raise FileNotFoundError(f"No kepler_catalog_*.csv in {PPOP_DATA_DIR}")
    frames = []
    for p in files:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in {
                "radius_p", "flux_p", "stype",
                "transiting_geometric", "detected", "bright_enough_kepler",
            })
            frames.append(d)
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(files)} Kepler P-Pop catalogs: {len(df):,} planets")

    # Normalise booleans
    for col in ["transiting_geometric", "detected", "bright_enough_kepler"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                       .isin(["true", "1", "yes", "y", "t"]).astype(bool))
        else:
            df[col] = True  # safe fallback

    df["radius_p"] = pd.to_numeric(df["radius_p"], errors="coerce")
    df["flux_p"]   = pd.to_numeric(df["flux_p"],   errors="coerce")
    return df.dropna(subset=["radius_p", "flux_p"]).reset_index(drop=True)


def compute_completeness_grid(ppop: pd.DataFrame) -> np.ndarray:
    """
    eta[i,j] = P(Kepler detects | transiting AND bright_enough_kepler) in cell (R_bin i, I_bin j).
    NaN where denominator < 5.
    """
    r     = ppop["radius_p"].to_numpy()
    f     = ppop["flux_p"].to_numpy()
    trans = ppop["transiting_geometric"].to_numpy(bool)
    obs   = ppop["bright_enough_kepler"].to_numpy(bool)
    det   = ppop["detected"].to_numpy(bool)

    nr  = len(COMP_R_EDGES) - 1
    ni  = len(COMP_I_EDGES) - 1
    eta = np.full((nr, ni), np.nan)

    for i in range(nr):
        for j in range(ni):
            cell = (
                (r >= COMP_R_EDGES[i]) & (r < COMP_R_EDGES[i+1]) &
                (f >= COMP_I_EDGES[j]) & (f < COMP_I_EDGES[j+1]) &
                trans & obs
            )
            denom = cell.sum()
            if denom >= 5:
                eta[i, j] = det[cell].sum() / denom

    valid = np.isfinite(eta)
    print(f"  Kepler completeness grid: {valid.sum()}/{eta.size} cells valid "
          f"(median eta = {np.nanmedian(eta):.3f})")
    return eta


def assign_weights(nasa: pd.DataFrame, eta: np.ndarray) -> np.ndarray:
    r  = nasa["radius_p"].to_numpy()
    f  = nasa["flux_p"].to_numpy()
    ri = np.clip(np.digitize(r, COMP_R_EDGES) - 1, 0, len(COMP_R_EDGES) - 2)
    fi = np.clip(np.digitize(f, COMP_I_EDGES) - 1, 0, len(COMP_I_EDGES) - 2)
    weights = np.empty(len(nasa), dtype=float)
    for k in range(len(nasa)):
        e = eta[ri[k], fi[k]]
        weights[k] = min(1.0 / e, MAX_WEIGHT) if (np.isfinite(e) and e > 0) else MAX_WEIGHT
    cap_frac = (weights >= MAX_WEIGHT).mean()
    print(f"  Weights: min={weights.min():.2f}  median={np.median(weights):.2f}  "
          f"max={weights.max():.2f}  ({cap_frac:.0%} capped)")
    if cap_frac > 0.5:
        print("  WARNING: >50% of weights hit cap -- run --n-catalogs 200+")
    return weights


# -- 3. Weighted statistics (identical to script 39) ---------------------------

def weighted_survival(radii, weights, r_grid):
    w_total = weights.sum()
    if w_total == 0:
        return np.zeros_like(r_grid, dtype=float)
    return np.array([weights[radii > rv].sum() / w_total for rv in r_grid])


def weighted_quantile(radii, weights, q):
    if len(radii) == 0 or weights.sum() == 0:
        return np.nan
    order = np.argsort(radii)
    cdf   = np.cumsum(weights[order]) / weights.sum()
    idx   = min(np.searchsorted(cdf, q), len(radii) - 1)
    return float(radii[order][idx])


def bootstrap_quantile_ci(radii, weights, q, n_boot, rng):
    point = weighted_quantile(radii, weights, q)
    n = len(radii)
    boot = [weighted_quantile(radii[rng.integers(0, n, size=n)],
                              weights[rng.integers(0, n, size=n)], q)
            for _ in range(n_boot)]
    arr = np.array(boot)
    return point, float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def bootstrap_count_ci(weights, n_boot, rng):
    point = float(weights.sum())
    n = len(weights)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot = [float(weights[rng.integers(0, n, size=n)].sum()) for _ in range(n_boot)]
    arr  = np.array(boot)
    return point, float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def permutation_test_q90(r_low, w_low, r_high, w_high, q, n_perm, rng):
    obs   = weighted_quantile(r_low, w_low, q) - weighted_quantile(r_high, w_high, q)
    all_r = np.concatenate([r_low, r_high])
    all_w = np.concatenate([w_low, w_high])
    n_low = len(r_low)
    null  = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(all_r))
        null[i] = (weighted_quantile(all_r[p[:n_low]], all_w[p[:n_low]], q)
                   - weighted_quantile(all_r[p[n_low:]], all_w[p[n_low:]], q))
    return obs, float((null <= obs).mean()), null


def permutation_test_count(w_low, w_high, n_perm, rng):
    obs   = float(w_low.sum() - w_high.sum())
    all_w = np.concatenate([w_low, w_high])
    n_low = len(w_low)
    null  = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(all_w))
        null[i] = all_w[p[:n_low]].sum() - all_w[p[n_low:]].sum()
    return obs, float((null <= obs).mean()), null


def weighted_ks_upper(r_low, w_low, r_high, w_high, n_perm=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(7)
    median_r = np.median(np.concatenate([r_low, r_high]))
    mask_l   = r_low  >= median_r
    mask_h   = r_high >= median_r
    if mask_l.sum() < 3 or mask_h.sum() < 3:
        return np.nan, np.nan

    def _ks_d(rl, wl, rh, wh):
        grid = np.sort(np.unique(np.concatenate([rl, rh])))
        cl   = np.array([wl[rl <= rv].sum() for rv in grid]) / wl.sum()
        ch   = np.array([wh[rh <= rv].sum() for rv in grid]) / wh.sum()
        return float(np.max(ch - cl))

    rl, wl = r_low[mask_l],  w_low[mask_l]
    rh, wh = r_high[mask_h], w_high[mask_h]
    D_obs  = _ks_d(rl, wl, rh, wh)
    all_r2 = np.concatenate([rl, rh])
    all_w2 = np.concatenate([wl, wh])
    nl     = len(rl)
    null_D = np.array([
        _ks_d(all_r2[p := rng.permutation(len(all_r2))][:nl], all_w2[p[:nl]],
              all_r2[p[nl:]], all_w2[p[nl:]])
        for _ in range(n_perm)
    ])
    return D_obs, float((null_D >= D_obs).mean())


# -- 4. Plotting ---------------------------------------------------------------

def _save(fig, fname):
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_completeness_grid(eta):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(eta, origin="lower", aspect="auto", cmap="viridis",
                   vmin=0, vmax=1,
                   extent=[0, len(COMP_I_EDGES)-1, 0, len(COMP_R_EDGES)-1])
    plt.colorbar(im, ax=ax, label="Detection fraction eta(R, I)")
    ax.set_xticks(np.arange(len(COMP_I_EDGES)-1) + 0.5)
    ax.set_xticklabels(
        [f"{v:.1g}" for v in 0.5 * (COMP_I_EDGES[:-1] + COMP_I_EDGES[1:])],
        rotation=50, ha="right", fontsize=6,
    )
    ax.set_yticks(np.arange(len(COMP_R_EDGES)-1) + 0.5)
    ax.set_yticklabels(
        [f"{v:.2f}" for v in 0.5 * (COMP_R_EDGES[:-1] + COMP_R_EDGES[1:])],
        fontsize=6,
    )
    ax.set_xlabel("Insolation bin centre (Iearth)")
    ax.set_ylabel("Radius bin centre (Rearth)")
    ax.set_title("Kepler P-Pop completeness grid  eta(R,I) = P(Kepler detects | transiting)\n"
                 "Used for both Kepler and K2 planets (K2 cold completeness is approximate)")
    for i in range(eta.shape[0]):
        for j in range(eta.shape[1]):
            if np.isfinite(eta[i, j]):
                ax.text(j+0.5, i+0.5, f"{eta[i,j]:.2f}",
                        ha="center", va="center", fontsize=4.5, color="white")
    fig.tight_layout()
    _save(fig, "fig1_completeness_grid.png")


def plot_survival_curves(groups, title_suffix):
    r_grid = np.linspace(0.5, 4.0, 400)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, use_w, panel in [
        (axes[0], False, "Raw (no completeness correction)"),
        (axes[1], True,  "Completeness-corrected  (w = 1/eta)"),
    ]:
        for label, color, radii, weights in groups:
            w    = weights if use_w else np.ones_like(weights)
            surv = weighted_survival(radii, w, r_grid)
            ax.plot(r_grid, surv, color=color, lw=2, label=label)
            ax.fill_between(r_grid, surv, alpha=0.08, color=color)
        ax.axvline(ROCKY_RADIUS_GAP, ls="--", color="gray", lw=0.9,
                   label=f"Radius gap ({ROCKY_RADIUS_GAP} Rearth)")
        ax.axvline(LHS1140B_RADIUS, ls=":", color="gold", lw=0.9,
                   label=f"LHS 1140 b ({LHS1140B_RADIUS} Rearth)")
        ax.set_xlabel("Planet radius  [Rearth]")
        ax.set_ylabel("S(R) = P(R_p > R)")
        ax.set_title(panel)
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25)
    fig.suptitle(f"Survival curves (Kepler+K2): planet radius by insolation bin\n{title_suffix}",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig2_survival_curves.png")


def plot_bootstrap_quantiles(groups, title_suffix):
    fig, ax = plt.subplots(figsize=(7, 4))
    for xi, (label, color, pt, lo, hi) in enumerate(groups):
        ax.errorbar(xi, pt, yerr=[[pt-lo], [hi-pt]],
                    fmt="o", color=color, ms=8, capsize=5, lw=1.5)
    ax.axhline(ROCKY_RADIUS_GAP, ls="--", color="gray", lw=0.9,
               label=f"Radius gap ({ROCKY_RADIUS_GAP} Rearth)")
    ax.axhline(LHS1140B_RADIUS,  ls=":",  color="gold", lw=0.9,
               label=f"LHS 1140 b ({LHS1140B_RADIUS} Rearth)")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel(f"Completeness-corrected Q{int(QUANTILE_TEST*100)}  [Rearth]")
    ax.set_title(f"Kepler+K2: Q90 by insolation bin  (95% bootstrap CI)\n{title_suffix}")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig3_bootstrap_quantile.png")


def plot_corrected_counts(groups, title_suffix, rocky_mode):
    fig, ax = plt.subplots(figsize=(7, 5))
    for xi, (label, color, pt, lo, hi) in enumerate(groups):
        ax.bar(xi, pt, color=color, alpha=0.75, width=0.6, label=label)
        ax.errorbar(xi, pt, yerr=[[pt-lo], [hi-pt]],
                    fmt="none", color="black", capsize=5, lw=1.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("Corrected transiting count  (sum of weights)")
    ax.set_title(
        f"Kepler+K2: corrected detection count -- rocky mode: {rocky_mode}\n"
        f"{title_suffix}\nNOTE: geometric transit prob. NOT corrected"
    )
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig4_corrected_counts.png")


def plot_permutation_null(obs, null, p_val, bins_compared, stat_label, fname):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=60, color="steelblue", alpha=0.7, density=True,
            label="Null distribution")
    ax.axvline(obs, color="crimson", lw=2, label=f"Observed: {obs:+.3f}")
    ax.set_xlabel(stat_label)
    ax.set_ylabel("Density")
    ax.set_title(
        f"Permutation test (Kepler+K2): {bins_compared}\n"
        f"H1: {stat_label.split('=')[0].strip()} < 0   "
        f"one-sided p = {p_val:.4f}  (N={N_PERMUTATION:,} shuffles)"
    )
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


# -- 5. Summary ----------------------------------------------------------------

def print_and_save_summary(nasa, weights, groups_data, perm_q90, perm_count,
                           ks_results, title_suffix, rocky_mode):
    lines = [
        "=" * 72,
        "KEPLER+K2: ROCKY PLANET PRIMORDIAL FORMATION TEST",
        "=" * 72,
        f"Sample    : {title_suffix}",
        f"Rocky mode: {rocky_mode}",
        f"N planets : {len(nasa):,}",
        f"Weights   : min={weights.min():.2f}  median={np.median(weights):.2f}"
        f"  max={weights.max():.2f}  cap={MAX_WEIGHT}",
        "",
        "-- Group sizes and corrected statistics ----------------------------",
    ]
    rng_sum = np.random.default_rng(99)
    for label, _, radii, w in groups_data:
        lines.append(
            f"  {label:<28}  N_det={len(radii):>4}  "
            f"N_corr={w.sum():>7.1f}  "
            f"Q50={weighted_quantile(radii, w, 0.50):.3f}  "
            f"Q90={weighted_quantile(radii, w, 0.90):.3f} Rearth"
        )

    lines += ["", f"-- Bootstrap Q{int(QUANTILE_TEST*100)} (95% CI) ----------"]
    rng_b = np.random.default_rng(99)
    for label, _, radii, w in groups_data:
        pt, lo, hi = bootstrap_quantile_ci(radii, w, QUANTILE_TEST, N_BOOTSTRAP, rng_b)
        lines.append(f"  {label:<28}  Q90 = {pt:.3f} [{lo:.3f}, {hi:.3f}] Rearth")

    lines += ["", "-- Bootstrap corrected count (95% CI) ---------------------"]
    rng_c = np.random.default_rng(77)
    for label, _, radii, w in groups_data:
        pt, lo, hi = bootstrap_count_ci(w, N_BOOTSTRAP, rng_c)
        lines.append(f"  {label:<28}  count = {pt:.1f} [{lo:.1f}, {hi:.1f}]")

    lines += ["", "-- Permutation test: Q90 -----------------------------------"]
    for la, lb, obs, pv in perm_q90:
        sig = "SIGNIFICANT" if pv < 0.05 else "marginal" if pv < 0.10 else "not significant"
        lines += [f"  {la} vs {lb}",
                  f"    DeltaQ90 = {obs:+.4f}  p = {pv:.4f}  [{sig}]"]

    lines += ["", "-- Permutation test: count ---------------------------------"]
    for la, lb, obs, pv in perm_count:
        sig = "SIGNIFICANT" if pv < 0.05 else "marginal" if pv < 0.10 else "not significant"
        lines += [f"  {la} vs {lb}",
                  f"    Delta_count = {obs:+.1f}   p = {pv:.4f}  [{sig}]",
                  "    (NOTE: geometric transit prob. not corrected)"]

    lines += ["", "-- Weighted KS upper-tail ----------------------------------"]
    for la, lb, D, p_ks in ks_results:
        lines.append(f"  {la} vs {lb}:  D = {D:.4f}   p ~= {p_ks:.4f}")

    lines += [
        "", "=" * 72,
        "KEPLER ADVANTAGE: 4-year baseline -> cold rocky completeness much higher than TESS.",
        "Compare eta(cold, small) in this grid vs script 39 (TESS) to see the improvement.",
        "K2 NOTE: K2 planets use Kepler P-Pop completeness (approximate; see docstring).",
        "=" * 72,
    ]
    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n  Saved: {out}")


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-catalogs", type=int, default=400,
                        help="Number of Kepler P-Pop catalogs (default: 400).")
    parser.add_argument("--star-types", type=str, default="F,G,K,M")
    parser.add_argument("--rocky-mode", type=str, default="all",
                        choices=["all", "mass-confirmed", "radius-gap"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    star_types = [s.strip().upper() for s in args.star_types.split(",")]
    title_sfx  = (f"Stars: {'+'.join(star_types)}  |  rocky: {args.rocky_mode}"
                  f"  |  P-Pop: {PPOP_DATA_DIR.name}  |  Kepler+K2")

    m_ref = r_ref = None
    shift = 0.0
    if args.rocky_mode == "mass-confirmed":
        print("\n-- Rocky threshold ------------------------------------------")
        m_ref, r_ref, shift = load_rocky_curve()

    print("\n-- 1. NASA PSCompPars (Kepler-discovered) ---------------------------")
    nasa_all = load_nasa(star_types, args.rocky_mode)
    nasa     = apply_rocky_filter(nasa_all, args.rocky_mode, m_ref, r_ref, shift)
    print(f"  Final sample: {len(nasa):,} planets")
    if len(nasa) < 10:
        print("  WARNING: fewer than 10 planets -- results unreliable.")

    print(f"\n-- 2. Kepler P-Pop completeness grid  ({args.n_catalogs} catalogs) --")
    ppop = load_kepler_ppop_subset(args.n_catalogs)
    eta  = compute_completeness_grid(ppop)
    plot_completeness_grid(eta)

    print("\n-- 3. Inverse-completeness weights ----------------------------------")
    weights     = assign_weights(nasa, eta)
    nasa        = nasa.copy()
    nasa["weight"] = weights

    print("\n-- 4. Group by insolation bin ----------------------------------------")
    groups_data = []
    for j in range(len(INSOL_EDGES) - 1):
        lo, hi = INSOL_EDGES[j], INSOL_EDGES[j+1]
        mask   = (nasa["flux_p"] >= lo) & (nasa["flux_p"] < hi)
        r      = nasa.loc[mask, "radius_p"].to_numpy()
        w      = nasa.loc[mask, "weight"].to_numpy()
        groups_data.append((INSOL_LABELS[j], INSOL_COLORS[j], r, w))
        print(f"  {INSOL_LABELS[j]}: N_detected={len(r)}  N_corrected={w.sum():.1f}")

    print("\n-- 5. Survival curves -----------------------------------------------")
    plot_survival_curves([(l, c, r, w) for l, c, r, w in groups_data], title_sfx)

    print(f"\n-- 6. Bootstrap Q{int(QUANTILE_TEST*100)} -----------------------------------")
    boot_q90 = []
    for label, color, radii, w in groups_data:
        pt, lo, hi = bootstrap_quantile_ci(radii, w, QUANTILE_TEST, N_BOOTSTRAP, rng)
        print(f"  {label}: Q90 = {pt:.3f}  [{lo:.3f}, {hi:.3f}] Rearth")
        boot_q90.append((label, color, pt, lo, hi))
    plot_bootstrap_quantiles(boot_q90, title_sfx)

    print(f"\n-- 7. Bootstrap corrected count -------------------------------------")
    boot_cnt = []
    for label, color, radii, w in groups_data:
        pt, lo, hi = bootstrap_count_ci(w, N_BOOTSTRAP, rng)
        print(f"  {label}: count = {pt:.1f}  [{lo:.1f}, {hi:.1f}]")
        boot_cnt.append((label, color, pt, lo, hi))
    plot_corrected_counts(boot_cnt, title_sfx, args.rocky_mode)

    print(f"\n-- 8. Permutation tests  (N={N_PERMUTATION}) -----------------------")
    lbl_l, _, r_l, w_l = groups_data[0]
    lbl_m, _, r_m, w_m = groups_data[1]
    lbl_h, _, r_h, w_h = groups_data[2]

    obs_q90_lh, pv_q90_lh, null_q90_lh = permutation_test_q90(
        r_l, w_l, r_h, w_h, QUANTILE_TEST, N_PERMUTATION, rng)
    obs_q90_lm, pv_q90_lm, _ = permutation_test_q90(
        r_l, w_l, r_m, w_m, QUANTILE_TEST, N_PERMUTATION, rng)
    perm_q90 = [
        (lbl_l, lbl_h, obs_q90_lh, pv_q90_lh),
        (lbl_l, lbl_m, obs_q90_lm, pv_q90_lm),
    ]
    print(f"  Q90 {lbl_l} vs {lbl_h}: delta={obs_q90_lh:+.4f}  p={pv_q90_lh:.4f}")
    print(f"  Q90 {lbl_l} vs {lbl_m}: delta={obs_q90_lm:+.4f}  p={pv_q90_lm:.4f}")
    plot_permutation_null(
        obs_q90_lh, null_q90_lh, pv_q90_lh,
        f"{lbl_l}  vs  {lbl_h}",
        f"DeltaQ{int(QUANTILE_TEST*100)} = Q90(low) - Q90(high)  [Rearth]",
        "fig5_permutation_q90.png",
    )

    obs_cnt_lh, pv_cnt_lh, null_cnt_lh = permutation_test_count(
        w_l, w_h, N_PERMUTATION, rng)
    obs_cnt_lm, pv_cnt_lm, _ = permutation_test_count(
        w_l, w_m, N_PERMUTATION, rng)
    perm_count = [
        (lbl_l, lbl_h, obs_cnt_lh, pv_cnt_lh),
        (lbl_l, lbl_m, obs_cnt_lm, pv_cnt_lm),
    ]
    print(f"  Count {lbl_l} vs {lbl_h}: delta={obs_cnt_lh:+.1f}  p={pv_cnt_lh:.4f}")
    plot_permutation_null(
        obs_cnt_lh, null_cnt_lh, pv_cnt_lh,
        f"{lbl_l}  vs  {lbl_h}",
        "Delta_count = corrected_count(low) - corrected_count(high)",
        "fig6_permutation_count.png",
    )

    ks_results = []
    for (la, ra, wa), (lb, rb, wb) in [
        ((lbl_l, r_l, w_l), (lbl_h, r_h, w_h)),
        ((lbl_l, r_l, w_l), (lbl_m, r_m, w_m)),
    ]:
        D, p_ks = weighted_ks_upper(ra, wa, rb, wb, n_perm=2000, rng=rng)
        print(f"  KS upper {la} vs {lb}: D={D:.4f}  p={p_ks:.4f}")
        ks_results.append((la, lb, D, p_ks))

    print("\n-- 9. Summary -------------------------------------------------------")
    print_and_save_summary(
        nasa, weights, groups_data,
        perm_q90, perm_count, ks_results,
        title_sfx, args.rocky_mode,
    )


if __name__ == "__main__":
    main()
