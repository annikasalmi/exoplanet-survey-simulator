"""
11_survival_tess_all.py

SCIENCE GOAL
------------
Test whether the apparent absence of large/rocky planets at low insolation
(I < 10 I_earth) is a real physical effect (primordial formation limit) or
purely a TESS detection bias.

At I > 50: many rocky planets sit on the rocky formation line -- these are
likely PHOTOEVAPORATED remnants (sub-Neptunes stripped of their envelopes).
At I < 10: photoevaporation is negligible, so rocky planets there are
PRIMORDIALLY FORMED.  The question: can large rocky planets form primordially
at low insolation, or does the apparent absence reflect real physics?

THREE ROCKY MODES (--rocky-mode):
  all            : all quality-filtered NASA PSCompPars planets
                   (original behavior -- tests all sizes, gas giants included)
  mass-confirmed : only mass-confirmed rocky planets
                   (radius_p <= rocky threshold from ref.ddat + LHS 1140 b anchor)
                   Same logic as script 36.  Requires RV masses -- biased against
                   cold long-period planets.  Expect N~5-15 at I<10.
  radius-gap     : rocky proxy = R_p < 1.65 R_earth (below the radius valley)
                   No mass measurement required -- all mass quality filters relaxed.
                   Justified at I<10: without photoevaporation, any R<1.65 planet
                   must be primordially rocky, so mass confirmation is unnecessary.
                   Gives ~3-5x more cold rocky candidates than mass-confirmed.

HOW THE COMPLETENESS CORRECTION WORKS:
  1. eta(R, I) = P(TESS detects | transiting AND observed) from P-Pop simulations.
     eta ~ 0.20-0.45 for cold + small (rocky) planets.
     eta ~ 0.91-1.00 for hot large planets.
  2. Each real detected planet gets weight w = 1/eta.
     Found in a cell where TESS is 25% complete -> counts as 4 planets.
  3. All statistics use these weights, approximating the true transiting population.

STATISTICAL TESTS (primary -> secondary):
  a) Permutation test: DeltaQ90 = Q90(low I) - Q90(high I)  [radius upper bound]
  b) Permutation test: Delta_count = corrected_count(low) - corrected_count(high)
     [tests whether rocky formation rate is equal across insolation bins]
  c) Bootstrap 95% CI on Q90 and corrected count per bin
  d) Weighted KS upper-tail statistic

INTERPRETING COUNT TEST:
  The corrected count = sum(1/eta) estimates how many transiting rocky planets
  exist at each insolation level (had TESS been 100% efficient).
  NOTE: geometric transit probability is NOT corrected here -- cold planets
  transit less often by geometry, so count(I<10) will naturally be lower.
  Count comparisons across bins must account for this caveat.
  Q90 test is unaffected by this (it tests radius distribution, not counts).

Usage:
    python scripts/11_survival_tess_all.py
    python scripts/11_survival_tess_all.py --rocky-mode mass-confirmed
    python scripts/11_survival_tess_all.py --rocky-mode radius-gap
    python scripts/11_survival_tess_all.py --rocky-mode mass-confirmed --star-types M
    python scripts/11_survival_tess_all.py --n-catalogs 400
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

OUT_DIR = ROOT / "output/plots" / "11_survival_tess_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Paths (mirror script 36) --------------------------------------------------

PPOP_DATA_DIR = ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1"

NASA_FLAGS_CACHE = (
    ROOT / "run" / "kepler" / "data" / "NASA"
    / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)

REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

# -- Rocky threshold constants (mirror script 36) ------------------------------

LHS1140B_MASS    = 5.60    # M_earth, Cadieux et al. 2024
LHS1140B_RADIUS  = 1.730   # R_earth
ROCKY_RADIUS_GAP = 1.65    # R_earth -- below the radius valley
                            # At I<10: no photoevaporation -> any R<1.65 is primordial rocky

# -- Science parameters --------------------------------------------------------

INSOL_EDGES  = np.array([0.1, 10.0, 50.0, 1e5])
INSOL_LABELS = ["I < 10 Iearth", "10 <= I < 50 Iearth", "I >= 50 Iearth"]
INSOL_COLORS = ["#4575b4", "#74add1", "#fdae61"]

# Coarse completeness grid: ~17 transiting planets per catalog x 100 catalogs
# = ~1700 total across 4x5=20 cells = ~85 per cell (reliable).
COMP_R_EDGES = np.array([0.5, 1.2, 2.0, 3.0, 4.0])
COMP_I_EDGES = np.array([0.1, 2.0, 10.0, 50.0, 500.0, 1e5])

# Quality thresholds (mirror script 36)
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
    "figure.dpi": 120,
    "savefig.dpi": 220,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
})


# -- Helpers -------------------------------------------------------------------

def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    return None


def infer_stype(teff: pd.Series) -> pd.Series:
    t = pd.to_numeric(teff, errors="coerce")
    s = pd.Series("Unknown", index=t.index, dtype=object)
    s[t >= 7500] = "A"
    s[(t >= 6000) & (t < 7500)] = "F"
    s[(t >= 5200) & (t < 6000)] = "G"
    s[(t >= 3700) & (t < 5200)] = "K"
    s[(t > 0)    & (t < 3700)]  = "M"
    return s


# -- Rocky threshold (ported from script 36) -----------------------------------

def load_rocky_curve() -> tuple[np.ndarray, np.ndarray, float]:
    """Load ref.ddat pure-rock boundary and compute LHS 1140 b shift."""
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
    print(f"  Rocky curve at LHS 1140 b mass ({LHS1140B_MASS} M_earth): "
          f"{r_at_lhs:.4f} R_earth  -->  shift = {shift:+.4f} R_earth")
    return m, r, shift


def rocky_threshold_at_mass(masses: np.ndarray, m_ref, r_ref, shift: float) -> np.ndarray:
    return np.interp(
        np.asarray(masses, dtype=float),
        m_ref, r_ref + shift,
        left=np.nan, right=np.nan,
    )


# -- 1. NASA PSCompPars loading ------------------------------------------------

def load_nasa(star_types: list[str], rocky_mode: str) -> pd.DataFrame:
    """
    Load and quality-filter NASA PSCompPars.
    rocky_mode='radius-gap' relaxes all mass-based filters (mass not needed).
    """
    if not NASA_FLAGS_CACHE.exists():
        raise FileNotFoundError(
            f"NASA cache not found:\n  {NASA_FLAGS_CACHE}\n"
            "Run script 36 first to download and cache it."
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
        "pl_insollim":  "insolation_limit_flag",
        "st_teff":      "teff_s",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    for col in ["mass_p", "radius_p", "flux_p",
                "mass_err_plus", "mass_err_minus",
                "radius_err_plus", "radius_err_minus",
                "mass_limit_flag", "radius_limit_flag", "teff_s"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Radius and insolation always required
    df = df.dropna(subset=["radius_p", "flux_p"]).copy()
    df = df[(df["radius_p"] > 0) & (df["flux_p"] > 0)].copy()

    # Mass required for 'all' and 'mass-confirmed'; relaxed for 'radius-gap'
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

    # Radius quality always applied
    if EXCLUDE_RADIUS_LIMITS:
        df = df[~df["radius_limit_flag"].fillna(0).ne(0)].copy()
    if REQUIRE_TWO_SIDED_RADIUS:
        df = df[df["radius_err_plus"].notna() & df["radius_err_minus"].notna()].copy()
    rad_err = pd.concat(
        [df["radius_err_plus"].abs(), df["radius_err_minus"].abs()], axis=1
    ).max(axis=1)
    df = df[~((rad_err / df["radius_p"]) > MAX_RADIUS_REL_UNCERTAINTY)].copy()

    # Star type
    if "stype" in df.columns:
        df["stype_clean"] = df["stype"].astype(str).str.strip().str.upper().str[0]
    else:
        df["stype_clean"] = infer_stype(df.get("teff_s", pd.Series(dtype=float)))
    df = df[df["stype_clean"].isin(star_types)].copy()

    # Science window
    df = df[
        (df["flux_p"]   >= COMP_I_EDGES[0]) & (df["flux_p"]   <= COMP_I_EDGES[-1]) &
        (df["radius_p"] >= COMP_R_EDGES[0]) & (df["radius_p"] <= COMP_R_EDGES[-1])
    ].copy()

    print(f"  NASA quality-filtered ({'/'.join(star_types)}, mode={rocky_mode}): {len(df):,}")
    return df.reset_index(drop=True)


def apply_rocky_filter(
    df: pd.DataFrame,
    rocky_mode: str,
    m_ref=None,
    r_ref=None,
    shift: float = 0.0,
) -> pd.DataFrame:
    """
    Filter to rocky planets.
    'all'           -> no filter
    'mass-confirmed'-> radius_p <= rocky_threshold_at_mass(mass_p)
    'radius-gap'    -> radius_p <= ROCKY_RADIUS_GAP (1.65 R_earth)
    """
    if rocky_mode == "all":
        return df.copy()

    if rocky_mode == "radius-gap":
        rocky = df[df["radius_p"] <= ROCKY_RADIUS_GAP].copy()
        print(f"  Radius-gap filter (R <= {ROCKY_RADIUS_GAP} R_earth): "
              f"{len(df):,} -> {len(rocky):,}")
        return rocky

    # mass-confirmed
    thresh = rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    mask = (df["radius_p"].to_numpy() <= thresh) & np.isfinite(thresh)
    rocky = df[mask].copy()
    rocky["rocky_threshold_radius"] = thresh[mask]
    print(f"  Mass-confirmed rocky filter: {len(df):,} -> {len(rocky):,}")
    return rocky


# -- 2. Completeness grid from P-Pop ------------------------------------------

def load_ppop_subset(n_catalogs: int) -> pd.DataFrame:
    files = sorted(PPOP_DATA_DIR.glob("tess_catalog_*.csv"))[:n_catalogs]
    if not files:
        raise FileNotFoundError(f"No tess_catalog_*.csv in {PPOP_DATA_DIR}")
    frames = []
    for p in files:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in {
                "radius_p", "flux_p", "stype",
                "tess_transiting_geometric", "transiting_geometric",
                "tess_observed", "tess_detected", "detected",
            })
            frames.append(d)
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded {len(files)} P-Pop catalogs: {len(df):,} planets")

    if "tess_transiting_geometric" not in df.columns and "transiting_geometric" in df.columns:
        df["tess_transiting_geometric"] = df["transiting_geometric"]
    if "tess_detected" not in df.columns and "detected" in df.columns:
        df["tess_detected"] = df["detected"]
    if "tess_observed" not in df.columns:
        df["tess_observed"] = True

    for col in ["tess_transiting_geometric", "tess_observed", "tess_detected"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                       .isin(["true", "1", "yes", "y", "t"]).astype(bool))

    df["radius_p"] = pd.to_numeric(df.get("radius_p"), errors="coerce")
    df["flux_p"]   = pd.to_numeric(df.get("flux_p"),   errors="coerce")
    return df.dropna(subset=["radius_p", "flux_p"]).reset_index(drop=True)


def compute_completeness_grid(ppop: pd.DataFrame) -> np.ndarray:
    """
    eta[i,j] = P(tess_detected | transiting AND observed) in cell (R_bin i, I_bin j).
    NaN where denominator < 5 (unreliable).
    """
    r     = ppop["radius_p"].to_numpy()
    f     = ppop["flux_p"].to_numpy()
    trans = ppop["tess_transiting_geometric"].to_numpy(bool)
    obs   = (ppop["tess_observed"].to_numpy(bool)
             if "tess_observed" in ppop.columns else np.ones(len(ppop), bool))
    det   = ppop["tess_detected"].to_numpy(bool)

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
    print(f"  Completeness grid: {valid.sum()}/{eta.size} cells valid "
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
          f"max={weights.max():.2f}  (capped at {MAX_WEIGHT})")
    if cap_frac > 0.5:
        print(f"  WARNING: {cap_frac:.0%} hit weight cap -- run --n-catalogs 150+")
    return weights


# -- 3. Weighted statistics ----------------------------------------------------

def weighted_survival(radii: np.ndarray, weights: np.ndarray, r_grid: np.ndarray) -> np.ndarray:
    w_total = weights.sum()
    if w_total == 0:
        return np.zeros_like(r_grid, dtype=float)
    return np.array([weights[radii > rv].sum() / w_total for rv in r_grid])


def weighted_quantile(radii: np.ndarray, weights: np.ndarray, q: float) -> float:
    if len(radii) == 0 or weights.sum() == 0:
        return np.nan
    order = np.argsort(radii)
    cdf   = np.cumsum(weights[order]) / weights.sum()
    idx   = min(np.searchsorted(cdf, q), len(radii) - 1)
    return float(radii[order][idx])


# -- 4. Bootstrap CIs ----------------------------------------------------------

def bootstrap_quantile_ci(
    radii: np.ndarray, weights: np.ndarray, q: float,
    n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    point = weighted_quantile(radii, weights, q)
    n = len(radii)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot.append(weighted_quantile(radii[idx], weights[idx], q))
    arr = np.array(boot)
    return point, float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def bootstrap_count_ci(
    weights: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap CI on sum(weights) -- the completeness-corrected detection count."""
    point = float(weights.sum())
    n = len(weights)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot = [float(weights[rng.integers(0, n, size=n)].sum()) for _ in range(n_boot)]
    arr  = np.array(boot)
    return point, float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


# -- 5. Permutation tests ------------------------------------------------------

def permutation_test_q90(
    r_low: np.ndarray, w_low: np.ndarray,
    r_high: np.ndarray, w_high: np.ndarray,
    q: float, n_perm: int, rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """
    One-sided permutation test: H1: Q_q(low I) < Q_q(high I).
    observed_diff = Q_q(low) - Q_q(high).  Negative supports H1.
    p-value = fraction of null shuffles with diff <= observed.
    """
    obs    = weighted_quantile(r_low, w_low, q) - weighted_quantile(r_high, w_high, q)
    all_r  = np.concatenate([r_low, r_high])
    all_w  = np.concatenate([w_low, w_high])
    n_low  = len(r_low)
    null   = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(all_r))
        null[i] = (weighted_quantile(all_r[p[:n_low]], all_w[p[:n_low]], q)
                   - weighted_quantile(all_r[p[n_low:]], all_w[p[n_low:]], q))
    return obs, float((null <= obs).mean()), null


def permutation_test_count(
    w_low: np.ndarray, w_high: np.ndarray,
    n_perm: int, rng: np.random.Generator,
) -> tuple[float, float, np.ndarray]:
    """
    One-sided permutation test: H1: corrected_count(low I) < corrected_count(high I).
    Statistic: Delta = sum(w_low) - sum(w_high).  Negative supports H1.
    p-value = fraction of null shuffles with Delta <= observed.
    """
    obs   = float(w_low.sum() - w_high.sum())
    all_w = np.concatenate([w_low, w_high])
    n_low = len(w_low)
    null  = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(len(all_w))
        null[i] = all_w[p[:n_low]].sum() - all_w[p[n_low:]].sum()
    return obs, float((null <= obs).mean()), null


def weighted_ks_upper(
    r_low: np.ndarray, w_low: np.ndarray,
    r_high: np.ndarray, w_high: np.ndarray,
    n_perm: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Weighted KS statistic on the upper half (R > median). Permutation p-value."""
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
    null_D = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(len(all_r2))
        null_D[k] = _ks_d(all_r2[p[:nl]], all_w2[p[:nl]],
                           all_r2[p[nl:]], all_w2[p[nl:]])
    return D_obs, float((null_D >= D_obs).mean())


# -- 6. Plotting ---------------------------------------------------------------

def plot_completeness_grid(eta: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(
        eta, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1,
        extent=[0, len(COMP_I_EDGES)-1, 0, len(COMP_R_EDGES)-1],
    )
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
    ax.set_title("P-Pop completeness grid  eta(R,I) = P(TESS detects | transiting)\n"
                 "Rocky planets live in bottom rows (R<2); cold planets in left columns")
    for i in range(eta.shape[0]):
        for j in range(eta.shape[1]):
            if np.isfinite(eta[i, j]):
                ax.text(j+0.5, i+0.5, f"{eta[i,j]:.2f}",
                        ha="center", va="center", fontsize=4.5, color="white")
    fig.tight_layout()
    _save(fig, "fig1_completeness_grid.png")


def plot_survival_curves(
    groups: list[tuple[str, str, np.ndarray, np.ndarray]],
    title_suffix: str,
) -> None:
    r_grid = np.linspace(0.5, 4.0, 300)
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
        ax.axvline(LHS1140B_RADIUS,  ls=":",  color="gold", lw=0.9,
                   label=f"LHS 1140 b ({LHS1140B_RADIUS} Rearth)")
        ax.set_xlabel("Planet radius  [Rearth]")
        ax.set_ylabel("S(R) = P(R_p > R)")
        ax.set_title(panel)
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"TESS Survival curves: planet radius by insolation bin\n{title_suffix}",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "fig2_survival_curves.png")


def plot_bootstrap_quantiles(
    groups: list[tuple[str, str, float, float, float]],
    title_suffix: str,
) -> None:
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
    ax.set_title(
        f"90th-percentile radius by insolation bin  (95% bootstrap CI)\n{title_suffix}"
    )
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig3_bootstrap_quantile.png")


def plot_corrected_counts(
    groups: list[tuple[str, str, float, float, float]],
    title_suffix: str,
    rocky_mode: str,
) -> None:
    """
    Bar chart: completeness-corrected detection count per insolation bin.
    = sum(1/eta) = estimated number of TRANSITING planets had TESS been perfect.
    CAVEAT: geometric transit probability NOT corrected; cold bins naturally lower.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for xi, (label, color, pt, lo, hi) in enumerate(groups):
        ax.bar(xi, pt, color=color, alpha=0.75, width=0.6, label=label)
        ax.errorbar(xi, pt, yerr=[[pt-lo], [hi-pt]],
                    fmt="none", color="black", capsize=5, lw=1.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_ylabel("Corrected transiting count  (sum of weights)")
    ax.set_title(
        f"Completeness-corrected detection count -- rocky mode: {rocky_mode}\n"
        f"{title_suffix}\n"
        f"NOTE: geometric transit prob. NOT corrected (cold bins transit less often)"
    )
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig4_corrected_counts.png")


def plot_permutation_null(
    obs: float, null: np.ndarray, p_val: float,
    bins_compared: str, stat_label: str, fname: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=60, color="steelblue", alpha=0.7, density=True,
            label="Null distribution")
    ax.axvline(obs, color="crimson", lw=2, label=f"Observed: {obs:+.3f}")
    ax.set_xlabel(stat_label)
    ax.set_ylabel("Density")
    ax.set_title(
        f"Permutation test: {bins_compared}\n"
        f"H1: {stat_label.split('=')[0].strip()} < 0   one-sided p = {p_val:.4f}  "
        f"(N={N_PERMUTATION:,} shuffles)"
    )
    ax.legend()
    fig.tight_layout()
    _save(fig, fname)


def _save(fig, fname: str) -> None:
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# -- 7. Summary report ---------------------------------------------------------

def print_and_save_summary(
    nasa: pd.DataFrame,
    weights: np.ndarray,
    groups_data: list[tuple],
    perm_q90: list[tuple],
    perm_count: list[tuple],
    ks_results: list[tuple],
    title_suffix: str,
    rocky_mode: str,
) -> None:
    lines = [
        "=" * 72,
        "ROCKY PLANET PRIMORDIAL FORMATION TEST",
        "=" * 72,
        f"Sample    : {title_suffix}",
        f"Rocky mode: {rocky_mode}",
        f"N planets : {len(nasa):,}",
        f"Weights   : min={weights.min():.2f}  median={np.median(weights):.2f}"
        f"  max={weights.max():.2f}  cap={MAX_WEIGHT}",
        "",
        "-- Group sizes and corrected statistics ----------------------------",
    ]
    for label, _, radii, w in groups_data:
        lines.append(
            f"  {label:<28}  N_det={len(radii):>4}  "
            f"N_corr={w.sum():>7.1f}  "
            f"Q50={weighted_quantile(radii, w, 0.50):.3f}  "
            f"Q90={weighted_quantile(radii, w, 0.90):.3f} Rearth"
        )

    lines += ["", f"-- Bootstrap Q{int(QUANTILE_TEST*100)} (95% CI, N={N_BOOTSTRAP}) ---"]
    for label, _, radii, w in groups_data:
        pt, lo, hi = bootstrap_quantile_ci(
            radii, w, QUANTILE_TEST, N_BOOTSTRAP, np.random.default_rng(99))
        lines.append(f"  {label:<28}  Q90 = {pt:.3f} [{lo:.3f}, {hi:.3f}] Rearth")

    lines += ["", "-- Bootstrap corrected count (95% CI) ---------------------"]
    for label, _, radii, w in groups_data:
        pt, lo, hi = bootstrap_count_ci(w, N_BOOTSTRAP, np.random.default_rng(77))
        lines.append(f"  {label:<28}  count = {pt:.1f} [{lo:.1f}, {hi:.1f}]")

    lines += ["", "-- Permutation test: Q90 (radius upper bound) -------------"]
    for la, lb, obs, pv in perm_q90:
        sig = ("SIGNIFICANT" if pv < 0.05
               else "marginal" if pv < 0.10
               else "not significant")
        lines += [
            f"  {la} vs {lb}",
            f"    DeltaQ90 = {obs:+.4f} Rearth   p = {pv:.4f}  [{sig}]",
        ]

    lines += ["", "-- Permutation test: corrected count (formation rate) -----"]
    for la, lb, obs, pv in perm_count:
        sig = ("SIGNIFICANT" if pv < 0.05
               else "marginal" if pv < 0.10
               else "not significant")
        lines += [
            f"  {la} vs {lb}",
            f"    Delta_count = {obs:+.1f}   p = {pv:.4f}  [{sig}]",
            f"    (NOTE: geometric transit prob. not corrected in count)",
        ]

    lines += ["", "-- Weighted KS upper-tail -----------------------------------"]
    for la, lb, D, p_ks in ks_results:
        lines.append(f"  {la} vs {lb}:  D = {D:.4f}   p ~= {p_ks:.4f}")

    lines += [
        "",
        "-- How to interpret --------------------------------------------------",
        "",
        f"  ROCKY MODE:",
        f"    all           -> all planets; large N but includes gas giants",
        f"    mass-confirmed-> mass-confirmed rocky; physically rigorous but RV-biased",
        f"                     against cold long-period planets (expect N~5-15 at I<10)",
        f"    radius-gap    -> R < {ROCKY_RADIUS_GAP} Rearth proxy; no mass required;",
        f"                     justified at I<10 (no photoevaporation to confuse it)",
        "",
        "  Q90 PERMUTATION (primary):",
        "    p < 0.05 -> radius upper bound is real after detection correction",
        "    p >= 0.10-> cannot distinguish from detection bias alone",
        "",
        "  COUNT PERMUTATION (secondary):",
        "    p < 0.05 -> fewer corrected cold rocky detections",
        "               (may be real OR still insufficient P-Pop statistics)",
        "    p >= 0.10-> corrected counts consistent with equal formation rate",
        "    CAUTION  -> cold bins have lower geometric transit rates by geometry;",
        "               this test does NOT correct for that factor",
        "",
        "  PRIMORDIAL FORMATION CONCLUSION:",
        "    If Q90 test p >= 0.10 AND count test p >= 0.10 after radius-gap mode:",
        "    -> apparent rocky planet deficit at I<10 is consistent with TESS",
        "       detection bias alone; primordial rocky formation at low insolation",
        "       is not statistically excluded.",
        "=" * 72,
    ]

    text = "\n".join(lines)
    print(text)
    out = OUT_DIR / "summary.txt"
    out.write_text(text, encoding="utf-8")
    print(f"\n  Saved: {out}")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-catalogs", type=int, default=400,
        help="Number of P-Pop tess_catalog CSVs for completeness grid (default: 400).",
    )
    parser.add_argument(
        "--star-types", type=str, default="F,G,K,M",
        help="Comma-separated star types to include (default: F,G,K,M).",
    )
    parser.add_argument(
        "--rocky-mode", type=str, default="all",
        choices=["all", "mass-confirmed", "radius-gap"],
        help=(
            "Rocky filter applied before analysis.\n"
            "  all           : all quality-filtered planets (original behavior)\n"
            "  mass-confirmed: planets with radius_p <= rocky_threshold_at_mass(mass_p)\n"
            "                  (LHS 1140 b anchor + ref.ddat; mirrors script 36)\n"
            "  radius-gap    : R_p < 1.65 R_earth proxy; mass filters relaxed;\n"
            "                  at I<10 no photoevaporation so proxy is reliable"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    star_types = [s.strip().upper() for s in args.star_types.split(",")]
    title_sfx  = (f"Stars: {'+'.join(star_types)}  |  rocky: {args.rocky_mode}"
                  f"  |  P-Pop: {PPOP_DATA_DIR.name}")

    # -- Load rocky curve if needed --------------------------------------------
    m_ref = r_ref = None
    shift = 0.0
    if args.rocky_mode == "mass-confirmed":
        print("\n-- Rocky threshold ------------------------------------------")
        m_ref, r_ref, shift = load_rocky_curve()

    # -- 1. Load and filter real planets ---------------------------------------
    print("\n-- 1. NASA PSCompPars -----------------------------------------------")
    nasa_all = load_nasa(star_types, args.rocky_mode)
    nasa     = apply_rocky_filter(nasa_all, args.rocky_mode, m_ref, r_ref, shift)
    print(f"  Final sample for analysis: {len(nasa):,} planets")
    if len(nasa) < 10:
        print("  WARNING: fewer than 10 planets -- results will be unreliable.")

    # -- 2. Completeness grid --------------------------------------------------
    print(f"\n-- 2. P-Pop completeness grid  ({args.n_catalogs} catalogs) ----------")
    ppop = load_ppop_subset(args.n_catalogs)
    eta  = compute_completeness_grid(ppop)
    plot_completeness_grid(eta)

    # -- 3. Assign weights -----------------------------------------------------
    print("\n-- 3. Inverse-completeness weights ---------------------------------")
    weights     = assign_weights(nasa, eta)
    nasa        = nasa.copy()
    nasa["weight"] = weights

    # -- 4. Group by insolation ------------------------------------------------
    print("\n-- 4. Group by insolation bin ---------------------------------------")
    groups_data: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for j in range(len(INSOL_EDGES) - 1):
        lo, hi = INSOL_EDGES[j], INSOL_EDGES[j+1]
        mask   = (nasa["flux_p"] >= lo) & (nasa["flux_p"] < hi)
        r      = nasa.loc[mask, "radius_p"].to_numpy()
        w      = nasa.loc[mask, "weight"].to_numpy()
        groups_data.append((INSOL_LABELS[j], INSOL_COLORS[j], r, w))
        print(f"  {INSOL_LABELS[j]}: N_detected={len(r)}  N_corrected={w.sum():.1f}")

    # -- 5. Survival curves ----------------------------------------------------
    print("\n-- 5. Survival curves -----------------------------------------------")
    plot_survival_curves([(l, c, r, w) for l, c, r, w in groups_data], title_sfx)

    # -- 6. Bootstrap Q90 ------------------------------------------------------
    print(f"\n-- 6. Bootstrap Q{int(QUANTILE_TEST*100)}  (N={N_BOOTSTRAP}) ----------")
    boot_q90 = []
    for label, color, radii, w in groups_data:
        pt, lo, hi = bootstrap_quantile_ci(radii, w, QUANTILE_TEST, N_BOOTSTRAP, rng)
        print(f"  {label:<28}: Q90 = {pt:.3f}  [{lo:.3f}, {hi:.3f}] Rearth")
        boot_q90.append((label, color, pt, lo, hi))
    plot_bootstrap_quantiles(boot_q90, title_sfx)

    # -- 7. Bootstrap corrected count ------------------------------------------
    print(f"\n-- 7. Bootstrap corrected count  (N={N_BOOTSTRAP}) ----------------")
    boot_cnt = []
    for label, color, radii, w in groups_data:
        pt, lo, hi = bootstrap_count_ci(w, N_BOOTSTRAP, rng)
        print(f"  {label:<28}: count = {pt:.1f}  [{lo:.1f}, {hi:.1f}]")
        boot_cnt.append((label, color, pt, lo, hi))
    plot_corrected_counts(boot_cnt, title_sfx, args.rocky_mode)

    # -- 8. Permutation tests --------------------------------------------------
    print(f"\n-- 8. Permutation tests  (N={N_PERMUTATION}) -----------------------")
    lbl_l, _, r_l, w_l = groups_data[0]   # I < 10
    lbl_m, _, r_m, w_m = groups_data[1]   # 10-50
    lbl_h, _, r_h, w_h = groups_data[2]   # I >= 50

    # Q90 permutation
    obs_q90_lh, pv_q90_lh, null_q90_lh = permutation_test_q90(
        r_l, w_l, r_h, w_h, QUANTILE_TEST, N_PERMUTATION, rng)
    obs_q90_lm, pv_q90_lm, _           = permutation_test_q90(
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
        f"DeltaQ{int(QUANTILE_TEST*100)} = Q{int(QUANTILE_TEST*100)}(low) - "
        f"Q{int(QUANTILE_TEST*100)}(high)  [Rearth]",
        "fig5_permutation_q90.png",
    )

    # Count permutation
    obs_cnt_lh, pv_cnt_lh, null_cnt_lh = permutation_test_count(
        w_l, w_h, N_PERMUTATION, rng)
    obs_cnt_lm, pv_cnt_lm, _           = permutation_test_count(
        w_l, w_m, N_PERMUTATION, rng)
    perm_count = [
        (lbl_l, lbl_h, obs_cnt_lh, pv_cnt_lh),
        (lbl_l, lbl_m, obs_cnt_lm, pv_cnt_lm),
    ]
    print(f"  Count {lbl_l} vs {lbl_h}: delta={obs_cnt_lh:+.1f}  p={pv_cnt_lh:.4f}")
    print(f"  Count {lbl_l} vs {lbl_m}: delta={obs_cnt_lm:+.1f}  p={pv_cnt_lm:.4f}")
    plot_permutation_null(
        obs_cnt_lh, null_cnt_lh, pv_cnt_lh,
        f"{lbl_l}  vs  {lbl_h}",
        "Delta_count = corrected_count(low) - corrected_count(high)",
        "fig6_permutation_count.png",
    )

    # KS upper-tail
    ks_results = []
    for (la, ra, wa), (lb, rb, wb) in [
        ((lbl_l, r_l, w_l), (lbl_h, r_h, w_h)),
        ((lbl_l, r_l, w_l), (lbl_m, r_m, w_m)),
    ]:
        D, p_ks = weighted_ks_upper(ra, wa, rb, wb, n_perm=2000, rng=rng)
        print(f"  KS upper {la} vs {lb}: D={D:.4f}  p={p_ks:.4f}")
        ks_results.append((la, lb, D, p_ks))

    # -- 9. Summary ------------------------------------------------------------
    print("\n-- 9. Summary -------------------------------------------------------")
    print_and_save_summary(
        nasa, weights, groups_data,
        perm_q90, perm_count, ks_results,
        title_sfx, args.rocky_mode,
    )


if __name__ == "__main__":
    main()
