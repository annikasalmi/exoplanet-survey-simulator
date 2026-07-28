"""
14_survival_kepler_tess.py

Cross-survey combination: TESS (script 39) + Kepler (script 40).

SCIENCE MOTIVATION
------------------
TESS and Kepler probe the same underlying planet population through different
detection lenses.  Combining them gives:
  1. Independent replication — if both surveys agree, the result is robust.
  2. Complementary power — Kepler sees cold (long-period) planets that TESS misses.
  3. Statistical gain — Fisher's method combines independent p-values to produce
     a single, more powerful test.

HOW THE COMBINATION WORKS
--------------------------
For each survey independently:
  - Load NASA PSCompPars (filtered to that survey's detections)
  - Build completeness grid eta(R, I) from that survey's P-Pop catalogs
  - Assign 1/eta weights
  - Compute Q90 per insolation bin and permutation p-value (H1: Q90_low < Q90_high)

Then combine:
  A) FISHER'S METHOD (p-value combination)
     chi2 = -2 * (ln(p_TESS) + ln(p_Kepler))
     Under H0, chi2 ~ chi2(4) (2 surveys × 2 degrees of freedom each)
     p_combined = P(chi2_4 > chi2_observed)
     Interpretation: one-sided test that BOTH surveys reject H0 in the same direction.

  B) INVERSE-VARIANCE WEIGHTED Q90
     sigma_i = (CI_hi - CI_lo) / (2 * 1.96)  from 95% bootstrap CI
     Q90_combined = sum(Q90_i / sigma_i^2) / sum(1 / sigma_i^2)
     sigma_combined = 1 / sqrt(sum(1 / sigma_i^2))
     This is the standard meta-analytic fixed-effects estimator.

  C) CONSISTENCY CHECK
     |Q90_TESS - Q90_Kepler| / sqrt(sigma_TESS^2 + sigma_Kepler^2)
     If > 2.0: surveys are inconsistent (heterogeneity); combination unreliable.
     If <= 2.0: consistent; combination valid.

Usage:
    python scripts/14_survival_kepler_tess.py
    python scripts/14_survival_kepler_tess.py --rocky-mode radius-gap
    python scripts/14_survival_kepler_tess.py --rocky-mode mass-confirmed --star-types F,G,K,M
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

try:
    from scipy.stats import chi2 as scipy_chi2
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False
    print("WARNING: scipy not found -- Fisher's combined p-value will use approximation.")


# -- Project root --------------------------------------------------------------

def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "run" / "tess").exists():
            return p
    return start.parents[2]


ROOT = find_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "my_outputs" / "14_survival_kepler_tess"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -- Survey configurations -----------------------------------------------------

SURVEYS = {
    "TESS": {
        "ppop_dir":        ROOT / "run" / "tess" / "data" / "Gaia_C_F_K_combined_cdpp_v1",
        "catalog_pattern": "tess_catalog_*.csv",
        "trans_col":       "tess_transiting_geometric",
        "obs_col":         "tess_observed",       # falls back to all-True if missing
        "det_col":         "tess_detected",
        "nasa_facility":   None,                  # use all TESS-surveyed transiting planets
        "tess_facility":   "Transiting Exoplanet Survey Satellite (TESS)",
        "color":           "#e6550d",
    },
    "Kepler": {
        "ppop_dir":        ROOT / "run" / "kepler" / "data" / "Gaia_C_F_K_combined",
        "catalog_pattern": "kepler_catalog_*.csv",
        "trans_col":       "transiting_geometric",
        "obs_col":         "bright_enough_kepler",
        "det_col":         "detected",
        "nasa_facility":   "Kepler",
        "color":           "#3182bd",
    },
}

NASA_FLAGS_CACHE = (
    ROOT / "run" / "kepler" / "data" / "NASA"
    / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv"
)
REF_CURVE_PATH = ROOT / "run" / "kepler" / "reference_curves" / "ref.ddat"

LHS1140B_MASS    = 5.60
LHS1140B_RADIUS  = 1.730
ROCKY_RADIUS_GAP = 1.65

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
MIN_BIN_N     = 3
N_BOOTSTRAP   = 4_000
N_PERMUTATION = 10_000
QUANTILE_TEST = 0.90

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 220,
    "font.size": 10, "axes.labelsize": 10,
    "axes.titlesize": 11, "legend.fontsize": 8,
})


# -- Shared helpers (mirrors scripts 39/40) ------------------------------------

def infer_stype(teff):
    t = pd.to_numeric(teff, errors="coerce")
    s = pd.Series("Unknown", index=t.index, dtype=object)
    s[t >= 7500] = "A"
    s[(t >= 6000) & (t < 7500)] = "F"
    s[(t >= 5200) & (t < 6000)] = "G"
    s[(t >= 3700) & (t < 5200)] = "K"
    s[(t > 0)    & (t < 3700)]  = "M"
    return s


def load_rocky_curve():
    if not REF_CURVE_PATH.exists():
        m = np.linspace(0.05, 30.0, 600)
        r = m ** 0.27
    else:
        ref = np.loadtxt(REF_CURVE_PATH, comments="#")
        m, r = ref[:, 0].astype(float), ref[:, 1].astype(float)
        ok = np.isfinite(m) & np.isfinite(r) & (m > 0) & (r > 0)
        m, r = m[ok], r[ok]
        idx = np.argsort(m)
        m, r = m[idx], r[idx]
    r_at_lhs = float(np.interp(LHS1140B_MASS, m, r))
    return m, r, LHS1140B_RADIUS - r_at_lhs


def rocky_threshold_at_mass(masses, m_ref, r_ref, shift):
    return np.interp(np.asarray(masses, dtype=float), m_ref, r_ref + shift,
                     left=np.nan, right=np.nan)


def load_nasa_for_survey(survey_name: str, cfg: dict,
                         star_types: list[str], rocky_mode: str) -> pd.DataFrame:
    raw = pd.read_csv(NASA_FLAGS_CACHE)
    rename = {
        "pl_name":      "planet_name", "hostname": "host_name",
        "pl_bmasse":    "mass_p",   "pl_bmasseerr1": "mass_err_plus",
        "pl_bmasseerr2":"mass_err_minus", "pl_bmasselim": "mass_limit_flag",
        "pl_bmassprov": "mass_provider",
        "pl_rade":      "radius_p",  "pl_radeerr1":  "radius_err_plus",
        "pl_radeerr2":  "radius_err_minus", "pl_radelim":   "radius_limit_flag",
        "pl_insol":     "flux_p",    "st_teff":       "teff_s",
    }
    df = raw.rename(columns={k: v for k, v in rename.items() if k in raw.columns})

    # Filter to this survey's discoveries
    facility = cfg["nasa_facility"]
    if facility and "disc_facility" in df.columns:
        df = df[df["disc_facility"].astype(str).str.strip() == facility].copy()
    elif survey_name == "TESS" and "disc_facility" in df.columns:
        tess_fac = cfg.get("tess_facility", "Transiting Exoplanet Survey Satellite (TESS)")
        df = df[df["disc_facility"].astype(str).str.strip() == tess_fac].copy()

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

    print(f"  [{survey_name}] quality-filtered ({'/'.join(star_types)}, mode={rocky_mode}): {len(df):,}")
    return df.reset_index(drop=True)


def apply_rocky_filter(df, rocky_mode, m_ref=None, r_ref=None, shift=0.0):
    if rocky_mode == "all":
        return df.copy()
    if rocky_mode == "radius-gap":
        rocky = df[df["radius_p"] <= ROCKY_RADIUS_GAP].copy()
        print(f"  Radius-gap filter: {len(df):,} -> {len(rocky):,}")
        return rocky
    thresh = rocky_threshold_at_mass(df["mass_p"].to_numpy(), m_ref, r_ref, shift)
    mask   = (df["radius_p"].to_numpy() <= thresh) & np.isfinite(thresh)
    rocky  = df[mask].copy()
    print(f"  Mass-confirmed filter: {len(df):,} -> {len(rocky):,}")
    return rocky


def load_ppop_subset(cfg: dict, n_catalogs: int) -> pd.DataFrame:
    ppop_dir = cfg["ppop_dir"]
    pattern  = cfg["catalog_pattern"]
    files    = sorted(ppop_dir.glob(pattern))[:n_catalogs]
    if not files:
        raise FileNotFoundError(f"No {pattern} in {ppop_dir}")
    trans_col = cfg["trans_col"]
    obs_col   = cfg["obs_col"]
    det_col   = cfg["det_col"]
    usecols_want = {"radius_p", "flux_p", "stype", trans_col, obs_col, det_col}
    frames = []
    for p in files:
        try:
            d = pd.read_csv(p, usecols=lambda c: c in usecols_want)
            frames.append(d)
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)

    for col in [trans_col, obs_col, det_col]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                       .isin(["true", "1", "yes", "y", "t"]).astype(bool))
        else:
            df[col] = True

    # Unify column names for compute_completeness_grid
    df["_trans"] = df[trans_col]
    df["_obs"]   = df[obs_col]
    df["_det"]   = df[det_col]

    df["radius_p"] = pd.to_numeric(df["radius_p"], errors="coerce")
    df["flux_p"]   = pd.to_numeric(df["flux_p"],   errors="coerce")
    return df.dropna(subset=["radius_p", "flux_p"]).reset_index(drop=True)


def compute_completeness_grid(ppop: pd.DataFrame) -> np.ndarray:
    r     = ppop["radius_p"].to_numpy()
    f     = ppop["flux_p"].to_numpy()
    trans = ppop["_trans"].to_numpy(bool)
    obs   = ppop["_obs"].to_numpy(bool)
    det   = ppop["_det"].to_numpy(bool)
    nr, ni = len(COMP_R_EDGES) - 1, len(COMP_I_EDGES) - 1
    eta = np.full((nr, ni), np.nan)
    for i in range(nr):
        for j in range(ni):
            cell = (
                (r >= COMP_R_EDGES[i]) & (r < COMP_R_EDGES[i+1]) &
                (f >= COMP_I_EDGES[j]) & (f < COMP_I_EDGES[j+1]) &
                trans & obs
            )
            if cell.sum() >= 5:
                eta[i, j] = det[cell].sum() / cell.sum()
    print(f"  Completeness grid: {np.isfinite(eta).sum()}/{eta.size} valid  "
          f"(median eta = {np.nanmedian(eta):.3f})")
    return eta


def assign_weights(nasa, eta):
    r  = nasa["radius_p"].to_numpy()
    f  = nasa["flux_p"].to_numpy()
    ri = np.clip(np.digitize(r, COMP_R_EDGES) - 1, 0, len(COMP_R_EDGES) - 2)
    fi = np.clip(np.digitize(f, COMP_I_EDGES) - 1, 0, len(COMP_I_EDGES) - 2)
    weights = np.empty(len(nasa), dtype=float)
    for k in range(len(nasa)):
        e = eta[ri[k], fi[k]]
        weights[k] = min(1.0 / e, MAX_WEIGHT) if (np.isfinite(e) and e > 0) else MAX_WEIGHT
    return weights


def weighted_quantile(radii, weights, q):
    if len(radii) == 0 or weights.sum() == 0:
        return np.nan
    order = np.argsort(radii)
    cdf   = np.cumsum(weights[order]) / weights.sum()
    return float(radii[order][min(np.searchsorted(cdf, q), len(radii) - 1)])


def bootstrap_quantile_ci(radii, weights, q, n_boot, rng):
    if len(radii) == 0:
        return np.nan, np.nan, np.nan
    point = weighted_quantile(radii, weights, q)
    n = len(radii)
    boot = [weighted_quantile(radii[idx := rng.integers(0, n, size=n)], weights[idx], q)
            for _ in range(n_boot)]
    arr = np.array(boot)
    return point, float(np.nanpercentile(arr, 2.5)), float(np.nanpercentile(arr, 97.5))


def permutation_test_q90(r_low, w_low, r_high, w_high, q, n_perm, rng):
    obs   = weighted_quantile(r_low, w_low, q) - weighted_quantile(r_high, w_high, q)
    all_r = np.concatenate([r_low, r_high])
    all_w = np.concatenate([w_low, w_high])
    n_low = len(r_low)
    null  = np.array([
        weighted_quantile(all_r[p := rng.permutation(len(all_r))][:n_low], all_w[p[:n_low]], q)
        - weighted_quantile(all_r[p[n_low:]], all_w[p[n_low:]], q)
        for _ in range(n_perm)
    ])
    return obs, float((null <= obs).mean())


# -- Fisher's combination ------------------------------------------------------

def fishers_combined_p(p_values: list[float]) -> float:
    """
    Fisher's method: chi2 = -2 * sum(log(p_i)), df = 2 * len(p_values).
    Returns combined one-sided p-value under H0 (all p_i uniform on [0,1]).
    """
    eps = 1e-300  # avoid log(0)
    chi2_stat = -2.0 * sum(np.log(max(p, eps)) for p in p_values)
    df = 2 * len(p_values)
    if _HAVE_SCIPY:
        return float(scipy_chi2.sf(chi2_stat, df))
    # Fallback: chi2 CDF approximation via Wilson-Hilferty
    z = ((chi2_stat / df) ** (1/3) - (1 - 2/(9*df))) / np.sqrt(2/(9*df))
    return float(0.5 * (1 - np.tanh(z * (1 + z**2 / 6) / np.sqrt(2))))


def inverse_variance_combine(estimates: list[tuple[float, float, float]]) -> tuple[float, float]:
    """
    Combine (point, lo, hi) estimates using inverse-variance weighting.
    sigma_i = (hi - lo) / (2 * 1.96).
    Returns (combined_estimate, combined_sigma).
    """
    filtered = [(pt, lo, hi) for pt, lo, hi in estimates
                if np.isfinite(pt) and hi > lo]
    if not filtered:
        return np.nan, np.nan
    sigmas  = [(hi - lo) / (2 * 1.96) for pt, lo, hi in filtered]
    points  = [pt for pt, lo, hi in filtered]
    weights = [1.0 / (s ** 2) for s in sigmas]
    combined = sum(p * w for p, w in zip(points, weights)) / sum(weights)
    sigma_c  = 1.0 / np.sqrt(sum(weights))
    return float(combined), float(sigma_c)


# -- Main survey runner --------------------------------------------------------

def run_survey(
    survey_name: str,
    star_types: list[str],
    rocky_mode: str,
    n_catalogs: int,
    rng: np.random.Generator,
    m_ref=None, r_ref=None, shift=0.0,
) -> dict:
    """Run the full survival analysis for one survey. Returns a stats dict."""
    cfg = SURVEYS[survey_name]
    print(f"\n{'='*60}")
    print(f"  SURVEY: {survey_name}")
    print(f"{'='*60}")

    nasa_all = load_nasa_for_survey(survey_name, cfg, star_types, rocky_mode)
    nasa     = apply_rocky_filter(nasa_all, rocky_mode, m_ref, r_ref, shift)
    if len(nasa) < 5:
        print(f"  WARNING: only {len(nasa)} planets -- skipping {survey_name}.")
        return {"survey": survey_name, "n": len(nasa), "skipped": True}

    ppop    = load_ppop_subset(cfg, n_catalogs)
    print(f"  Loaded {len(ppop):,} P-Pop planets from {survey_name}")
    eta     = compute_completeness_grid(ppop)
    weights = assign_weights(nasa, eta)
    nasa    = nasa.copy()
    nasa["weight"] = weights

    groups = []
    for j in range(len(INSOL_EDGES) - 1):
        lo, hi = INSOL_EDGES[j], INSOL_EDGES[j+1]
        mask   = (nasa["flux_p"] >= lo) & (nasa["flux_p"] < hi)
        r      = nasa.loc[mask, "radius_p"].to_numpy()
        w      = nasa.loc[mask, "weight"].to_numpy()
        groups.append({"label": INSOL_LABELS[j], "color": INSOL_COLORS[j],
                       "r": r, "w": w})
        print(f"  {INSOL_LABELS[j]}: N={len(r)}  N_corr={w.sum():.1f}")

    g_low  = groups[0]
    g_high = groups[2]

    # Bootstrap Q90
    q90_results = []
    for g in groups:
        pt, lo, hi = bootstrap_quantile_ci(g["r"], g["w"], QUANTILE_TEST, N_BOOTSTRAP, rng)
        g["q90_pt"], g["q90_lo"], g["q90_hi"] = pt, lo, hi
        q90_results.append((pt, lo, hi))
        print(f"  Q90 {g['label']}: {pt:.3f} [{lo:.3f}, {hi:.3f}]")

    # Permutation test (low vs high)
    if len(g_low["r"]) < MIN_BIN_N:
        print(f"  SKIP permutation: cold bin has {len(g_low['r'])} planets "
              f"(need >= {MIN_BIN_N}).")
        obs_q90, p_q90 = np.nan, np.nan
    else:
        obs_q90, p_q90 = permutation_test_q90(
            g_low["r"], g_low["w"], g_high["r"], g_high["w"],
            QUANTILE_TEST, N_PERMUTATION, rng,
        )
        print(f"  Q90 permutation ({INSOL_LABELS[0]} vs {INSOL_LABELS[2]}): "
              f"delta={obs_q90:+.4f}  p={p_q90:.4f}")

    return {
        "survey":    survey_name,
        "n":         len(nasa),
        "skipped":   False,
        "groups":    groups,
        "eta":       eta,
        "obs_q90":   obs_q90,
        "p_q90":     p_q90,
        "q90_low":   (g_low["q90_pt"], g_low["q90_lo"], g_low["q90_hi"]),
        "q90_high":  (g_high["q90_pt"], g_high["q90_lo"], g_high["q90_hi"]),
    }


# -- Plotting ------------------------------------------------------------------

def _save(fig, fname):
    out = OUT_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_completeness_comparison(results: dict) -> None:
    """Side-by-side completeness grids for TESS vs Kepler."""
    valid_results = {k: v for k, v in results.items() if not v.get("skipped")}
    if len(valid_results) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (sname, res) in zip(axes, valid_results.items()):
        eta = res["eta"]
        im = ax.imshow(eta, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=1,
                       extent=[0, len(COMP_I_EDGES)-1, 0, len(COMP_R_EDGES)-1])
        plt.colorbar(im, ax=ax, label="eta(R, I)")
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
        ax.set_title(f"{sname}: P(detect | transiting)")
        ax.set_xlabel("Insolation bin centre (Iearth)")
        ax.set_ylabel("Radius bin centre (Rearth)")
        for i in range(eta.shape[0]):
            for j in range(eta.shape[1]):
                if np.isfinite(eta[i, j]):
                    ax.text(j+0.5, i+0.5, f"{eta[i,j]:.2f}",
                            ha="center", va="center", fontsize=5, color="white")
    fig.suptitle("Completeness comparison: TESS vs Kepler\n"
                 "Left columns = cold planets; bottom rows = rocky planets",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "fig1_completeness_comparison.png")


def plot_survival_comparison(results: dict, title_suffix: str) -> None:
    """Survival curves from both surveys in one panel (corrected only)."""
    valid_results = {k: v for k, v in results.items() if not v.get("skipped")}
    r_grid = np.linspace(0.5, 4.0, 300)

    def weighted_survival(radii, weights, grid):
        w_tot = weights.sum()
        return np.array([weights[radii > rv].sum() / w_tot for rv in grid]) if w_tot else grid * 0

    survey_styles = {"TESS": "-", "Kepler": "--"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for sname, res in valid_results.items():
        ls = survey_styles.get(sname, "-")
        for g in res["groups"]:
            if len(g["r"]) == 0:
                continue
            surv = weighted_survival(g["r"], g["w"], r_grid)
            label = f"{sname}: {g['label']}"
            ax.plot(r_grid, surv, color=g["color"], lw=2, ls=ls, label=label)
    ax.axvline(ROCKY_RADIUS_GAP, ls="--", color="gray", lw=0.9,
               label=f"Radius gap ({ROCKY_RADIUS_GAP} Rearth)")
    ax.axvline(LHS1140B_RADIUS,  ls=":",  color="gold", lw=0.9,
               label=f"LHS 1140 b ({LHS1140B_RADIUS} Rearth)")
    ax.set_xlabel("Planet radius  [Rearth]")
    ax.set_ylabel("S(R) = P(R_p > R)  [completeness-corrected]")
    ax.set_xlim(0.5, 3.5)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Completeness-corrected survival curves: TESS (solid) vs Kepler (dashed)\n{title_suffix}")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig2_survival_comparison.png")


def plot_q90_comparison(results: dict, combined_q90_low, combined_q90_high,
                        title_suffix: str) -> None:
    """Q90 estimates per survey + combined, for low and high insolation."""
    valid_results = {k: v for k, v in results.items() if not v.get("skipped")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    bin_titles = [INSOL_LABELS[0], INSOL_LABELS[2]]
    bin_q90s   = [
        [(res["q90_low"])  for res in valid_results.values()],
        [(res["q90_high"]) for res in valid_results.values()],
    ]
    combined = [combined_q90_low, combined_q90_high]

    for ax, btitle, survey_q90s, comb in zip(axes, bin_titles, bin_q90s, combined):
        for xi, (sname, (pt, lo, hi)) in enumerate(zip(valid_results, survey_q90s)):
            color = SURVEYS[sname]["color"]
            ax.errorbar(xi, pt, yerr=[[pt-lo], [hi-pt]],
                        fmt="o", color=color, ms=8, capsize=5, lw=1.5,
                        label=sname)
        if np.isfinite(comb[0]):
            pt_c, sig_c = comb
            ax.axhline(pt_c, color="black", lw=1.5, ls="--",
                       label=f"Combined: {pt_c:.3f} ± {sig_c:.3f}")
            ax.axhspan(pt_c - sig_c, pt_c + sig_c, alpha=0.1, color="black")
        ax.axhline(ROCKY_RADIUS_GAP, ls="--", color="gray", lw=0.9)
        ax.set_xticks(range(len(valid_results)))
        ax.set_xticklabels(list(valid_results.keys()))
        ax.set_ylabel("Q90 radius  [Rearth]")
        ax.set_title(btitle)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Q90 per survey + inverse-variance combined\n{title_suffix}", fontsize=11)
    fig.tight_layout()
    _save(fig, "fig3_q90_comparison.png")


def plot_pvalue_comparison(results: dict, p_combined: float, title_suffix: str) -> None:
    """Bar plot of survey p-values + combined Fisher's p."""
    valid_results = {k: v for k, v in results.items() if not v.get("skipped")}
    names = list(valid_results.keys()) + ["Fisher\nCombined"]
    pvals = [res["p_q90"] for res in valid_results.values()] + [p_combined]
    colors = [SURVEYS[n]["color"] for n in valid_results] + ["#31a354"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, pvals, color=colors, alpha=0.8, width=0.5)
    ax.axhline(0.05,  color="red",    lw=1.5, ls="--", label="p=0.05 (significant)")
    ax.axhline(0.10,  color="orange", lw=1.2, ls=":",  label="p=0.10 (marginal)")
    ax.set_ylabel("One-sided p-value\nH1: Q90(low I) < Q90(high I)")
    ax.set_title(
        f"Survey p-values + Fisher's combined test\n{title_suffix}\n"
        f"(p < 0.05 = cold rocky planets have smaller upper radius bound)"
    )
    for bar, p in zip(bars, pvals):
        if np.isfinite(p):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{p:.4f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                    "N/A", ha="center", va="bottom", fontsize=9, color="gray")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig4_pvalue_comparison.png")


# -- Summary report ------------------------------------------------------------

def save_summary(results: dict, p_combined: float,
                 combined_q90_low, combined_q90_high,
                 title_suffix: str, rocky_mode: str) -> None:
    valid_results = {k: v for k, v in results.items() if not v.get("skipped")}
    p_values = [res["p_q90"] for res in valid_results.values()]

    lines = [
        "=" * 72,
        "CROSS-SURVEY COMBINATION: TESS + KEPLER",
        "ROCKY PLANET PRIMORDIAL FORMATION TEST",
        "=" * 72,
        f"Rocky mode : {rocky_mode}",
        f"Sample     : {title_suffix}",
        "",
        "-- Per-survey results -------------------------------------------",
    ]
    for sname, res in valid_results.items():
        g_low  = res["groups"][0]
        g_high = res["groups"][2]
        pt_l, lo_l, hi_l = res["q90_low"]
        pt_h, lo_h, hi_h = res["q90_high"]
        _pq90 = res["p_q90"]
        if np.isnan(_pq90):
            sig = "N/A"
            perm_line = "    Permutation p = N/A (cold bin too small)"
        else:
            sig = "SIGNIFICANT" if _pq90 < 0.05 else "marginal" if _pq90 < 0.10 else "not significant"
            perm_line = f"    Permutation p = {_pq90:.4f}  [{sig}]"
        obs_str = f"{res['obs_q90']:+.4f}" if np.isfinite(res["obs_q90"]) else "N/A"
        lines += [
            f"  Survey: {sname}",
            f"    N planets (after filters): {res['n']:,}",
            f"    Cold  ({INSOL_LABELS[0]}): N={len(g_low['r'])}  "
            f"Q90={pt_l:.3f} [{lo_l:.3f}, {hi_l:.3f}] Rearth",
            f"    Hot   ({INSOL_LABELS[2]}): N={len(g_high['r'])}  "
            f"Q90={pt_h:.3f} [{lo_h:.3f}, {hi_h:.3f}] Rearth",
            f"    Delta Q90 (cold-hot) = {obs_str} Rearth",
            perm_line,
            "",
        ]

    finite_p  = [p for p in p_values if np.isfinite(p)]
    chi2_str  = f"{-2.0 * sum(np.log(max(p, 1e-300)) for p in finite_p):.3f}" if finite_p else "N/A"
    pcomb_str = f"{p_combined:.4f}" if np.isfinite(p_combined) else "N/A"
    sig_str   = (f"({'NOT ' if p_combined >= 0.05 else ''}SIGNIFICANT at alpha=0.05)"
                 if np.isfinite(p_combined) else "(insufficient data for significance test)")
    lines += [
        "-- Fisher's combined test ----------------------------------------",
        "  Individual p-values: " + "  ".join(
            f"{sn}={pv:.4f}" if np.isfinite(pv) else f"{sn}=N/A"
            for sn, pv in zip(valid_results, p_values)),
        f"  chi2 statistic: {chi2_str}",
        f"  Fisher's combined p = {pcomb_str}",
        f"  {sig_str}",
        "",
        "-- Combined Q90 (inverse-variance weighted) ----------------------",
    ]
    for label, comb in [(INSOL_LABELS[0], combined_q90_low), (INSOL_LABELS[2], combined_q90_high)]:
        pt_c, sig_c = comb
        if np.isfinite(pt_c):
            lines.append(f"  {label}: Q90_combined = {pt_c:.3f} +/- {sig_c:.3f} Rearth")
        else:
            lines.append(f"  {label}: Q90_combined = N/A (insufficient data)")

    delta_c = combined_q90_low[0] - combined_q90_high[0]
    sigma_delta = np.sqrt(combined_q90_low[1]**2 + combined_q90_high[1]**2)
    z_score = delta_c / sigma_delta if sigma_delta > 0 else np.nan
    delta_str = f"{delta_c:+.3f}" if np.isfinite(delta_c) else "N/A"
    z_str     = f"{z_score:.2f}" if np.isfinite(z_score) else "N/A"
    lines += [
        f"  Delta Q90 combined (cold-hot) = {delta_str} Rearth",
        f"  Z-score = {z_str}  (|Z| > 2 suggests significant difference)",
        "",
        "-- Consistency check (TESS vs Kepler) ----------------------------",
    ]
    if len(valid_results) >= 2:
        svs = list(valid_results.values())
        q90_lows = [s["q90_low"] for s in svs]
        diff = q90_lows[0][0] - q90_lows[1][0]
        se_diff = np.sqrt(((q90_lows[0][2]-q90_lows[0][1])/(2*1.96))**2 +
                          ((q90_lows[1][2]-q90_lows[1][1])/(2*1.96))**2)
        z_cons = diff / se_diff if se_diff > 0 else np.nan
        lines += [
            f"  Q90_cold: TESS={q90_lows[0][0]:.3f}  Kepler={q90_lows[1][0]:.3f}",
            f"  Difference Z-score = {z_cons:.2f}",
            f"  {'INCONSISTENT (|Z|>2): combination may be unreliable.' if abs(z_cons) > 2 else 'CONSISTENT (|Z|<=2): combination valid.'}",
        ]

    lines += [
        "",
        "-- Interpretation -----------------------------------------------",
        "  Fisher's p < 0.05: BOTH surveys find evidence of a radius upper",
        "    bound difference between cold and hot rocky planets.",
        "  Fisher's p >= 0.10: No consistent evidence across surveys.",
        "  Consistency Z < 2: surveys are compatible; combined result is robust.",
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
    parser.add_argument("--n-catalogs", type=int, default=300,
                        help="P-Pop catalogs per survey (default: 300).")
    parser.add_argument("--star-types", type=str, default="F,G,K,M")
    parser.add_argument("--rocky-mode", type=str, default="mass-confirmed",
                        choices=["all", "mass-confirmed", "radius-gap"],
                        help="Default: mass-confirmed (requires measured mass + below rocky M-R curve).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--surveys", type=str, default="TESS,Kepler",
                        help="Comma-separated list of surveys to include.")
    args = parser.parse_args()

    rng        = np.random.default_rng(args.seed)
    star_types = [s.strip().upper() for s in args.star_types.split(",")]
    surveys    = [s.strip() for s in args.surveys.split(",")]
    title_sfx  = (f"Stars: {'+'.join(star_types)}  |  rocky: {args.rocky_mode}"
                  f"  |  surveys: {'+'.join(surveys)}")

    m_ref = r_ref = shift = None
    shift = 0.0
    if args.rocky_mode == "mass-confirmed":
        m_ref, r_ref, shift = load_rocky_curve()

    # -- Run each survey -------------------------------------------------------
    results = {}
    for sname in surveys:
        if sname not in SURVEYS:
            print(f"WARNING: unknown survey '{sname}', skipping.")
            continue
        results[sname] = run_survey(
            sname, star_types, args.rocky_mode,
            args.n_catalogs, rng, m_ref, r_ref, shift,
        )

    valid = {k: v for k, v in results.items() if not v.get("skipped")}
    if len(valid) == 0:
        print("ERROR: no surveys returned valid results.")
        return

    # -- Fisher's combination --------------------------------------------------
    p_values = [res["p_q90"] for res in valid.values()]
    finite_p = [p for p in p_values if np.isfinite(p)]
    if len(finite_p) >= 2:
        p_combined = fishers_combined_p(finite_p)
    elif len(finite_p) == 1:
        p_combined = finite_p[0]
        print("  NOTE: Only one survey has a valid p-value; using it directly.")
    else:
        p_combined = np.nan
        print("  NOTE: No surveys have valid p-values; Fisher combination not possible.")
    print(f"\n{'='*60}")
    p_str = f"{p_combined:.4f}" if np.isfinite(p_combined) else "N/A"
    print(f"  Fisher's combined p-value: {p_str}")
    print(f"{'='*60}")

    # -- Inverse-variance Q90 combination -------------------------------------
    combined_q90_low  = inverse_variance_combine([res["q90_low"]  for res in valid.values()])
    combined_q90_high = inverse_variance_combine([res["q90_high"] for res in valid.values()])
    print(f"  Combined Q90 cold: {combined_q90_low[0]:.3f} +/- {combined_q90_low[1]:.3f} Rearth")
    print(f"  Combined Q90 hot:  {combined_q90_high[0]:.3f} +/- {combined_q90_high[1]:.3f} Rearth")

    # -- Figures ---------------------------------------------------------------
    print("\n-- Figures ----------------------------------------------------------")
    plot_completeness_comparison(results)
    plot_survival_comparison(results, title_sfx)
    plot_q90_comparison(results, combined_q90_low, combined_q90_high, title_sfx)
    plot_pvalue_comparison(results, p_combined, title_sfx)

    # -- Summary ---------------------------------------------------------------
    save_summary(results, p_combined, combined_q90_low, combined_q90_high,
                 title_sfx, args.rocky_mode)


if __name__ == "__main__":
    main()
