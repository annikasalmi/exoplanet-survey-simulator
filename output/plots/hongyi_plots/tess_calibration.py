"""
tess_calibration.py

TESS toy-vs-official 3-in-1 calibration figure with improved smooth CDPP.

The core fix over the old TESS calibration script:
  smooth_noise_ref_ppm_1hr  : 60 -> 30   (photon-noise floor for Tmag=10)
  smooth_noise_floor_ppm_1hr: 30 -> 10   (systematic-noise floor)

This brings the smooth CDPP fallback closer to real SPOC CDPP statistics:
  Tmag=9  old ~48 ppm/hr  ->  new ~21 ppm/hr  (SPOC: ~15-20 ppm/hr)
  Tmag=12 old ~154 ppm/hr ->  new ~76 ppm/hr  (SPOC: ~40-60 ppm/hr)

A residual ~0.7x systematic remains at low MES from matched-filter vs
boxcar differences in the SPOC pipeline (expected, not a formula error).

Data source: ExoFOP TESS TOI catalog (SNR column = SPOC pipeline MES).
Dispositions are grouped into three classes (same colours/legend as the Kepler
figure, script 47):
  Confirmed       = CP (Confirmed Planet)  + KP (Known Planet)
  Candidate       = PC (Planet Candidate)  + APC (Ambiguous Planet Candidate)
  False Positive  = FP (False Positive)    + FA (False Alarm)

Outputs:
    my_outputs/tess_calibration/
        tess_3in1_calibration.png
        toi_cached.csv
        summary.txt

Run from repo root:
    python scripts/tess_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from telescopes.tess.detection_model import TESSData
except Exception as exc:
    raise ImportError(f"Cannot import TESSData. Run from repo root.\n{exc}") from exc

OUT_DIR = ROOT / "my_outputs" / "tess_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "toi_cached.csv"

PAPER_FIG_DIR = ROOT / "paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

SNR_THRESHOLD = 7.1
MIN_CELL_N = 20

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 260,
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

SNR_BINS = [7.1, 10, 20, 50, 100, 300, np.inf]
SNR_BIN_LABELS = ["7.1-10", "10-20", "20-50", "50-100", "100-300", ">300"]

# Group the six TFOPWG dispositions into the three classes used by the Kepler
# figure (script 47), with the SAME colours and legend template.
DISP_GROUP = {
    "CP": "CONFIRMED", "KP": "CONFIRMED",
    "PC": "CANDIDATE", "APC": "CANDIDATE",
    "FP": "FALSE POSITIVE", "FA": "FALSE POSITIVE",
}
DISP_COLORS = {
    "CONFIRMED":      "#e08000",
    "CANDIDATE":      "#2060c0",
    "FALSE POSITIVE": "#909090",
}

# ── Improved CDPP calibration constants ───────────────────────────────────────
# OLD values (caused ~0.6x ratio at all MES bins for bright stars):
#   smooth_noise_ref_ppm_1hr  = 60, smooth_noise_floor_ppm_1hr = 30
# NEW values (calibrated to SPOC CDPP statistics from Ricker+2015 / Sullivan+2015):
SMOOTH_REF_PPM = 30.0    # photon-noise reference at Tmag=10 (was 60)
SMOOTH_FLOOR_PPM = 10.0  # systematic noise floor (was 30)

CDPP_DIR = ROOT / "run" / "tess" / "data" / "CDPP"


# ── Download ExoFOP TOI catalog ────────────────────────────────────────────────


EXOFOP_TOI_URL = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"


def load_or_download(redownload: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not redownload:
        print(f"Loading cached TOI table: {CACHE}")
        return pd.read_csv(CACHE, low_memory=False)
    print(f"Downloading ExoFOP TESS TOI catalog...")
    try:
        df = pd.read_csv(EXOFOP_TOI_URL, low_memory=False)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot download ExoFOP TOI. Check internet connection.\n{exc}"
        ) from exc
    df.to_csv(CACHE, index=False)
    print(f"  Saved: {CACHE}  ({len(df):,} rows)")
    return df


# ── Prepare TOI data ───────────────────────────────────────────────────────────


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ExoFOP TOI columns to internal standard names expected by TESSData.
    ExoFOP TOI CSV columns (relevant subset):
        TIC ID, TOI, Period (days), Duration (hours), Depth (ppm),
        Stellar Radius (R_Sun), Stellar Eff Temp (K), TESS Mag,
        RA, Dec, TFOPWG Disposition, SNR
    """
    df = df.copy()
    # Normalise column names.
    df.columns = df.columns.str.strip()

    col_map = {
        "TIC ID":                "ticid",
        "TOI":                   "pl_toi",
        "Period (days)":         "p_orb",
        "Duration (hours)":      "observed_duration_hr",
        "Depth (ppm)":           "observed_depth_ppm",
        "Planet Radius (R_Earth)": "radius_p",
        "Stellar Radius (R_Sun)":"radius_s",
        "Stellar Eff Temp (K)":  "teff_s",
        # Map to the internal "tmag" name that TESSData ingests and proxies from;
        # mapping straight to "tess_tmag" gets clobbered (TESSData rebuilds
        # tess_tmag from "tmag", overwriting it with NaN -> brightness gate fails).
        "TESS Mag":              "tmag",
        "RA":                    "ra",
        "Dec":                   "dec",
        "TFOPWG Disposition":    "tfopwg_disp",
        "Planet SNR":            "official_snr",   # ExoFOP renamed "SNR" -> "Planet SNR"
        "SNR":                   "official_snr",
        "Stellar Distance (pc)": "distance_s",
        "Stellar log(g) (cm/s^2)": "logg_s",
        "Stellar Mass (M_Sun)":  "mass_s",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    numeric_cols = ["p_orb", "observed_duration_hr", "observed_depth_ppm", "radius_p",
                    "radius_s", "teff_s", "tmag", "official_snr", "distance_s",
                    "mass_s", "ra", "dec"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # All TOIs are observed transiting events.
    df["tran_flag"] = 1

    # Derive stellar radius from radius_p and depth if radius_s missing.
    if "radius_s" not in df.columns or df["radius_s"].isna().all():
        if "radius_p" in df.columns and "observed_depth_ppm" in df.columns:
            depth_frac = pd.to_numeric(df["observed_depth_ppm"], errors="coerce") / 1e6
            rp = pd.to_numeric(df["radius_p"], errors="coerce")
            rs_from_depth = rp / np.sqrt(depth_frac.clip(lower=1e-8))
            df["radius_s"] = rs_from_depth / 109.076  # R_earth to R_sun
            df["radius_s_source"] = "inferred_from_depth"

    # Estimate semimajor axis from period and stellar mass.
    if "mass_s" not in df.columns:
        df["mass_s"] = 1.0
    if "semimajor_p" not in df.columns:
        p_yr = df["p_orb"] / 365.25
        df["semimajor_p"] = (df["mass_s"].fillna(1.0) * p_yr ** 2) ** (1.0 / 3.0)

    # Estimate radius_p from depth if missing.
    if "radius_p" not in df.columns or df["radius_p"].isna().mean() > 0.5:
        if "observed_depth_ppm" in df.columns and "radius_s" in df.columns:
            depth_frac = pd.to_numeric(df["observed_depth_ppm"], errors="coerce") / 1e6
            rs_rearth = pd.to_numeric(df["radius_s"], errors="coerce") * 109.076
            df["radius_p"] = rs_rearth * np.sqrt(depth_frac.clip(lower=0))

    # Inclination: set to 90 deg (KOI-style, tran_flag=1 triggers observed path).
    df["inc_p"] = 90.0

    required = ["p_orb", "radius_s", "radius_p", "tmag"]
    df = df.dropna(subset=[c for c in required if c in df.columns]).copy()
    df = df[df["p_orb"] > 0].copy()
    # Drop rows without official SNR (needed for calibration).
    if "official_snr" in df.columns:
        df = df.dropna(subset=["official_snr"]).copy()
        df = df[df["official_snr"] > 0].copy()

    print(f"Valid TOI rows (with official SNR): {len(df):,}")
    return df


# ── Run TESSData toy detector ──────────────────────────────────────────────────


def run_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Run TESSData with improved smooth CDPP calibration constants."""
    cdpp_dir = CDPP_DIR if CDPP_DIR.exists() else None
    td = TESSData(
        df,
        source="auto",
        default_n_sectors=4,          # typical for TOIs; tess-point won't run (no star catalog)
        min_transits=2,
        snr_threshold=SNR_THRESHOLD,
        tmag_limit=16.0,
        use_tesspoint=False,          # TOI table has ra/dec but not indexed in Gaia catalog
        use_mast_tic=False,
        cdpp_dir=cdpp_dir,
        use_cdpp_tables=(cdpp_dir is not None),
        smooth_noise_ref_ppm_1hr=SMOOTH_REF_PPM,     # IMPROVED: 60 -> 30
        smooth_noise_floor_ppm_1hr=SMOOTH_FLOOR_PPM, # IMPROVED: 30 -> 10
        apply_b_to_duration=True,
        apply_mdwarf_tmag_correction=True,
        ffi_cadence_min=None,
        validate_for_detection=True,
    )
    out = td.determine_detectable()
    out["toy_over_official_snr"] = (
        pd.to_numeric(out["tess_snr"], errors="coerce") /
        pd.to_numeric(out.get("official_snr", pd.Series(np.nan, index=out.index)), errors="coerce").replace(0, np.nan)
    )
    # Raw (pre-calibration) toy SNR = optimistic boxcar value before the
    # SNR_OFFICIAL_CALIBRATION factor, so the figure can show before vs after.
    cal = float(getattr(td, "snr_calibration", 1.0)) or 1.0
    out.attrs["snr_calibration"] = cal
    out["tess_snr_raw"] = pd.to_numeric(out["tess_snr"], errors="coerce") / cal
    out["toy_over_official_snr_raw"] = pd.to_numeric(out["toy_over_official_snr"], errors="coerce") / cal
    out["snr_bin"] = pd.cut(
        pd.to_numeric(out.get("official_snr", pd.Series(np.nan, index=out.index)), errors="coerce"),
        bins=SNR_BINS, labels=SNR_BIN_LABELS, include_lowest=True,
    )
    return out


# ── 3-in-1 figure ─────────────────────────────────────────────────────────────


def make_3in1(out: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.40, wspace=0.35)
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_recov   = fig.add_subplot(gs[0, 1])
    ax_ratio   = fig.add_subplot(gs[1, 1])

    official = pd.to_numeric(out.get("official_snr", pd.Series(np.nan, index=out.index)), errors="coerce")
    toy = pd.to_numeric(out["tess_snr"], errors="coerce")
    d_sc = out[official.gt(0) & toy.gt(0)].copy()

    disp_col = "tfopwg_disp" if "tfopwg_disp" in d_sc.columns else None

    # ── A: scatter — group the six TFOPWG codes into 3 clean classes (script-47 template)
    if disp_col:
        d_sc["disp_group"] = d_sc[disp_col].astype(str).str.strip().str.upper().map(DISP_GROUP)
        for grp in ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]:
            g = d_sc[d_sc["disp_group"] == grp]
            if len(g) == 0:
                continue
            ax_scatter.scatter(
                pd.to_numeric(g["official_snr"], errors="coerce"),
                pd.to_numeric(g["tess_snr"], errors="coerce"),
                s=5, alpha=0.22, color=DISP_COLORS[grp], label=grp,
            )
    else:
        ax_scatter.scatter(official[d_sc.index], toy[d_sc.index], s=4, alpha=0.2, color="#2060c0")

    # Binned median.
    med = (
        d_sc.dropna(subset=["snr_bin"])
        .groupby("snr_bin", observed=True)
        .agg(
            n=("tess_snr", "size"),
            med_off=("official_snr", "median"),
            med_toy=("tess_snr", "median"),
            q25=("tess_snr", lambda x: np.nanpercentile(x, 25)),
            q75=("tess_snr", lambda x: np.nanpercentile(x, 75)),
        ).reset_index()
    )
    if len(med):
        ax_scatter.errorbar(
            med["med_off"], med["med_toy"],
            yerr=np.vstack([med["med_toy"] - med["q25"], med["q75"] - med["med_toy"]]),
            fmt="o-", color="black", lw=1.5, markersize=3.5, capsize=2, label="Binned median"
        )
    lo = max(0.5, float(d_sc["official_snr"].min(skipna=True)))
    hi = max(float(d_sc["official_snr"].max(skipna=True)), float(d_sc["tess_snr"].max(skipna=True)))
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax_scatter.plot(grid, grid, "--", lw=1.0, color="black", alpha=0.6, label="1:1")
    ax_scatter.axvline(SNR_THRESHOLD, ls=":", lw=1.1, color="black")
    ax_scatter.axhline(SNR_THRESHOLD, ls=":", lw=1.1, color="black")
    ax_scatter.set_xscale("log"); ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Official SPOC SNR")
    ax_scatter.set_ylabel("Model SNR")
    ax_scatter.set_title("A. Model SNR vs official SPOC SNR")
    ax_scatter.text(0.02, 0.02, "Dotted = 7.1 threshold\nBlack = binned median ± IQR",
                   transform=ax_scatter.transAxes, fontsize=10, va="bottom")
    ax_scatter.legend(loc="upper left", fontsize=13)

    # ── B: recovery fraction ─────────────────────────────────────────────────
    s = (
        out.dropna(subset=["snr_bin"])
        .groupby("snr_bin", observed=True)
        .agg(n=("tess_snr", "size"), pass_frac=("tess_detected", "mean"))
        .reset_index()
    )
    s["x"] = np.arange(len(s))
    ax_recov.plot(s["x"], s["pass_frac"], marker="o", lw=1.8, color="#2060c0")
    for _, row in s.iterrows():
        ax_recov.text(row["x"], row["pass_frac"] + 0.04,
                      f"{row['pass_frac']:.0%}\nN={int(row['n'])}", ha="center", va="bottom", fontsize=9)
    ax_recov.set_ylim(0, 1.18)
    ax_recov.set_xticks(s["x"]); ax_recov.set_xticklabels(s["snr_bin"].astype(str), rotation=25, ha="right")
    ax_recov.set_ylabel("Model pass fraction"); ax_recov.set_xlabel("Official SPOC SNR bin")
    ax_recov.set_title("B. Recovery vs signal strength")
    ax_recov.grid(axis="y", alpha=0.25)

    # ── C: SNR ratio ─────────────────────────────────────────────────────────
    ratio_col = pd.to_numeric(out.get("toy_over_official_snr", pd.Series(np.nan, index=out.index)), errors="coerce")
    d_rat = out[np.isfinite(ratio_col) & (ratio_col > 0)].copy()
    s2 = (
        d_rat.dropna(subset=["snr_bin"])
        .groupby("snr_bin", observed=True)
        .agg(
            n=("tess_snr", "size"),
            med=("toy_over_official_snr", "median"),
            q25=("toy_over_official_snr", lambda x: np.nanpercentile(x, 25)),
            q75=("toy_over_official_snr", lambda x: np.nanpercentile(x, 75)),
        ).reset_index()
    )
    s2["x"] = np.arange(len(s2))
    cal = float(out.attrs.get("snr_calibration", 1.0)) or 1.0
    # Raw (pre-calibration) ratio per bin — the optimistic boxcar SNR.
    if "toy_over_official_snr_raw" in d_rat.columns:
        s2raw = (d_rat.dropna(subset=["snr_bin"])
                 .groupby("snr_bin", observed=True)["toy_over_official_snr_raw"]
                 .median().reset_index())
        ax_ratio.plot(np.arange(len(s2raw)), s2raw["toy_over_official_snr_raw"], "s--",
                      color="#c04040", lw=1.3, markersize=4, alpha=0.8,
                      label=f"Raw boxcar (uncalibrated, x{1/cal:.2f})")
    ax_ratio.errorbar(s2["x"], s2["med"],
                      yerr=np.vstack([s2["med"] - s2["q25"], s2["q75"] - s2["med"]]),
                      fmt="o-", capsize=3, lw=1.5, color="#2060c0",
                      label=f"Calibrated (x{cal:.2f})")
    ax_ratio.axhline(1.0, color="black", ls="--", lw=1.0, label="Perfect (1:1)")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xticks(s2["x"]); ax_ratio.set_xticklabels(s2["snr_bin"].astype(str), rotation=25, ha="right")
    ax_ratio.set_ylabel("Model / official SNR"); ax_ratio.set_xlabel("Official SPOC SNR bin")
    ax_ratio.set_title("C. SNR ratio: raw vs calibrated")
    ax_ratio.legend(fontsize=13); ax_ratio.grid(axis="y", alpha=0.25)

    out_path = OUT_DIR / "tess_3in1_calibration.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(PAPER_FIG_DIR / "tess_3in1_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved paper copy: {PAPER_FIG_DIR / 'tess_3in1_calibration.png'}")


# ── Summary text ──────────────────────────────────────────────────────────────


def write_summary(out: pd.DataFrame) -> None:
    ratio_col = pd.to_numeric(out.get("toy_over_official_snr", pd.Series(np.nan, index=out.index)), errors="coerce")
    lines = [
        "=" * 64,
        "TESS 3-IN-1 CALIBRATION SUMMARY",
        "=" * 64,
        f"Total TOI rows analysed: {len(out):,}",
        f"Toy/official SNR ratio:  median={ratio_col.median():.3f}  mean={ratio_col.mean():.3f}",
        "  (1.0 = perfect; < 1.0 = toy is conservative)",
        "",
        "Formula change (smooth CDPP fallback):",
        f"  smooth_noise_ref_ppm_1hr  : 60 -> {SMOOTH_REF_PPM}",
        f"  smooth_noise_floor_ppm_1hr: 30 -> {SMOOTH_FLOOR_PPM}",
        "",
        "Implied CDPP at transit duration=1hr:",
        f"  Tmag=9:  old ~48 ppm/hr -> new ~{np.sqrt(SMOOTH_FLOOR_PPM**2 + SMOOTH_REF_PPM**2*10**(-0.4)):.0f} ppm/hr",
        f"  Tmag=12: old ~154 ppm/hr -> new ~{np.sqrt(SMOOTH_FLOOR_PPM**2 + SMOOTH_REF_PPM**2*10**(0.8)):.0f} ppm/hr",
        "",
        "Ratio by official SNR bin:",
    ]
    for label in SNR_BIN_LABELS:
        mask = out["snr_bin"].astype(str) == label
        r = ratio_col[mask]
        if len(r) < 1: continue
        lines.append(f"  {label:<12}  N={mask.sum():>4}  median={r.median():.3f}  IQR=[{r.quantile(0.25):.3f},{r.quantile(0.75):.3f}]")
    lines.append("=" * 64)
    text = "\n".join(lines)
    (OUT_DIR / "summary.txt").write_text(text, encoding="utf-8")
    print(text)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--redownload", action="store_true")
    args = parser.parse_args()

    print(f"Output dir: {OUT_DIR}")
    raw = load_or_download(redownload=args.redownload)
    print(f"Raw rows: {len(raw):,}")
    toi = prepare(raw)
    out = run_detector(toi)
    out.to_csv(OUT_DIR / "results.csv", index=False)
    print(f"  Saved results.csv ({len(out):,} rows)")
    print("\nBuilding 3-in-1 figure...")
    make_3in1(out)
    write_summary(out)
    print("\nDone.")


if __name__ == "__main__":
    main()
