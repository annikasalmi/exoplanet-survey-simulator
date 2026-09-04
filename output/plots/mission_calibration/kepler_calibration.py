"""
kepler_calibration.py

Kepler toy-vs-official 3-in-1 calibration figure with improved CDPP.

The core fix over script 28:
  - Fetches rrmscdpp01p5 … rrmscdpp15p0 columns from the Kepler stellar
    table (q1_q17_dr25_stellar) by JOINing with the KOI cumulative table.
  - KeplerData then uses the real per-target CDPP instead of the
    magnitude-based fallback (100 ppm at Kp=12), which was the primary
    cause of the flat ~0.6x under-estimate in toy MES.

Outputs (single combined figure):
    my_outputs/kepler_calibration/
        kepler_3in1_calibration.png   — scatter | recovery | ratio
        koi_stellar_cached.csv        — downloaded / cached KOI+stellar data
        summary.txt                   — key calibration numbers

Run from repo root:
    python output/plots/mission_calibration/kepler_calibration.py
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

from tools.paths import LIFESIM_OUTER_DIR, KOI_CUMULATIVE_CSV
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from telescopes.kepler.detection_model import KeplerData
except Exception as exc:
    raise ImportError(f"Cannot import KeplerData. Run from repo root.\n{exc}") from exc

OUT_DIR = ROOT / "output" / "plots" / "mission_calibration" / "kepler_calibration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE = OUT_DIR / "koi_stellar_cached.csv"

PAPER_FIG_DIR = ROOT / "paper" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

MES_THRESHOLD = 7.1
MIN_CELL_N = 20
DOWNLOAD_NASA_DATA = False  # Set to True to download fresh data, False to use local CSV

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 260,
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

CDPP_COLS = [
    "rrmscdpp01p5", "rrmscdpp02p0", "rrmscdpp02p5", "rrmscdpp03p0",
    "rrmscdpp03p5", "rrmscdpp04p5", "rrmscdpp05p0", "rrmscdpp06p0",
    "rrmscdpp07p5", "rrmscdpp09p0", "rrmscdpp10p5", "rrmscdpp12p0",
    "rrmscdpp12p5", "rrmscdpp15p0",
]

MES_BINS = [7.1, 10, 20, 50, 100, 300, np.inf]
MES_BIN_LABELS = ["7.1-10", "10-20", "20-50", "50-100", "100-300", ">300"]

DISP_COLORS = {
    "CONFIRMED":   "#e08000",
    "CANDIDATE":   "#2060c0",
    "FALSE POSITIVE": "#909090",
}


# ── Download ──────────────────────────────────────────────────────────────────


def nasa_tap_url(query: str) -> str:
    return (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        f"?query={quote(query)}&format=csv"
    )


def load_or_download(redownload: bool = False) -> pd.DataFrame:
    """KOI cumulative + stellar CDPP.

    Reads the copy in data/ by default. Only downloads when DOWNLOAD_NASA_DATA is
    True (or redownload is passed), and writes what it fetches back to data/ so
    the next run is offline.
    """
    local = Path(KOI_CUMULATIVE_CSV)
    if local.exists() and not (DOWNLOAD_NASA_DATA or redownload):
        print(f"Loading local KOI table: {local}")
        df = pd.read_csv(local, comment="#", low_memory=False)
        missing = [c for c in ("koi_max_mult_ev", "koi_num_transits")
                   if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"{local.name} lacks {missing}. Set DOWNLOAD_NASA_DATA = True to "
                "refresh it from the NASA Exoplanet Archive.")
        return df

    if not (DOWNLOAD_NASA_DATA or redownload):
        raise RuntimeError(
            f"{local} not found. Set DOWNLOAD_NASA_DATA = True to fetch the KOI "
            "cumulative table from the NASA Exoplanet Archive.")

    cdpp_select = ", ".join(f"s.{c}" for c in CDPP_COLS)
    query = f"""
    SELECT
        c.kepid, c.kepoi_name, c.koi_disposition, c.koi_pdisposition,
        c.koi_period, c.koi_prad, c.koi_sma, c.koi_insol,
        c.koi_depth, c.koi_duration, c.koi_incl,
        c.koi_steff, c.koi_srad, c.koi_smass, c.koi_kepmag,
        c.koi_max_sngle_ev, c.koi_max_mult_ev, c.koi_model_snr, c.koi_num_transits,
        c.koi_tce_delivname,
        s.dataspan, s.dutycycle,
        {cdpp_select}
    FROM cumulative c
    JOIN keplerstellar s ON c.kepid = s.kepid
    WHERE c.koi_period IS NOT NULL
      AND c.koi_prad   IS NOT NULL
      AND c.koi_srad   IS NOT NULL
      AND c.koi_depth  IS NOT NULL
      AND c.koi_duration IS NOT NULL
    """
    # NASA retired the one-row-per-target q1_q17_dr25_stellar table; keplerstellar
    # holds the same rrmscdpp* columns but with one row per (target, data release),
    # so the JOIN multiplies each KOI.  Collapse back to one row per KOI by keeping
    # the longest-baseline (largest dataspan) row that carries a valid CDPP value.
    print("Downloading KOI + stellar CDPP from NASA ExoplanetArchive (JOIN query)...")
    df = pd.read_csv(nasa_tap_url(query), low_memory=False)
    n_raw = len(df)
    df["dataspan"] = pd.to_numeric(df["dataspan"], errors="coerce")
    df["rrmscdpp07p5"] = pd.to_numeric(df["rrmscdpp07p5"], errors="coerce")
    key = "kepoi_name" if "kepoi_name" in df.columns else "kepid"
    df = (df[df["rrmscdpp07p5"].notna()]
          .sort_values("dataspan", ascending=False)
          .drop_duplicates(subset=key, keep="first")
          .reset_index(drop=True))
    print(f"  Deduped keplerstellar rows: {n_raw:,} -> {len(df):,} (one per {key})")
    df.to_csv(KOI_CUMULATIVE_CSV, index=False)
    print(f'Saved to: {KOI_CUMULATIVE_CSV}')
    print(f"  Saved: {CACHE}  ({len(df):,} rows)")
    return df


# ── Prepare and run detector ───────────────────────────────────────────────────


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to DR25 TCEs, set tran_flag=1 (all KOIs are transit-like events)."""
    numeric_cols = [
        "koi_period", "koi_prad", "koi_sma", "koi_insol",
        "koi_depth", "koi_duration", "koi_incl", "koi_steff",
        "koi_srad", "koi_smass", "koi_kepmag",
        "koi_max_sngle_ev", "koi_max_mult_ev", "koi_model_snr", "koi_num_transits",
        "dataspan", "dutycycle",
    ] + CDPP_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Restrict to DR25 KOIs when available.
    if "koi_tce_delivname" in df.columns:
        is_dr25 = df["koi_tce_delivname"].astype(str).str.contains("Q1_Q17_DR25", case=False, na=False)
        if is_dr25.any():
            before = len(df); df = df[is_dr25].copy()
            print(f"DR25 filter: {before:,} -> {len(df):,}")

    required = ["koi_period", "koi_prad", "koi_srad", "koi_depth", "koi_duration"]
    df = df.dropna(subset=required).copy()
    df = df[(df["koi_period"] > 0) & (df["koi_prad"] > 0) & (df["koi_srad"] > 0)].copy()
    df["tran_flag"] = 1   # KOIs are already transit-like events
    print(f"Valid KOI rows: {len(df):,}")
    return df


def run_detector(df: pd.DataFrame) -> pd.DataFrame:
    """Run KeplerData; it will use real rrmscdpp* columns when present."""
    kd = KeplerData(
        df,
        source="koi",
        validate_for_detection=True,
        min_transits=3,
        mes_threshold=MES_THRESHOLD,
        kepler_mag_limit=16.0,
        fallback_cdpp_ppm=50.0,    # better calibrated fallback (was 100)
        use_kepmag_cdpp_fallback=True,
        use_observed_transit_flag_for_nasa=True,
        use_observed_transit_depth_for_nasa=True,
        assume_bright_if_kepmag_missing_for_nasa=False,
        estimate_missing_semimajor_axis=True,
    )
    out = kd.determine_detectable()
    if "kepoi_name" not in out.columns and "planet_name" in out.columns:
        out["kepoi_name"] = out["planet_name"]
    out["toy_over_koi_mes"] = (
        pd.to_numeric(out["kepler_mes"], errors="coerce") /
        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce").replace(0, np.nan)
    )
    # Raw (pre-calibration) toy MES = the optimistic boxcar value before the
    # MES_OFFICIAL_CALIBRATION factor, so the figure can show before vs after.
    cal = float(getattr(kd, "mes_calibration", 1.0)) or 1.0
    out.attrs["mes_calibration"] = cal
    out["kepler_mes_raw"] = pd.to_numeric(out["kepler_mes"], errors="coerce") / cal
    out["toy_over_koi_mes_raw"] = pd.to_numeric(out["toy_over_koi_mes"], errors="coerce") / cal
    out["koi_mes_bin"] = pd.cut(
        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce"),
        bins=MES_BINS, labels=MES_BIN_LABELS, include_lowest=True,
    )
    # Tag which rows used real CDPP vs fallback.
    out["used_real_cdpp"] = out.get("kepler_cdpp_source", pd.Series("unknown", index=out.index)).str.contains("rrmscdpp", na=False)
    return out


# ── 3-in-1 figure ─────────────────────────────────────────────────────────────


def make_3in1(out: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(14, 8))
    # Layout: scatter spans full height on left; two stacked panels on right.
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.40, wspace=0.35)
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_recov   = fig.add_subplot(gs[0, 1])
    ax_ratio   = fig.add_subplot(gs[1, 1])

    # ── A: scatter toy MES vs official MES ──────────────────────────────────
    d_sc = out[
        pd.to_numeric(out["koi_max_mult_ev"], errors="coerce").gt(0) &
        pd.to_numeric(out["kepler_mes"], errors="coerce").gt(0)
    ].copy()

    for disp, g in d_sc.groupby("koi_disposition", dropna=False):
        c = DISP_COLORS.get(str(disp), "#707070")
        ax_scatter.scatter(g["koi_max_mult_ev"], g["kepler_mes"],
                           s=5, alpha=0.22, color=c, label=str(disp))

    # Binned median with IQR.
    med = (
        d_sc.dropna(subset=["koi_mes_bin"])
        .groupby("koi_mes_bin", observed=True)
        .agg(
            n=("kepoi_name", "size"),
            med_koi=("koi_max_mult_ev", "median"),
            med_toy=("kepler_mes", "median"),
            q25=("kepler_mes", lambda x: np.nanpercentile(x, 25)),
            q75=("kepler_mes", lambda x: np.nanpercentile(x, 75)),
        ).reset_index()
    )
    med = med[np.isfinite(med["med_koi"]) & np.isfinite(med["med_toy"])]
    if len(med):
        ax_scatter.errorbar(med["med_koi"], med["med_toy"],
                            yerr=np.vstack([med["med_toy"] - med["q25"], med["q75"] - med["med_toy"]]),
                            fmt="o-", color="black", linewidth=1.5, markersize=3.5, capsize=2,
                            label="Binned median")

    lo = max(0.5, min(d_sc["koi_max_mult_ev"].min(), d_sc["kepler_mes"].min()))
    hi = max(d_sc["koi_max_mult_ev"].max(), d_sc["kepler_mes"].max())
    grid = np.logspace(np.log10(lo), np.log10(hi), 200)
    ax_scatter.plot(grid, grid, "--", lw=1.0, color="black", alpha=0.6, label="1:1")
    ax_scatter.axvline(MES_THRESHOLD, ls=":", lw=1.1, color="black")
    ax_scatter.axhline(MES_THRESHOLD, ls=":", lw=1.1, color="black")
    ax_scatter.set_xscale("log"); ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel("Official KOI MES")
    ax_scatter.set_ylabel("Model MES")
    ax_scatter.set_title("A. Model MES vs official KOI MES")
    ax_scatter.text(0.02, 0.02, "Dotted lines = 7.1 threshold\nBlack = binned median ± IQR",
                   transform=ax_scatter.transAxes, fontsize=10, va="bottom")
    ax_scatter.legend(loc="upper left", fontsize=13)

    # ── B: recovery fraction ─────────────────────────────────────────────────
    s = (
        out.dropna(subset=["koi_mes_bin"])
        .groupby("koi_mes_bin", observed=True)
        .agg(n=("kepoi_name", "size"), pass_frac=("detected", "mean"))
        .reset_index()
    )
    s["x"] = np.arange(len(s))
    ax_recov.plot(s["x"], s["pass_frac"], marker="o", linewidth=1.8, color="#2060c0")
    for _, row in s.iterrows():
        ax_recov.text(row["x"], row["pass_frac"] + 0.04,
                      f"{row['pass_frac']:.0%}\nN={int(row['n'])}", ha="center", va="bottom", fontsize=9)
    ax_recov.set_ylim(0, 1.18)
    ax_recov.set_xticks(s["x"]); ax_recov.set_xticklabels(s["koi_mes_bin"].astype(str), rotation=25, ha="right")
    ax_recov.set_ylabel("Model pass fraction"); ax_recov.set_xlabel("Official KOI MES bin")
    ax_recov.set_title("B. Recovery vs signal strength")
    ax_recov.grid(axis="y", alpha=0.25)

    # ── C: MES ratio ─────────────────────────────────────────────────────────
    d_rat = out[np.isfinite(out.get("toy_over_koi_mes", pd.Series(np.nan, index=out.index))) &
                (out.get("toy_over_koi_mes", pd.Series(0, index=out.index)) > 0)].copy()
    s2 = (
        d_rat.dropna(subset=["koi_mes_bin"])
        .groupby("koi_mes_bin", observed=True)
        .agg(
            n=("kepoi_name", "size"),
            med=("toy_over_koi_mes", "median"),
            q25=("toy_over_koi_mes", lambda x: np.nanpercentile(x, 25)),
            q75=("toy_over_koi_mes", lambda x: np.nanpercentile(x, 75)),
        ).reset_index()
    )
    s2["x"] = np.arange(len(s2))
    cal = float(out.attrs.get("mes_calibration", 1.0)) or 1.0
    # Raw (pre-calibration) ratio per bin — the optimistic boxcar MES.
    if "toy_over_koi_mes_raw" in d_rat.columns:
        s2raw = (d_rat.dropna(subset=["koi_mes_bin"])
                 .groupby("koi_mes_bin", observed=True)["toy_over_koi_mes_raw"]
                 .median().reset_index())
        ax_ratio.plot(np.arange(len(s2raw)), s2raw["toy_over_koi_mes_raw"], "s--",
                      color="#c04040", lw=1.3, markersize=4, alpha=0.8,
                      label=f"Raw boxcar (uncalibrated, x{1/cal:.2f})")
    ax_ratio.errorbar(s2["x"], s2["med"],
                      yerr=np.vstack([s2["med"] - s2["q25"], s2["q75"] - s2["med"]]),
                      fmt="o-", capsize=3, linewidth=1.5, color="#2060c0",
                      label=f"Calibrated (x{cal:.2f})")
    ax_ratio.axhline(1.0, color="black", ls="--", lw=1.0, label="Perfect agreement")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xticks(s2["x"]); ax_ratio.set_xticklabels(s2["koi_mes_bin"].astype(str), rotation=25, ha="right")
    ax_ratio.set_ylabel("Model / official MES"); ax_ratio.set_xlabel("Official KOI MES bin")
    ax_ratio.set_title("C. MES ratio: raw vs calibrated")
    ax_ratio.legend(fontsize=13); ax_ratio.grid(axis="y", alpha=0.25)

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = OUT_DIR / "kepler_3in1_calibration.png"
    plt.savefig(out_path, bbox_inches="tight")
    plt.savefig(PAPER_FIG_DIR / "kepler_3in1_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    print(f"  Saved paper copy: {PAPER_FIG_DIR / 'kepler_3in1_calibration.png'}")


# ── Summary text ──────────────────────────────────────────────────────────────


def write_summary(out: pd.DataFrame) -> None:
    n = len(out)
    n_real_cdpp = out.get("used_real_cdpp", pd.Series(False, index=out.index)).sum()
    ratio = pd.to_numeric(out["toy_over_koi_mes"], errors="coerce")
    lines = [
        "=" * 64,
        "KEPLER 3-IN-1 CALIBRATION SUMMARY",
        "=" * 64,
        f"Total KOI rows:          {n:>8,}",
        f"Rows with real rrmscdpp: {n_real_cdpp:>8,}  ({n_real_cdpp/n:.1%})",
        f"Rows using fallback CDPP:{n-n_real_cdpp:>8,}  ({(n-n_real_cdpp)/n:.1%})",
        "",
        f"Toy/official MES ratio:  median={ratio.median():.3f}  mean={ratio.mean():.3f}",
        "  (1.0 = perfect; < 1.0 = toy is conservative)",
        "",
        "Ratio by official MES bin:",
    ]
    for label in MES_BIN_LABELS:
        mask = out["koi_mes_bin"].astype(str) == label
        r = ratio[mask]
        lines.append(f"  {label:<12}  N={mask.sum():>4}  median={r.median():.3f}  IQR=[{r.quantile(0.25):.3f},{r.quantile(0.75):.3f}]")
    lines += [
        "",
        "Key formula change:",
        "  OLD: CDPP = fallback(100 ppm @ Kp=12, scaled by mag)",
        "  NEW: CDPP = real rrmscdpp from q1_q17_dr25_stellar (fallback=50 ppm @ Kp=12)",
        "=" * 64,
    ]
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
    koi = prepare(raw)
    out = run_detector(koi)
    out.to_csv(OUT_DIR / "results.csv", index=False)
    print(f"  Saved results.csv ({len(out):,} rows)")
    print("\nBuilding 3-in-1 figure...")
    make_3in1(out)
    write_summary(out)
    print("\nDone.")


if __name__ == "__main__":
    main()
