"""
04_tess_detector_check.py

TESS P-Pop detector verification (transiting planets only).
Loads the single-file universe catalog, filters to transiting planets,
then shows where in the detection pipeline losses occur.

Input:  run/tess/data/Gaia_cdpp_v1/tess_catalog_0.csv
Output: output/plots/04_tess_detector_check/

Run from repo root:
    python scripts/04_tess_detector_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

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
CATALOG = ROOT / "run" / "tess" / "data" / "Gaia_cdpp_v1" / "tess_catalog_0.csv"
OUT_DIR = ROOT / "output/plots" / "04_tess_detector_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SNR_THRESHOLD = 7.1
TMAG_LIMIT = 16.0
MIN_CELL_N = 10

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 260,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

KEEP_COLS = [
    "radius_p", "p_orb", "semimajor_p", "flux_p",
    "radius_s", "mass_s", "teff_s", "stype",
    "transiting_geometric", "tess_transiting_geometric",
    "tess_tmag", "tess_n_sectors", "tess_observed",
    "tess_cdpp_ppm", "tess_snr", "tess_snr_threshold", "tess_tmag_limit",
    "tess_n_transits", "tess_enough_transits",
    "tess_star_bright_enough", "tess_depth_pass", "tess_detected",
    "detected", "tess_transit_depth_ppm", "transit_depth_ppm",
    "tess_reason_category", "reason_category", "miss_reason",
]


def load_catalog() -> pd.DataFrame:
    print(f"Loading {CATALOG.name} ...")
    avail = pd.read_csv(CATALOG, nrows=0).columns.tolist()
    cols = [c for c in KEEP_COLS if c in avail]
    df = pd.read_csv(CATALOG, usecols=cols, low_memory=False)

    for col in ["radius_p", "p_orb", "tess_tmag", "tess_snr", "tess_cdpp_ppm",
                "tess_n_sectors", "tess_n_transits", "radius_s", "teff_s", "flux_p",
                "tess_transit_depth_ppm", "transit_depth_ppm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["detected", "tess_detected", "tess_transiting_geometric", "transiting_geometric",
                "tess_star_bright_enough", "tess_enough_transits", "tess_depth_pass", "tess_observed"]:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False}).fillna(False).astype(bool)

    n_total = len(df)
    # Filter to transiting (geometric) planets only — removes "not_transiting" dominance
    df = df[df["tess_transiting_geometric"].astype(bool)].copy()
    print(f"  Total planets: {n_total:,}   Transiting: {len(df):,}   Detected: {df['detected'].sum():,}")
    return df


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_bin"] = pd.cut(df["p_orb"], bins=[0, 3, 10, 30, 100, 300, np.inf],
        labels=["<3", "3-10", "10-30", "30-100", "100-300", ">300"], include_lowest=True)
    df["radius_bin"] = pd.cut(df["radius_p"], bins=[0, 1.0, 1.5, 2.0, 4.0, np.inf],
        labels=["<1", "1-1.5", "1.5-2", "2-4", ">4"], include_lowest=True)
    df["snr_bin"] = pd.cut(df["tess_snr"],
        bins=[0, 3, SNR_THRESHOLD, 10, 20, 50, np.inf],
        labels=["<3", f"3-{SNR_THRESHOLD}", f"{SNR_THRESHOLD}-10", "10-20", "20-50", ">50"],
        include_lowest=True)
    df["sector_bin"] = pd.cut(df["tess_n_sectors"].clip(0, 27),
        bins=[-0.5, 0.5, 1.5, 3.5, 6.5, 12.5, 27.5],
        labels=["0", "1", "2-3", "4-6", "7-12", "13+"], include_lowest=True)
    return df


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_tmag_vs_snr(df: pd.DataFrame) -> None:
    d = df[np.isfinite(df["tess_tmag"]) & (df["tess_tmag"] > 0) &
           np.isfinite(df["tess_snr"]) & (df["tess_snr"] > 0)].copy()
    det = d[d["detected"]]; mis = d[~d["detected"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, sharex=True)
    for ax, sub, title, color in [
        (axes[0], mis, f"Missed ({len(mis):,})", "#e07020"),
        (axes[1], det, f"Detected ({len(det):,})", "#2060c0"),
    ]:
        ax.scatter(sub["tess_tmag"], sub["tess_snr"], s=3, alpha=0.18, color=color)
        ax.axvline(TMAG_LIMIT, color="black", linestyle="--", linewidth=1.2, label="Tmag limit")
        ax.axhline(SNR_THRESHOLD, color="red", linestyle="--", linewidth=1.2, label="SNR threshold")
        ax.set_yscale("log"); ax.set_xlabel("TESS T magnitude")
        ax.set_ylabel("Toy TESS SNR (log)"); ax.set_title(f"A. {title} (transiting only)")
        ax.legend(fontsize=8); ax.grid(alpha=0.15)
    plt.suptitle("A. Tmag vs SNR — transiting planets only, TESS P-Pop universe", y=1.01)
    savefig(OUT_DIR / "A_tmag_vs_snr.png")
    print("  Saved A_tmag_vs_snr.png")


def plot_recovery_by_snr(df: pd.DataFrame) -> None:
    df = add_bins(df)
    s = (df.dropna(subset=["snr_bin"]).groupby("snr_bin", observed=True)
         .agg(n=("detected", "size"), pass_fraction=("detected", "mean")).reset_index())
    s["x"] = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(s["x"], s["pass_fraction"], color="#2060c0", alpha=0.75, width=0.6)
    for _, row in s.iterrows():
        ax.text(row["x"], row["pass_fraction"] + 0.025,
                f"{row['pass_fraction']:.0%}\nN={int(row['n']):,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 1.18); ax.set_xticks(s["x"])
    ax.set_xticklabels(s["snr_bin"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Detection fraction"); ax.set_xlabel("Toy TESS SNR bin")
    ax.set_title("B. Recovery fraction vs SNR — transiting planets\nTESS P-Pop universe")
    ax.grid(axis="y", alpha=0.2)
    savefig(OUT_DIR / "B_recovery_curve.png")
    print("  Saved B_recovery_curve.png")


def plot_heatmap_radius_period(df: pd.DataFrame) -> None:
    df = add_bins(df)
    frac = df.groupby(["radius_bin", "period_bin"], observed=True)["detected"].mean().unstack("period_bin")
    count = df.groupby(["radius_bin", "period_bin"], observed=True)["detected"].size().unstack("period_bin")
    display = frac.where(count >= MIN_CELL_N)
    cmap = plt.cm.viridis.copy(); cmap.set_bad("0.88")

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    im = ax.imshow(display.to_numpy(dtype=float), vmin=0, vmax=1, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(frac.columns))); ax.set_yticks(np.arange(len(frac.index)))
    ax.set_xticklabels([str(x) for x in frac.columns], rotation=30, ha="right")
    ax.set_yticklabels([str(y) for y in frac.index])
    ax.set_xlabel("Period [days]"); ax.set_ylabel("Planet radius [R_earth]")
    ax.set_title("C. Detection fraction (transiting planets) — TESS P-Pop\nGrey = N < 10")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            n = count.iloc[i, j] if not pd.isna(count.iloc[i, j]) else 0
            f = frac.iloc[i, j]
            if n == 0: continue
            if n < MIN_CELL_N:
                ax.text(j, i, f"N={int(n)}", ha="center", va="center", fontsize=7, color="0.45")
            else:
                ax.text(j, i, f"{f:.0%}\nN={int(n)}", ha="center", va="center", fontsize=7.5,
                        color="black" if f > 0.55 else "white")
    plt.colorbar(im, ax=ax, label="Detection fraction")
    savefig(OUT_DIR / "C_heatmap_radius_period.png")
    print("  Saved C_heatmap_radius_period.png")


def plot_sector_detection(df: pd.DataFrame) -> None:
    df = add_bins(df)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sec = df["tess_n_sectors"].dropna().clip(0, 27)
    ax.hist(sec, bins=np.arange(-0.5, 28.5, 1), color="#2060c0", alpha=0.75)
    ax.axvline(sec.median(), color="black", linestyle="--", linewidth=1.3, label=f"Median = {sec.median():.0f}")
    ax.set_xlabel("TESS sectors observed"); ax.set_ylabel("Count (transiting planets)")
    ax.set_title(f"D-left. Sector distribution\n({len(df):,} transiting planets)")
    ax.legend(); ax.grid(axis="y", alpha=0.2)

    ax2 = axes[1]
    s = (df.dropna(subset=["sector_bin"]).groupby("sector_bin", observed=True)
         .agg(n=("detected", "size"), pass_fraction=("detected", "mean")).reset_index())
    s["x"] = np.arange(len(s))
    ax2.bar(s["x"], s["pass_fraction"], color="#2060c0", alpha=0.75, width=0.6)
    for _, row in s.iterrows():
        ax2.text(row["x"], row["pass_fraction"] + 0.01,
                 f"{row['pass_fraction']:.1%}\nN={int(row['n']):,}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(s["x"]); ax2.set_xticklabels(s["sector_bin"].astype(str), rotation=0)
    ax2.set_ylim(0, max(s["pass_fraction"].max() * 1.35, 0.1))
    ax2.set_xlabel("Sector bin"); ax2.set_ylabel("Detection fraction")
    ax2.set_title("D-right. Detection fraction by sector count")
    ax2.grid(axis="y", alpha=0.2)
    savefig(OUT_DIR / "D_sector_detection.png")
    print("  Saved D_sector_detection.png")


def plot_miss_reasons(df: pd.DataFrame) -> None:
    """Loss budget + loss reason by star type (TESS analog of Kepler script 45 part D)."""
    order = ["detected", "too_shallow_or_low_snr", "too_few_transits",
             "not_observed_in_time_window", "not_observed_by_tess",
             "host_star_too_faint", "other"]
    colors = {"detected": "#2060c0", "too_shallow_or_low_snr": "#c04040",
              "too_few_transits": "#a0a020", "not_observed_in_time_window": "#806080",
              "not_observed_by_tess": "#40a0a0", "host_star_too_faint": "#b07050",
              "other": "#909090"}

    # Key the "detected" slice off the actual (recalibrated) detected flag, not the
    # catalog's reason_category, which can be stale after detection recalibration;
    # missed planets keep their recorded miss reason (unknown -> "other").
    df2 = df.copy()
    miss_reasons = order[1:-1]
    df2["reason_plot"] = np.where(
        df2["detected"].astype(bool), "detected",
        df2["reason_category"].where(df2["reason_category"].isin(miss_reasons), other="other"))
    counts = df2["reason_plot"].value_counts().reindex([r for r in order if r in df2["reason_plot"].values], fill_value=0)

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    ax_pie.pie(counts.values, labels=counts.index, colors=[colors.get(r, "#909090") for r in counts.index],
               autopct=lambda p: f"{p:.1f}%" if p > 1 else "", startangle=90)
    ax_pie.set_title("E-left. Loss budget (transiting planets)\nTESS P-Pop universe")

    stype_order = ["M", "K", "G", "F"]
    pivot = df2.groupby(["stype", "reason_plot"]).size().unstack("reason_plot", fill_value=0)
    pivot = pivot.reindex(index=[s for s in stype_order if s in pivot.index])
    for col in order:
        if col not in pivot.columns: pivot[col] = 0
    pivot = pivot[order]; frac = pivot.div(pivot.sum(axis=1), axis=0)
    bottom = np.zeros(len(frac))
    for col in order:
        ax_bar.bar(np.arange(len(frac)), frac[col], bottom=bottom, color=colors.get(col, "#909090"), label=col, width=0.6)
        bottom += frac[col].values
    ax_bar.set_xticks(np.arange(len(frac)))
    ax_bar.set_xticklabels([f"{st}\nN={int(pivot.loc[st].sum()):,}" for st in frac.index], rotation=0)
    ax_bar.set_ylim(0, 1); ax_bar.set_ylabel("Fraction"); ax_bar.set_xlabel("Star type")
    ax_bar.set_title("E-right. Loss reason by star type")
    ax_bar.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    savefig(OUT_DIR / "E_miss_reasons.png")
    print("  Saved E_miss_reasons.png")


def write_summary(df: pd.DataFrame) -> None:
    n_trans = len(df)
    n_bright = df["tess_star_bright_enough"].sum()
    n_enough = (df["tess_star_bright_enough"] & df["tess_enough_transits"]).sum()
    n_det = df["detected"].sum()
    reasons = df["reason_category"].value_counts()
    sec = df["tess_n_sectors"].dropna()
    lines = [
        "=" * 64,
        "TESS P-Pop DETECTOR VERIFICATION (transiting planets only)",
        f"Catalog: {CATALOG.name}",
        "=" * 64,
        f"Transiting planets:         {n_trans:>10,}",
        f"  + bright enough:          {n_bright:>10,}  ({n_bright/n_trans:.1%})",
        f"  + enough transits:        {n_enough:>10,}  ({n_enough/n_trans:.1%})",
        f"Detected:                   {n_det:>10,}  ({n_det/n_trans:.1%})",
        "",
        f"Sector stats: median={sec.median():.0f}, mean={sec.mean():.1f}, p25={sec.quantile(0.25):.0f}, p75={sec.quantile(0.75):.0f}",
        "",
        "Loss reasons (transiting only):",
    ]
    for reason, cnt in reasons.items():
        lines.append(f"  {reason:<35} {cnt:>6,}  ({cnt/n_trans:.1%})")
    lines += ["", "Detected by star type:"]
    for st, g in df.groupby("stype"):
        nd = g["detected"].sum()
        lines.append(f"  {st}:  {nd:>5,} / {len(g):>5,}  ({nd/len(g):.2%})")
    lines.append("=" * 64)
    text = "\n".join(lines)
    (OUT_DIR / "summary_stats.txt").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    print(f"Output dir: {OUT_DIR}")
    df = load_catalog()
    df = add_bins(df)
    print("\nGenerating figures...")
    plot_tmag_vs_snr(df)
    plot_recovery_by_snr(df)
    plot_heatmap_radius_period(df)
    plot_sector_detection(df)
    plot_miss_reasons(df)
    write_summary(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
