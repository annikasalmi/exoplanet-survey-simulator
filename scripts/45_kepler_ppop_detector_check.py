"""
45_kepler_ppop_detector_check.py

Kepler P-Pop detector verification (transiting planets only).
Loads the single-file universe catalog, filters to transiting planets,
then shows where in the detection pipeline losses occur.

Input:  run/kepler/data/Gaia/kepler_catalog_0.csv
Output: my_outputs/45_kepler_ppop_detector_check/

Run from repo root:
    python scripts/45_kepler_ppop_detector_check.py
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

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "run" / "kepler" / "data" / "Gaia" / "kepler_catalog_0.csv"
OUT_DIR = ROOT / "my_outputs" / "45_kepler_ppop_detector_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MES_THRESHOLD = 7.1
MIN_CELL_N = 10

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 260,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 9, "ytick.labelsize": 9,
})

KEEP_COLS = [
    "radius_p", "p_orb", "semimajor_p", "flux_p",
    "radius_s", "mass_s", "teff_s", "stype",
    "transiting_geometric",
    "kepler_mag_used", "kepler_cdpp_ppm",
    "n_transits_keplerish", "transit_depth_ppm",
    "kepler_mes", "kepler_mes_threshold",
    "kepler_enough_transits", "kepler_star_bright_enough", "kepler_depth_pass",
    "detected", "reason_category", "miss_reason",
]


def load_catalog() -> pd.DataFrame:
    print(f"Loading {CATALOG.name} ...")
    avail = pd.read_csv(CATALOG, nrows=0).columns.tolist()
    cols = [c for c in KEEP_COLS if c in avail]
    df = pd.read_csv(CATALOG, usecols=cols, low_memory=False)

    for col in ["radius_p", "p_orb", "kepler_mes", "kepler_cdpp_ppm",
                "kepler_mag_used", "n_transits_keplerish", "transit_depth_ppm",
                "radius_s", "teff_s", "flux_p", "kepler_mes_threshold"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["detected", "transiting_geometric", "kepler_star_bright_enough",
                "kepler_enough_transits", "kepler_depth_pass"]:
        if col in df.columns:
            df[col] = df[col].map({"True": True, "False": False, True: True, False: False}).fillna(False).astype(bool)

    n_total = len(df)
    df = df[df["transiting_geometric"].astype(bool)].copy()
    print(f"  Total planets: {n_total:,}   Transiting: {len(df):,}   Detected: {df['detected'].sum():,}")
    return df


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["period_bin"] = pd.cut(df["p_orb"], bins=[0, 3, 10, 30, 100, 300, np.inf],
        labels=["<3", "3-10", "10-30", "30-100", "100-300", ">300"], include_lowest=True)
    df["radius_bin"] = pd.cut(df["radius_p"], bins=[0, 1.0, 1.5, 2.0, 4.0, np.inf],
        labels=["<1", "1-1.5", "1.5-2", "2-4", ">4"], include_lowest=True)
    df["mes_bin"] = pd.cut(df["kepler_mes"],
        bins=[0, 5, MES_THRESHOLD, 10, 20, 50, 100, np.inf],
        labels=["<5", f"5-{MES_THRESHOLD}", f"{MES_THRESHOLD}-10", "10-20", "20-50", "50-100", ">100"],
        include_lowest=True)
    return df


def savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def plot_mes_distribution(df: pd.DataFrame) -> None:
    d = df[np.isfinite(df["kepler_mes"]) & (df["kepler_mes"] > 0)].copy()
    det = d[d["detected"]]
    mis = d[~d["detected"]]
    bins = np.logspace(np.log10(0.5), np.log10(d["kepler_mes"].quantile(0.995)), 55)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(mis["kepler_mes"], bins=bins, color="#e07020", alpha=0.65, label=f"Missed ({len(mis):,})")
    ax.hist(det["kepler_mes"], bins=bins, color="#2060c0", alpha=0.80, label=f"Detected ({len(det):,})")
    ax.axvline(MES_THRESHOLD, color="black", linestyle="--", linewidth=1.5, label=f"Threshold {MES_THRESHOLD}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Toy Kepler MES")
    ax.set_ylabel("Count (log)")
    ax.set_title("A. Toy MES — transiting planets only\nKepler P-Pop universe")
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    savefig(OUT_DIR / "A_mes_distribution.png")
    print("  Saved A_mes_distribution.png")


def plot_recovery_curve(df: pd.DataFrame) -> None:
    df = add_bins(df)
    s = (df.dropna(subset=["mes_bin"]).groupby("mes_bin", observed=True)
         .agg(n=("detected", "size"), pass_fraction=("detected", "mean")).reset_index())
    s["x"] = np.arange(len(s))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(s["x"], s["pass_fraction"], color="#2060c0", alpha=0.75, width=0.6)
    for _, row in s.iterrows():
        ax.text(row["x"], row["pass_fraction"] + 0.025,
                f"{row['pass_fraction']:.0%}\nN={int(row['n']):,}", ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 1.18)
    ax.set_xticks(s["x"]); ax.set_xticklabels(s["mes_bin"].astype(str), rotation=25, ha="right")
    ax.set_ylabel("Detection fraction"); ax.set_xlabel("Toy Kepler MES bin")
    ax.set_title("B. Recovery fraction vs MES — transiting planets\nKepler P-Pop universe")
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
    ax.set_title("C. Detection fraction (transiting planets) — Kepler P-Pop\nGrey = N < 10")
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


def plot_miss_reasons(df: pd.DataFrame) -> None:
    order = ["detected", "too_shallow_or_low_mes", "too_few_transits", "host_star_too_faint", "other"]
    df2 = df.copy()
    df2["reason_plot"] = df2["reason_category"].where(df2["reason_category"].isin(order[:-1]), other="other")
    counts = df2["reason_plot"].value_counts().reindex([r for r in order if r in df2["reason_plot"].values], fill_value=0)

    colors = {"detected": "#2060c0", "too_shallow_or_low_mes": "#c04040",
              "too_few_transits": "#a0a020", "host_star_too_faint": "#806080", "other": "#909090"}

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    ax_pie.pie(counts.values, labels=counts.index, colors=[colors.get(r, "#909090") for r in counts.index],
               autopct=lambda p: f"{p:.1f}%" if p > 1 else "", startangle=90)
    ax_pie.set_title("D-left. Loss budget (transiting planets)\nKepler P-Pop universe")

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
    ax_bar.set_title("D-right. Loss reason by star type")
    ax_bar.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    savefig(OUT_DIR / "D_miss_reasons.png")
    print("  Saved D_miss_reasons.png")


def plot_cdpp_vs_mag(df: pd.DataFrame) -> None:
    d = df[np.isfinite(df["kepler_cdpp_ppm"]) & (df["kepler_cdpp_ppm"] > 0) &
           np.isfinite(df["kepler_mag_used"]) & (df["kepler_mag_used"] > 0)].copy()
    det = d[d["detected"]]; mis = d[~d["detected"]]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.scatter(mis["kepler_mag_used"], mis["kepler_cdpp_ppm"], s=4, alpha=0.15,
               color="#e07020", label=f"Transiting, missed ({len(mis):,})")
    ax.scatter(det["kepler_mag_used"], det["kepler_cdpp_ppm"], s=6, alpha=0.35,
               color="#2060c0", label=f"Detected ({len(det):,})")
    ax.axvline(16.0, color="black", linestyle="--", linewidth=1.3, label="Kp = 16 limit")
    ax.set_yscale("log"); ax.set_xlabel("Kepler magnitude (Kp)")
    ax.set_ylabel("CDPP [ppm] (log)")
    ax.set_title("E. CDPP vs brightness — transiting planets\nKepler P-Pop universe")
    ax.legend(); ax.grid(axis="y", alpha=0.2)
    savefig(OUT_DIR / "E_cdpp_vs_mag.png")
    print("  Saved E_cdpp_vs_mag.png")


def write_summary(df: pd.DataFrame) -> None:
    n_trans = len(df)
    n_bright = df["kepler_star_bright_enough"].sum()
    n_enough = (df["kepler_star_bright_enough"] & df["kepler_enough_transits"]).sum()
    n_det = df["detected"].sum()
    reasons = df["reason_category"].value_counts()
    lines = [
        "=" * 64,
        "KEPLER P-Pop DETECTOR VERIFICATION (transiting planets only)",
        f"Catalog: {CATALOG.name}",
        "=" * 64,
        f"Transiting planets:     {n_trans:>10,}",
        f"  + bright enough:      {n_bright:>10,}  ({n_bright/n_trans:.1%})",
        f"  + enough transits:    {n_enough:>10,}  ({n_enough/n_trans:.1%})",
        f"Detected:               {n_det:>10,}  ({n_det/n_trans:.1%})",
        "",
        "Loss reasons (among transiting):",
    ]
    for reason, cnt in reasons.items():
        lines.append(f"  {reason:<32} {cnt:>6,}  ({cnt/n_trans:.1%})")
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
    plot_mes_distribution(df)
    plot_recovery_curve(df)
    plot_heatmap_radius_period(df)
    plot_miss_reasons(df)
    plot_cdpp_vs_mag(df)
    write_summary(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
