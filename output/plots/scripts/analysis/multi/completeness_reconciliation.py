"""
32_completeness_reconciliation.py — FIX for "your own two analyses disagree".

Two earlier numbers looked contradictory for the SAME cold corner:

    script 60 (cold_corner_metric_power) : joint completeness ~ 1.3%   (cold, pooled)
    script 53 / 64 (forward-model)       : joint completeness ~ 62%    (cold, M dwarfs)

They are NOT in conflict.  They are completeness against DIFFERENT denominators, and
the gap between them IS a physical result, not a bug.  Any cold rocky planet must pass
TWO independent gates to become a confirmed NASA planet:

    (1) GEOMETRIC TRANSIT      P_geom  = does its orbit cross the star at all?
    (2) DETECT + MASS | TRANSIT P_conf = given it transits, is the transit detected AND
                                         the mass RV-measurable?

    absolute completeness  =  P_geom  x  P_conf|transit

  * Script 60's 1.3% is the ABSOLUTE completeness (it includes the geometric-transit
    gate and pools all star types) — the right number for a blind-survey YIELD.
  * Script 53/64's 62% is P_conf|transit for M dwarfs — the right number for the
    radius-limit question, because the NASA overlay planets ALL already transit, so the
    correct denominator is "planets that transit", not "planets that exist".

This script computes the full decomposition per star type and shows the two numbers
reconcile:  pooled absolute reproduces ~script-60, and M's conditional reproduces ~62%.
The cold corner is geometry-starved (long period -> rarely transits) AND, for FGK,
RV-starved; M dwarfs keep P_conf high because their cold orbits are short-period.

Run from repo root:
    python scripts/32_completeness_reconciliation.py
"""

from __future__ import annotations

import argparse
import importlib.util
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

from tools.paths import LIFESIM_OUTER_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S44 = _load_module("s44", ROOT / "plot" / "script plots" / "56_kepler_tess_rocky_fgkm_gaia60pc.py")
S53 = _load_module("s53", ROOT / "plot" / "script plots" / "58_rv_rocky_fgkm_test.py")

OUT_DIR = ROOT / "output/plots" / "32_completeness_reconciliation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAR_ORDER      = S44.STAR_ORDER
COLD_INSOLATION = S44.COLD_INSOLATION
RADIUS_LIMITS   = S44.RADIUS_LIMITS


def cold_large_mask(df: pd.DataFrame, r_large: float) -> np.ndarray:
    """Cold (I<10) AND large radius (R > r_large) — the 'large cold rocky' corner."""
    f = pd.to_numeric(df["flux_p"], errors="coerce").to_numpy()
    r = pd.to_numeric(df["radius_p"], errors="coerce").to_numpy()
    return (f < COLD_INSOLATION) & (r > r_large)


def decompose(df: pd.DataFrame, r_large: float) -> dict:
    """Per star type (+ pooled): N_exist, P_geom, P_conf|transit, absolute."""
    df = df.copy()
    df["joint"] = df["detected_case"].to_numpy() & df["rv_ok"].to_numpy()
    region_all = cold_large_mask(df, r_large)

    rows = {}
    for st in STAR_ORDER + ["ALL"]:
        sel = region_all if st == "ALL" else (region_all & (df["stype_clean"].to_numpy() == st))
        sub = df[sel]
        n_exist = len(sub)
        if n_exist == 0:
            rows[st] = dict(n_exist=0, p_geom=np.nan, p_conf=np.nan, absolute=np.nan,
                            n_trans=0, n_conf=0)
            continue
        trans = sub["denominator_case"].to_numpy()
        conf = sub["joint"].to_numpy()
        n_trans = int(trans.sum())
        n_conf = int(conf.sum())
        p_geom = n_trans / n_exist
        p_conf = (n_conf / n_trans) if n_trans > 0 else np.nan
        absolute = n_conf / n_exist
        rows[st] = dict(n_exist=n_exist, p_geom=p_geom, p_conf=p_conf,
                        absolute=absolute, n_trans=n_trans, n_conf=n_conf)
    return rows


def plot_decomposition(rows: dict, r_large: float, args) -> Path:
    sts = [s for s in STAR_ORDER + ["ALL"] if rows[s]["n_exist"] > 0]
    x = np.arange(len(sts))
    w = 0.27
    p_geom = [rows[s]["p_geom"] for s in sts]
    p_conf = [rows[s]["p_conf"] for s in sts]
    absol  = [rows[s]["absolute"] for s in sts]

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    b1 = ax.bar(x - w, p_geom, w, color="#4575b4", label="P(geometric transit)")
    b2 = ax.bar(x,      p_conf, w, color="#91bfdb",
                label="P(detect + RV mass | transits)  ← the 53/64 number")
    b3 = ax.bar(x + w,  absol,  w, color="#d73027",
                label="absolute = P_geom × P_conf  ← the script-60 number")
    for bars, vals in [(b1, p_geom), (b2, p_conf), (b3, absol)]:
        for rect, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(rect.get_x() + rect.get_width() / 2, v + 0.012,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(sts)
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("host star type")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper center")
    ax.set_title(
        f"Cold large-rocky completeness decomposed (I<{COLD_INSOLATION:g}, R>{r_large:g} R⊕, rocky)\n"
        "The '1.3% vs 62%' disagreement = two different denominators: geometric-transit "
        "gate (blue) × confirmable-given-transit (light) = absolute yield (red)",
        fontsize=11)
    out = OUT_DIR / f"cold_completeness_decomposition_R{r_large:g}.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


def write_summary(rows: dict, r_large: float, args):
    lines = [
        "=" * 84,
        "RECONCILING THE COLD-COMPLETENESS 'DISAGREEMENT'",
        "=" * 84,
        f"Region: cold (I < {COLD_INSOLATION:g}) AND large (R > {r_large:g} R⊕), rocky P-Pop, "
        f"{args.n_catalogs or 'all'} universes, RV={args.instrument} N={args.n_obs}",
        "",
        f"{'stype':<6}{'N_exist':>9}{'P_geom':>9}{'P_conf|tr':>11}{'absolute':>10}"
        f"{'(N_trans)':>11}{'(N_conf)':>10}",
        "-" * 84,
    ]
    for st in STAR_ORDER + ["ALL"]:
        r = rows[st]
        if r["n_exist"] == 0:
            lines.append(f"{st:<6}{0:>9}{'-':>9}{'-':>11}{'-':>10}{'-':>11}{'-':>10}")
            continue
        lines.append(
            f"{st:<6}{r['n_exist']:>9,}{r['p_geom']:>9.1%}"
            f"{(r['p_conf'] if np.isfinite(r['p_conf']) else float('nan')):>11.1%}"
            f"{r['absolute']:>10.1%}{r['n_trans']:>11,}{r['n_conf']:>10,}")
    m = rows["M"]; allr = rows["ALL"]
    lines += [
        "-" * 84,
        "",
        "RECONCILIATION",
        f"  * script 53/64 number  = P_conf|transit for M dwarfs        = "
        f"{m['p_conf']:.0%}"
        if np.isfinite(m["p_conf"]) else "  * (no M-dwarf cold-large planets)",
        f"  * script 60 number     = ALL-pooled ABSOLUTE completeness   = "
        f"{allr['absolute']:.1%}"
        if np.isfinite(allr["absolute"]) else "  * (no pooled planets)",
        f"  * identity check (M):    P_geom x P_conf = {m['p_geom']:.1%} x {m['p_conf']:.0%}"
        f" = {m['p_geom']*m['p_conf']:.1%} = absolute ({m['absolute']:.1%})"
        if np.isfinite(m["p_conf"]) else "",
        "",
        "READING",
        "  The two analyses never disagreed.  P_geom (does it transit?) is the big",
        "  filter in the cold corner: long-period orbits rarely transit, so the ABSOLUTE",
        "  yield is tiny (script 60).  But the NASA planets we plot ALL transit already,",
        "  so the radius-limit question must condition on transiting planets — and there",
        "  M dwarfs stay ~60% confirmable (script 53/64) because their cold habitable-zone",
        "  orbits are SHORT-period (transit often, large RV K).  Use absolute for survey",
        "  YIELD; use P_conf|transit for the radius-limit (overlay) question.",
        "=" * 84,
    ]
    text = "\n".join([l for l in lines if l != ""] if False else lines)
    (OUT_DIR / f"summary_R{r_large:g}.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-catalogs", type=int, default=None)
    ap.add_argument("--instrument", default="best")
    ap.add_argument("--snr-threshold", type=float, default=5.0)
    ap.add_argument("--n-obs", type=int, default=100)
    ap.add_argument("--mag-target", type=float, default=12.0)
    ap.add_argument("--r-large", type=float, default=1.4,
                    help="radius above which a cold rocky planet counts as 'large' "
                         "(1.4 matches script 60's injection)")
    args = ap.parse_args()

    print("=" * 70)
    print("32_completeness_reconciliation.py")
    print("=" * 70)

    m_ref, r_ref = S44.load_rocky_reference_curve()
    shift = S44.compute_rocky_threshold_shift(m_ref, r_ref)

    ppop = S53.load_tess_ppop_for_rv(args.n_catalogs)
    ppop = S53.restrict_ppop_to_rocky(ppop, m_ref, r_ref, shift)
    ppop = S53.add_rv_ok(ppop, args.instrument, args.snr_threshold, args.n_obs, args.mag_target)

    rows = decompose(ppop, args.r_large)
    plot_decomposition(rows, args.r_large, args)
    write_summary(rows, args.r_large, args)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
