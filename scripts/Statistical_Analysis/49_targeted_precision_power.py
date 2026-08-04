"""49_targeted_precision_power.py — further-work forecast for the relaxed-threshold (v2) analysis.

The cold rocky desert census saturated at six candidates (script 48): relaxing the cuts adds no
seventh planet, so the near-term experiment is no longer finding new cold planets but
re-measuring the six we already have. Four of the six carry 13-28% mass uncertainties. This
script forecasts that targeted-precision program with the SAME methods as the strict-cut note
(scripts 44/45/46): the Gaussian-overlap sigma gap

    T = |mu_obs - mu_scn| / sqrt(sd_obs^2 + sd_scn^2)

on the detected volatile fraction, with the observed sample built by the per-planet error redraw
of script 17 (mc_nasa) and the scenario bells built by its detect+noise Monte Carlo (mc_universe).

Three analyses, all under the relaxed cuts (30% mass / 10% radius, R>=1.35, cold super-Earth cut
S<50 & M>2):
  1. precision power          — the sigma gap to escape-only and to primordial-rocky as the six
                                candidates' mass precision improves from today to ~2%, in the
                                P-Pop universe (prior-dependent, near-ambiguous under v2) and the
                                flat-Otegi control (still favors escape-only).
  2. value of a re-measurement — which single candidate, pinned to 5% mass, moves the gap most.
  3. the prior wall           — the residual disagreement between the P-Pop and flat verdicts that
                                no amount of precision on the six can close, plus the four-relation
                                mass-radius spread inside the flat control.

Reuses (does not reimplement): script 20 (S77: RELATIONS, build_arrays, SEED) and script 17
(S72: load_nasa, mc_nasa, mc_universe, build_pool, puffy_frac, load_silicate, BOX,
MASS_THRESHOLD, gauss). Relaxed cuts are set on S72 exactly as script 48 does; every number
traces to this run.

Outputs (my_outputs/49_targeted_precision_power/ + paper/figures_v2/):
    precision_power.png, remeasurement_value.png, precision_table.csv, candidate_value.csv

Run (from repo root, PYTHONPATH set):
    python "scripts/Statistical_Analysis/49_targeted_precision_power.py"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S77 = _load("s77", str(ROOT / "scripts" / "Statistical_Analysis" / "20_flat_rocky_mr_vs_nasa.py"))
S72 = S77.S72

# relaxed thresholds, identical to script 48
S72.NASA_MASS_PREC = 0.30
S72.NASA_RAD_PREC = 0.10
S72.N_REPEATS = 4000

COLD_INSOL = 50.0
CORNER_RADIUS = 1.35
COLD_CUT = dict(mass_min=2.0, insol_max=COLD_INSOL)
SEED = S77.SEED

OUT_DIR = ROOT / "my_outputs" / "49_targeted_precision_power"
FIGS_V2 = ROOT / "paper" / "figures_v2"

# mass-precision ladder for the six candidates (today's worst is ~28%); radius kept as published,
# since the campaign is RV mass follow-up
TARGET_PRECS = np.array([0.28, 0.25, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02])
PIN_PREC = 0.05

# the six occupants of the relaxed-cut desert (paper_v2 Table 2), for a name cross-check only
EXPECTED = {"LHS 1140 b", "TOI-1452 b", "LHS 1903 e", "TOI-198 b", "TOI-771 b", "TOI-1468 b"}


def tension(a, b):
    if a.size == 0 or b.size == 0:
        return np.nan
    sd = np.sqrt(a.std() ** 2 + b.std() ** 2)
    return abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan


def load_nasa_named():
    """S72.load_nasa's precision-passing sample with planet names attached, row-aligned with
    S72.load_nasa()'s array order (same boolean mask on the same table)."""
    df = pd.read_csv(S72.NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    prec = (np.maximum(me1, me2) / m <= S72.NASA_MASS_PREC) & (np.maximum(re1, re2) / r <= S72.NASA_RAD_PREC)
    keep = meas & prec & r.between(S72.BOX["r_lo"], S72.BOX["r_hi"]) & m.between(S72.BOX["m_lo"], S72.BOX["m_hi"])
    name = df.get("pl_name", pd.Series("", index=df.index)).astype(str)
    d = S72.load_nasa()
    d["name"] = name[keep].to_numpy()
    assert d["name"].size == d["m"].size, "named NASA load misaligned with S72.load_nasa"
    return d


def find_candidates(nasa, m_sil, r_sil):
    """The six cold-rocky-desert occupants inside the precision-passing sample, by central value:
    S<50, M>2, R>=1.35, below the silicate line."""
    below = nasa["r"] <= np.interp(nasa["m"], m_sil, r_sil)
    sel = (nasa["ins"] < COLD_INSOL) & (nasa["m"] > COLD_CUT["mass_min"]) & (nasa["r"] >= CORNER_RADIUS) & below
    return np.flatnonzero(sel)


def shrink(nasa, idx, mass_prec):
    """Copy of the NASA dict with the candidates at `idx` re-measured to `mass_prec` fractional
    mass error (symmetric); radius errors left at their published values."""
    d = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in nasa.items()}
    d["me1"][idx] = d["me2"][idx] = mass_prec * d["m"][idx]
    return d


def obs_bell(nasa, m_sil, r_sil):
    """Observed cold-super-Earth volatile-fraction bell (common random numbers: fresh seeded rng
    so the only thing that changes across precision settings is the candidates' error bars)."""
    rng = np.random.default_rng(SEED)
    nv, n_eff = S72.mc_nasa(nasa, COLD_CUT, m_sil, r_sil, rng)
    return nv, n_eff


def scenario_bells(arr, m_sil, r_sil):
    rng = np.random.default_rng(SEED)
    s_esc, n_esc = S72.mc_universe(arr, True, COLD_CUT, m_sil, r_sil, rng)   # drop rocky M>2
    s_prim, n_prim = S72.mc_universe(arr, False, COLD_CUT, m_sil, r_sil, rng)  # keep all
    return dict(esc=s_esc, prim=s_prim, n_esc=n_esc, n_prim=n_prim)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_V2.mkdir(parents=True, exist_ok=True)
    m_sil, r_sil = S72.load_silicate()

    nasa = load_nasa_named()
    idx = find_candidates(nasa, m_sil, r_sil)
    names = list(nasa["name"][idx])
    print("=" * 78)
    print(f"Six-candidate identification (relaxed cuts {S72.NASA_MASS_PREC:.0%}/{S72.NASA_RAD_PREC:.0%}, "
          f"R>={CORNER_RADIUS}, S<{COLD_INSOL:g}, M>{COLD_CUT['mass_min']:g}):")
    print(f"  found {len(idx)}: {names}")
    print(f"  paper_v2 Table 2 cross-check: match={set(names) == EXPECTED}")
    cur_prec = np.maximum(nasa["me1"][idx], nasa["me2"][idx]) / nasa["m"][idx]
    for nm, pc in sorted(zip(names, cur_prec), key=lambda t: -t[1]):
        print(f"    {nm:<14} current mass precision {pc:.1%}")

    print("\n" + "=" * 78)
    print("Building scenario bells (cold super-Earth cut, relaxed precision) ...")
    universes = {
        "P-Pop": S72.build_pool("ppop", m_sil, r_sil),
        "flat-Otegi": S77.build_arrays(dict(mr_C=1.03, mr_beta=0.29), m_sil, r_sil),
    }
    bells = {u: scenario_bells(arr, m_sil, r_sil) for u, arr in universes.items()}
    for u, b in bells.items():
        print(f"  {u:<12} escape-only mu={b['esc'].mean():.3f} sd={b['esc'].std():.3f} (N={b['n_esc']:.0f})  "
              f"primordial mu={b['prim'].mean():.3f} sd={b['prim'].std():.3f} (N={b['n_prim']:.0f})")

    # ---- 1. precision power: sweep the candidates' mass precision ----
    print("\n" + "=" * 78)
    print("PRECISION POWER: sigma gap vs candidate mass precision")
    print("=" * 78)
    rows = []
    # today's baseline first (no shrink)
    ladder = [("today", None)] + [(f"{p:.0%}", p) for p in TARGET_PRECS]
    for u, b in bells.items():
        print(f"\n[{u}]  {'target':>8}{'mu_obs':>9}{'sd_obs':>9}{'T(escape)':>11}{'T(primordial)':>15}")
        for lab, p in ladder:
            src = nasa if p is None else shrink(nasa, idx, p)
            nv, n_eff = obs_bell(src, m_sil, r_sil)
            t_esc = tension(nv, b["esc"])
            t_prim = tension(nv, b["prim"])
            print(f"  {lab:>8}{nv.mean():>9.3f}{nv.std():>9.3f}{t_esc:>10.1f}σ{t_prim:>14.1f}σ")
            rows.append(dict(universe=u, target=lab, target_prec=(np.nan if p is None else p),
                             n_eff=n_eff, mu_obs=nv.mean(), sd_obs=nv.std(),
                             T_escape=t_esc, T_primordial=t_prim))
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT_DIR / "precision_table.csv", index=False)

    # ---- 2. value of a re-measurement: pin one candidate at a time ----
    print("\n" + "=" * 78)
    print(f"VALUE OF A RE-MEASUREMENT: pin one candidate to {PIN_PREC:.0%} mass, dT to primordial")
    print("=" * 78)
    vrows = []
    nv0 = {u: obs_bell(nasa, m_sil, r_sil)[0] for u in bells}
    base = {u: dict(esc=tension(nv0[u], bells[u]["esc"]), prim=tension(nv0[u], bells[u]["prim"])) for u in bells}
    for u, b in bells.items():
        print(f"\n[{u}]  baseline T(escape)={base[u]['esc']:.2f}σ  T(primordial)={base[u]['prim']:.2f}σ")
        print(f"  {'candidate':<14}{'dT(escape)':>12}{'dT(primordial)':>16}")
        for j in idx:
            nv, _ = obs_bell(shrink(nasa, np.array([j]), PIN_PREC), m_sil, r_sil)
            d_esc = tension(nv, b["esc"]) - base[u]["esc"]
            d_prim = tension(nv, b["prim"]) - base[u]["prim"]
            print(f"  {nasa['name'][j]:<14}{d_esc:>11.3f}σ{d_prim:>15.3f}σ")
            vrows.append(dict(universe=u, candidate=nasa["name"][j],
                              dT_escape=d_esc, dT_primordial=d_prim))
    vtab = pd.DataFrame(vrows)
    vtab.to_csv(OUT_DIR / "candidate_value.csv", index=False)

    # ---- 3. the prior wall: four-relation flat spread + P-Pop vs flat residual ----
    print("\n" + "=" * 78)
    print("PRIOR WALL: baseline gaps across the four flat relations, and P-Pop vs flat")
    print("=" * 78)
    print(f"  {'flat relation':<22}{'T(escape)':>11}{'T(primordial)':>15}")
    esc_gaps, prim_gaps = [], []
    for name, eq, applies, kw in S77.RELATIONS:
        arr = S77.build_arrays(kw, m_sil, r_sil)
        b = scenario_bells(arr, m_sil, r_sil)
        nv, _ = obs_bell(nasa, m_sil, r_sil)
        te, tp = tension(nv, b["esc"]), tension(nv, b["prim"])
        esc_gaps.append(te); prim_gaps.append(tp)
        print(f"  {name:<22}{te:>10.1f}σ{tp:>14.1f}σ")
    print(f"  flat control escape-only gap spans {min(esc_gaps):.1f}-{max(esc_gaps):.1f}sigma; "
          f"primordial {min(prim_gaps):.1f}-{max(prim_gaps):.1f}sigma")
    print(f"  P-Pop     escape-only {base['P-Pop']['esc']:.1f}sigma  primordial {base['P-Pop']['prim']:.1f}sigma "
          f"(near-ambiguous); the P-Pop-vs-flat disagreement is the prior wall precision cannot close")

    make_figures(tab, vtab, bells, base, m_sil, r_sil, nasa, idx)
    print("\n--> done.")


def make_figures(tab, vtab, bells, base, m_sil, r_sil, nasa, idx):
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
                         "legend.fontsize": 10})

    # Fig 1: precision-power curves, one panel per universe
    worst_now = float(np.nanmax(np.maximum(nasa["me1"][idx], nasa["me2"][idx]) / nasa["m"][idx])) * 100
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, u in zip(axes, bells):
        sub = tab[(tab.universe == u) & tab.target_prec.notna()].sort_values("target_prec", ascending=False)
        x = sub.target_prec.to_numpy() * 100
        ax.plot(x, sub.T_primordial, "o-", color="tab:blue", lw=2, label="gap to primordial-rocky")
        ax.plot(x, sub.T_escape, "s-", color="tab:orange", lw=2, label="gap to escape-only")
        ax.axhline(3, color="0.5", ls="--", lw=1); ax.axhline(5, color="0.5", ls=":", lw=1)
        ax.text(x.max(), 3.05, "3σ", fontsize=9, color="0.4", va="bottom", ha="left")
        ax.text(x.max(), 5.05, "5σ", fontsize=9, color="0.4", va="bottom", ha="left")
        ax.axvline(worst_now, color="tab:green", lw=1.3)
        ax.invert_xaxis()
        ax.set_xlabel("mass precision on the six candidates [%]")
        ax.set_title(u); ax.grid(alpha=0.2); ax.legend(loc="best")
    axes[0].set_ylabel(r"$\sigma$ gap from the observed cold sample")
    fig.suptitle("Precision power: re-measuring the six cold candidates sharpens the observed volatile fraction\n"
                 "left improves →; green line marks today's worst candidate precision", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for dest in (OUT_DIR / "precision_power.png", FIGS_V2 / "precision_power.png"):
        fig.savefig(dest, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Fig 2: per-candidate value bars (P-Pop, the ambiguous universe)
    sub = vtab[vtab.universe == "P-Pop"].sort_values("dT_primordial")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ["tab:blue" if v >= 0 else "tab:red" for v in sub.dT_primordial]
    ax.barh(sub.candidate, sub.dT_primordial, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"change in the $\sigma$ gap to primordial-rocky from pinning this planet to 5% mass")
    ax.set_title("Value of a re-measurement (P-Pop universe): which of the six to re-measure first\n"
                 "positive = sharpens the test, negative = a planet that settles onto the line dilutes it",
                 fontsize=11)
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    for dest in (OUT_DIR / "remeasurement_value.png", FIGS_V2 / "remeasurement_value.png"):
        fig.savefig(dest, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> figures saved to {OUT_DIR} and {FIGS_V2}")


if __name__ == "__main__":
    main()
