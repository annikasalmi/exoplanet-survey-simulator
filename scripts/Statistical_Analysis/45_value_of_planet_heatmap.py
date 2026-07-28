"""
45_value_of_planet_heatmap.py — Method 2: value-of-a-planet map.

Question: WHICH new NASA planet (at what insolation, radius) would help most to resolve the
flat-A-vs-flat-B puffy-fraction tension in the cold super-Earth corner (mass>2 M⊕, insol<50 I⊕)?

Machinery: reused from script 77 (RELATIONS, build_arrays, SEED, MR_SCATTER_DEX) and script 72
(via S77.S72: mc_universe, mc_nasa, load_nasa, load_silicate, BOX, MASS_THRESHOLD, N_REPEATS).
For each of the 4 rocky M-R relations: build the flat-universe pool, take the flat-B bell
(mc_universe, drop=False, cold-corner cut) ONCE — it does not change when a planet is injected
into NASA — and the baseline NASA bell (mc_nasa, cold-corner cut) ONCE. Tension is the same
Gaussian-overlap heuristic used throughout this repo: |mu1-mu2| / sqrt(sigma1^2+sigma2^2).

For each cell of a 12x12 log(insolation) x log(radius) grid spanning S72.BOX:
  - the injected planet sits at the cell center (radius, insolation fixed);
  - its TRUE mass is drawn (25x, seeded) from the flat-A ("no rocky super-Earths dropped")
    truth pool of DETECTED planets falling in that cell — flat A predicts the composition
    there, so this is the physically motivated source for "what mass would such a planet have";
    a cell with zero flat-A pool planets is left NaN (flat A predicts nothing there);
  - its measurement errors are the MEDIAN relative mass/radius errors of the current
    precision-passing NASA sample (a future planet is assumed measured like a typical current
    one) — computed once from S72.load_nasa()'s me1/me2/re1/re2 columns;
  - the row is appended to the NASA arrays mc_nasa actually uses (m, r, ins, me1, me2, re1, re2)
    and mc_nasa is re-run with the cold-corner cut; delta_sigma = tension_after - tension_before,
    averaged over the 25 mass draws.

Performance note: mc_nasa has no bootstrap (small n, python loop over N_REPEATS only) — it is
NOT the N_SAMPLE-driven cost of mc_universe. Benchmarking showed mc_nasa at N_REPEATS=4000
takes ~0.12 s/call; with up to 4 relations x 144 cells x 25 draws that is too slow (~30 min),
so N_REPEATS is reduced to 1000 for the injection scan only (restored to 4000 — S77's setting —
immediately after) and mc_universe/the baseline mc_nasa call stay at the full 4000. A spot-check
cell is evaluated at both settings and the relative difference printed (target ~10%).

Interpretation caveats (delta_sigma is a DESCRIPTIVE statistic, not an information value):
  - delta_sigma can be NEGATIVE: an injected planet whose drawn mass straddles the silicate
    line under measurement noise classifies ~50/50 puffy and dilutes NASA's high puffy
    fraction (0.75) toward flat B's, lowering the Gaussian-overlap tension. In a likelihood
    framework a confirmed planet never carries negative evidence — negativity here is an
    estimator property of the sample proportion, not evidence loss.
  - mc_nasa applies the mass>2 cut to the OBSERVED (re-noised) mass, so injected planets with
    drawn true mass near 2 M⊕ fail the cut in many repeats; near-zero cells at low mass are
    damped by exclusion, not necessarily low value.
  - baseline and injected tensions are independent MC draws (no common random numbers), so
    per-cell values carry ~0.01-0.02 sigma of MC noise; the ordering among the top cells is
    not significant — quote the high-value REGION, not a ranked list.

Outputs (my_outputs/45_value_of_planet_heatmap/):
    value_heatmap.png (2x2 panels, one per relation, delta_sigma heatmap) + value_table.csv

Run:
    python "scripts/Statistical_Analysis/45_value_of_planet_heatmap.py"
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
S72 = S77.S72                    # already loaded by S77, N_REPEATS = 4000

OUT_DIR = os.path.join(ROOT, "my_outputs", "45_value_of_planet_heatmap")
COLD_CUT = dict(mass_min=2.0, insol_max=50.0)
N_CELLS = 12
N_MASS_DRAWS = 25
N_REPEATS_SCAN = 1000
MDWARF_TEFF_MAX = 3900.0


def load_nasa_teff():
    """Duplicates S72.load_nasa's precision-passing mask (not modifying S72) so the host
    star Teff column (present in the underlying NASA CSV but not loaded by S72) can be
    attached, row-aligned with S72.load_nasa()'s output."""
    df = pd.read_csv(S72.NASA_FILE)
    m = pd.to_numeric(df["pl_bmasse"], errors="coerce")
    r = pd.to_numeric(df["pl_rade"], errors="coerce")
    ins = pd.to_numeric(df["pl_insol"], errors="coerce")
    teff = pd.to_numeric(df["st_teff"], errors="coerce")
    me1 = pd.to_numeric(df["pl_bmasseerr1"], errors="coerce").abs()
    me2 = pd.to_numeric(df["pl_bmasseerr2"], errors="coerce").abs()
    re1 = pd.to_numeric(df["pl_radeerr1"], errors="coerce").abs()
    re2 = pd.to_numeric(df["pl_radeerr2"], errors="coerce").abs()
    prov = df.get("pl_bmassprov", pd.Series("", index=df.index)).astype(str)
    meas = prov.str.contains("Mass|Msini", case=False, na=False) & ~prov.str.contains("Calc", case=False, na=False)
    prec = (np.maximum(me1, me2) / m <= S72.NASA_MASS_PREC) & (np.maximum(re1, re2) / r <= S72.NASA_RAD_PREC)
    keep = meas & prec & r.between(S72.BOX["r_lo"], S72.BOX["r_hi"]) & m.between(S72.BOX["m_lo"], S72.BOX["m_hi"])
    return dict(m=m[keep].to_numpy(), r=r[keep].to_numpy(), ins=ins[keep].to_numpy(), teff=teff[keep].to_numpy())


def tension(a, b):
    if a.size == 0 or b.size == 0:
        return np.nan
    sd = np.sqrt(a.std() ** 2 + b.std() ** 2)
    return abs(a.mean() - b.mean()) / sd if sd > 0 else np.nan


def inject_row(nasa, mass_c, radius_c, insol_c, rel_mass_err, rel_rad_err):
    me = rel_mass_err * mass_c
    re = rel_rad_err * radius_c
    return dict(m=np.append(nasa["m"], mass_c), r=np.append(nasa["r"], radius_c),
                ins=np.append(nasa["ins"], insol_c),
                me1=np.append(nasa["me1"], me), me2=np.append(nasa["me2"], me),
                re1=np.append(nasa["re1"], re), re2=np.append(nasa["re2"], re))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = S72.load_silicate()
    rng = np.random.default_rng(S77.SEED)
    nasa = S72.load_nasa()
    nasa_teff = load_nasa_teff()
    assert nasa_teff["m"].size == nasa["m"].size, "teff-augmented NASA load misaligned with S72.load_nasa"

    rel_mass_err = float(np.median(np.maximum(nasa["me1"], nasa["me2"]) / nasa["m"]))
    rel_rad_err = float(np.median(np.maximum(nasa["re1"], nasa["re2"]) / nasa["r"]))
    print(f"--> median NASA relative errors: mass={rel_mass_err:.3f}, radius={rel_rad_err:.3f} "
          f"(applied to the injected planet)")

    is_m = nasa_teff["teff"] < MDWARF_TEFF_MAX
    has_teff = np.isfinite(nasa_teff["teff"])
    corner = nasa_teff["ins"] < COLD_CUT["insol_max"]
    n_corner = int(corner.sum())
    n_corner_m = int((corner & is_m & has_teff).sum())
    n_corner_fgk = int((corner & ~is_m & has_teff).sum())
    n_corner_noteff = int((corner & ~has_teff).sum())
    print(f"--> cold corner (insol<{COLD_CUT['insol_max']:g}): {n_corner_m} of {n_corner} "
          f"precision-passing planets orbit M dwarfs (Teff<{MDWARF_TEFF_MAX:g} K); "
          f"{n_corner_fgk} orbit FGK hosts; {n_corner_noteff} have no Teff.")

    insol_edges = np.geomspace(S72.BOX["f_lo"], S72.BOX["f_hi"], N_CELLS + 1)
    radius_edges = np.geomspace(S72.BOX["r_lo"], S72.BOX["r_hi"], N_CELLS + 1)
    insol_centers = np.sqrt(insol_edges[:-1] * insol_edges[1:])
    radius_centers = np.sqrt(radius_edges[:-1] * radius_edges[1:])

    N_REPEATS_FULL = S72.N_REPEATS
    print(f"\n--> building {len(S77.RELATIONS)} rocky-relation pools + baselines "
          f"(N_REPEATS={N_REPEATS_FULL})...")
    rel_data = []
    for name, eq, applies, kw in S77.RELATIONS:
        arr = S77.build_arrays(kw, m_sil, r_sil)
        mass, radius, flux, puffy, det = arr
        flatA_mask = det & ~((~puffy) & (mass > S72.MASS_THRESHOLD))
        fi = np.digitize(flux, insol_edges) - 1
        ri = np.digitize(radius, radius_edges) - 1
        sB, nB = S72.mc_universe(arr, False, COLD_CUT, m_sil, r_sil, rng)
        nv0, n0 = S72.mc_nasa(nasa, COLD_CUT, m_sil, r_sil, rng)
        tension0 = tension(nv0, sB)
        print(f"  {name:<24} flat-B: mu={sB.mean():.3f} sd={sB.std():.3f} N_eff={nB:.0f}   "
              f"NASA0: mu={nv0.mean():.3f} sd={nv0.std():.3f} N={n0}   tension0={tension0:.2f}sigma")
        rel_data.append(dict(name=name, mass=mass, radius=radius, flatA_mask=flatA_mask,
                             fi=fi, ri=ri, sB=sB, tension0=tension0))

    # ---- spot-check: reduced N_REPEATS vs full, on the first non-empty cold-corner cell ----
    d0 = rel_data[0]
    spot_cell = None
    for ci in range(N_CELLS):
        if insol_centers[ci] >= COLD_CUT["insol_max"]:
            continue
        for ri_ in range(N_CELLS):
            cell_masses = d0["mass"][d0["flatA_mask"] & (d0["fi"] == ci) & (d0["ri"] == ri_)]
            if cell_masses.size > 0:
                spot_cell = (ci, ri_, cell_masses)
                break
        if spot_cell:
            break

    if spot_cell:
        ci, ri_, cell_masses = spot_cell
        draws = rng.choice(cell_masses, size=N_MASS_DRAWS, replace=True)
        for tag, nrep in [("full", N_REPEATS_FULL), ("scan", N_REPEATS_SCAN)]:
            S72.N_REPEATS = nrep
            deltas = np.empty(N_MASS_DRAWS)
            for k, md in enumerate(draws):
                inj = inject_row(nasa, md, radius_centers[ri_], insol_centers[ci], rel_mass_err, rel_rad_err)
                nv_after, _ = S72.mc_nasa(inj, COLD_CUT, m_sil, r_sil, rng)
                deltas[k] = tension(nv_after, d0["sB"]) - d0["tension0"]
            if tag == "full":
                dv_full = deltas.mean()
            else:
                dv_scan = deltas.mean()
        S72.N_REPEATS = N_REPEATS_FULL
        reldiff = abs(dv_scan - dv_full) / abs(dv_full) if dv_full != 0 else np.nan
        print(f"\n--> spot-check cell (insol~{insol_centers[ci]:.2f}, radius~{radius_centers[ri_]:.2f}): "
              f"delta_sigma full(N={N_REPEATS_FULL})={dv_full:.4f}  scan(N={N_REPEATS_SCAN})={dv_scan:.4f}  "
              f"relative diff={reldiff:.1%}")
    else:
        print("\n--> spot-check: no non-empty cold-corner cell found in relation 0 pool (unexpected).")

    # ---- full grid scan at reduced N_REPEATS ----
    S72.N_REPEATS = N_REPEATS_SCAN
    print(f"\n--> scanning {N_CELLS}x{N_CELLS} grid x {len(rel_data)} relations x {N_MASS_DRAWS} mass draws "
          f"(N_REPEATS={N_REPEATS_SCAN})...")
    for d in rel_data:
        grid = np.full((N_CELLS, N_CELLS), np.nan)   # [radius_row, insol_col]
        for ri_ in range(N_CELLS):
            for ci in range(N_CELLS):
                cell_masses = d["mass"][d["flatA_mask"] & (d["fi"] == ci) & (d["ri"] == ri_)]
                if cell_masses.size == 0:
                    continue
                draws = rng.choice(cell_masses, size=N_MASS_DRAWS, replace=True)
                deltas = np.empty(N_MASS_DRAWS)
                for k, md in enumerate(draws):
                    inj = inject_row(nasa, md, radius_centers[ri_], insol_centers[ci], rel_mass_err, rel_rad_err)
                    nv_after, _ = S72.mc_nasa(inj, COLD_CUT, m_sil, r_sil, rng)
                    deltas[k] = tension(nv_after, d["sB"]) - d["tension0"]
                grid[ri_, ci] = deltas.mean()
        d["grid"] = grid
        print(f"  {d['name']:<24} done ({np.isfinite(grid).sum()}/{N_CELLS * N_CELLS} cells populated)")
    S72.N_REPEATS = N_REPEATS_FULL

    # ---- table ----
    rows = []
    stack = np.stack([d["grid"] for d in rel_data], axis=0)   # (relation, radius, insol)
    any_finite = np.isfinite(stack).any(axis=0)
    mean_grid = np.full(any_finite.shape, np.nan)
    mean_grid[any_finite] = np.nanmean(stack[:, any_finite], axis=0)
    for ri_ in range(N_CELLS):
        for ci in range(N_CELLS):
            row = dict(insol_center=insol_centers[ci], radius_center=radius_centers[ri_])
            for d in rel_data:
                row[f"delta_sigma_{d['name']}"] = d["grid"][ri_, ci]
            row["delta_sigma_mean"] = mean_grid[ri_, ci]
            rows.append(row)
    table = pd.DataFrame(rows)
    out_csv = os.path.join(OUT_DIR, "value_table.csv")
    table.to_csv(out_csv, index=False)

    top5 = table.dropna(subset=["delta_sigma_mean"]).nlargest(5, "delta_sigma_mean")
    print("\n--> top-5 highest-value cells (mean delta_sigma over relations):")
    for _, r in top5.iterrows():
        print(f"    insol={r['insol_center']:.2f} I⊕  radius={r['radius_center']:.2f} R⊕  "
              f"delta_sigma_mean={r['delta_sigma_mean']:.3f}")
    print("    (per-cell MC noise ~0.01-0.02 sigma: ordering among these cells is NOT "
          "significant; quote the high-value region, not the ranking)")

    # ---- figure ----
    vmax = np.nanmax(np.abs(stack)) if np.isfinite(stack).any() else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(15, 12.5), constrained_layout=True)
    X, Y = np.meshgrid(insol_edges, radius_edges)
    pcs = []
    for ax, d in zip(axes.flat, rel_data):
        pc = ax.pcolormesh(X, Y, d["grid"], cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat")
        pcs.append(pc)
        for ri_ in range(N_CELLS):
            for ci in range(N_CELLS):
                if not np.isfinite(d["grid"][ri_, ci]):
                    ax.add_patch(plt.Rectangle((insol_edges[ci], radius_edges[ri_]),
                                               insol_edges[ci + 1] - insol_edges[ci],
                                               radius_edges[ri_ + 1] - radius_edges[ri_],
                                               hatch="xx", fill=False, ec="0.75", lw=0))
        sel_m = is_m & has_teff
        sel_fgk = ~is_m & has_teff
        ax.scatter(nasa_teff["ins"][sel_m], nasa_teff["r"][sel_m], s=22, marker="o",
                   facecolor="tab:orange", edgecolor="k", linewidth=0.4, zorder=5, label="NASA (M dwarf host)")
        ax.scatter(nasa_teff["ins"][sel_fgk], nasa_teff["r"][sel_fgk], s=22, marker="^",
                   facecolor="tab:cyan", edgecolor="k", linewidth=0.4, zorder=5, label="NASA (FGK host)")
        ax.axvline(COLD_CUT["insol_max"], color="k", ls="--", lw=1.3, zorder=6)
        ax.axvspan(insol_edges[0], COLD_CUT["insol_max"], color="0.9", alpha=0.35, zorder=0)
        ax.text(COLD_CUT["insol_max"] * 0.85, radius_edges[-1] * 0.97, "Cold Rocky Desert\n(insol<50)",
               color="0.25", fontsize=8, ha="right", va="top")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(insol_edges[0], insol_edges[-1]); ax.set_ylim(radius_edges[0], radius_edges[-1])
        ax.set_title(d["name"], fontsize=12)
        ax.set_xlabel(r"insolation [$I_\oplus$]"); ax.set_ylabel(r"planet radius [$R_\oplus$]")
        ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)

    fig.colorbar(pcs[0], ax=axes, shrink=0.85, pad=0.02,
                label=r"$\Delta\sigma$ = change in overlap tension from confirming one planet in this cell")
    fig.suptitle("Value of a planet: change in the flat-A-vs-flat-B overlap tension per confirmed NASA planet\n"
                 "cold-corner tension, true mass drawn from the flat-A truth pool per cell; "
                 "hatched = flat A predicts no planets there\n"
                 "negative = the planet dilutes NASA's puffy fraction toward flat B (estimator effect, not negative evidence)\n"
                 f"cold corner: {n_corner_m} of {n_corner} precision-passing planets orbit M dwarfs "
                 f"(Teff<{MDWARF_TEFF_MAX:g} K); G/FGK-host cold super-Earths: currently N={n_corner_fgk}",
                 fontsize=12)
    out_png = os.path.join(OUT_DIR, "value_heatmap.png")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n--> Saved: {out_png}")
    print(f"--> Saved: {out_csv}")


if __name__ == "__main__":
    main()
