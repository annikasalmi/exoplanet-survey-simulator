"""
52_mc_mr_density_priors.py — 2x3 mass-radius probability-density map built from 4,000 Monte
Carlo noise realizations of each of the two priors in 41_bayesian_cold_rocky_desert.py, so the
two priors' predicted mass-radius densities (with realistic measurement scatter) can be
compared side by side, panel by panel.

Same noise model as predicted_frac() in 41_bayesian_cold_rocky_desert.py (the function that
produces every f_k in Table 1 of the note): take the universe's already joint-detected
(transit+RV) planets in an insolation panel, apply the same M>2 Mearth super-Earth cut Table 1's
panels use, and on each of N_MC draws perturb every survivor's mass and radius by independent
log-normal noise (MASS_FRAC_ERR, RAD_FRAC_ERR). Stacking all N_MC noised draws into one 2D
histogram and dividing by (draws x cell area) gives a probability density on the mass-radius
plane that integrates to 1 over the box — this script does that instead of collapsing each draw
to the single volatile-fraction number Table 1 reports, so the same Monte Carlo procedure that
produced f_esc and f_prim is shown directly as a picture. Escape-only's defining population is
small (M<2) stripped-rocky planets, which the M>2 cut removes from view here, same as it removes
them from Table 1 — the cut keeps this figure an apples-to-apples match to the table, not a
complete picture of what each prior contains.

Rows = the two priors (rocky_formation, escape_only). Columns = the three insolation panels
(I<10, I<50, I>50), same as Figure 1/2 of the note. No Bayes factor or odds on this figure;
it is a direct density picture, not a model-comparison statistic.

Run:
    python scripts/Statistical_Analysis/52_mc_mr_density_priors.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter

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


S41 = _load("s41", str(ROOT / "scripts" / "Statistical_Analysis" / "41_bayesian_cold_rocky_desert.py"))

OUT_DIR = os.path.join(ROOT, "my_outputs", "52_mc_mr_density_priors")
N_MC = 4000        # Monte Carlo noise realizations per prior per panel
BATCH = 50         # draws per histogram2d call, bounds peak memory
RNG_SEED = 1        # independent of the note's RNG_SEED=0, so this figure is not entangled
                    # with the table numbers it visualizes


def mc_density(pool, lo, hi, rng, mass_min=None, n_mc=N_MC, batch=BATCH):
    """Stack n_mc noise-perturbed draws of the panel's joint-detected planets into a 2D
    histogram on the (M_EDGES, R_EDGES) grid, then normalize to a density integrating to 1.
    mass_min, if given, cuts on the NOISED mass each draw, exactly as predicted_frac() does in
    41_bayesian_cold_rocky_desert.py — boundary planets can flip in/out of the cut draw to draw,
    the same way they can flip in/out of Table 1's f_k."""
    sel = pool["det"] & (pool["flux"] >= lo) & (pool["flux"] < hi)
    m0, r0 = pool["mass"][sel], pool["radius"][sel]
    n = m0.size
    accum = np.zeros((S41.M_EDGES.size - 1, S41.R_EDGES.size - 1))
    done = 0
    while done < n_mc:
        b = min(batch, n_mc - done)
        mo = np.repeat(m0[None, :], b, axis=0) * np.exp(
            rng.normal(0.0, S41.MASS_FRAC_ERR, size=(b, n)))
        ro = np.repeat(r0[None, :], b, axis=0) * np.exp(
            rng.normal(0.0, S41.RAD_FRAC_ERR, size=(b, n)))
        if mass_min:
            keep = mo > mass_min
            mo, ro = mo[keep], ro[keep]
        H, _, _ = np.histogram2d(mo.ravel(), ro.ravel(), bins=[S41.M_EDGES, S41.R_EDGES])
        accum += H
        done += b
    cell_area = (S41.M_EDGES[1] - S41.M_EDGES[0]) * (S41.R_EDGES[1] - S41.R_EDGES[0])
    total = accum.sum()
    density = accum / (total * cell_area) if total > 0 else accum
    return density, n


def fig_mc_density(univ, nasa, m_sil, r_sil, rng):
    ms = np.linspace(S41.BOX["m_lo"], S41.BOX["m_hi"], 250)
    r_sil_line = np.interp(ms, m_sil, r_sil)
    norm = Normalize(vmin=np.log10(S41.BOX["f_lo"]), vmax=np.log10(S41.BOX["f_hi"]))
    X, Y = np.meshgrid(S41.M_CENT, S41.R_CENT)

    densities = {}
    for key, _, _ in S41.UNIVERSES:
        for lbl, lo, hi in S41.INSOL_BINS:
            D, n = mc_density(univ[key], lo, hi, rng, mass_min=S41.MASS_MIN)
            densities[(key, lbl)] = D
            print(f"    {key:<16} {lbl:<8} detected-in-panel n={n:>7}  {N_MC} MC draws  (M>{S41.MASS_MIN})")

    fig, axes = plt.subplots(len(S41.UNIVERSES), 3, figsize=(17, 9.5),
                             sharex=True, sharey=True, constrained_layout=True)
    sc = None
    for j, (blabel, lo, hi) in enumerate(S41.INSOL_BINS):
        vmax = max(densities[(key, blabel)].max() for key, _, _ in S41.UNIVERSES)
        for i, (key, plabel, _) in enumerate(S41.UNIVERSES):
            ax = axes[i, j]
            D = densities[(key, blabel)]
            ax.imshow(D.T, origin="lower",
                      extent=[S41.BOX["m_lo"], S41.BOX["m_hi"], S41.BOX["r_lo"], S41.BOX["r_hi"]],
                      aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                      interpolation="bilinear", zorder=0)
            sm = gaussian_filter(D, 0.8)
            levels = np.linspace(0.2 * vmax, 0.8 * vmax, 3)
            if vmax > 0:
                ax.contour(X, Y, sm.T, levels=levels, colors="white", linewidths=0.7,
                          alpha=0.7, zorder=2)
            ax.plot(ms, r_sil_line, "k--", lw=1.3, zorder=3)
            s = S41._nasa_overlay(ax, nasa, lo, hi, norm, mass_min=S41.MASS_MIN)
            if s is not None:
                sc = s
            ax.set_xlim(S41.BOX["m_lo"], S41.BOX["m_hi"])
            ax.set_ylim(S41.BOX["r_lo"], S41.BOX["r_hi"])
            if i == 0:
                ax.set_title(f"{blabel} $I_\\oplus$", fontsize=12)
            if j == 0:
                ax.set_ylabel(f"{plabel}\n" + r"radius [$R_\oplus$]", fontsize=11)
            if i == len(S41.UNIVERSES) - 1:
                ax.set_xlabel(r"planet mass [$M_\oplus$]")
            cb = fig.colorbar(ax.images[0], ax=ax, location="right", shrink=0.85, pad=0.02)
            cb.ax.tick_params(labelsize=7)

    if sc is not None:
        cb2 = fig.colorbar(sc, ax=axes, location="bottom", shrink=0.4, pad=0.02, aspect=50)
        cb2.set_label(r"NASA planets: log(Insolation Flux [$I_\oplus$])")
    fig.suptitle(f"Mass-radius probability density from {N_MC:,} Monte Carlo noise draws per "
                 "prior (log-normal mass/radius measurement noise, same procedure as Table 1's "
                 f"$f_k$), $M>{S41.MASS_MIN:.0f}\\,M_\\oplus$, vs confirmed NASA planets\n"
                 "rows = priors; columns = insolation panels; color scale shared within each "
                 "column and normalized to integrate to 1 over the box; black dashed = silicate "
                 "line", fontsize=12)
    out = os.path.join(OUT_DIR, "mc_mr_density_priors.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"--> Saved: {out}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    m_sil, r_sil = S41.load_silicate()
    rng = np.random.default_rng(RNG_SEED)
    nasa = S41.load_nasa(precision=True)

    print("--> loading cached universes (rocky_formation, escape_only) ...")
    univ = S41.build_universes(m_sil, r_sil)

    print(f"--> {N_MC:,} MC noise draws per prior per panel ...")
    fig_mc_density(univ, nasa, m_sil, r_sil, rng)


if __name__ == "__main__":
    main()
