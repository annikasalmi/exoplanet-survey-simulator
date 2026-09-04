"""Figure 1 variant: radius floor 1.35 R_earth, precision cuts 10% radius / 30% mass,
Cold Corner drawn as a shaded band under the silicate curve instead of the red ellipse.

Reuses the NASA loading, quality filtering and silicate curve from
rocky_scatter_gaia60pc.py; only the three thresholds and the corner marker differ.
Writes to output/plots/47_fig1_relaxed_cuts/ only, never to paper/figures/.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "rocky_scatter_gaia60pc", ROOT / "important_plots" / "rocky_scatter_gaia60pc.py")
s15 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s15)

CORNER_RADIUS = 1.35
MAX_RADIUS_REL_UNCERTAINTY = 0.10
MAX_MASS_REL_UNCERTAINTY = 0.30

OUT_DIR = ROOT / "output/plots" / "47_fig1_relaxed_cuts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CORNER_COLOR = "#ff9ec4"
CORNER_LABEL = (r"Cold Rocky Desert ($R>%.2f\,R_\oplus$, $I<50$)" % CORNER_RADIUS)

XLIM = (0.0, 12.0)
LABEL_MODES = ["corner", "corner", "none"]


def draw_corner_band(ax, m_line, r_curve):
    ax.fill_between(m_line, CORNER_RADIUS, r_curve,
                    where=np.isfinite(r_curve) & (r_curve >= CORNER_RADIUS),
                    color=CORNER_COLOR, alpha=0.35, lw=0, zorder=1)


def corner_mask(panel, m_ref, r_ref):
    r_sil = np.interp(panel["mass_p"], m_ref, r_ref, left=np.nan, right=np.nan)
    return (panel["radius_p"] >= CORNER_RADIUS) & (panel["radius_p"] <= r_sil)


def label_selection(panel, m_ref, r_ref, mode):
    if mode == "corner":
        return panel[corner_mask(panel, m_ref, r_ref)]
    return panel.iloc[:0]


LABEL_X = 8.8
LABEL_Y_TOP = 1.80
LABEL_Y_BOTTOM = 0.78


def annotate_names(ax, sub):
    if len(sub) == 0:
        return
    order = sub.index[np.argsort(-sub["radius_p"].to_numpy())]
    ys = (np.linspace(LABEL_Y_TOP, LABEL_Y_BOTTOM, len(order)) if len(order) > 1
          else np.array([LABEL_Y_TOP]))
    for i, y in zip(order, ys):
        ax.annotate(sub.at[i, "planet_name"],
                    xy=(sub.at[i, "mass_p"], sub.at[i, "radius_p"]),
                    xytext=(LABEL_X, y), textcoords="data",
                    ha="left", va="center", fontsize=9, color="0.15", zorder=7,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="0.45",
                                    shrinkA=0, shrinkB=3))


def plot_panels(m_ref, r_ref, nasa_win, color_map, major, counts):
    m_line = np.linspace(XLIM[0] + 1e-3, XLIM[1], 600)
    r_curve = np.interp(m_line, m_ref, r_ref, left=np.nan, right=np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True,
                             constrained_layout=True)

    print("\nMass-radius insolation panels:")
    for (ax, (title, sel, draw_corner), mode) in zip(
            axes, s15.MR_INSOL_PANELS, LABEL_MODES):
        panel = nasa_win[sel(nasa_win["flux_p"].to_numpy())].copy()
        fac = panel["discovery_facility"].fillna("Unknown")
        labelled = label_selection(panel, m_ref, r_ref, mode)
        print(f"  {title:16s}  N={len(panel):3d}  "
              f"in corner={int(corner_mask(panel, m_ref, r_ref).sum()):2d}  "
              f"labelled({mode})={len(labelled)}: {list(labelled['planet_name'])}")

        if draw_corner:
            draw_corner_band(ax, m_line, r_curve)
        ax.plot(m_line, r_curve, "k--", lw=1.6, zorder=5)

        for f in major + [None]:
            sub = panel[fac == f] if f else panel[~fac.isin(major)]
            if len(sub) == 0:
                continue
            ax.errorbar(
                sub["mass_p"], sub["radius_p"],
                xerr=[sub["mass_err_minus"].abs().fillna(0).values,
                      sub["mass_err_plus"].abs().fillna(0).values],
                yerr=[sub["radius_err_minus"].abs().fillna(0).values,
                      sub["radius_err_plus"].abs().fillna(0).values],
                fmt="o", ms=5, color=color_map[f] if f else s15.OTHER_COLOR,
                alpha=0.9 if f else 0.75,
                elinewidth=0.7, capsize=2, ecolor="0.6", zorder=4 if f else 3,
            )

        annotate_names(ax, labelled)

        ax.set_xlim(*XLIM)
        ax.set_ylim(*s15.RADIUS_LIMITS)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel(r"Mass [$M_\oplus$]", fontsize=12)
        ax.grid(alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=11)

    axes[0].set_ylabel(r"Radius [$R_\oplus$]", fontsize=12)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[f],
               markersize=7, label=f"{s15.short_facility(f)}  (N={counts[f]})")
        for f in major
    ]
    n_other = int(len(nasa_win) - sum(counts[f] for f in major))
    if n_other > 0:
        handles.append(Line2D([0], [0], marker="o", linestyle="", color=s15.OTHER_COLOR,
                              markersize=7, label=f"Other facilities  (N={n_other})"))
    handles.append(Line2D([0], [0], color="black", lw=1.6, ls="--",
                          label=f"Silicate rocky curve ({s15.ROCKY_CURVE_PATH.name})"))
    handles.append(Patch(facecolor=CORNER_COLOR, alpha=0.35, label=CORNER_LABEL))

    fig.legend(handles=handles, loc="outside lower center",
               ncol=min(len(handles), 5), fontsize=10, framealpha=0.9,
               title="Observed planets — colored by discovery facility",
               title_fontsize=11)

    out = OUT_DIR / "rocky_mr_insolation_3panel_r135_prec10_30.png"
    fig.savefig(out, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")
    return out


def main():
    s15.MAX_MASS_REL_UNCERTAINTY = MAX_MASS_REL_UNCERTAINTY
    s15.MAX_RADIUS_REL_UNCERTAINTY = MAX_RADIUS_REL_UNCERTAINTY

    print("47_fig1_relaxed_cuts.py")
    print("=" * 70)
    print(f"radius floor  : {CORNER_RADIUS} R_earth")
    print(f"radius prec   : <= {MAX_RADIUS_REL_UNCERTAINTY:.0%}")
    print(f"mass prec     : <= {MAX_MASS_REL_UNCERTAINTY:.0%}")
    print(f"output dir    : {OUT_DIR}\n")

    m_ref, r_ref = s15.load_rocky_reference_curve()
    shift = s15.compute_rocky_threshold_shift(m_ref, r_ref)
    nasa_all, _ = s15.load_and_filter_nasa(m_ref, r_ref, shift)
    nasa_win = s15.restrict_to_window(nasa_all)
    print(f"\nprecision sample in window: N={len(nasa_win)}")

    color_map, major, counts = s15.build_facility_styles(nasa_win)
    plot_panels(m_ref, r_ref, nasa_win, color_map, major, counts)
    nasa_win.to_csv(OUT_DIR / "precision_sample_in_window.csv", index=False)


if __name__ == "__main__":
    main()
