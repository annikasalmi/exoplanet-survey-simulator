"""Original discrepancy display, evaluated from the shared population-comparison draws.

The saved sidecar retains the shared count likelihood and posterior calculation.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from science.population_comparison import BINS, ComparisonConfig, run_comparison
from tools.paths import PAPER_FIGURES_DIR
from tools.plotting_constants import PAPER_STYLE

N_DRAWS = ComparisonConfig().draws
LABELS = {"rocky_formation": "Sub-Neptune + Super-Earth",
          "escape_only": "Sub-Neptune"}

BIG_TEXT_STYLE = {
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 26,
}



def main():
    summary, arrays = run_comparison(ComparisonConfig(draws=N_DRAWS))
    plt.rcParams.update(PAPER_STYLE | BIG_TEXT_STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), layout="constrained")
    for i, (ax, (blabel, _, _)) in enumerate(zip(axes, BINS)):
        result = summary["bins"][blabel]
        n_obs = result["n_observed"]
        f_obs = result["k_observed"] / n_obs
        sig_obs = arrays[f"b{i}_observed"].std()
        if not np.isfinite(sig_obs) or sig_obs <= 0:
            raise ValueError(f"{blabel}: discrepancy display requires nonzero observed uncertainty")
        stats, points = {}, {}
        for key in ("rocky_formation", "escape_only"):
            fk = arrays[f"b{i}_{key}"]
            stats[key] = ((fk - f_obs) / sig_obs) ** 4
            points[key] = ((fk.mean() - f_obs) / sig_obs) ** 4
        # an n_obs-planet survey can only give x_k^(1/4) = m * step for integer m. Put edges
        # halfway between allowed values (so no bin is empty), merging neighbours until each
        # bin is >= 1/20 of the range (so the narrow cells near 0 don't spike the density).
        step = 1.0 / (n_obs * sig_obs)
        m_max = int(np.ceil(max(s.max() for s in stats.values()) ** 0.25 / step)) + 1
        lattice = ((np.arange(m_max) + 0.5) * step) ** 4
        min_width = lattice[-1] / 20
        edges = [0.0]
        for e in lattice:
            if e - edges[-1] >= min_width:
                edges.append(e)
        if edges[-1] < lattice[-1]:
            edges[-1] = lattice[-1]
        for key in ("rocky_formation", "escape_only"):
            c = "C0" if key == "rocky_formation" else "C1"
            ax.hist(stats[key], bins=edges, color=c, alpha=0.45, density=True)
            ax.axvline(points[key], color=c, ls="--", lw=3, label=LABELS[key])
            print(f"    {key:<16} x_k^2 point = {points[key]:.2f}  "
                  f"(draws: {stats[key].mean():.2f} +- {stats[key].std():.2f})")
        ax.set_xlim(left=-0.05)
        ax.set_title(f"{blabel} $I_\\oplus$\n" + r"$M>2.5\,M_\oplus$, $R>1.4\,R_\oplus$")
        ax.set_xlabel(r"$x_k^2 = (f_k - f_{\rm obs})^4\,/\,\hat\sigma_{\rm obs}^4$")
        ax.set_ylabel("Probability density")
        if i == 0:
            ax.legend(loc="upper right", handlelength=1.4)

    fig.suptitle("MCMC Volatile-Fraction Discrepancy Relative to Observations", fontsize=30)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = PAPER_FIGURES_DIR / f"mc_comparison_statistic_{N_DRAWS}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    out.with_name(out.stem+"_posterior.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
