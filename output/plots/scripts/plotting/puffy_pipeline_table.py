"""
64_puffy_pipeline_table.py — clean one-page table explaining, step by step, how the
PUFFY-FRACTION normal distribution is built (model universes vs NASA). Styled to match the
"Planet Generation" table: light header, green NASA column, thin rules, short phrases.

Run:
    python scripts/64_puffy_pipeline_table.py
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

from tools.paths import LIFESIM_OUTER_DIR
ROOT = Path(LIFESIM_OUTER_DIR)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(ROOT, "output/plots", "64_puffy_pipeline_table")
TITLE = "Puffy-Fraction Distribution — Build Pipeline"
HEAD = ("Step", "What we do  (flat / P-Pop universes)", "NASA — what's different")

# (step, model logic, NASA difference)
ROWS = [
    ("1 · Population",
     "Build the universe pool; box-restrict R∈[0.5,2.2] R⊕, M∈[0.1,12] M⊕. Tag PUFFY if "
     "R > R_silicate(M), else ROCKY. Universe A drops rocky M>2 M⊕; B keeps all.",
     "Real confirmed + transiting planets with a MEASURED mass (RV/TTV), passing the "
     "precision test (frac mass err ≤25%, radius ≤8%): 292 → ~125 survive. No pool built."),

    ("2 · Detect\n(selection)",
     "Keep planets passing TRANSIT (Kepler) AND RV (best HARPS/NIRPS). This sets which planets "
     "are observable.",
     "Already observed — no detector run. The sample is itself transit+RV-selected."),

    ("3 · Sample\n(one draw)",
     "Draw 20,000 planets with replacement; keep the detected ones → N_det survive (~120–250).",
     "No resample: keep the FULL fixed precision-passing set (~125) every universe."),

    ("4 · Add noise\n(one draw)",
     "Observe each: m_obs = M·exp(Normal(0, 0.20)),  r_obs = R·exp(Normal(0, 0.046)). "
     "Sizes = NASA median errors (mass 20%, radius 4.6%).",
     "Perturb each planet by its OWN published ± error bar (asymmetric)."),

    ("5 · Classify\n(one draw)",
     "PUFFY if r_obs > R_silicate(m_obs), else ROCKY.",
     "Same rule, on the perturbed values."),

    ("6 · One number\n(one draw)",
     "Puffy fraction for this draw:  p = N_puffy / N_det.",
     "p = N_puffy / 292."),

    ("7 · Repeat",
     "Redo steps 3–6  ten thousand times → 10,000 values of p. Their wobble IS the bell's width.",
     "Re-perturb the SAME fixed set 10,000 times → 10,000 values of p (no resample)."),

    ("8 · Fit μ, σ",
     "μ = mean(p),  σ = std(p).   By the CLT:  σ ≈ √(p(1−p) / N_det).",
     "μ = mean(p), σ = std(p). Narrower than a bootstrap: only measurement error moves p, "
     "not sampling of which planets are in the set (the 'real' universe is one fixed set)."),

    ("9 · Result ✓",
     "Bell = Normal(μ, σ). Closest universe to NASA = smallest |μ_model − μ_NASA| / √(σ² + σ²).",
     "The green reference bell every model is compared against."),
]

W_STEP, W_MODEL, W_NASA = 14, 74, 54
X0, X1, X2, XR = 0.012, 0.150, 0.620, 0.992
GREEN = "#e3efdc"
GREEN_HEAD = "#cfe0c2"
GRAY_HEAD = "#f1f2f4"
LINE = "#c9cdd4"
PAD = 0.85
UNIT_IN = 0.30


def wrap(s, w):
    out = []
    for part in s.split("\n"):
        out += textwrap.wrap(part, w) or [""]
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for step, model, nasa in ROWS:
        s, m, n = wrap(step, W_STEP), wrap(model, W_MODEL), wrap(nasa, W_NASA)
        rows.append((s, m, n, max(len(s), len(m), len(n))))

    title_h, head_h = 1.7, 1.5
    total = title_h + head_h + sum(nl + PAD for *_, nl in rows)
    fig, ax = plt.subplots(figsize=(15.5, total * UNIT_IN))
    ax.set_xlim(0, 1); ax.set_ylim(0, total); ax.axis("off")

    y = total
    ax.text(X0, y - 1.05, TITLE, fontsize=16, fontweight="bold", color="#222")
    y -= title_h
    table_top = y

    # green NASA column spans header + body
    ax.add_patch(plt.Rectangle((X2 - 0.012, 0), 1 - (X2 - 0.012), table_top, color=GREEN, zorder=0))
    # header bands
    ax.add_patch(plt.Rectangle((0, y - head_h), X2 - 0.012, head_h, color=GRAY_HEAD, zorder=0.5))
    ax.add_patch(plt.Rectangle((X2 - 0.012, y - head_h), 1 - (X2 - 0.012), head_h, color=GREEN_HEAD, zorder=0.5))
    hy = y - head_h / 2
    ax.text(X0, hy, HEAD[0], fontsize=12, fontweight="bold", va="center", color="#222")
    ax.text(X1, hy, HEAD[1], fontsize=12, fontweight="bold", va="center", color="#222")
    ax.text(X2, hy, HEAD[2], fontsize=12, fontweight="bold", va="center", color="#2c4a23")
    y -= head_h
    ax.plot([0, 1], [y, y], color="#8a909c", lw=1.1, zorder=2)

    for s, m, n, nl in rows:
        rh = nl + PAD
        top = y - 0.55
        for i, ln in enumerate(s):
            ax.text(X0, top - i, ln, fontsize=10.6, fontweight="bold", va="center", color="#1d2330")
        for i, ln in enumerate(m):
            ax.text(X1, top - i, ln, fontsize=10.2, va="center", color="#15191f")
        for i, ln in enumerate(n):
            ax.text(X2, top - i, ln, fontsize=10.0, va="center", color="#22341a")
        y -= rh
        ax.plot([0, 1], [y, y], color=LINE, lw=0.7, zorder=2)

    ax.plot([0, 1], [table_top, table_top], color="#8a909c", lw=1.1, zorder=2)
    out_png = os.path.join(OUT_DIR, "puffy_fraction_pipeline.png")
    fig.savefig(out_png, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"--> Saved: {out_png}")


if __name__ == "__main__":
    main()
