"""50_rv_followup_scorecard.py — guidance product for RV follow-up of the cold-corner planets.

Question answered: for the cold (I<10 I_earth) super-Earths in the relaxed-cut census, is the
CURRENTLY published mass/radius precision good enough to decide rocky-vs-volatile, and if not,
what mass precision would a follow-up campaign need to reach?

Two payoffs are reported side by side because they point different ways:
  1. PER-PLANET composition payoff (large): can mass+radius classify THIS planet as rocky or
     volatile-rich? The acceptance line is the required-precision result of
     Plotnykov et al. 2024 (MNRAS): at ~2% radius error, iron-mass-fraction to 8 wt% needs
     5-11% mass precision; a water world's water content to 10 wt% needs 9-20% mass precision.
  2. POPULATION-test payoff (small): how much does pinning that one planet to 5% mass move the
     sigma gap of the cold-corner volatile-fraction test? Read from script 49's candidate_value.csv
     (the prior-wall result: the population test is prior-limited, so per-planet payoff is tiny).

This script is THIN: NASA values are read from the same PSCompPars file the pipeline uses; the
population payoff is read from script 49's output; the Plotnykov thresholds are literature
constants, flagged as such. Run script 49 first so candidate_value.csv exists.

Outputs (output/plots/50_rv_followup_scorecard/):
    followup_scorecard.csv, followup_scorecard.png

Run (repo root, PYTHONPATH set):
    python "scripts/statistical_analysis/50_rv_followup_scorecard.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NASA_FILE = (ROOT / "run" / "kepler" / "data" / "NASA"
             / "NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv")
CANDIDATE_VALUE_CSV = ROOT / "output/plots" / "49_targeted_precision_power" / "candidate_value.csv"
OUT_DIR = ROOT / "output/plots" / "50_rv_followup_scorecard"

# the cold (I<10) occupants of the relaxed-cut census; the two TOIs are TESS discoveries
COLD_INSOL = 10.0
COLD_PLANETS = ["LHS 1140 b", "TOI-1452 b", "LHS 1903 e", "TOI-198 b"]

# Plotnykov et al. 2024 (MNRAS) required-precision line, quoted at ~2% radius error.
# These are LITERATURE targets, not repo-derived numbers.
PLOTNYKOV_RADIUS_ASSUMED = 0.02
ROCKY_MASS_TARGET = 0.11    # <= this mass precision -> iron-mass-fraction to 8 wt%
WATER_MASS_TARGET = 0.20    # <= this mass precision -> water content to 10 wt%


def load_cold_planets():
    df = pd.read_csv(NASA_FILE, comment="#")
    cols = ["pl_name", "disc_facility", "pl_bmasse", "pl_bmasseerr1", "pl_bmasseerr2",
            "pl_rade", "pl_radeerr1", "pl_radeerr2", "pl_insol"]
    sub = df[df["pl_name"].isin(COLD_PLANETS)][cols].copy()
    for c in cols[2:]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    sub["mass_prec"] = np.maximum(sub["pl_bmasseerr1"].abs(), sub["pl_bmasseerr2"].abs()) / sub["pl_bmasse"]
    sub["rad_prec"] = np.maximum(sub["pl_radeerr1"].abs(), sub["pl_radeerr2"].abs()) / sub["pl_rade"]
    sub["is_toi"] = sub["pl_name"].str.startswith("TOI")
    return sub.sort_values("pl_insol").reset_index(drop=True)


def classify_status(mass_prec, rad_prec):
    """Can current mass+radius classify this planet's composition, per the Plotnykov line?"""
    if mass_prec <= ROCKY_MASS_TARGET and rad_prec <= PLOTNYKOV_RADIUS_ASSUMED * 3:
        return "classifiable"
    if mass_prec <= WATER_MASS_TARGET:
        return "volatile-only"      # can bound water content but not iron-mass-fraction
    return "needs follow-up"


def load_population_payoff():
    if not CANDIDATE_VALUE_CSV.exists():
        print(f"  WARNING: {CANDIDATE_VALUE_CSV} not found; run script 49 first. "
              f"Population payoff left as NaN.")
        return {}
    v = pd.read_csv(CANDIDATE_VALUE_CSV)
    v = v[v["universe"] == "P-Pop"]
    return dict(zip(v["candidate"], v["dT_primordial"]))


def build_table():
    sub = load_cold_planets()
    payoff = load_population_payoff()
    rows = []
    for _, r in sub.iterrows():
        rows.append(dict(
            planet=r["pl_name"], is_toi=bool(r["is_toi"]), insol=r["pl_insol"],
            mass=r["pl_bmasse"], mass_prec=r["mass_prec"], rad=r["pl_rade"], rad_prec=r["rad_prec"],
            status=classify_status(r["mass_prec"], r["rad_prec"]),
            mass_target_rocky=ROCKY_MASS_TARGET, mass_target_water=WATER_MASS_TARGET,
            shrink_factor_to_rocky=r["mass_prec"] / ROCKY_MASS_TARGET,
            dT_population=payoff.get(r["pl_name"], np.nan),
        ))
    return pd.DataFrame(rows)


def make_figure(tab):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                         "legend.fontsize": 9})
    STATUS_COLOR = {"classifiable": "tab:green", "volatile-only": "tab:orange",
                    "needs follow-up": "tab:red"}
    order = tab.sort_values("mass_prec")
    y = np.arange(len(order))
    labels = [f"{p}{'  (TOI)' if t else ''}" for p, t in zip(order.planet, order.is_toi)]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.2))

    # Panel A: current mass precision vs the required-precision line
    for yi, (_, r) in zip(y, order.iterrows()):
        c = STATUS_COLOR[r["status"]]
        axA.barh(yi, r["mass_prec"] * 100, color=c, alpha=0.85, height=0.55, zorder=3)
        axA.text(r["mass_prec"] * 100 + 0.6, yi, f"{r['mass_prec']*100:.0f}%",
                 va="center", ha="left", fontsize=9, color="0.2")
    lr = axA.axvline(ROCKY_MASS_TARGET * 100, color="tab:green", ls="--", lw=1.4, zorder=2,
                     label="rocky line ($\\leq$11% mass)")
    lw = axA.axvline(WATER_MASS_TARGET * 100, color="tab:orange", ls=":", lw=1.6, zorder=2,
                     label="water line ($\\leq$20% mass)")
    axA.set_yticks(y); axA.set_yticklabels(labels)
    axA.set_xlabel("current mass precision  $\\sigma_M/M$  [%]")
    axA.set_xlim(0, max(35, order.mass_prec.max() * 100 + 8))
    axA.set_ylim(-0.6, len(order) - 0.4)
    axA.set_title("A. Is the mass good enough to classify composition?\n"
                  "(required-precision line: Plotnykov et al. 2024, at ~2% radius)", fontsize=10.5)
    axA.grid(axis="x", alpha=0.25)
    status_handles = [plt.Rectangle((0, 0), 1, 1, color=STATUS_COLOR[k]) for k in STATUS_COLOR]
    axA.legend(status_handles + [lr, lw],
               list(STATUS_COLOR.keys()) + ["rocky line ($\\leq$11%)", "water line ($\\leq$20%)"],
               loc="lower right", framealpha=0.9, fontsize=8)

    # Panel B: population-test payoff of pinning this planet to 5% mass (prior wall)
    payoff = order.dropna(subset=["dT_population"])
    if len(payoff):
        yb = np.arange(len(payoff))
        cols = ["tab:blue" if v >= 0 else "tab:red" for v in payoff.dT_population]
        axB.barh(yb, payoff.dT_population, color=cols, alpha=0.85, height=0.55)
        axB.set_yticks(yb)
        axB.set_yticklabels([f"{p}{'  (TOI)' if t else ''}"
                             for p, t in zip(payoff.planet, payoff.is_toi)])
        axB.axvline(0, color="k", lw=0.8)
        axB.set_xlabel("change in cold-corner $\\sigma$ gap\nfrom pinning this planet to 5% mass")
    axB.set_title("B. Population-test payoff (P-Pop): tiny\n"
                  "the rocky-vs-volatile census is prior-limited (script 49 prior wall)",
                  fontsize=10.5)
    axB.grid(axis="x", alpha=0.25)

    fig.suptitle("RV follow-up scorecard for the cold (I<10) super-Earths: "
                 "per-planet composition is worth chasing; the population test is not",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "followup_scorecard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tab = build_table()
    pd.set_option("display.width", 140, "display.max_columns", 20)
    print("=" * 92)
    print("RV FOLLOW-UP SCORECARD — cold (I<10) super-Earths in the relaxed-cut census")
    print("=" * 92)
    show = tab.copy()
    show["mass_prec"] = (show["mass_prec"] * 100).round(0)
    show["rad_prec"] = (show["rad_prec"] * 100).round(0)
    show["dT_population"] = show["dT_population"].round(3)
    show["shrink_factor_to_rocky"] = show["shrink_factor_to_rocky"].round(1)
    print(show[["planet", "is_toi", "insol", "mass", "mass_prec", "rad_prec",
                "status", "shrink_factor_to_rocky", "dT_population"]].to_string(index=False))
    tab.to_csv(OUT_DIR / "followup_scorecard.csv", index=False)
    print(f"\n  Saved: {OUT_DIR / 'followup_scorecard.csv'}")
    make_figure(tab)
    print("\n--> done.")


if __name__ == "__main__":
    main()
