---
name: rv-joint-verdict-script55
description: Script 55 joint-completeness verdict — cold rocky radius limit survives full transit×RV correction
metadata:
  type: project
---

`scripts/55_rv_joint_completeness_verdict.py` is the capstone of the RV half (2026-06): it
re-runs the project's central Q90 cold-vs-hot test under the professor's joint completeness
η = η_transit × η_RV, plus a forward-model yield test. Reuses S44 (NASA rocky sample, 90% line)
and S53 (rocky P-Pop + RVData verdicts).

**FINAL result (all 401 catalogs):**
- TEST 1 (Q90, one-sided permutation H1: Q90_cold < Q90_hot):
  unweighted p=0.0000 → 1/η_transit p=0.0000 → **1/η_joint p=0.0074**. The "cold rocky planets
  are smaller" signal SURVIVES the full selection correction (delta −0.50→−0.36 R⊕;
  Q90 cold 1.25 vs hot 1.61).
- TEST 2 (yield, scaled to hot region, Poisson):
  cold all-radii: observed 11 vs expected 5.2 → NO deficit (excess; dedicated M-dwarf programs
  oversample cold targets relative to hot-region scaling — a caveat, not a bug).
  **red window (cold AND large): observed 1 (LHS 1140 b) vs expected 5.1 → p=0.038 deficit.**
  The deficit is specifically in RADIUS at low insolation, not in cold counts.
- Both independent tests agree at p<0.05: the cold rocky upper radius limit is NOT explained
  by transit+RV selection — evidence for a real astrophysical bound.

**DETOUR + CORRECTION (2026-06-15→16):** I briefly added a correlated-jitter floor
(jitter_red_frac=0.5) to be "conservative." User intuition flagged the RV detector as way too
strict (flat-control RV panel only 4.9% detected, dark even in the high-K corner). DIRECT
CALIBRATION proved them right: at f=0.5 the detector recovered only **10.8%** of 342 real
small-planet RV detections; at **f=0 it recovers 57.6%, matching the 58.5%** actually published
at ≥5σ (see [[rv-survey-model-decision]] for the acid test). **Reverted to f=0** — it is correctly
calibrated, not optimistic. Lesson: the Q90/yield tests are SELF-NORMALIZING and cannot detect an
overall-completeness miscalibration — always validate the detector against real recovered planets,
not against the self-normalizing science tests.

**FINAL numbers, calibrated detector f=0 (2026-06-16):** η_joint median=0.42 (healthy).
- TEST 1 (Q90, 1/η_joint): **p=0.0077** (cold 1.13 vs hot 1.88 R⊕). Two-survey (script 57):
  Kepler p≈0.000, TESS p=0.0069.
- TEST 2 (yield): red window FGKM E=44 O=1 **p<0.0001**; M-stars E=5.6 O=1 **p=0.024**.
- CAVEAT (new): the 1/η_transit-only weighting is NON-MONOTONIC (p=0.43, vs unweighted 0.000 and
  1/η_joint 0.0077) — transit-only correction is unstable here (MAX_WEIGHT cap + sparse cells);
  the physically complete 1/η_joint is the one to trust and it is significant. Worth understanding.

**Calibration backing** (script 54): K_model/K_published median 1.001 (±2%, N=2,510 vs pl_rvamp).
NOTE the all-planet σ_K ratio (0.19) is giants-dominated/misleading; small-planet σ_K matches
published (0.27 vs 0.31 m/s).
Cache: `run/kepler/data/NASA/NASA_PSCompPars_rvamp_calibration.csv` (has pl_rvamp/msinie/orbeccen).

**Caveats to repeat in any writeup:** P-Pop occurrence is the null's cold/hot prior; 90% line is
fit mostly on hot planets and extrapolated; η grids FGKM-combined; hot-region scaling removes
normalization but not shape errors; cold "excess" reflects targeted M-dwarf follow-up effort.

Related: [[rv-survey-model-decision]], [[rv-rocky-redwindow-script53]].
