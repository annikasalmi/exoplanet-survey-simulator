---
name: rv-robustness-tests-56-57-58
description: Three robustness tests on the cold rocky radius deficit (occurrence prior, Fisher Kepler+TESS, TTV channel)
metadata:
  type: project
---

Follow-ups to [[rv-joint-verdict-script55]], stress-testing the cold-large rocky red-window
deficit (script 55: obs 1 vs exp 5.1, p_red=0.038). All reuse S55/S53/S44 machinery.

**Script 56 — occurrence-prior robustness** (`56_yield_occurrence_prior_robustness.py`).
Reweights P-Pop from its generator prior (empirical (lnR,lnP) density = Bergsten/Dressing)
to analytic SAG13 baseline/opt/pess via importance weights, re-runs the hot-scaled Poisson
yield. FULL 401-cat result: p_red = generator 0.038, SAG13 baseline 0.080, opt 0.075, pess 0.090.
→ Deficit is robust: stays marginal-significant (~0.04–0.09) under every prior; SAG13 predicts
slightly more cold-large planets (E_red 4.0–4.3 vs 5.1) so slightly weaker but same direction.

**Script 57 — Kepler + TESS two-survey agreement** (`57_fisher_kepler_tess_joint.py`).
REWRITTEN 2026-06-15: Fisher chi-square was too opaque for the user. Headline is now a
DUMBBELL plot — per survey, Q90(cold) vs Q90(hot) dots with bootstrap CIs, shown RAW vs
1/η_joint-corrected. Cold sits below hot for BOTH surveys and survives correction = the
robustness, no combined statistic needed. Output `kepler_tess_cold_vs_hot_dumbbell.png`.
Conservative-detector FULL run: 1/η_joint Kepler p=0.0009, TESS p=0.0005 (Fisher kept only as a
printed footnote ≈0.0000, NOT featured). This is the corroboration of script 55 TEST 1 (Q90),
the robust test. CAVEAT: same NASA sample; independence is in the completeness lens (like script 41).

**Script 58 — TTV mass channel** (`58_ttv_mass_channel.py`).
Adds ttv_ok (mass measurable via TTV) for detected transiting pairs near first-order MMR
(Δ≤0.05, Lithwick+2012 amplitude ≥3 min). mass_measurable = rv_ok OR ttv_ok.
Result (FULL 401-cat): RV-only p_red=0.038 → RV-or-TTV p_red=0.039 (E_red 5.1→5.0; TTV-only
E_red=0.0). Detected near-resonant transiting pairs are rare in this nearby volume-limited
P-Pop, and the few TTV-measurable planets sit in the HOT region (short periods, more transits),
so adding TTV does NOT fill the cold-large red window. → The RV-only deficit is a conservative
bound. CAVEAT: toy TTV, generous Δ/amplitude.

Bottom line: all three independent stress tests leave the cold-large rocky radius deficit
intact — it is not an artifact of the occurrence prior, single-survey choice, or the RV-only
mass channel.
