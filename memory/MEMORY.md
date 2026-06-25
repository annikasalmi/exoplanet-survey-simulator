# Memory index

- [RV survey model decision](rv-survey-model-decision.md) — why the RV detector models HARPS/HARPS-N + locked constants
- [P-Pop stellar mass is placeholder](ppop-stellar-mass-is-placeholder.md) — mass_s is constant 1 M☉ in kg; derive M★ from radius
- [Rocky RV red-window (script 53)](rv-rocky-redwindow-script53.md) — rocky-planet RV-test on script-44 figure; cold limit is a major RV artifact
- [Joint-completeness verdict (script 55)](rv-joint-verdict-script55.md) — Q90 + yield tests under η_transit×η_RV (calibrated detector f=0): Q90 1/η_joint p=0.0077, yield red-window p<0.0001; validate detector against REAL recovered planets, not self-normalizing tests
- [Robustness tests 56-58](rv-robustness-tests-56-57-58.md) — occurrence-prior swap, Fisher Kepler+TESS (p=0.0004), TTV channel: cold rocky deficit robust
- [Flat control generator](uniform-box-control-generator.md) — fully-flat synthetic catalogue (all params independent, no Gaia/P-Pop priors) to map Kepler/TESS/RV detection edges (uniform_generator + flat_detect + scripts 30-34)
- [Kepler/TESS detector calibration](kepler-tess-detector-calibration.md) — toy MES/SNR ran hot vs official; factors Kepler 0.84 / TESS 0.66 (threshold-anchored) added + baked into catalogs via run/apply_detection_calibration.py; watch the script-48 tmag bug
- [Cold-completeness reconciliation](cold-completeness-reconciliation.md) — why 1.3% vs 62% isn't a conflict (P_geom×P_conf|transit), puffy-gap = mostly RV selection, mentor verdict on scripts 56-63 (scripts 67/69)
- [Prefers simple vanilla methods](prefers-simple-vanilla-methods.md) — keep puffy-fraction normal as headline; don't push ABC/Bayes/energy-distance; density is the only vanilla cross-check
- [P-Pop generation bottleneck](ppop-generation-bottleneck.md) — catalogue gen time (~5–14 h/10^6) is dominated by Chen2017/Forecaster mass model (~78%), NOT the planet distribution (Bergsten vs SAG13 <1%) or fsolve geometry
- [G/K background top-up](gk-background-topup.md) — run/generate_gk_8000.py adds G/K-only Gaia-60pc universes (files from 8001) to reach 8000 transiting/type; ~113 G + ~303 K per universe; script 44 now stacks 8001+
