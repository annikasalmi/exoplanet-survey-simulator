---
name: uniform-box-control-generator
description: Fully-flat synthetic control catalogue for stress-testing Kepler/TESS/RV detectors
metadata:
  type: project
---

A fully-FLAT synthetic planet catalogue exists for mapping detector sensitivity
with no astrophysical prior baked in (NOT P-Pop: no Gaia stars, no occurrence
rates, no Chen2017 M-R, no P-Pop orbit model).

- `run/ppop/uniform_generator.py` — `generate_flat_catalog(n_planets, seed, ...)` and
  `get_or_build_catalog(cache_path, ...)`. Every parameter drawn flat & independent within
  P-Pop bounds: radius_p U[0.5,2.2], mass_p logU[0.1,12] (independent of radius -> M-R scatter),
  p_orb logU[0.5,20000], ecc_p U[0,0.9], inc isotropic, teff_s U[2300,9700], distance_s U[1,60].
  Derived (fundamental physics, not priors): radius_s/mass_s from teff via MS interp, l_sun=R²(T/5772)⁴,
  semimajor from Kepler's 3rd law, flux_p = l_sun/a² (insolation COUPLED to orbit, rejection-bounded to
  [1e-2,1e4]). Caches to `my_outputs/uniform_box/flat_catalog.csv`.
- `run/ppop/flat_detect.py` — `run_kepler/run_tess/run_rv/run_all`; loads the 3 detector .py files via
  importlib so lifesim's Qt import is never triggered (sandbox Python 3.13 lacks PyQt5).
- Scripts: `30/31/32` per-detector caveman tests; `33_flat_detfrac_insolation_radius.py`
  (1×3 Kepler/TESS/RV detected-fraction on radius-insolation, no FGKM split, denominator = observable
  set = transiting for K/T, bright-enough for RV — same as script 44); `34_flat_mass_radius_by_insolation.py`
  (per-detector 1×3 M-R panels split I<10 / 10-50 / ≥50, detected planets coloured by insolation,
  black-dashed pure-rock theory curve `run/kepler/reference_curves/ref.ddat`);
  `37_flat_detfrac_vs_radius_and_mass.py` (detection FRACTION vs radius and vs mass, one curve per
  detector — the unambiguous view: transit rises with radius/flat in mass, RV rises with mass/flat in radius).

Two-universe rocky/puffy distinguishability (silicate boundary Hongyi-silicon.ddat; above=puffy, below=rocky):
- `49_two_universe_rocky_puffy_overlap.py` — puffy-only (A) vs rocky+puffy (B), detected by transit(Kepler) AND
  RV(best HARPS+NIRPS), MC sampling distributions of scalar discriminators + OVL/AUC. Result: FLAT trivially
  separable (independent M-R overproduces dense RV-easy rocky planets — an artifact); PPop realistic: # detected
  is indistinguishable (OVL 0.85) but the detected-radius shape separates. mean-radius separation is partly
  tautological (puffy≡larger radius); density-offset is the honest discriminator.
- `51_flat_puffy_rocky_ratio_vs_nasa.py` — flat only, M>2 M⊕, three ratio distributions (puffy_frac,
  total/puffy, rocky/puffy) for universe B vs NASA bootstrap, with A=puffy-only as reference line.
  Result: NASA puffy_frac≈0.43 firmly REJECTS puffy-only (predicts 1.0) but sits ~2-3σ off flat-B (≈0.27 —
  flat is too rocky-heavy because independent M-R overcreates dense planets). The 3 ratios are monotonic
  transforms of each other (same OVL).
- `56_flat_puffy_rocky_ratio_vs_nasa.py` — single puffy-fraction metric, flat+P-Pop universes A/B vs NASA
  bootstrap, now in 1×3 panels (transit-only / RV-only / transit+RV). Each bell = N_REPEATS draws of
  N_SAMPLE planets (default 10000×20000); bell width is pure sampling spread σ=√(p(1−p)/N_det), verified
  predicted==measured to 4 dp. p is the EMERGENT detected puffy fraction (puffy.mean()), never hardcoded.
  KEY per-detector result (undetected→detected puffy frac): transit barely moves it (flat B 0.59→0.62,
  ppop B 0.44→0.43); RV moves it OPPOSITELY for the two universes — flat B 0.59→0.28 (independent M-R makes
  many heavy-but-small rocky planets RV loves) vs ppop B 0.44→0.52 (correlated M-R ties high mass to large
  radius); transit+RV gives flat B 0.32, ppop B 0.51, with NASA (0.40) BRACKETED between them. RV is the
  real discriminator between independent vs correlated M-R. N_det shrinks 20000→~120-240 under both, so the
  detected bells widen to ~0.03 (≈ NASA's 0.029).
- `56b_pool_param_diagnostics.py` — WHY transit favours flat but RV favours P-Pop. ROOT CAUSE is the
  STELLAR sample, not the planets: flat draws Teff uniform[2300,9700] (median ~6230 K, only 16% M dwarf,
  median R*=1.21 Rsun) while P-Pop/Gaia is 86% M dwarf (median R*=0.30 Rsun). Transit favours flat —
  geometric transit prob ∝ R*, so flat 4.5% vs ppop 2.2% transiting; but detect-given-transit is ~93% for
  BOTH (ppop's tiny stars give 1480 vs 108 ppm depth, cancelling flat's bigger planets), so SNR is NOT the
  lever, geometry is. RV favours P-Pop — K∝Mp·M*^-2/3·P^-1/3 is ~5× higher for ppop (M-dwarf host ~2.2×,
  shorter period ~1.45×, higher Forecaster mass ~1.7×); flat also wastes 34% of planets at <0.5 Me where RV
  is blind. Same stellar property, OPPOSITE sign: transit likes big stars, RV likes small stars. Uniform-Teff
  over-weights hot stars vs a realistic volume sample, so the detection split is partly a sampling artifact —
  sample stars from a real MF and it largely vanishes.
- `57_universe_metrics_vs_nasa.py` — honest "which universe matches OBSERVED?" test: forward-models each
  universe to the observed plane (transit+RV detect → add NASA-like measurement noise mass 20%/radius 4.7%),
  three metrics — puffy fraction (curve-dep), mean distance to silicate curve (curve-dep continuous), median
  bulk density 5.513·M/R³ g/cc (CURVE-FREE) — vs NASA bootstrap WITH per-planet asymmetric-error propagation.
  VERDICT: P-Pop B (with rocky) is closest on all three (1.1–1.7σ); all A (no massive rocky) universes 5–7σ
  off → massive rocky planets robustly must exist. KEY catch: the curve-free DENSITY metric exposes flat B as
  unphysical — median density 10.7 g/cc (denser than iron!) because independent M-R + RV selection keeps
  heavy-but-small planets; puffy fraction alone (0.33, 2.4σ) hid this. Always cross-check with a curve-free
  metric. Error propagation widens NASA (solid vs dashed green) and is the honest uncertainty; NASA's
  transiting+measured-mass subset is itself transit+RV-selected, so that detection mode is the fair match.
- `run/ppop/universe_metrics.py` — metric library (pure numpy/scipy/sklearn, no Qt): puffy_fraction,
  mean_distance_to_curve (curve-dep); median_density, mr_scatter (slope/intercept/sigma_int dex),
  logdensity_scatter, density_bimodality (Sarle BC + GMM dBIC) (curve-FREE); energy_distance (2-sample
  logM-logR + permutation p); kde_mean_loglik (Bayes factor). `58_all_metrics_scorecard.py` runs them all
  vs NASA. KEY finding: metrics SPLIT — LOCATION metrics (puffy, dist-to-curve, median density, energy
  distance) pick P-Pop B as NASA-like; but SCATTER/shape metrics (MR intrinsic scatter 0.169 dex, log-density
  scatter 0.626 dex, bimodality) show NASA has LARGE diversity that P-Pop B's tight Chen2017/Forecaster
  relation UNDER-produces (0.08/0.25 dex) and flat B matches only unphysically. KDE-loglik is gamed by flat
  B's diffuseness (hedging) — trust energy distance (penalises location) over likelihood. Conclusion: data
  want P-Pop B's LOCATION + MORE intrinsic M-R scatter than Forecaster gives → real compositional diversity.
- `60_cold_corner_metric_power.py` — self-validation by INJECTION-RECOVERY (the way to check a metric's
  usefulness): inject large rocky planets (R 1.4-2.0, ρ 7.5 g/cc) on real cold vs hot P-Pop orbits, detect,
  measure each metric's AUC at separating rocky-present vs -absent observed catalogues. VERDICT: large-rocky
  joint completeness 1.25% cold vs 7.6% hot (~6x selection bias against cold large rocky); metric power weak
  in cold (AUC 0.51-0.57) vs strong hot (0.64-0.88); median DENSITY is the most powerful metric in both. NASA
  has 44 cold measured-mass planets (puffy~0.52) so cold rocky planets DO exist — but their FRACTION in the
  top-left corner is selection-limited and the metrics can barely constrain it. So "rocky as big as puffy" is
  supported GLOBALLY (warm-dominated detections) but only weakly in the COLD corner specifically.
- `62_mstar_conditioned.py` — condition on host-star type (works for the flat universe too: it draws
  Teff~U[2300,9700], keep Teff<3900). Removing the stellar confound: on M dwarfs (NASA N=89) the puffy
  fraction tension of the B universes drops to consistency — flat B 0.27σ, P-Pop B 0.98σ — while the A
  (no massive rocky) universes stay rejected 5-6σ. So "massive rocky planets exist among M dwarfs" is robust.
  BUT flat B is STILL unphysical on median density (~10 g/cc) even M-dwarf-conditioned (M-R independence is
  star-independent); only P-Pop B passes all three location metrics.
- `63_energy_bayes_2d.py` — the two PROPER 2-D (logM-logR) comparisons. (A) energy distance shown as a
  distribution with a NASA-vs-NASA null (floor 0.014): P-Pop B 0.050 (2.2σ from null, by far closest),
  others 0.12-0.18 (3.8-4.8σ). NOTE: "mean distance to silicate curve" is a 1-D gap, NOT the energy distance.
  (B) ERROR-CONVOLVED KDE Bayes factor (MC over each NASA planet's error ellipse) FIXES the earlier hedging
  pathology where diffuse flat B won the naive likelihood: now P-Pop B is highest, flat B 10^34x less likely,
  flat A 10^58x, P-Pop A 10^70x. Both proper methods now AGREE with the scalar metrics: P-Pop B is decisively
  the best match to observed. flat B is excluded by density + convolved Bayes despite matching puffy fraction.
- `65_occurrence_correction.py` — inverse-detection-efficiency test of the cold "red window"
  (insol<10, radius>1.4) for ROCKY planets: inject rocky planets across (insolation, radius), measure
  transit+RV completeness η, divide NASA rocky counts by η. RESULT: red window has only 3 observed NASA
  rocky planets at median η=1.6% (BELOW the 2% trust floor → formally UNCONSTRAINED; naive N/η=189±109);
  hot control (insol>100) η=7%, 46 observed. VERDICT for "do large cold rocky planets only exist via
  atmosphere stripping, or is it bias?": current transit+RV-selected data CANNOT decide — the cold deficit
  is bias-dominated/unknowable (η too low to correct), and the face-value correction (3→~189) is consistent
  with NO physical deficit. To actually decide, need higher cold completeness (microlensing/direct imaging,
  or condition on M dwarfs where cold=short period=detectable) or the insolation-dependent radius-valley
  slope test (Van Eylen 2018). Caveats: η is a toy-detector proxy for NASA's true heterogeneous selection,
  uses P-Pop's M-dwarf-heavy star/orbit mix, and one representative rocky mass per radius.
- `66_occurrence_correction_mdwarf.py` — same correction CONDITIONED ON M dwarfs (Teff<3900), red window
  insol<10 & R>1.7. SURPRISE: M-dwarf conditioning did NOT help — red-window η went 1.6%→1.7% (1.0x),
  N_obs stayed 2. WHY: (1) P-Pop/Gaia is already 86% M dwarfs so "all stars" was already ≈ M dwarfs
  (conditioning redundant for this pool); (2) the binding constraint is the GEOMETRIC TRANSIT PROBABILITY
  (~R*/a, ~1.5% Sun vs ~4% M-dwarf at the I=10 cold edge — M dwarfs ~2-3x better but still tiny), NOT the RV
  signal (script 60: detect-given-transit ~93%), so joint completeness ≈ geometric transit prob; (3) deep
  tension: rocky/puffy classification REQUIRES radius → requires transit → pays the geometric penalty, so
  cold transiting rocky planets are intrinsically rare and cannot be rescued by stellar-type conditioning
  within transit+RV. N_obs=2 also makes it Poisson-unconstrained (2/0.017≈118±83). TTV won't help (it aids
  the MASS side, but transit geometry is the bottleneck). Real paths: bigger M-dwarf transit survey to
  accumulate N, or the radius-valley-slope physical test (doesn't need to detect cold rocky directly).
- `68_radius_valley_slope.py` — VANILLA test (histograms + one straight-line fit; no KDE/GMM) of whether the
  cold red-window deficit is PHYSICAL: slice transiting planets (R 1-4, σ_R/R<10%, N=1267) by insolation,
  find the radius-valley dip per slice, fit log R_valley = β log I. Recovers Van Eylen 2018's canonical
  period slope: All β_period_eq=-0.090±0.014 (≡ β_insol=+0.067). RESULT IS STELLAR-TYPE DEPENDENT:
  FGK (N=1076) β_insol=+0.078±0.008 (~10σ POSITIVE = thermal atmosphere stripping / photoevaporation) →
  valley extrapolates to R=1.50 at I=10, BELOW 1.7, so large cold rocky planets are PHYSICALLY rare for FGK
  → the cold deficit IS real for Sun-like stars. M dwarf (N=191) β_insol=-0.007±0.042 (FLAT, consistent with
  zero) → no thermal-mass-loss signal (matches Cloutier & Menou 2020 that M-dwarf valley differs) → the
  M-dwarf cold deficit is NOT explained by stripping, so it stays bias/unconstrained (cf scripts 65/66).
  Methodological win: the deep, literature-matching result came from the simplest possible method.
  SILICATE OVERLAY (panels a/c/d): the empirical radius valley sits ON the project's silicate pure-rock
  limit — valley R≈1.64-1.73 = silicate max-rocky radius at M≈4.5-5.5 M⊕ (band 1.47-2.02 over 3-10 M⊕).
  So the observed radius gap IS the rock/volatile boundary, unifying the radius-valley result with the
  M-R composition framework used everywhere else in the project.
- `70_universe_dists_cuts.py` — "which universe does NASA imply?" repeated under 4 sub-sample cuts
  (mass>2, mass>2.5, insol<30, insol<50), puffy fraction + median density, transit+RV + error-prop.
  ANSWER (professor's key question): P-Pop B (correlated M-R, WITH rocky) is closest under EVERY cut, and
  essentially PERFECT in the cold (insol<30: puffy 0.2σ, density 0.9σ; insol<50: 0.2σ/0.7σ). The "no massive
  rocky" A universes are rejected everywhere (3-9σ); flat B rejected by impossible density (8-14 g/cc).
  NASA puffy fraction RISES toward low insolation (0.43 for mass>2 → 0.53 for insol<30) — more un-stripped
  puffy planets in the cold, exactly the photoevaporation direction — and P-Pop B tracks that shift.
  NASA box sample = 292 measured-mass (285 Mass + 7 Msini) transiting planets, R 0.53-2.20 (med 1.52),
  mass 0.10-11.87 (med 4.0), insol 0.1-6660 (MED 96 → uncut sample is HOT-dominated). Bell WIDTH is NOT made
  by repeating the MC: it is finite-N (σ≈√(p(1−p)/N)) + per-planet measurement error (mass 20%/radius 4.6%,
  real published pl_*err1/2); bootstrap MEASURES it, more repeats just smooth it. Counts after cuts: mass>2
  N=230, mass>2.5 N=206, insol<30 N=85, insol<50 N=112.
- `71_universe_mr_scatter_cuts.py` — 1×4 M-R SCATTER (not bells): sampled detected+noised planets of all 4
  universes + NASA (open black circles) on the M-R plane with the silicate curve, cuts all/mass>2/insol<50/
  combined. Puffy fraction = fraction above the curve, made visual. Answers WHY flat A / P-Pop A are not at
  puffy=1 after mass>2: ZERO real rocky with true mass>2; the ~0.86 is pure measurement noise (~1/3 rocky-
  below-2 leaking up across the observed-mass cut, ~2/3 truly-puffy planets near the curve label-flipped by
  the 20%/4.6% noise). Noise-free value is exactly 1.000.
- `72_puffy_cuts_flat_ppop.py` — puffy-fraction bells, 2×4, FLAT (top)/P-Pop (bottom) rows × 4 cuts, transit+RV
  + error-prop (like 70 but puffy-only, split by universe type). P-Pop B closest to NASA under every cut
  (1.5-1.7σ mass cuts, 0.2-0.6σ insol cuts); flat B 1.7-3.3σ; A's 4-8σ.
- `73_puffy_detection_flat_ppop.py` — like 56 (INTRINSIC detected puffy fraction, NO measurement noise) but
  2×3, FLAT/P-Pop rows × detection mode (transit/RV/both). Cleanly shows RV does OPPOSITE things: flat B
  0.59→0.29 (RV rockifies), P-Pop B 0.44→0.52 (RV puffifies); transit barely moves either. Going forward the
  user wants puffy-fraction-ONLY in the distribution graphs (median density dropped).

**Why:** occurrence-rate P-Pop clumps planets in the small/short-period corner, so detector output
reflects the prior. A flat box maps the detector edge directly.

**How to apply:** insolation coupled to orbit + log-uniform spacing were the user's chosen options.
On radius-insolation, RV reads ~flat/empty BY DESIGN — RV detection is mass-driven and mass is
independent of radius; the M-R plane (script 34) is the RV-appropriate view (detections pile up at
high mass / high insolation). See [[ppop-stellar-mass-is-placeholder]]. Earlier PPop/Gaia-based
UniformBox approach was replaced by this fully-flat one (UniformBox.py deleted).
