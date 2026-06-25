---
name: rv-survey-model-decision
description: Why the RV detector models HARPS/HARPS-N and the constants/data facts behind it
metadata:
  type: project
---

The RV-completeness work (the radial-velocity sibling of the Kepler/TESS transit
detection-fraction maps) models the **HARPS / HARPS-N family** as its one survey.

**Why:** In the small-planet box (R≤2.2 R⊕, M≤12 M⊕) of the NASA RM-filtered file
(`run/kepler/data/NASA/NASA_PSCompPars_transiting_confirmed_RM_insolation_errors_limits.csv`),
of 1,717 planets only **296 have a genuinely measured mass** (provenance `Mass`/`Msini`);
the other 1,421 are "M-R relationship / Calculated" and MUST be excluded (circular).
Of the 296: 240 RV, 56 TTV. The RV masses are dominated by HARPS-N (Bonomo 2023 = 28)
and HARPS (southern), HIRES second. HARPS/HARPS-N are twin spectrographs (R≈115,000),
so a one- or two-survey model shares constants — satisfies the "keep the two similar" ask.

**Locked constants** (in `lifesim/core/rv_data.py`, all parameters; recalibrated 2026-06 to
published small-planet campaigns — TOI-1453 Roy+2025, K2-136c Mayo+2023):
- detection threshold = **K/σ_K ≥ 5** (5σ "secure mass"; analog of 7.1 MES)
- **σ_K = √(2/N_obs)·σ_RV** (white-noise scaling). A `jitter_red_frac` (f) knob exists to add a
  correlated-noise floor σ_K² = (2/N)·[σ_instr²+σ_phot²+((1−f)σ_jit)²] + (f·σ_jit)², but **default
  f = 0** after direct calibration. **CRITICAL CALIBRATION (2026-06-16):** validate σ_K against the
  342 real SMALL (R≤2.2) planets with published pl_rvamp/pl_rvamperr — NOT the all-planet median.
  At f=0: model median σ_K=0.27 m/s (published 0.31), median K/σ_K=5.7 (published 6.4), and the
  detector recovers **57.6%** at ≥5σ — matching the **58.5%** of those planets actually published
  at ≥5σ. So f=0 is CORRECTLY calibrated, not "optimistic." A red floor f>0 double-counts activity
  (real teams GP-model it out) and rejects real detections: f=0.5 recovers only **10.8%** of real
  small-planet RV measurements — way too strict. PITFALL: the all-planet σ_K ratio (0.19 at f=0,
  0.49 at f=0.5) is GIANTS-DOMINATED (median Kpub=29 m/s, big published systematics) and is
  MISLEADING — it made f=0 look too optimistic when it is right for the small planets that matter.
  Acid test for any future detector tweak: % of real small-planet RV detections recovered (~58%).
- **N_obs = 100** (real HARPS-N small-planet campaigns: 93–100 RV points; the whiteboard "40–50"
  was a survey-detection number, too few for mass measurement — caused an over-strict K floor).
- **Adaptive-exposure noise model**: σ_RV = √(σ_instr² + σ_phot² + σ_jitter²); σ_phot stays at the
  floor (~1 m/s) for targets brighter than `phot_full_mag` (V=12 HARPS / J=11 NIRPS) and only
  degrades beyond — because surveys integrate longer on fainter stars. The OLD fixed-exposure
  `10^(0.2(V−8))` made typical targets ~2× too noisy. HARPS σ_instr=0.8, jitter G/K≈1.2–1.3, M≈2.5.
- Validated: bright quiet G/K → σ_RV ≈ 1.75–1.8 m/s (documented 1.56–1.6 ✓); min detectable mass at
  P=20d ≈ 4–5 M⊕ (HARPS G/K), 2.5 M⊕ (NIRPS M) — matches the documented 1–3 M⊕ capability.
- K = 28.43 m/s · (Mp sin i/M_Jup)·(M★/M☉)^-2/3·(P/yr)^-1/3 / √(1-e²)  (validated: Earth→0.089, Jupiter→12.47 m/s)
- NOTE the earlier "K=3.1 m/s floor" was NEVER a documented constant — it was emergent = median σ_RV ×
  threshold × √(2/N). The documented HARPS-N floor is K ≈ 1–2 m/s.

**Result — red-window hypothesis CONFIRMED** (`scripts/51_rv_redwindow_verification.py`,
runs RVData on the SAME transiting P-Pop planets as the TESS map): in the cold red window
(I<10, above 90% line) transit detect stays high but RV mass-detect collapses — M: 64% vs 2%;
G: 43% vs 0%; K: 53% vs 16%. RV detect rises monotonically cold→hot (K ∝ P^-1/3). So RV is the
bottleneck there. M-R-plane version is `scripts/50_rv_mr_detection.py`. **sin i is negligible
for the red window** (transiting ⇒ i≈90°); it only matters for non-transiting planets (script 50).

**NIRPS added as a second instrument preset** (`RVData(instrument="NIRPS")`): NIR band J,
where M dwarfs are bright (BC_J>0) and jitter is lower, so it complements HARPS exactly
where HARPS fails. Validated: M-dwarf median σ_RV drops 38 m/s (HARPS/V) → 6.5 m/s (NIRPS/J).
Keep them SEPARATE (don't average noise); script 50 supports `--instrument best` = per-planet
max(HARPS,NIRPS). Script 50 now emits THREE planes (each 2x4, sin i rows × FGKM): mass-radius,
insolation-radius, insolation-mass (the RV-native one). NASA overlay is split RV(red○)/TTV(blue△)
via reflink-author heuristic — TTV points correctly land in RV-empty cells.

**Script 53** (`scripts/53_rv_2x4_rocky_fgkm_rvtest.py`) is the publication-style companion to
script 44: same 2x4 FGKM insolation-radius styling (imports S44's machinery), but background =
ROCKY P-Pop only (below LHS 1140 b threshold) and rows = transit-test vs transit+RV joint test
(the rocky sample's full selection function). Key result (all 401 catalogs, instrument=best):
red-window false-negatives rise transit→joint: G 100%→100%, K 57%→71%, M 45%→74%. So the cold
empty region is mostly joint selection; the M panel's residual ~26% pass-fraction is the
discriminating region where a real radius limit could still be tested.

**Key data gotchas** (see [[ppop-stellar-mass-is-placeholder]]):
- P-Pop `mass_s` is a constant placeholder (1.989e30 = 1 M☉ in kg) → derive M★ from radius.
- P-Pop `ecc_p` is all 0 (circular); `inc_p` is radians; NASA `pl_orbincl` is degrees.
- P-Pop is volume-limited (Gaia ~60 pc), so the RV background must be conditioned on an
  RV-target brightness cut (default V≤12), the RV analog of the transit "P(detect|transiting)".

How to apply: use `apply_sini=True/False` to show the M sin i projection penalty (user
wants both). Validate K against `pl_rvamp` if the archive is re-pulled with that column
(current cache lacks `pl_rvamp`, `pl_msinie`, `pl_orbeccen`).
