---
name: kepler-tess-detector-calibration
description: Kepler/TESS toy detectors were ~1.2x optimistic vs official pipelines; calibration factors added and baked into catalogs
metadata:
  type: project
---

The toy Kepler MES and TESS SNR detectors over-detected: measured against the
official pipelines they ran ~1.2x hot, so they crossed the 7.1 threshold too
readily (figure 44/scripts 35/36/44 looked "too yellow", esp. M dwarfs).

**Calibration measured (scripts 47/48, toy vs official):**
- Kepler toy MES / official DR25 `koi_max_mult_ev`: median **1.19** (uses real `rrmscdpp*`, so it's a formula/shape effect, not noise-model error).
- TESS toy SNR / official SPOC `Planet SNR` (ExoFOP TOI): **rises toward the cut — ~1.52 in the SNR 7.1–20 bins, median 1.41 overall**.

**Fix (multiplicative calibration on the detection statistic):**
- `KeplerData.MES_OFFICIAL_CALIBRATION = 0.84` (≈1/median) → applied in `calc_kepler_mes` (also rescales `one_sigma_depth_ppm`). Constructor param `mes_calibration` (None→default, 1.0→raw). Threshold bin ends ~0.94 (mildly conservative).
- `TESSData.SNR_OFFICIAL_CALIBRATION = 0.66` → applied in `snr()`. Constructor param `snr_calibration`. **Anchored at the threshold region** (not the median) so the calibrated ratio is ≈1.0 exactly at the SNR≥7.1 cut (7.1–10→0.999, 10–20→1.004); conservative above; median ~0.93.
- After fix, scripts 47/48 show the threshold region ≈1.0 and recovery curves rise properly (TESS panel B 67%→98%). 47/48 overlay raw-vs-calibrated in panel C.

**CAUTION — TESS factor was re-derived after a bug fix.** script 48 mapped ExoFOP
`"TESS Mag"`→`tess_tmag`, but TESSData ingests magnitude as `tmag` and rebuilds
`tess_tmag` from it — so `tess_tmag` came out all-NaN: (a) the brightness gate
failed for every TOI → panel B was a flat 0%, and (b) the smooth-CDPP fallback was
corrupted, biasing the toy SNR low so the *first* measured ratio (1.20) and factor
(0.83) were WRONG. Fix: map `"TESS Mag"`→`tmag` in script 48. Always sanity-check
`tess_tmag` non-null and panel B before trusting a TESS calibration number.

**Propagation to already-generated catalogs:** `run/apply_detection_calibration.py`
rescales stored `kepler_mes`/`tess_snr` in place (×factor), rescales derived
noise cols, and recomputes the boolean `detected*` columns from unchanged gate
columns AND statistic≥7.1 (threshold model verified to reproduce detected* 100%).
Idempotent via a `detection_calibration` stamp column, and **re-calibration-safe**:
if a file is already stamped, it applies only the delta (target/prior) so changing a
factor and re-running corrects rather than double-applies. Covers the dirs feeding
fig 44 (`Gaia_C_F_K_combined`, `Gaia_C_F_K_combined_cdpp_v1`) and the single-file
dirs for scripts 45/46 (`Gaia`, `Gaia_cdpp_v1`). Effect: detected-among-transiting
Kepler combined 80.6→78.1%, TESS combined 59.5→51.3% (0.66); cold rocky false-negatives rose
(e.g. TESS-M I<10 window ~45% missed vs near-0 before).

**Why surgical rescale not regeneration:** calibration is a pure scalar; full
regen would re-roll TESS random transit phase and change results beyond the
calibration. **To re-derive the factors**, re-run scripts 47 and 48.

**Data-source drift fixed while running 47/48:** NASA retired `q1_q17_dr25_stellar`
(47 now JOINs `keplerstellar`, deduped to longest-dataspan row per KOI); ExoFOP
renamed catalog col `SNR`→`Planet SNR` (48 rename map). Also: importing
`lifesim.core.*` pulls PyQt5 via `lifesim/__init__.py` (installed); standalone
scripts load the module via importlib to avoid it. Related: [[uniform-box-control-generator]].
