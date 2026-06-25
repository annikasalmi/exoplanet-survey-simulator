---
name: ppop-stellar-mass-is-placeholder
description: P-Pop catalog mass_s is a constant 1 M_sun in kg, not a real stellar mass
metadata:
  type: project
---

In the generated P-Pop catalogs (e.g. `run/tess/data/Gaia_cdpp_v1/tess_catalog_0.csv`),
the `mass_s` column is a **constant 1.989e30 for every star** — i.e. 1 M☉ expressed in kg,
a placeholder, NOT a real per-star stellar mass. `radius_s`, `teff_s`, `l_sun` ARE clean
and track spectral type correctly.

**Why it matters:** any calc needing M★/M☉ (e.g. RV semi-amplitude's (M★/M☉)^-2/3 term)
cannot use `mass_s` directly — it would treat every M dwarf as 1 M☉ and lose the ~2.5×
K boost that low-mass stars give. The transit detectors dodged this because P-Pop already
populates `semimajor_p`, so their Kepler's-3rd-law fallback (which fills mass with 1.0)
never fires.

How to apply: derive M★ from `radius_s` on the lower main sequence (M/M☉ ≈ R/R☉, good for
K/M dwarfs) when `mass_s` isn't in plausible solar units; NASA `st_mass` IS real solar
units so use it there. This is what `lifesim/core/rv_data.py` `stellar_mass_msun()` does.
Related: [[rv-survey-model-decision]].
