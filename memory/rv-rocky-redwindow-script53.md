---
name: rv-rocky-redwindow-script53
description: Script 53 — rocky-planet RV-test version of the script-44 FGKM red-window figure
metadata:
  type: project
---

`scripts/53_rv_rocky_fgkm_redwindow.py` answers "is the cold upper-radius limit for rocky
planets real, or an RV-confirmation artifact?" It mirrors script 44's FGKM insolation-radius
figure but stacks an RV-test on the transit-test, on rocky-filtered P-Pop.

Design: reuses script 44 helpers (rocky threshold, NASA loader, 90% line, facility overlay,
contours). The **rocky cut (ref.ddat shifted to LHS 1140 b) is applied to the P-Pop background
too**, not just the NASA overlay — so puffy planets are removed and the test is conservative
(rocky = denser = more massive = larger K = easier RV). Two rows: top = transit-test
P(transit detected | rocky, transiting); bottom = transit AND RV mass-measurable. `--mission
kepler` (default) uses kepler_p_detect from run/kepler/data/Gaia (clean 4-yr cold baseline);
`--mission tess` uses the combined cdpp catalogs. RV = best of HARPS/NIRPS, N=100.

**Result (Kepler, best, N=100), cold red window (I<10, above 90% line) MISS fraction:**
M (N=1989): transit 10% -> transit+RV 45%.  K (N=76): 33% -> 59%.  G (N=11): 55% -> 82%.
So for M dwarfs transit detects 90% (matches script 44's 9% FN) but only 55% are confirmable
once RV is required -> RV removes ~35pp. The upper-radius limit is therefore a MAJOR RV
artifact, but NOT total: 55% would still be confirmable at N=100/best, so part of the
emptiness may be genuine low occurrence + the multi-year RV baseline cost (the model has NO
survey-baseline gate, so it UNDERstates cold difficulty). See [[rv-survey-model-decision]].
