---
name: cold-completeness-reconciliation
description: Why the "1.3% vs 62%" cold-completeness numbers are not in conflict, plus the puffy-gap attribution
metadata:
  type: project
---

The cold-corner completeness "disagreement" (script 60's ~1.3% vs script 53/64's ~62%)
is NOT a contradiction — it is two completenesses against different denominators, and the
factorization itself is the result. Verified in `scripts/67_completeness_reconciliation.py`:

  absolute completeness = P_geom × P_conf|transit

Cold (I<10), large (R>1.4), rocky TESS P-Pop, all 10 universes, RV=best N=100:
- M dwarfs: P_geom=1.9%, P_conf|transit=58%, absolute=1.1% (identity 1.9%×58%=1.1% holds).
- Pooled ALL: absolute=1.0% ≈ script 60's 1.3%.
So script 60's ~1% = pooled ABSOLUTE (geometric-transit gate dominates the cold loss — long
period rarely transits). Script 53/64's ~60% = P_conf|transit for M dwarfs. Use ABSOLUTE for
blind-survey YIELD; use P_conf|transit for the radius-limit (NASA overlay) question, because
every NASA overlay planet already transits. M dwarfs stay ~60% confirmable because their cold
habitable-zone orbits are SHORT-period (transit often, large RV K). See [[rv-survey-model-decision]].

ROBUSTNESS / GAP (`scripts/69_metric_robustness.py`):
- Cosmic variance: detected puffy fraction across the 10 Gaia P-Pop universes = mean 0.49,
  SD 0.015; the model-vs-NASA puffy gap (0.086) is ~6× that SD, so the single-catalog
  (kepler_catalog_0) conclusions of scripts 56-58 are robust — not a draw artifact.
- NASA-model puffy gap is MOSTLY RV selection: tightening K/σ_K from 5→15 drops the detected
  puffy fraction 0.51→0.43 (RV preferentially confirms dense/rocky planets), closing ~3/4 of
  the 0.51→0.40 gap. The residual ~0.03 does NOT close even at K/σ_K=15 → a real P-Pop
  over-production of puffy planets (occurrence) or a small silicate-curve offset. That residual
  is exactly what the planned M-R sensitivity band should bound.

MENTOR VERDICT on the puffy-fraction program (scripts 56-63): curve-DEPENDENCE is fine and
appropriate (the silicate curve IS the rocky/puffy hypothesis boundary; the question can't be
posed curve-free). The real weakness is the puffy FRACTION being (a) binary at the curve
(noise flips labels) → use the continuous signed distance-to-curve instead, and (b)
marginalized over insolation → all global scalars are powerless in the cold corner (script 60
AUC≈0.5). Strongest inference tools that match the 53 radius-insolation graph's significance:
median bulk density and 2-D energy distance, each CONDITIONED on insolation + star type.
Drop the flat universe as a primary null (strawman); the real test is P-Pop with vs without
massive rocky (script 63: removing massive rocky worsens the NASA fit → they exist).
