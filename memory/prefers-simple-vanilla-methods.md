---
name: prefers-simple-vanilla-methods
description: User wants simple/vanilla statistical methods (puffy-fraction normal), not advanced tools
metadata:
  type: feedback
---

The user is a student whose statistics comfort level is the **puffy-fraction normal distribution** (script 56): a fraction with a binomial/bootstrap "bell" error bar, which they can describe — method AND result — clearly to their mentors. Do NOT push advanced statistical machinery (ABC/ABC-SMC, hierarchical Bayes, 2-D energy distance, KDE Bayes factors) as the recommended path, even though those scripts exist (57–63).

**Why:** Their stated philosophy — "the best scientific result never comes from over-using statistical tools; it comes from presenting deep science behind vanilla, simple, clean methods." Clarity of explanation and hand-checkable numbers are the priority. Being able to describe the method to mentors is itself a hard constraint.

**How to apply:** Default to the simplest method that supports the claim. The only "upgrade" consistent with their philosophy is the curve-free **median bulk density** (ρ = 5.513·M/R³) as a vanilla cross-check, because it's just a physics formula and neutralizes the silicate-curve uncertainty without new statistics. Improvements should be honesty/sensitivity checks (curve-nudge sensitivity, state box-edge meaning, only claim where counts exist), NOT new tools. The fancy scripts (63 etc.) are back-pocket cross-checks ("I checked the sophisticated way too and got the same answer"), never the lead. See [[cold-completeness-reconciliation]] and [[rv-rocky-redwindow-script53]].

**UPDATE 2026-07-03:** the corner puffy-fraction normal bell is DEMOTED — user judged the script-77 corner version circular ("worthless": NASA≈flat A after the corner cut just restates that the observed corner lacks big rocky planets; it can't separate formation from selection). Vanilla is still the constraint, but it must now be completeness-aware vanilla: Poisson on completeness-weighted counts, binomial rocky-share, Agresti–Coull two-proportion vs a selection-only control (scripts 79/80, [[corner-occupancy-rocky-fraction-tests]]). They also asked for a FEW "scientifically interesting, less obvious" metrics rather than the script-58 many-metric scorecard. Hierarchical/Bayes machinery is no longer strictly off-limits: they explicitly ordered option E built ("build E now" → script 81), but its role is back-pocket rigorous cross-check confirming the vanilla tests — 79/80 remain the mentor-facing headline.

**UPDATE 2026-07-06:** they then ordered "plan 5" built (classifier/SBI full-catalog likelihood ratio → script 82), explicitly demanding "clear architecture and explainability" and a plain-words ("caveman") summary of the effects. So ML/SBI methods are now acceptable to them when (a) the script is staged transparently (STEP 1..5 banners matching the plan), (b) validity is frequentist-by-simulation not "trust the network", and (c) every result has a plain-language reading. Explainability is the constraint, not the tool class. They also push on forward-model bias ("any way to further take out bias in my simulator?") — answer that question proactively in designs.
