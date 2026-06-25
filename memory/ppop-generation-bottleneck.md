---
name: ppop-generation-bottleneck
description: P-Pop catalogue generation time is dominated by the Chen2017/Forecaster mass model, not the planet distribution
metadata:
  type: project
---

End-to-end P-Pop catalogue generation (`SystemGenerator.SimulateUniverses`, the
path `run_sim.py` uses) costs **~5–14 h per 10^6 planets single-threaded** on
this box (~2× run-to-run scatter from machine load), ~1.5–3 h with `run_sim`'s
5 workers. Measured via `my_outputs/bench_generation_e2e.py` +
`my_outputs/profile_generation.py` on a fixed Gaia-60pc star subset.

cProfile attribution (load-independent fractions):
- **~75–80%: Chen2017 / Forecaster mass model** — `PPop/MassModels/Forecaster/mr_forecast.py::Rpost2M` → `func.py::ProbRGivenM` (radius→mass probabilistic inversion, once per planet incl. He2019-rejected draws). THIS is the bottleneck.
- ~10%: Dressing2015 M-dwarf draw (scipy KDE `resample`) — the 73% M-dwarf majority.
- ~4–6%: `getmaxrpproj` geometry `fsolve` (`PPop/System.py:202`).
- **<1%: the FGK planet distribution draw itself (Bergsten2022 OR SAG13)** — doesn't reach top-18 by self-time.

**Why it matters:** choosing Bergsten2022 vs SAG13 (Annika's Kopparapu-2018 FGK
baseline) changes generation time by <1% — both ~the same multi-hour runtime,
ordering even flips between samples (noise-dominated). Pick Bergsten for the
SCIENCE (stellar-mass-dependent FGK occurrence), not speed. An EARLIER note that
isolated the distribution draw (~23–31 s/10^6) was misleading: that bare draw is
<1% of the real pipeline. To actually speed up generation, optimize the
Forecaster mass model (`Rpost2M`) or swap to a deterministic mass–radius
relation — NOT the distribution and NOT the fsolve. Related:
[[ppop-stellar-mass-is-placeholder]].
