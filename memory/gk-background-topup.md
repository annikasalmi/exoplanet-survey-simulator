---
name: gk-background-topup
description: How to top up the Gaia-60pc G/K detection background to 8000 transiting planets (run/generate_gk_8000.py)
metadata:
  type: project
---

Script 44's per-type detection background only smooths once a panel holds
>= ~3000 transiting-in-window planets. The Gaia-60pc stacks (universes 0..9)
held only **~1128 G / ~3026 K (Kepler)** and **~1124 G / ~3009 K (TESS)** — G is
far short. Per-universe yield (transiting-in-window = script-44 def: flux 0.1-1e4,
radius 0.6-2.2, transiting_geometric): **~113 G/universe, ~303 K/universe**. So
reaching 8000 needs **~62 more G-universes** (the bottleneck) + **~17 K-universes**.

**`run/generate_gk_8000.py`** generates the top-up. Key design:
- Generates G-ONLY and K-ONLY universes (filters the Gaia catalogue by spectral
  type) so it doesn't waste compute on ~45 extra K-universes. G/K use Bergsten2022
  (vectorized, ~8 ms/star) so this is MUCH faster than a full M-dwarf-dominated
  universe — see [[ppop-generation-bottleneck]] (M dwarfs + Dressing KDE are the slow part).
- Kepler & TESS universe i share ONE P-Pop draw (same seed): generate once, run
  BOTH detectors -> ~half cost vs running run_Kepler + run_TESS separately.
- New files numbered from **8001** (kepler_catalog_8001.csv / tess_catalog_8001.csv)
  in run/kepler/data/Gaia and run/tess/data/Gaia_cdpp_v1 — never overwrites 0..9.
- Restartable: recounts all existing catalogs on startup, resumes. Run:
  `python run/generate_gk_8000.py` (or `--smoke` to self-test, `--types K`,
  `--max-universes N`).

**DONE (2026-06-25):** ran to completion — 76 top-up universes written
(**G: 8001-8058**, **K: 8059-8076**) to run/kepler/data/Gaia and
run/tess/data/Gaia_cdpp_v1. Final transiting-in-window totals all clear 8000:
**G** Kepler=8093 / TESS=8089, **K** Kepler=8242 / TESS=8225. Realized per-universe
yield: ~118 G (first draw was a low 81), ~290 K. Script 44 re-rendered with these
stacks — G/K panels now smooth (titles read the topped-up N). Note the baseline
universes 0..9 are FULL F/G/K/M (~172k rows each, M-dominated), so script 44's
CSV load is slow (~30-40 min for the 86+86 stacked files).

**Consumers must stack 8001+**: script
`44_2x4_kepler_tess_rocky-fgkm-detection_gaia60pc.py` was updated so `_ppop_files`
additively stacks any index >= `EXTRA_START_INDEX (8001)` on top of 0..9 (single-
type files only add to their own panel). Other consumers (e.g. script 53) still
read only 0..9 — widen them the same way if they need the denser G/K background.
