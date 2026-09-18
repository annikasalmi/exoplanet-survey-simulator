# results/

Everything the pipelines generate. Git ignores all of it except this file, so
nothing here is a source of truth — any of it can be deleted and regenerated.

```
catalogs/     simulated planet populations, one directory per pipeline
  kepler/  tess/  rv/  hwo/  lifesim/  flat_universe/
figures/
  simulation/   plot_all output, per run: <sim>_<nruns>_<catalog>/
  analysis/     the analysis scripts, one directory each
  calibration/  plotting/scripts/calibration/: detectors against real mission
                data, and the three planet generators against each other
  other/        the standalone scripts in plotting/scripts/
paper/        publication figures (README.md "Paper figures" says which script makes each)
logs/         per-run logs written by sim.py
```

Catalogs are large — a single Gaia-60pc universe is roughly 300 MB per pipeline,
and regenerating one takes about 20 minutes. Input data is not here; it lives in
`data/` and is tracked.

Paths come from `tools.paths`, not from string literals, so this layout can be
moved by editing that one module.
