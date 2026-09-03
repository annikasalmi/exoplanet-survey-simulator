# Forked PPop

This is a forked and modified copy of [P-pop](https://github.com/kammerje/P-pop) by Jens Kammerer, MIT
licensed (see `LICENSE` here). It was forked in April 2025; P-Pop has changed since then.
It is copied in rather than pip-installed because upstream P-pop cannot be imported as a library due to the hyphen -.
`MassModels/Forecaster/` is another copy of [Forecaster](https://github.com/chenjj2/forecaster) by Jingjing Chen, also MIT.

## Forked changes

- Seeded random numbers as opposed to np.random.
- Outputted DataFrames instead of .txt; no need to save files locally when PPop is frequently rerun. 
It is possible to turn on saving to .txt is possible to turn on.
- Added new files: `StarCatalogs/gaia.py`, `StarCatalogs/build_gaia_60pc.py` and `PlanetDistributions/SAG13_extrap.py`.
- Cleaned up names for consinstency: renamed some files/functions to follow snake_case; added a tools.paths file.

This is an old fork, so there are some newer distributions that have not been copied over.
These include: `Kaminski2025`, `Bryson2021*` and `Dressing2015Extrap`; the
catalogs `LTC4`, `HPIC`, `alphaCenA` and `Sun10pc`; the `Teff_range` filter; the
`'mc'` scenario; the `WDSsep` and `name` output columns; and a fix to
`Fernandes2019Symm`, where upstream now has `c0 = 0.84 / 10.` and this copy still has
`0.84`.