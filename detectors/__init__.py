"""
Detection models, one per facility. Each takes a planet catalog and adds
detectability columns via `determine_detectable()`.

Written for this project, not taken from LIFEsim.

Kepler, TESS and RV import `lifesim.core.data.Data` inside a try/except and fall
back to `Data = None`, so they still work where lifesim is not installed. HWO
imports it outright and needs it. This file imports nothing, so importing one
detector does not drag in the others.
"""
