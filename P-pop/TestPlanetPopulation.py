"""
# =============================================================================
# P-POP
# A Monte-Carlo tool to simulate exoplanet populations
# =============================================================================
"""


# =============================================================================
# IMPORTS
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np

import ReadPlanetPopulation as RPP


# =============================================================================
# SETUP
# =============================================================================

# Select the name of the planet population table to be read.
PathPlanetTable = 'TestPlanetPopulation.txt' # str

# Select a model for the computation of the habitable zone.
Model = 'MS'
#Model = 'POST-MS'

# Select the name of the photometry tables to be read and give a tag to each
# of them (leave empty if you don't want to read any photometry tables).
PathPhotometryTable = ['TestPlanetPopulation_MIRI.F560W.txt',
                       'TestPlanetPopulation_MIRI.F1000W.txt',
                       'TestPlanetPopulation_MIRI.F1500W.txt'] # list of str
Tag = ['F560W', 'F1000W', 'F1500W'] # list of str


# =============================================================================
# READ PLANET POPULATION
# =============================================================================

# The next five lines read the planet population table and its photometry.
PP = RPP.PlanetPopulation(PathPlanetTable)
PP.ComputeHZ(Model)
for i in range(len(PathPhotometryTable)):
    PP.appendPhotometry(PathPhotometryTable[i],
                        Tag[i])

print('Number of planets in the planet population table')
print(len(PP.Rp))

print('Planet radius (Rearth) of the first planet in the planet population table')
print(PP.Rp[0])

print('Host star radius (Rsun) of the first planet in the planet population table')
print(PP.Rs[0])

print('Display the header of the '+Tag[0]+' photometry')
print(PP.Phot[Tag[0]]['HEAD'])

print('Display the data of the '+Tag[0]+' photometry of the first planet in the planet population table')
for i in range(len(PP.Phot[Tag[0]]['HEAD'])):
    print(PP.Phot[Tag[0]]['HEAD'][i]+': '+str(PP.Phot[Tag[0]]['DATA'][i][0]))

import pdb; pdb.set_trace()
