import pandas as pd
import numpy as np

from PPop.PlanetDistributions.SAG13 import PlanetDistribution
from PPop.Star import Star

PlanetDistribution = PlanetDistribution(Scenario='baseline')
Star = Star(
    Name='TestStar',
    Dist=1.0,
    Stype='G',
    Rad=1.0,
    Teff=5778,
    Mass=1.0,
    RA=180.0,
    Dec=0.0,
    Vmag=5.0,
    Jmag=None,
    Hmag=None,
    WDSsep=None,
    WDSdmag=None,
    lGal=None,
    bGal=None
)

r_p_all = []
for i in range(500):
    n_planets = np.random.randint(1, 6)
    r_p, _ = PlanetDistribution.draw(Star=Star, Nplanets=n_planets)
    r_p_all.extend(r_p)
