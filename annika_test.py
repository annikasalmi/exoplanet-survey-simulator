import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, expon, gamma

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

PlanetDistribution.SummaryPlot(block=False)

r_p_all = []
for i in range(500):
    n_planets = np.random.randint(1, 6)
    r_p, _ = PlanetDistribution.draw(Star=Star, Nplanets=n_planets)
    r_p_all.extend(r_p)

counts,bins,ignored=plt.hist(r_p_all, bins=30, alpha=0.5, color='green', label='Simulated Planet Radii')
# data=r_p_all
# # Fit a Gamma distribution to the data
# shape_est, loc_est, scale_est = gamma.fit(data, floc=0)  # Fit Gamma distribution and fix location to 0
# # Plot the fitted Gamma distribution
# x = np.linspace(min(data), max(data), 1000)
# gamma_values = gamma.pdf(x, shape_est, loc_est, scale_est)

# plt.plot(x, gamma_values, 'k', linewidth=2, label=f'Fitted Gamma Distribution\n(Shape={shape_est:.2f}, Scale={scale_est:.2f})')

# Adding titles and labels
plt.xlabel('Data')
# plt.ylabel('Density')
plt.title('Histogram')
plt.legend()
plt.grid(True)

plt.show()
a=1