"""
# =============================================================================
# P-POP
# A Monte-Carlo tool to simulate exoplanet populations
#
# Authors: Jens Kammerer, Sascha Quanz, Emile Fontanet
# Version: 5.0.0
# Last edited: 26.03.2021
# =============================================================================
#
# P-pop is introduced in Kammerer & Quanz 2018
# (https://ui.adsabs.harvard.edu/abs/2018A%26A...609A...4K/abstract). Please
# cite this paper if you use P-pop for your research.
#
# P-pop makes use of forecaster from Chen & Kipping 2017
# (https://ui.adsabs.harvard.edu/abs/2017ApJ...834...17C/abstract).
"""


# Don't print annoying warnings. Comment out if you want to see them.
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# IMPORTS
# =============================================================================

# Import your own catalogs, distributions and models here.
import SystemGenerator
from StarCatalogs import CrossfieldBrightSample, ExoCat_1, LTC_2, LTC_3
from PlanetDistributions import Fressin2013, Burke2015, Dressing2015, SAG13,\
                                Weiss2018, Weiss2018KDE, HabitableNominal, \
                                HabitablePessimistic, Fernandes2019symm
from ScalingModels import BinarySuppression
from MassModels import Chen2017
from EccentricityModels import Circular
from StabilityModels import He2019
from OrbitModels import Random
from AlbedoModels import Uniform
from ExozodiModels import Ertel2020, Ertel2018


# =============================================================================
# SETUP
# =============================================================================

# Select the catalogs, distributions and models which you want to use here.

# Select the star catalog, the spectral types, the distance range and the
# declination range which should be included here.
#StarCatalog = CrossfieldBrightSample # used in Kammerer & Quanz 2018
#StarCatalog = ExoCat_1 # used by NASA
#StarCatalog = LTC_2 # LIFE Target Catalog (version 2)
StarCatalog = LTC_3 # LIFE Target Catalog (version 3)
Stypes = ['A', 'F', 'G', 'K', 'M'] # list of str
Dist_range = [0., 20.] # pc, list of float, [min, max]
Dec_range = [-90., 90.] # deg, list of float, [min, max]

# Select the planet distributions, the scenario, and the scaling model which
# should be used here. A different planet distribution can be assigned to each
# spectral type.
# dict, StarCatalog.Stype as keys, PlanetDistribution as data
# StypeToModel = {'A': Fressin2013, 'F': Fressin2013, 'G': Fressin2013, 'K': Fressin2013, 'M': Fressin2013}
#StypeToModel = {'F': Burke2015, 'G': Burke2015, 'K': Burke2015, 'M': Dressing2015}
#StypeToModel = {'F': SAG13, 'G': SAG13, 'K': SAG13, 'M': Dressing2015} # used in the ESA Voyage 2050 White Paper
#StypeToModel = {'F': SAG13, 'G': SAG13, 'K': SAG13, 'M': SAG13}
#StypeToModel = {'F': Weiss2018, 'G': Weiss2018, 'K': Weiss2018, 'M': Weiss2018}
#StypeToModel = {'F': Weiss2018KDE, 'G': Weiss2018KDE, 'K': Weiss2018KDE, 'M': Weiss2018KDE}
#StypeToModel = {'A': SAG13, 'F': SAG13, 'G': SAG13, 'K': SAG13, 'M': Dressing2015}
StypeToModel = {'A': SAG13, 'F': SAG13, 'G': SAG13, 'K': SAG13, 'M': SAG13}
#StypeToModel = {'A': HabitableNominal, 'F': HabitableNominal, 'G': HabitableNominal, 'K': HabitableNominal, 'M': HabitableNominal}
#StypeToModel = {'A': HabitablePessimistic, 'F': HabitablePessimistic, 'G': HabitablePessimistic, 'K': HabitablePessimistic, 'M': HabitablePessimistic}
#StypeToModel = {'A': Fernandes2019symm, 'F': Fernandes2019symm, 'G': Fernandes2019symm, 'K': Fernandes2019symm, 'M': Fernandes2019symm}
Scenario = 'baseline'
#Scenario = 'pessimistic' # for 1-sigma lower error bars on planet distribution
#Scenario = 'optimistic' # for 1-sigma upper error bars on planet distribution
#ScalingModel = None
ScalingModel = BinarySuppression

# Select the mass model, the eccentricity model, the stability model, the orbit
# model, the albedo model and the exozodiacal dust model which should be used
# here.
MassModel = Chen2017 # Forecaster
EccentricityModel = Circular
#StabilityModel = None
StabilityModel = He2019
OrbitModel = Random
AlbedoModel = Uniform
#ExozodiModel = Ertel2018
ExozodiModel = Ertel2020

# Select whether you want to display summary plots after loading the catalogs,
# distributions and models selected above, how many test draws should be
# done for generating these plots, and where you want to save them.
SummaryPlots = True
#SummaryPlots = False
Ntest = 100000 # int
#Ntest = 10000 # int
#FigDir = None # if you don't want to save the summary plots
FigDir = 'Figures/' # should end with a slash ("/")
#block = True
block = False

# Select a name for the output planet population table and how many universes
# should be simulated.
Name = 'TestPlanetPopulation' # str
Nuniverses = 10 # int


# =============================================================================
# P-POP
# =============================================================================

# Don't modify the following code.
SysGen = SystemGenerator.SystemGenerator(StarCatalog,
                                         StypeToModel,
                                         ScalingModel,
                                         MassModel,
                                         EccentricityModel,
                                         StabilityModel,
                                         OrbitModel,
                                         AlbedoModel,
                                         ExozodiModel,
                                         Stypes,
                                         Dist_range, # pc
                                         Dec_range, # deg
                                         Scenario,
                                         SummaryPlots,
                                         Ntest,
                                         FigDir,
                                         block)
SysGen.SimulateUniverses(Name,
                         Nuniverses)
