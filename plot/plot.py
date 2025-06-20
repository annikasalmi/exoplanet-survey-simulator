from plot.plot_by_type import PlotPlanetType
from plot.plot_detections import PlanetDetectionPlotter
from plot.plot_rejections import PlanetRejectionPlotter


def plot_all(df, nruns=1, star_catalog='Gaia', sim_name='HWO'):

    if sim_name == 'hwo':
        sim_name = 'HWO'
    elif sim_name == 'lifesim':
        sim_name = 'LIFEsim'    
    PlotPlanetType(df=df, name=sim_name, nruns=nruns, star_catalog=star_catalog).plot_all()
    PlanetDetectionPlotter(df=df, name=sim_name, nruns=nruns, star_catalog=star_catalog).plot_all()
    if sim_name == 'HWO':
        PlanetRejectionPlotter(df, nruns=nruns, star_catalog=star_catalog, name=sim_name).plot_all()