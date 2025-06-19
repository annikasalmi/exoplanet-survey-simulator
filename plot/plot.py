from plot.plot_by_type import plot_by_type
from plot.plot_detections import plot_detections
from plot.plot_rejections import plot_rejections


def plot_all(df, nruns=1, star_catalog='Gaia', sim_name='hwo'):

    if sim_name == 'hwo':
        name = 'HWO'
    elif sim_name == 'lifesim':
        name = 'LIFEsim'

    plot_by_type(df=df, name=name, nruns=nruns, star_catalog=star_catalog)
    plot_detections(df=df, name=name, nruns=nruns, star_catalog=star_catalog)
    if name == 'HWO':
        plot_rejections(df, nruns=nruns, star_catalog=star_catalog, name=name)