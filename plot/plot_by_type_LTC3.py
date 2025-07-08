import matplotlib.pyplot as plt
from plot.plot_by_type import PlotPlanetType

class PlotPlanetTypeLTC3(PlotPlanetType):
    def _create_overlay_bars(self, ax, x, total_heights, detected_heights, 
                           total_errors=None, detected_errors=None,
                           bar_width=0.8, total_color='lightgray', 
                           detected_color='green', detected_hatch=None,
                           add_total_label=True, detected_label='Detected'):
        """Override to only plot detected bars (no gray background)."""
        # Only plot detected bars
        ax.bar(x, detected_heights, width=bar_width, color=detected_color,
               alpha=0.8, edgecolor='black', yerr=detected_errors, capsize=3,
               bottom=None, hatch=detected_hatch,
               label=detected_label, ecolor='black')
        if detected_errors is not None:
            return detected_heights, detected_errors
        else:
            return detected_heights, None

# The following function can be called from plot.py or run_sim.py

def plot_by_type_LTC3(df, nruns=1, star_catalog='LTC_3', name='LIFEsim'):
    plotter = PlotPlanetTypeLTC3(df=df, nruns=nruns, star_catalog=star_catalog, name=name)
    plotter.plot_all() 