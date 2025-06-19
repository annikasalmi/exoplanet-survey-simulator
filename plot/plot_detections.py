import os
import numpy as np
import matplotlib.pyplot as plt

from plot.helpers import make_output_dir

def plot_efficiency_rocky(df, nruns=1, star_catalog='Gaia', name='hwo'):

    # Filter only rocky eHZ planets
    rocky_ehz = df[df['radius_bin'] == 'Rocky HZ']

    # Bin edges
    bins = np.linspace(125, 305, 40)  # ~4.6 K per bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Histogram of all and detected
    total_counts, _ = np.histogram(rocky_ehz['temp_p'], bins=bins)
    try:
        mask = rocky_ehz['detected']
    except KeyError:
        mask = rocky_ehz['detectable']

    detected_counts, _ = np.histogram(
        rocky_ehz[mask]['temp_p'],
        bins=bins
    )

    # Avoid divide-by-zero
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(detected_counts, total_counts)
        efficiency[np.isnan(efficiency)] = 0.0

    # === Plot ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plots
    ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', color='lightgrey', label='Rocky, eHZ planets present')
    ax1.bar(bin_centers, detected_counts, width=np.diff(bins), align='center', color='green', label='Rocky, eHZ planets detectable')

    ax1.set_ylabel("Number of rocky, eHZ planets")
    ax1.set_xlabel("Temperature [K]")
    ax1.set_xlim(bins[0], bins[-1])

    # Secondary axis for detection efficiency
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Detection efficiency')
    ax2.set_ylabel("Detection efficiency")
    ax2.set_ylim(0, 1.0)

    # Legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    ax1.set_title(f'Efficiency in Detecting Planets for {name} for {nruns} Runs\nStar Catalog: {star_catalog}')

    plt.tight_layout()
    data_dir = make_output_dir(name, nruns, star_catalog)

    plt.savefig(os.path.join(data_dir, f"detection_efficiency_rocky_{name}_nruns{nruns}_{star_catalog}.png"), 
                             dpi=300, bbox_inches='tight')
    


def plot_detection_efficiency(df, nruns=1, star_catalog='Gaia', name='hwo', 
                              category_column=None, category_label=None):
    """
    Plot detection efficiency for planetary candidates based on temperature.
    
    Parameters:
    - df: DataFrame containing at least 'temp_p' and 'detected' or 'detectable' columns
    - nruns: Number of runs used in the simulation
    - star_catalog: Name of the star catalog used
    - name: Name for output files
    - category_column: Optional column to filter by category (e.g., 'radius_bin')
    - category_label: Label within category_column to filter (e.g., 'Rocky HZ')
    """

    # Optional filtering by category
    if category_column and category_label:
        df = df[df[category_column] == category_label]

    # Bin setup
    bins = np.linspace(125, 305, 40)  # ~4.6 K per bin
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Total count per bin
    total_counts, _ = np.histogram(df['temp_p'], bins=bins)

    # Handle detection mask flexibly
    try:
        mask = df['detected']
    except KeyError:
        mask = df['detectable']

    detected_counts, _ = np.histogram(df[mask]['temp_p'], bins=bins)

    # Compute efficiency safely
    with np.errstate(divide='ignore', invalid='ignore'):
        efficiency = np.true_divide(detected_counts, total_counts)
        efficiency[np.isnan(efficiency)] = 0.0

    # === Plotting ===
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Histogram bars
    ax1.bar(bin_centers, total_counts, width=np.diff(bins), align='center', 
            color='lightgrey', label='All planets present')
    ax1.bar(bin_centers, detected_counts, width=np.diff(bins), align='center', 
            color='green', label='Planets detectable')

    ax1.set_ylabel("Number of planets")
    ax1.set_xlabel("Temperature [K]")
    ax1.set_xlim(bins[0], bins[-1])

    # Efficiency curve on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(bin_centers, efficiency, 'r--', linewidth=2, label='Detection efficiency')
    ax2.set_ylabel("Detection efficiency")
    ax2.set_ylim(0, 1.0)

    # Combine legends
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='upper left')

    # Title
    category_str = f" ({category_label})" if category_label else ""
    ax1.set_title(f'Detection Efficiency{category_str} for {name} ({nruns} runs)\nStar Catalog: {star_catalog}')

    plt.tight_layout()

    # Output
    data_dir = make_output_dir(name, nruns, star_catalog)
    outfile = f"detection_efficiency_{name}_nruns{nruns}_{star_catalog}.png"
    plt.savefig(os.path.join(data_dir, outfile), dpi=300, bbox_inches='tight')

def plot_detection_mr(df, nruns=1, star_catalog='Gaia', name='hwo'):
    plt.figure(figsize=(8,6))

    # Separate by detection status
    detected = df[df['detected'] == 1]
    not_detected = df[df['detected'] == 0]

    # Plot detected (green) and not detected (red)
    plt.scatter(not_detected['radius_p'], not_detected['mass_p'],
                color='red', alpha=0.6, label='Not Detected')
    plt.scatter(detected['radius_p'], detected['mass_p'],
                color='green', alpha=0.6, label='Detected')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel("Planet Radius ($R_\\oplus$)")
    plt.ylabel("Planet Mass ($M_\\oplus$)")
    plt.title("Detection Likelihood by Planet Radius and Mass (Log Scale)")
    plt.legend()
    plt.tight_layout()

    # Save figure
    data_dir = make_output_dir(name, nruns, star_catalog)
    outfile = f"detection_mr_log_{name}_nruns{nruns}_{star_catalog}.png"
    plt.savefig(os.path.join(data_dir, outfile), dpi=300, bbox_inches='tight')

def plot_detection_vs_distance_color(df, nruns=1, star_catalog='Gaia', name='hwo'):
    xvars = ['flux_p', 'maxangsep', 'flux_ratio', 'photon_rate', 'planet_flux', 'temp_p']
    if name == 'LIFEsim':
        xtitles = {
            'flux_p': 'Planet Flux ($W/m^2$)',
            'maxangsep': 'Maximum Angular Separation (arcsec)',
        }
    else:
        xtitles = {
            'flux_p': 'Planet Flux ($W/m^2$)',
            'maxangsep': 'Maximum Angular Separation (arcsec)',
            'flux_ratio': 'Planet/Star Flux Ratio',
            'photon_rate': 'Photon Rate (photons/s/m²)',
        }

    data_dir = make_output_dir(name, nruns, star_catalog)

    for var in xvars:
        # Filter to valid, positive values
        df_plot = df[(df[var] > 0) & (df['distance_s'] > 0)]

        # Separate by detection
        detected = df_plot[df_plot['detected'] == 1]
        not_detected = df_plot[df_plot['detected'] == 0]

        plt.figure(figsize=(8,6))
        plt.scatter(not_detected[var], not_detected['distance_s'],
                    color='red', alpha=0.6, label='Not Detected')
        plt.scatter(detected[var], detected['distance_s'],
                    color='green', alpha=0.6, label='Detected')

        plt.xscale('log')  # log scale appropriate for most x variables
        plt.yscale('log')  # distance can also span orders of magnitude

        plt.xlabel(xtitles.get(var, var))
        plt.ylabel("Distance to Star (pc)")
        plt.title(f"{xtitles.get(var, var)} vs. Distance to Star")
        plt.legend()
        plt.tight_layout()

        outfile = f"{var}_vs_distance_{name}_nruns{nruns}_{star_catalog}.png"
        plt.savefig(os.path.join(data_dir, outfile), dpi=300, bbox_inches='tight')
        plt.close()

def plot_detections(df, nruns=1, star_catalog='Gaia', name='hwo'):
    plot_detection_efficiency(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_efficiency_rocky(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_detection_mr(df, nruns=nruns, star_catalog=star_catalog, name=name)
    plot_detection_vs_distance_color(df, nruns=nruns, star_catalog=star_catalog, name=name)