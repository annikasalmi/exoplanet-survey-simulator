import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import PLOTS_DIR
import tools.constants as const

def make_output_dir(name, nruns, star_catalog):
    out_dir = os.path.join(PLOTS_DIR, str(name)+'_'+str(nruns)+'_'+str(star_catalog))
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def temp_zone(temp):
    '''
    Assigns a temperature zone based on the temperature value.'''
    if temp > 600:
        return 'hot'
    elif temp > 300:
        return 'warm'
    else:
        return 'cold'

def assign_category(row):
    '''
    Assigns a category based on the planet's radius, habitability, and star type.'''
    r = row['radius_p']
    hab = row['habitable']
    stype = row['stype']

    if r < 1.5 and hab:
        return 'Rocky eHZ'
    elif r < 1.8 and hab and stype in ['G', 'K']:
        return 'Exo-Earth Candidates'
    elif 1.0 <= r < 2.0:
        return 'Rocky + Super-Earths'
    elif 2.0 <= r < 4.0:
        return 'Sub-Neptunes'
    elif 4.0 <= r < 8.0:
        return 'Sub-Jovians'
    else:
        return None
    

def get_rejection_reason(row):
    if not row['iwa_pass']:
        return 'IWA'
    elif not row['flux_pass']:
        return 'Flux Ratio'
    elif not row['min_photons_pass']:
        return 'Min Photons'
    else:
        return 'Detected'
    

# Function to compute mean and std by run
def pivot_stats_temp(df_filtered):
    grouped = df_filtered.groupby(['run', 'category', 'temp_zone']).size().reset_index(name='count')
    pivot = grouped.pivot_table(index='run', columns=['category', 'temp_zone'], values='count', fill_value=0)
    mean = pivot.mean(axis=0).reset_index(name='count')
    std = pivot.std(axis=0).reset_index(name='error')
    return pd.merge(mean, std, on=['category', 'temp_zone'])


def pivot_stats_radius(df_grouped):
    pivot = df_grouped.pivot_table(index='run', columns=['stype', 'radius_bin'], values='count', fill_value=0)
    mean = pivot.mean(axis=0).reset_index(name='count')
    std = pivot.std(axis=0).reset_index(name='error')
    merged = pd.merge(mean, std, on=['stype', 'radius_bin'])
    return merged