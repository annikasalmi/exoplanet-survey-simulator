import numpy as np
import pandas as pd
from tools import physics_constants as const

def load_and_filter_exoplanets(csv_path, instrument='LIFE'):
    """
    Load and filter exoplanet CSV with permissive, calculation-specific filtering.
    Returns a DataFrame ready for LIFEsim or plotting.
    
    Parameters:
    -----------
    csv_path : str
        Path to the exoplanet CSV file
    instrument : str, optional
        'LIFE' or 'HWO' compute a direct-imaging flux ratio (18.5 and 2.5 microns).
        'Kepler' or 'TESS' instead select that mission's discoveries by
        disc_facility; a flux ratio is not meaningful for a transit survey and
        is not computed.
    """
    # NASA Exoplanet Archive exports carry '#' comment headers.
    df = pd.read_csv(csv_path, comment='#', low_memory=False)

    facility_filters = {
        'KEPLER': 'Kepler',
        'TESS': 'Transiting Exoplanet Survey Satellite (TESS)',
    }
    facility = facility_filters.get(instrument.upper())
    if facility is not None:
        if 'disc_facility' not in df.columns:
            raise KeyError(
                f"{csv_path} has no disc_facility column, so {instrument} "
                "discoveries cannot be selected from it.")
        df = df[df['disc_facility'] == facility].copy()

    # Only filter for nulls when needed for calculations
    # For angular separation calculation
    mask_angsep = df['sy_dist'].notnull() & df['pl_orbsmax'].notnull()
    df_angsep = df[mask_angsep].copy()
    df_angsep['AngSep'] = (df_angsep['pl_orbsmax'] / df_angsep['sy_dist']) * 206265
    df['AngSep'] = df_angsep['AngSep']

    # For flux ratio calculation
    mask_flux = df['pl_rade'].notnull() & df['st_rad'].notnull() & df['pl_eqt'].notnull() & df['st_teff'].notnull()
    df_flux = df[mask_flux].copy()
    
    # Set wavelength based on instrument. Transit missions get no flux ratio:
    # the quantity describes direct imaging contrast and does not apply.
    if instrument.upper() == 'LIFE':
        wavelength = 18.5e-6    # 18.5 microns (mid-IR)
    elif instrument.upper() == 'HWO':
        wavelength = 2.5e-6     # 2.5 microns (near-IR)
    elif facility is not None:
        wavelength = None
    else:
        raise ValueError(
            f"Unknown instrument: {instrument}. "
            "Use 'LIFE', 'HWO', 'Kepler' or 'TESS'.")
    def planck(wavelength, T):
        exponent = const.h * const.c / (wavelength * const.k * T)
        exp_term = np.where(exponent > 100, np.inf, np.exp(exponent))
        B_lambda = (2 * const.h * const.c**2) / (wavelength**5) / (exp_term - 1)
        return np.where(np.isinf(exp_term), 0, B_lambda)
    if wavelength is None:
        for col in ['Fp', 'fp', 'flux_ratio_value_best']:
            df[col] = np.nan
    else:
        df_flux['Rp_m'] = df_flux['pl_rade'] * const.R_earth
        df_flux['Rs_m'] = df_flux['st_rad'] * const.R_sun
        df_flux['B_planet'] = planck(wavelength, df_flux['pl_eqt'])
        df_flux['B_star'] = planck(wavelength, df_flux['st_teff'])
        df_flux['Fp'] = (df_flux['Rp_m'] / df_flux['Rs_m'])**2 * (df_flux['B_planet'] / df_flux['B_star'])
        df_flux['fp'] = df_flux['Fp']
        df_flux['flux_ratio_value_best'] = df_flux['Fp']
        for col in ['Fp', 'fp', 'flux_ratio_value_best']:
            df[col] = df_flux[col]

    # Critical columns needed for detection calculations
    # These must be present for a planet to be properly evaluated for detection
    critical_columns = [
        'pl_rade',      # Planet radius - needed for flux calculations
        'sy_dist',      # Distance - needed for angular separation
        'st_rad',       # Star radius - needed for flux calculations
        'pl_eqt',       # Planet equilibrium temperature - needed for flux calculations
        'st_teff'       # Star effective temperature - needed for flux calculations
    ]
    
    # Remove planets missing ANY critical columns
    # This ensures detection percentages are calculated against planets that can actually be evaluated
    missing_critical = df[critical_columns].isnull().any(axis=1)
    
    # Keep only planets with all critical data
    df = df[~missing_critical].copy()

    # Comprehensive column mapping for all use cases
    col_map = {
        # Basic plotting columns
        'luminosity_s': lambda df: 10**df['st_lum'] if 'st_lum' in df.columns else np.nan,
        'distance_s': 'sy_dist',
        'radius_p': 'pl_rade',
        'radius_s': 'st_rad',
        'p_orb': 'pl_orbper',
        'semimajor_p': 'pl_orbsmax',
        'temp_s': 'st_teff',
        'temp_p': 'pl_eqt',
        'mass_p': 'pl_bmasse',
        'detected': True,
        'detected_best': True,
        'detected_worst': True,
        'flux_ratio_value_best': 'flux_ratio_value_best',
        'maxangsep': 'AngSep',
        'z': 0.1,
        'stype': 'G',
        'habitable': lambda df: (df['pl_rade'] < 1.5) & (df['pl_eqt'] >= 270) & (df['pl_eqt'] <= 390),
        'run': 0,
        
        # Additional LIFEsim columns
        'Rp': 'pl_rade',
        'Porb': 'pl_orbper',
        'Mp': 'pl_bmasse',
        'ep': 0.0,
        'ecc_p': 0.0,
        'ip': 0.0,
        'inc_p': 0.0,
        'Omegap': 0.0,
        'large_omega_p': 0.0,
        'omegap': 0.0,
        'small_omega_p': 0.0,
        'thetap': 'AngSep',
        'Abond': 0.3,
        'AgeomVIS': 0.3,
        'AgeomMIR': 0.3,
        'ap': 'pl_orbsmax',
        'rp': 'pl_orbsmax',
        'AngSep': 'AngSep',
        'maxAngSep': 'AngSep',
        'Fp': 'Fp',
        'fp': 'fp',
        'Tp': 'pl_eqt',
        'Msun': 'st_mass',
        'Nuniverse': 1,
        'Nstar': 1,
        'nstar': 1,
        'ra': 0.0,
        'dec': 0.0,
        'name_s': 'st_refname',
        'id': lambda df: np.arange(len(df)),
        'Rs': 'st_rad',
        'Ms': 'st_mass',
        'Ts': 'st_teff',
        'Ds': 'sy_dist',
        'RA': 0.0,
        'Dec': 0.0,
    }
    
    for col, val in col_map.items():
        if callable(val):
            df[col] = val(df)
        elif isinstance(val, str) and val in df.columns:
            df[col] = df[val]
        else:
            df[col] = val
    
    # Assign star types based on luminosity
    df.loc[df['luminosity_s'] < 0.1, 'stype'] = 'M'
    df.loc[(df['luminosity_s'] >= 0.1) & (df['luminosity_s'] < 0.6), 'stype'] = 'K'
    
    # Ensure all numeric columns are properly converted
    numeric_columns = [
        'radius_p', 'Rp', 'Porb', 'Mp', 'semimajor_p', 'ap', 'rp',
        'temp_p', 'Tp', 'temp_s', 'Ts', 'radius_s', 'Rs', 'Ms', 'Msun',
        'distance_s', 'Ds', 'luminosity_s', 'mass_p'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Add radius binning for plotting
    bins = [0, 1.5, 3.0, 6.0]
    labels = ['<1.5', '1.5–3.0', '3.0–6.0']
    df['radius_bin'] = pd.cut(df['radius_p'], bins=bins, labels=labels, include_lowest=True)

    return df 