# =============================================================================
# STARCATALOG FOR GAIA DR3 DATA  (volume-limited, M-dwarf-complete to 60 pc)
# =============================================================================
#
# Input file: gaia_within_60pc.csv  (built by build_gaia_60pc.py)
# Columns:    SOURCE_ID, ra, dec, parallax, parallax_error,
#             phot_g_mean_mag, bp_rp, ruwe
#
# Stellar parameters (Teff, radius, mass) are NOT taken from Gaia's GSP-Phot /
# FLAME products, because those are missing for most faint M dwarfs and would
# silently drop the bulk of the nearby population.  Instead we derive Teff, radius
# and mass for EVERY star from its BP-RP colour using a main-sequence relation
# (anchored on Pecaut & Mamajek 2013), so no star is lost for lacking a parameter.
# White dwarfs and giants are removed with simple colour-magnitude cuts.
# =============================================================================

import os

import astropy.table as at
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from tools.paths import PPOP_DIR


# -----------------------------------------------------------------------------
# Main-sequence colour -> (Teff, radius, mass) relation.
# Anchor points (BP-RP, Teff[K], R[Rsun], M[Msun]) along the dwarf sequence,
# A0V -> M9V.  Monotonic in BP-RP; values interpolated and clamped at the ends.
# -----------------------------------------------------------------------------
_MS_BPRP = np.array([-0.04, 0.49, 0.59, 0.77, 0.92, 1.00, 1.45,
                     1.85, 2.20, 2.50, 2.85, 3.10, 3.40, 4.20, 5.00])
_MS_TEFF = np.array([9700., 7200., 6500., 5900., 5500., 5280., 4400.,
                     3870., 3550., 3400., 3150., 3000., 2850., 2500., 2300.])
_MS_RAD = np.array([2.10, 1.50, 1.30, 1.10, 0.92, 0.85, 0.70,
                    0.59, 0.45, 0.36, 0.26, 0.20, 0.15, 0.11, 0.09])
_MS_MASS = np.array([2.10, 1.60, 1.30, 1.05, 0.92, 0.88, 0.68,
                     0.57, 0.44, 0.35, 0.21, 0.15, 0.10, 0.085, 0.08])


def _teff_from_bprp(bp_rp):
    return np.interp(bp_rp, _MS_BPRP, _MS_TEFF)


def _radius_from_bprp(bp_rp):
    return np.interp(bp_rp, _MS_BPRP, _MS_RAD)


def _mass_from_bprp(bp_rp):
    return np.interp(bp_rp, _MS_BPRP, _MS_MASS)


def _classify_sptype(teff):
    """Spectral type from effective temperature (dwarf scale)."""
    if teff > 7500:
        return 'A'
    elif teff > 6000:
        return 'F'
    elif teff > 5200:
        return 'G'
    elif teff > 3700:
        return 'K'
    elif teff > 2000:
        return 'M'
    return 'X'  # unknown / out of range


class StarCatalog():

    def __init__(self,
                 Stypes=['A', 'F', 'G', 'K', 'M'],
                 Dist_range=[0, 60],   # pc
                 Dec_range=[-90, 90],  # deg
                 Path=os.path.join(PPOP_DIR, 'StarCatalogs', 'gaia_within_60pc.csv')):
        self.SC = self.read(Stypes, Dist_range, Dec_range, Path)

    def read(self, Stypes, Dist_range, Dec_range, Path):
        print('--> Reading Gaia star catalog: %s' % os.path.basename(Path))

        df = pd.read_csv(Path)

        # --- distance from parallax (mas) ---
        plx = pd.to_numeric(df['parallax'], errors='coerce').to_numpy()
        bp_rp = pd.to_numeric(df['bp_rp'], errors='coerce').to_numpy()
        gmag = pd.to_numeric(df['phot_g_mean_mag'], errors='coerce').to_numpy()
        ra = pd.to_numeric(df['ra'], errors='coerce').to_numpy()
        dec = pd.to_numeric(df['dec'], errors='coerce').to_numpy()
        name = df['SOURCE_ID'].astype(str).to_numpy()

        good = np.isfinite(plx) & (plx > 0) & np.isfinite(bp_rp) & np.isfinite(gmag)
        dist = np.full_like(plx, np.nan)
        dist[good] = 1000.0 / plx[good]

        # --- derive stellar parameters from colour (main sequence) ---
        teff = _teff_from_bprp(bp_rp)
        rad = _radius_from_bprp(bp_rp)
        mass = _mass_from_bprp(bp_rp)

        # absolute G magnitude, for white-dwarf / giant rejection
        with np.errstate(invalid='ignore', divide='ignore'):
            abs_g = gmag - 5.0 * np.log10(dist) + 5.0

        # White dwarfs: blue but very sub-luminous.  Giants: red but very luminous.
        is_wd = (bp_rp < 1.5) & (abs_g > 8.0)
        is_giant = (bp_rp > 1.0) & (abs_g < 3.0)
        keep_phys = good & ~is_wd & ~is_giant

        Nin = int(good.sum())

        # --- build per-star records and apply the selection cuts ---
        names, dists, stypes = [], [], []
        rads, teffs, masses = [], [], []
        ras, decs = [], []

        for i in np.where(keep_phys)[0]:
            st = _classify_sptype(teff[i])
            if st not in Stypes:
                continue
            if not (Dist_range[0] <= dist[i] <= Dist_range[1]):
                continue
            if not (Dec_range[0] <= dec[i] <= Dec_range[1]):
                continue
            names.append(name[i])
            dists.append(dist[i])
            stypes.append(st)
            rads.append(rad[i])
            teffs.append(teff[i])
            masses.append(mass[i])
            ras.append(ra[i])
            decs.append(dec[i])

        n = len(names)
        # Galactic coordinates (vectorised).
        if n > 0:
            coord = SkyCoord(ra=np.array(ras) * u.deg, dec=np.array(decs) * u.deg, frame='icrs')
            lGal = coord.galactic.l.deg
            bGal = coord.galactic.b.deg
        else:
            lGal = np.array([])
            bGal = np.array([])

        nan = np.full(n, np.nan)
        inf = np.full(n, np.inf)
        SC_out = at.Table(
            data=[np.array(names, dtype='S32'),
                  np.array(dists, dtype='d'),
                  np.array(stypes, dtype='c'),
                  np.array(rads, dtype='d'),
                  np.array(teffs, dtype='d'),
                  np.array(masses, dtype='d'),
                  np.array(ras, dtype='d'),
                  np.array(decs, dtype='d'),
                  nan, nan, nan,   # Vmag, Jmag, Hmag (unknown)
                  inf, inf,        # WDSsep, WDSdmag (no companion info)
                  np.asarray(lGal, dtype='d'),
                  np.asarray(bGal, dtype='d')],
            names=('Name', 'Dist', 'Stype', 'Rad', 'Teff', 'Mass',
                   'RA', 'Dec', 'Vmag', 'Jmag', 'Hmag',
                   'WDSsep', 'WDSdmag', 'lGal', 'bGal'))

        # --- report ---
        print('--> Including %d / %d stars (%.1f%%) after type/distance/quality cuts'
              % (n, Nin, 100.0 * n / Nin if Nin else 0.0))
        if n > 0:
            uniq, cnt = np.unique(np.array(stypes), return_counts=True)
            comp = ', '.join('%s:%d' % (s, c) for s, c in zip(uniq, cnt))
            print('--> Spectral-type census: %s' % comp)
            print('--> Distance in [%.2f, %.2f] pc' % (np.min(dists), np.max(dists)))

        return SC_out

    def SummaryPlot(self, FigDir=None, block=True):
        f, ax = plt.subplots(2, 3)
        ax[0, 0].hist(self.SC['Dist'], bins=25)
        ax[0, 0].grid(axis='y')
        ax[0, 0].set_xlabel('Distance [pc]')
        ax[0, 0].set_ylabel('Number')

        ax[0, 1].hist(self.SC['Mass'], bins=25)
        ax[0, 1].grid(axis='y')
        ax[0, 1].set_xlabel('Mass [$M_\\odot$]')
        ax[0, 1].set_ylabel('Number')

        ax[0, 2].hist(self.SC['Rad'], bins=25)
        ax[0, 2].grid(axis='y')
        ax[0, 2].set_xlabel('Radius [$R_\\odot$]')
        ax[0, 2].set_ylabel('Number')

        ax[1, 0].hist(self.SC['Teff'], bins=25)
        ax[1, 0].grid(axis='y')
        ax[1, 0].set_xlabel('Effective temperature [K]')
        ax[1, 0].set_ylabel('Number')

        ax[1, 1].scatter(self.SC['Teff'], self.SC['Mass'], c=self.SC['Teff'], cmap='jet_r', s=2)
        ax[1, 1].invert_xaxis()
        ax[1, 1].grid()
        ax[1, 1].set_xlabel('Effective temperature [K]')
        ax[1, 1].set_ylabel('Mass [$M_\\odot$]')

        plt.subplot(236, projection='aitoff')
        plt.grid()
        plt.scatter(((self.SC['RA'] * np.pi / 180. + np.pi) % (2. * np.pi)) - np.pi,
                    self.SC['Dec'] * np.pi / 180., s=2)
        plt.xlabel('Right ascension [deg]')
        plt.ylabel('Declination [deg]')

        plt.tight_layout()
        if FigDir:
            plt.savefig(os.path.join(FigDir, 'StarCatalog.pdf'))
        plt.show(block=block)
        plt.close()
