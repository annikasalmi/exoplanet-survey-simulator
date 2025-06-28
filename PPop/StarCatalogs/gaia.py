# =============================================================================
# STARCATALOG FOR GAIA DR3 DATA
# =============================================================================

import astropy.table as at
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from astropy.coordinates import SkyCoord
import astropy.units as u

from tools.paths import PPOP_DIR  # Adjust if needed

# =============================================================================
# P-POP STYLE GAIA STAR CATALOG CLASS
# =============================================================================

import astropy.table as at
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
from astropy.coordinates import SkyCoord
import astropy.units as u

from tools.paths import PPOP_DIR  # adjust as needed

class StarCatalog():
    def __init__(self,
                 Stypes=['A', 'F', 'G', 'K', 'M'],
                 Dist_range=[0, 30],  # pc
                 Dec_range=[-90, 90],  # deg
                 Path=os.path.join(PPOP_DIR, 'StarCatalogs', 'gaia_within_20pc.csv')):
        self.SC = self.read(Stypes, Dist_range, Dec_range, Path)

    def classify_sptype(self, teff):
        """Rough spectral type classification by Teff"""
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
        else:
            return 'X'  # unknown/invalid

    def read(self, Stypes, Dist_range, Dec_range, Path):
        # print(f'--> Reading Gaia star catalog: {Path}')

        # Read input CSV as list of dicts
        with open(Path, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)

        Nin = 0
        Name, Dist, Stype = [], [], []
        Rad, Teff, Mass = [], [], []
        RA, Dec = [], []
        Vmag, Jmag, Hmag = [], [], []
        WDSsep, WDSdmag = [], []
        lGal, bGal = [], []

        for row in data:
            try:
                source_id = row['SOURCE_ID']
                ra = float(row['ra'])
                dec = float(row['dec'])
                parallax = float(row['parallax'])
                gmag = float(row['phot_g_mean_mag'])
                teff = float(row['teff_gspphot'])
                radius = float(row['radius_gspphot'])
                mass = float(row['mass_flame'])
            except (ValueError, KeyError):
                continue

            if parallax <= 0:
                continue
            dist_pc = 1000.0 / parallax

            if not (Dist_range[0] <= dist_pc <= Dist_range[1]):
                continue
            if not (Dec_range[0] <= dec <= Dec_range[1]):
                continue

            sptype = self.classify_sptype(teff)
            if sptype not in Stypes:
                continue

            Nin += 1
            Name.append(source_id)
            Dist.append(dist_pc)
            Stype.append(sptype)
            Rad.append(radius)
            Teff.append(teff)
            Mass.append(mass)
            RA.append(ra)
            Dec.append(dec)
            Vmag.append(np.nan)
            Jmag.append(np.nan)
            Hmag.append(np.nan)
            WDSsep.append(np.inf)
            WDSdmag.append(np.inf)

            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
            lGal.append(coord.galactic.l.deg)
            bGal.append(coord.galactic.b.deg)

        # Build output astropy Table
        SC_out = at.Table(
            names=('Name', 'Dist', 'Stype', 'Rad', 'Teff', 'Mass',
                   'RA', 'Dec', 'Vmag', 'Jmag', 'Hmag',
                   'WDSsep', 'WDSdmag', 'lGal', 'bGal'),
            dtype=('S32', 'd', 'c', 'd', 'd', 'd',
                   'd', 'd', 'd', 'd', 'd',
                   'd', 'd', 'd', 'd'))

        for i in range(len(Name)):
            SC_out.add_row([
                Name[i], Dist[i], Stype[i], Rad[i], Teff[i], Mass[i],
                RA[i], Dec[i], Vmag[i], Jmag[i], Hmag[i],
                WDSsep[i], WDSdmag[i], lGal[i], bGal[i]
            ])

        # print(f'--> Included {len(SC_out)} / {Nin} stars ({len(SC_out)/float(Nin)*100:.2f}%)')
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

