import astropy.table as at
import pandas as pd

class GaiaStarCatalog:
    def __init__(self, 
                 path='gaia_within_20pc.csv', 
                 dist_range=[0, 20], 
                 dec_range=[-90, 90]):
        """
        Parameters
        ----------
        path: str
            Path to the Gaia CSV file.
        dist_range: list
            Distance range (pc) to include.
        dec_range: list
            Declination range (deg) to include.
        """
        self.catalog = self.read_csv(path, dist_range, dec_range)
    
    def read_csv(self, path, dist_range, dec_range):
        print(f"--> Reading Gaia catalog from {path}")
        
        df = pd.read_csv(path)

        # Calculate distance from parallax (parallax in mas → distance in pc)
        df["Dist"] = 1000. / df["parallax"]

        # Apply filters
        df = df[df["Dist"].between(dist_range[0], dist_range[1])]
        df = df[df["dec"].between(dec_range[0], dec_range[1])]

        # Create an astropy Table
        table = at.Table.from_pandas(df)
        print(f"--> Final catalog contains {len(table)} stars")
        return table