import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from astroquery.gaia import Gaia

# Query all stars within 20 parsecs (parallax > 50 mas)
query = """
SELECT
    source_id, ra, dec, parallax, parallax_error, phot_g_mean_mag,
    pmra, pmdec, radial_velocity
FROM gaiadr3.gaia_source
WHERE parallax > 50
AND parallax IS NOT NULL
AND phot_g_mean_mag IS NOT NULL
"""

# Launch and get results
job = Gaia.launch_job(query)
results = job.get_results()

# Save to CSV
results.write("gaia_within_20pc.csv", format="csv", overwrite=True)
print("Saved as 'gaia_within_20pc.csv'")