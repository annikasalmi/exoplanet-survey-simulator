import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from astroquery.gaia import Gaia

# Query all stars within 20 parsecs (parallax > 50 mas)
query = """SELECT
    gs.source_id,
    gs.ra, gs.dec,
    gs.parallax, gs.parallax_error,
    gs.phot_g_mean_mag,
    gs.pmra, gs.pmdec,
    gs.radial_velocity,
    ap.teff_gspphot,
    ap.radius_gspphot,
    ap.mass_flame,
    gs.bp_rp
FROM gaiadr3.gaia_source AS gs
LEFT JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id
WHERE gs.parallax > 50
  AND gs.parallax IS NOT NULL
  AND gs.phot_g_mean_mag IS NOT NULL
  AND ap.teff_gspphot IS NOT NULL
  AND ap.radius_gspphot IS NOT NULL
  AND ap.mass_flame IS NOT NULL

"""

# Launch and get results
job = Gaia.launch_job(query)
results = job.get_results()

# Save to CSV
results.write("gaia_within_20pc.csv", format="csv", overwrite=True)
print("Saved as 'gaia_within_20pc.csv'")