from run.hwo.hwo_run_multiple import run_single

df = run_single(i=0, star_catalog="Gaia")

print("\nHWO single run worked.")
print("Number of planets:", len(df))
print("Columns:")
print(df.columns.tolist())

print("\nDetection columns:")
for col in ["detected_best", "detected_worst", "iwa_pass_best", "flux_pass_best"]:
    if col in df.columns:
        print(col, df[col].value_counts(dropna=False).to_dict())

print("\nFirst 5 rows:")
print(df.head()) 