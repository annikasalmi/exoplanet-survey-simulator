import numpy as np
from run.hwo.hwo_run_multiple import main

df = main(
    parallel=False,
    nruns=np.arange(2),
    star_catalog="Gaia",
    run_anew=True,
)

print("\nHWO multiple-run mode worked.")
print("Number of rows:", len(df))
print("Run counts:")
print(df["run"].value_counts().sort_index())

print("\nDetection summary:")
for col in ["detected_best", "detected_worst"]:
    if col in df.columns:
        print(col)
        print(df[col].value_counts(dropna=False))