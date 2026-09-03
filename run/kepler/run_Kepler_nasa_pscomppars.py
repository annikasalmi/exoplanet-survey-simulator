from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd


def find_project_root(start_path: Path) -> Path:
    start_path = start_path.resolve()
    for p in [start_path] + list(start_path.parents):
        if (p / "run" / "kepler").exists() and (p / "lifesim").exists():
            return p
        if (p / "run" / "kepler").exists():
            return p
    return start_path.parents[1]


ROOT = find_project_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.kepler_data import KeplerData  # noqa: E402


NASA_DATA_DIR = ROOT / "run" / "kepler" / "data" / "NASA"
NASA_INPUT_CSV = NASA_DATA_DIR / "NASA_PSCompPars_transiting_confirmed_RM_insolation.csv"
NASA_OUTPUT_CSV = NASA_DATA_DIR / "kepler_catalog_nasa_pscomppars.csv"

# Optional: keep all official transiting confirmed planets in your file.
# Set to "Kepler" if you only want planets discovered by Kepler.
FILTER_DISC_FACILITY = None
# FILTER_DISC_FACILITY = "Kepler"

FALLBACK_CDPP_PPM = 100.0
MISSION_DURATION_DAYS = 4 * 365.25
MIN_TRANSITS = 3
MES_THRESHOLD = 7.1
KEPLER_MAG_LIMIT = 16.0


def run_nasa_pscomppars(
    input_csv: Path = NASA_INPUT_CSV,
    output_csv: Path = NASA_OUTPUT_CSV,
    filter_disc_facility: str | None = FILTER_DISC_FACILITY,
) -> pd.DataFrame:
    print("Project root:", ROOT)
    print("Loading NASA PSCompPars CSV:")
    print(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            f"Could not find NASA input CSV:\n{input_csv}\n\n"
            "Put NASA_PSCompPars_transiting_confirmed_RM_insolation.csv in run/kepler/data/NASA/."
        )

    df = pd.read_csv(input_csv)
    print(f"Raw NASA rows: {len(df):,}")

    if filter_disc_facility is not None and "disc_facility" in df.columns:
        before = len(df)
        df = df[df["disc_facility"].astype(str) == filter_disc_facility].copy()
        print(f"Filtered disc_facility == {filter_disc_facility!r}: {before:,} -> {len(df):,}")

    # The NASA file is already transiting confirmed planets. For NASA, we use tran_flag
    # as the transit condition rather than recalculating geometry from noisy/incomplete parameters.
    model = KeplerData(
        df,
        source="pscomppars",
        validate_for_detection=True,
        fallback_cdpp_ppm=FALLBACK_CDPP_PPM,
        mission_duration_days=MISSION_DURATION_DAYS,
        min_transits=MIN_TRANSITS,
        mes_threshold=MES_THRESHOLD,
        kepler_mag_limit=KEPLER_MAG_LIMIT,
        use_observed_transit_flag_for_nasa=True,
        use_observed_transit_depth_for_nasa=True,
        assume_bright_if_kepmag_missing_for_nasa=True,
    )

    out = model.determine_detectable()
    out["run"] = 0
    out["model_input_file"] = input_csv.name
    out["model_population"] = "NASA_PSCompPars_transiting_confirmed_RM_insolation"

    # Useful radius bins for older plotting code.
    out["radius_bin"] = pd.cut(
        out["radius_p"],
        bins=[0, 1.5, 3.0, 6.0, float("inf")],
        labels=["<1.5", "1.5–3.0", "3.0–6.0", ">6.0"],
        include_lowest=True,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    print("\nSaved Kepler-on-NASA catalog:")
    print(output_csv)

    print("\nDetector summary:")
    for col in [
        "transiting_geometric",
        "bright_enough_kepler",
        "kepler_enough_transits",
        "kepler_depth_good",
        "detected",
    ]:
        if col in out.columns:
            print(f"\n{col}:")
            print(out[col].value_counts(dropna=False))

    print("\nSource diagnostics:")
    for col in ["kepler_mag_source", "transit_depth_source", "kepler_cdpp_source", "transiting_source"]:
        if col in out.columns:
            print(f"\n{col}:")
            print(out[col].value_counts(dropna=False).head(12))

    print("\nRows:", len(out))
    print("Columns:", out.columns.tolist())

    return out


def main():
    return run_nasa_pscomppars()


if __name__ == "__main__":
    main()
