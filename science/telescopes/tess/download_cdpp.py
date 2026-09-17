"""Download the per-sector SPOC RMS CDPP tables (sectors 1..MAX_SECTOR, ~230 MB) from the MAST TCE
bulk-download page into results/catalogs/tess/CDPP, skipping files already there. tess_calibration.py
and build_reference_data.py read them; the TESS P-Pop pipeline does not.
Run from repo root:  python science/telescopes/tess/download_cdpp.py
"""

import re
import sys
import urllib.request
from pathlib import Path

from tools.paths import TESS_DATA_DIR
from science.telescopes.tess.build_reference_data import MAX_SECTOR

PAGE_URL = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html"
FILE_URL = "https://archive.stsci.edu/missions/tess/catalogs/cdpp/{}"
CDPP_DIR = Path(TESS_DATA_DIR) / "CDPP"


def single_sector_files(page: str) -> list[str]:
    """File names of the single-sector tables for sectors 1..MAX_SECTOR. The page also lists
    multi-sector runs (s0001-s0069); those are skipped."""
    names = set()
    for m in re.finditer(r"tess\d+-s(\d{4})-s(\d{4})-\d+_rms-cdpp\.csv", page):
        if m.group(1) == m.group(2) and 1 <= int(m.group(1)) <= MAX_SECTOR:
            names.add(m.group(0))
    return sorted(names)


def main():
    with urllib.request.urlopen(PAGE_URL) as r:
        names = single_sector_files(r.read().decode("utf-8", "replace"))
    if len(names) != MAX_SECTOR:
        sys.exit(f"Expected {MAX_SECTOR} single-sector CDPP tables on {PAGE_URL}, found {len(names)}.")

    CDPP_DIR.mkdir(parents=True, exist_ok=True)
    missing = [n for n in names if not (CDPP_DIR / n).exists()]
    print(f"{len(names) - len(missing)} of {len(names)} CDPP tables already in {CDPP_DIR}")
    for i, name in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {name}", flush=True)
        tmp = CDPP_DIR / (name + ".part")
        urllib.request.urlretrieve(FILE_URL.format(name), tmp)
        tmp.rename(CDPP_DIR / name)


if __name__ == "__main__":
    main()
