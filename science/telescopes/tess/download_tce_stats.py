"""Download the SPOC DV TCE statistics tables from the MAST bulk-download page into
results/catalogs/tess/TCE. They carry tce_max_mult_ev, the MES the pipeline actually detected on,
which the ExoFOP TOI table does not publish. One table per multi-sector run, so a TOI must be read
from the run named in its Source column.
Run from repo root:  python science/telescopes/tess/download_tce_stats.py
"""

import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

from tools.paths import TESS_DATA_DIR

PAGE_URL = "https://archive.stsci.edu/tess/bulk_downloads/bulk_downloads_tce.html"
FILE_URL = "https://archive.stsci.edu/missions/tess/catalogs/tce/{}"
TCE_DIR = Path(TESS_DATA_DIR) / "TCE"
KEEP = ["ticid", "tce_plnt_num", "tce_period", "tce_max_mult_ev", "tce_model_snr"]


def available_spans() -> dict[tuple[int, int], str]:
    """(first sector, last sector) -> file name, for every multi-sector run on the page."""
    page = urllib.request.urlopen(PAGE_URL).read().decode("utf-8", "replace")
    spans = {}
    for name in sorted(set(re.findall(r"[\w\-]+_dvr-tcestats\.csv", page))):
        m = re.search(r"-s(\d{4})-s(\d{4})_dvr", name)
        if m:
            spans[(int(m.group(1)), int(m.group(2)))] = name
    return spans


def fetch(spans_wanted, force: bool = False) -> dict[tuple[int, int], Path]:
    """Download only the runs asked for; each is trimmed to the few columns the figure needs."""
    TCE_DIR.mkdir(parents=True, exist_ok=True)
    have = available_spans()
    out = {}
    for span in sorted(set(spans_wanted)):
        if span not in have:
            continue
        slim = TCE_DIR / f"tcestats_s{span[0]:04d}_s{span[1]:04d}.csv"
        if slim.exists() and not force:
            out[span] = slim
            continue
        url = FILE_URL.format(have[span])
        print(f"  downloading {have[span]} ...", flush=True)
        raw = TCE_DIR / have[span]
        urllib.request.urlretrieve(url, raw)
        df = pd.read_csv(raw, comment="#", low_memory=False)
        df[[c for c in KEEP if c in df.columns]].to_csv(slim, index=False)
        raw.unlink()
        out[span] = slim
    return out


def main():
    spans = sorted(available_spans())
    print(f"{len(spans)} multi-sector runs listed; pass spans via tess_calibration instead of "
          f"downloading all of them.")
    if "--all" in sys.argv:
        fetch(spans)


if __name__ == "__main__":
    main()
