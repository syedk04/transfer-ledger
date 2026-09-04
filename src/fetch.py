"""Download and cache the raw transfermarkt-datasets CSVs.

We use plain `requests` rather than pandas.read_csv(url) directly for two
reasons: (1) we want the files to persist on disk as a cache so re-running
the pipeline doesn't re-hit the network every time, and (2) streaming the
download in chunks avoids holding a ~40MB file entirely in memory before
it's even written out.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path

import requests

from config import DATA_BASE_URL, RAW_DATA_DIR, RAW_FILES


def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk, writing to a temp file first.

    Writing to `dest.tmp` then renaming avoids leaving a corrupt/partial
    file at the final path if the download is interrupted midway.
    """
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.rename(dest)


def fetch_raw_data(force: bool = False) -> dict[str, Path]:
    """Download each raw CSV (decompressed) into RAW_DATA_DIR, caching on disk.

    Parameters
    ----------
    force:
        If True, re-download and overwrite even if a cached copy exists.

    Returns
    -------
    Mapping of table name -> path to the decompressed .csv file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    for name, gz_filename in RAW_FILES.items():
        csv_path = RAW_DATA_DIR / f"{name}.csv"
        gz_path = RAW_DATA_DIR / gz_filename

        if csv_path.exists() and not force:
            print(f"[fetch] {name}: using cached {csv_path}")
            paths[name] = csv_path
            continue

        url = f"{DATA_BASE_URL}/{gz_filename}"
        print(f"[fetch] {name}: downloading {url}")
        _download(url, gz_path)

        with gzip.open(gz_path, "rb") as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        gz_path.unlink()  # keep only the decompressed CSV in the cache

        print(f"[fetch] {name}: saved {csv_path}")
        paths[name] = csv_path

    return paths


if __name__ == "__main__":
    fetch_raw_data()
