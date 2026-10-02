from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "imu.csv"
DATASET_URL = "https://archive.ics.uci.edu/static/public/755/accelerometer+gyro+mobile+phone+dataset.zip"
CSV_NAME = "accelerometer_gyro_mobile_phone_dataset.csv"


def download_dataset() -> pd.DataFrame:
    with urlopen(DATASET_URL, timeout=60) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        matches = [name for name in archive.namelist() if name.endswith(CSV_NAME)]
        if not matches:
            raise FileNotFoundError(f"Could not find {CSV_NAME} in the UCI archive")
        with archive.open(matches[0]) as csv_file:
            return pd.read_csv(csv_file)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = download_dataset()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Downloaded {len(data):,} IMU rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
