from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


SENSOR_COLUMNS = ["accX", "accY", "accZ", "gyroX", "gyroY", "gyroZ"]
REQUIRED_COLUMNS = {*SENSOR_COLUMNS, "timestamp", "Activity"}


@dataclass(frozen=True)
class PipelineConfig:
    fallback_sampling_hz: float = 50.0
    lowpass_cutoff_hz: float = 5.0
    filter_order: int = 4
    window_seconds: float = 2.0
    overlap: float = 0.5


def _canonical_name(name: object) -> str:
    lookup = {
        "accx": "accX",
        "accy": "accY",
        "accz": "accZ",
        "gyrox": "gyroX",
        "gyroy": "gyroY",
        "gyroz": "gyroZ",
        "timestamp": "timestamp",
        "activity": "Activity",
    }
    key = str(name).strip().lower()
    return lookup.get(key, str(name).strip())


def load_imu(path: Path | str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data = data.rename(columns={column: _canonical_name(column) for column in data.columns})

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for column in SENSOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data[SENSOR_COLUMNS].isna().any().any():
        raise ValueError("Sensor columns contain missing or non-numeric values")

    if data["Activity"].isna().any():
        raise ValueError("Activity contains missing values")

    return data.reset_index(drop=True)


def timestamp_seconds(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        raw = numeric.to_numpy(dtype=float)
        if len(raw) < 2:
            return np.zeros(len(raw), dtype=float)
        positive = np.diff(raw)
        positive = positive[positive > 0]
        if len(positive) == 0:
            return np.arange(len(raw), dtype=float)
        median_delta = float(np.median(positive))
        scale = 1.0
        if median_delta > 1e7:
            scale = 1e9
        elif median_delta > 1e4:
            scale = 1e6
        elif median_delta > 10:
            scale = 1e3
        return (raw - raw[0]) / scale

    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.notna().all():
        return (parsed - parsed.iloc[0]).dt.total_seconds().to_numpy()

    return np.arange(len(values), dtype=float)


def estimate_sampling_hz(data: pd.DataFrame, fallback_hz: float = 50.0) -> float:
    seconds = timestamp_seconds(data["timestamp"])
    if len(seconds) < 3:
        return fallback_hz

    deltas = np.diff(seconds)
    deltas = deltas[(deltas > 0) & np.isfinite(deltas)]
    if len(deltas) < max(2, len(seconds) // 20):
        return fallback_hz

    median_delta = float(np.median(deltas))
    hz = 1.0 / median_delta if median_delta > 0 else fallback_hz
    if not 5 <= hz <= 500:
        return fallback_hz
    return hz


def add_magnitudes(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["acc_mag"] = np.sqrt((result[["accX", "accY", "accZ"]] ** 2).sum(axis=1))
    result["gyro_mag"] = np.sqrt((result[["gyroX", "gyroY", "gyroZ"]] ** 2).sum(axis=1))
    return result


def lowpass_filter(data: pd.DataFrame, sampling_hz: float, cutoff_hz: float = 5.0, order: int = 4) -> pd.DataFrame:
    if cutoff_hz <= 0 or cutoff_hz >= sampling_hz / 2:
        raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency")

    result = data.copy()
    sos = butter(order, cutoff_hz, btype="lowpass", fs=sampling_hz, output="sos")
    for column in SENSOR_COLUMNS:
        values = result[column].to_numpy(dtype=float)
        if len(values) < 30:
            result[f"{column}_filtered"] = values
        else:
            result[f"{column}_filtered"] = sosfiltfilt(sos, values)
    return add_magnitudes(result)


def sensor_quality_report(data: pd.DataFrame, sampling_hz: float) -> dict:
    seconds = timestamp_seconds(data["timestamp"])
    deltas = np.diff(seconds)
    expected_delta = 1.0 / sampling_hz
    positive = deltas[deltas > 0]
    gap_count = int(np.sum(positive > expected_delta * 2.5)) if len(positive) else 0
    repeated_timestamps = int(np.sum(deltas == 0))

    extreme_rates = {}
    flatline_rates = {}
    for column in SENSOR_COLUMNS:
        values = data[column].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad == 0:
            extreme_rate = 0.0
        else:
            robust_z = 0.6745 * (values - median) / mad
            extreme_rate = float(np.mean(np.abs(robust_z) > 6))
        extreme_rates[column] = round(extreme_rate, 6)
        flatline_rates[column] = round(float(np.mean(np.diff(values) == 0)), 6) if len(values) > 1 else 0.0

    return {
        "rows": int(len(data)),
        "estimated_sampling_hz": round(float(sampling_hz), 3),
        "timestamp_gap_count": gap_count,
        "repeated_timestamp_count": repeated_timestamps,
        "extreme_value_rate": extreme_rates,
        "flatline_step_rate": flatline_rates,
    }


def _dominant_frequency(values: np.ndarray, sampling_hz: float) -> float:
    centered = values - np.mean(values)
    if np.allclose(centered, 0):
        return 0.0
    frequencies = np.fft.rfftfreq(len(centered), d=1.0 / sampling_hz)
    power = np.abs(np.fft.rfft(centered)) ** 2
    if len(power) <= 1:
        return 0.0
    index = int(np.argmax(power[1:]) + 1)
    return float(frequencies[index])


def _window_stats(values: np.ndarray, prefix: str, sampling_hz: float) -> dict:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_rms": float(np.sqrt(np.mean(values ** 2))),
        f"{prefix}_range": float(np.max(values) - np.min(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_dominant_hz": _dominant_frequency(values, sampling_hz),
    }


def build_windows(
    data: pd.DataFrame,
    sampling_hz: float,
    window_seconds: float = 2.0,
    overlap: float = 0.5,
) -> pd.DataFrame:
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be between 0 and 1")

    window_size = max(16, int(round(sampling_hz * window_seconds)))
    stride = max(1, int(round(window_size * (1 - overlap))))
    feature_columns = [f"{column}_filtered" for column in SENSOR_COLUMNS]

    missing = set(feature_columns).difference(data.columns)
    if missing:
        raise ValueError("Filtered sensor columns are required before windowing")

    rows: list[dict] = []
    labels = data["Activity"].to_numpy()

    start = 0
    while start + window_size <= len(data):
        stop = start + window_size
        label_window = labels[start:stop]
        values, counts = np.unique(label_window, return_counts=True)
        majority_index = int(np.argmax(counts))
        majority_label = values[majority_index]
        purity = counts[majority_index] / window_size

        if purity >= 0.9:
            row: dict[str, float | int | str] = {
                "start_index": start,
                "end_index": stop - 1,
                "activity": majority_label,
                "label_purity": round(float(purity), 4),
            }
            for column in feature_columns:
                row.update(_window_stats(data[column].iloc[start:stop].to_numpy(dtype=float), column, sampling_hz))

            acc_mag = np.sqrt(
                sum(data[f"{axis}_filtered"].iloc[start:stop].to_numpy(dtype=float) ** 2 for axis in ("accX", "accY", "accZ"))
            )
            gyro_mag = np.sqrt(
                sum(data[f"{axis}_filtered"].iloc[start:stop].to_numpy(dtype=float) ** 2 for axis in ("gyroX", "gyroY", "gyroZ"))
            )
            row.update(_window_stats(acc_mag, "acc_mag", sampling_hz))
            row.update(_window_stats(gyro_mag, "gyro_mag", sampling_hz))
            rows.append(row)

        start += stride

    if not rows:
        raise ValueError("No windows were created. Check sampling rate and dataset length")
    return pd.DataFrame(rows)
