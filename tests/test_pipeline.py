from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from generate_demo_data import make_demo_data
from imu_pipeline import (
    SENSOR_COLUMNS,
    build_windows,
    estimate_sampling_hz,
    load_imu,
    lowpass_filter,
    sensor_quality_report,
)
from model import train_baselines


def test_demo_data_has_expected_shape():
    data = make_demo_data(samples=1000)
    assert len(data) == 1000
    assert set(SENSOR_COLUMNS).issubset(data.columns)
    assert {"timestamp", "Activity"}.issubset(data.columns)


def test_sampling_rate_is_estimated_from_seconds():
    data = make_demo_data(samples=1000, sampling_hz=50.0)
    assert estimate_sampling_hz(data) == pytest.approx(50.0, rel=0.01)


def test_lowpass_filter_adds_filtered_columns_and_magnitudes():
    data = make_demo_data(samples=1000)
    filtered = lowpass_filter(data, sampling_hz=50.0)
    for column in SENSOR_COLUMNS:
        assert f"{column}_filtered" in filtered.columns
    assert "acc_mag" in filtered.columns
    assert "gyro_mag" in filtered.columns
    assert np.isfinite(filtered[[f"{c}_filtered" for c in SENSOR_COLUMNS]].to_numpy()).all()


def test_window_features_are_created():
    data = make_demo_data(samples=3000)
    filtered = lowpass_filter(data, sampling_hz=50.0)
    windows = build_windows(filtered, sampling_hz=50.0)
    assert len(windows) > 20
    assert "accX_filtered_std" in windows.columns
    assert "acc_mag_dominant_hz" in windows.columns
    assert windows["label_purity"].min() >= 0.9


def test_quality_report_flags_rows_and_sampling_rate():
    data = make_demo_data(samples=1000)
    quality = sensor_quality_report(data, sampling_hz=50.0)
    assert quality["rows"] == 1000
    assert quality["estimated_sampling_hz"] == 50.0
    assert set(quality["extreme_value_rate"]) == set(SENSOR_COLUMNS)


def test_model_baselines_train_on_demo_windows():
    data = make_demo_data(samples=6000)
    filtered = lowpass_filter(data, sampling_hz=50.0)
    windows = build_windows(filtered, sampling_hz=50.0)
    results = train_baselines(windows)
    assert results["feature_count"] > 20
    assert len(results["models"]) == 2
    assert 0 <= results["models"][0]["macro_f1"] <= 1


def test_loader_rejects_missing_sensor_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": [0, 1], "Activity": [0, 1]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing required columns"):
        load_imu(path)
