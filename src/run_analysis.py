from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imu_pipeline import PipelineConfig, build_windows, estimate_sampling_hz, load_imu, lowpass_filter, sensor_quality_report
from model import train_baselines


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = ROOT / "figures"
SITE_ASSET_DIR = ROOT / "site" / "assets"


def save_signal_plot(data: pd.DataFrame, sampling_hz: float, destination: Path) -> None:
    count = min(len(data), int(sampling_hz * 8))
    time = np.arange(count) / sampling_hz
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(time, data["accX"].iloc[:count], linewidth=1, alpha=0.55, label="Raw accX")
    ax.plot(time, data["accX_filtered"].iloc[:count], linewidth=1.6, label="Filtered accX")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Sensor value")
    ax.set_title("Raw and low-pass filtered accelerometer signal")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_activity_plot(windows: pd.DataFrame, destination: Path) -> None:
    grouped = windows.groupby("activity")["acc_mag_std"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(grouped["activity"].astype(str), grouped["mean"], yerr=grouped["std"], capsize=4)
    ax.set_xlabel("Activity label")
    ax.set_ylabel("Mean window acceleration variability")
    ax.set_title("Acceleration variability by activity")
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_confusion_plot(model_result: dict, destination: Path) -> None:
    matrix = np.asarray(model_result["confusion_matrix"])
    labels = model_result["labels"]
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    image = ax.imshow(matrix)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f'{model_result["name"]} confusion matrix')
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def run(input_path: Path) -> dict:
    config = PipelineConfig()
    raw = load_imu(input_path)
    sampling_hz = estimate_sampling_hz(raw, config.fallback_sampling_hz)
    filtered = lowpass_filter(raw, sampling_hz, config.lowpass_cutoff_hz, config.filter_order)
    quality = sensor_quality_report(raw, sampling_hz)
    windows = build_windows(filtered, sampling_hz, config.window_seconds, config.overlap)
    model_results = train_baselines(windows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SITE_ASSET_DIR.mkdir(parents=True, exist_ok=True)

    windows.to_csv(OUTPUT_DIR / "window_features.csv", index=False)
    (OUTPUT_DIR / "sensor_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "model_results.json").write_text(json.dumps(model_results, indent=2), encoding="utf-8")

    save_signal_plot(filtered, sampling_hz, FIGURE_DIR / "filter-comparison.png")
    save_activity_plot(windows, FIGURE_DIR / "activity-variability.png")
    save_confusion_plot(model_results["models"][0], FIGURE_DIR / "confusion-matrix.png")

    for name in ("filter-comparison.png", "activity-variability.png", "confusion-matrix.png"):
        (SITE_ASSET_DIR / f"generated-{name}").write_bytes((FIGURE_DIR / name).read_bytes())

    summary = {
        "source_file": input_path.name,
        "rows": int(len(raw)),
        "sampling_hz": round(float(sampling_hz), 2),
        "window_count": int(len(windows)),
        "activity_count": int(windows["activity"].nunique()),
        "quality": quality,
        "models": model_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze accelerometer and gyroscope time-series data")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "imu.csv")
    args = parser.parse_args()
    summary = run(args.input)
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f'Analyzed {summary["rows"]:,} rows into {summary["window_count"]:,} windows')
    print(f'Estimated sampling rate: {summary["sampling_hz"]} Hz')
    print(f'Best baseline model: {summary["models"]["best_model"]}')


if __name__ == "__main__":
    main()
