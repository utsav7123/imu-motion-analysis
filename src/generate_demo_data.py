from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "imu_demo.csv"


def make_demo_data(samples: int = 6000, sampling_hz: float = 50.0, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(samples) / sampling_hz
    activity = np.where((time.astype(int) // 20) % 2 == 0, 0, 1)

    walk = activity == 1
    step_wave = np.sin(2 * np.pi * 1.8 * time)
    arm_wave = np.sin(2 * np.pi * 0.9 * time + 0.4)

    frame = pd.DataFrame(
        {
            "accX": rng.normal(0, 0.05, samples) + walk * 0.55 * step_wave,
            "accY": rng.normal(0, 0.05, samples) + walk * 0.30 * arm_wave,
            "accZ": 1.0 + rng.normal(0, 0.04, samples) + walk * 0.25 * np.abs(step_wave),
            "gyroX": rng.normal(0, 0.02, samples) + walk * 0.35 * arm_wave,
            "gyroY": rng.normal(0, 0.02, samples) + walk * 0.22 * step_wave,
            "gyroZ": rng.normal(0, 0.02, samples) + walk * 0.18 * np.cos(2 * np.pi * 1.8 * time),
            "timestamp": time,
            "Activity": activity,
        }
    )

    for index in (850, 2220, 4780):
        if index < len(frame):
            frame.loc[index, "accX"] += 2.5

    return frame


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = make_demo_data()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(data):,} demo rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
