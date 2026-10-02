# IMU Motion Analysis

I built this project around a question that comes up quickly when working with motion sensors:

**Can I turn noisy accelerometer and gyroscope readings into a clean, explainable activity signal and catch data quality problems before they reach a model?**

The project downloads a public IMU dataset, validates the raw readings, estimates the sampling rate, applies a low-pass Butterworth filter, extracts fixed-window time-series features, trains two baseline activity classifiers, and creates a small report for reviewing the results.

The goal is not to hide the work behind a notebook. Each step is kept separate so I can inspect the sensor data, filtering, features, model results, and quality checks on their own.

## Dataset

The main analysis uses the **Accelerometer Gyro Mobile Phone** dataset from the UCI Machine Learning Repository, dataset ID 755. It contains 31,991 sequential records with three accelerometer axes, three gyroscope axes, a timestamp, and an activity label for standing or walking.

Dataset citation:

> AlSahly, A. (2022). Accelerometer Gyro Mobile Phone [Dataset]. UCI Machine Learning Repository. https://doi.org/10.3390/s22176513

License: CC BY 4.0.

The repository does not commit the downloaded dataset. `src/download_data.py` retrieves it from UCI when needed.

## What the project does

- Validates accelerometer, gyroscope, timestamp, and activity fields
- Estimates sampling frequency from timestamp spacing
- Applies a Butterworth low-pass filter to reduce high-frequency noise
- Calculates acceleration and angular velocity magnitudes
- Segments the signal into overlapping two-second windows
- Extracts mean, standard deviation, RMS, range, median, and dominant frequency features
- Checks timestamp gaps, repeated timestamps, flatline behaviour, and extreme sensor values
- Trains Logistic Regression and Random Forest baseline classifiers
- Reports accuracy, macro F1, per-class metrics, and a confusion matrix
- Produces plots for raw vs filtered signals and activity-level variability
- Includes automated tests and GitHub Actions

## Project structure

```text
imu-motion-analysis/
├── src/
│   ├── download_data.py
│   ├── generate_demo_data.py
│   ├── imu_pipeline.py
│   ├── model.py
│   ├── run_analysis.py
│   └── build_site_data.py
├── tests/
├── docs/
├── site/
├── .github/workflows/
├── requirements.txt
└── README.md
```

## Run with the real UCI dataset

```bash
python -m pip install -r requirements.txt
python src/download_data.py
python src/run_analysis.py --input data/imu.csv
python src/build_site_data.py
pytest -q
```

## Run without downloading anything

A synthetic IMU stream is included only for development and automated tests. It mimics a simple standing and walking pattern so the full pipeline can be tested offline.

```bash
python src/generate_demo_data.py
python src/run_analysis.py --input data/imu_demo.csv
python src/build_site_data.py
pytest -q
```

## Filtering

The pipeline uses a fourth-order Butterworth low-pass filter with a default 5 Hz cutoff. The cutoff is intentionally configurable because a real wearable study should choose filtering parameters based on the movement being measured and the device sampling rate.

I keep both the raw and filtered signals so it is easy to compare what the filter changed instead of treating filtering as a black box.

## Window features

The default window is two seconds with 50 percent overlap. A window is used only when at least 90 percent of its samples share the same activity label. This avoids training on windows that mostly sit across an activity transition.

For each sensor axis and the accelerometer and gyroscope magnitudes, the pipeline calculates:

- Mean
- Standard deviation
- RMS
- Range
- Median
- Dominant frequency

## Model evaluation

The project compares Logistic Regression with Random Forest. These are baseline models, not an attempt to claim the most accurate possible activity-recognition system.

The UCI dataset used here does not provide a subject identifier in the published feature list, so the baseline evaluation uses a stratified train/test split at the window level. That limitation is documented because a subject-independent split would be more appropriate when subject IDs are available.

## Sensor quality checks

Before modelling, the project produces a quality report containing:

- Estimated sampling frequency
- Timestamp gaps
- Repeated timestamps
- Flatline step rate for each sensor axis
- Robust extreme-value rate for each sensor axis

These checks are meant to surface problems for investigation, not automatically declare a sensor faulty.

## Dashboard

The `site/` folder contains a lightweight project report that can be deployed through GitHub Pages. It shows the size of the analyzed dataset, estimated sampling rate, number of analysis windows, best baseline model, classification metrics, and generated plots.

## What I would improve next

- Evaluate on a dataset with participant IDs and use subject-independent validation
- Add band-pass and median filtering comparisons
- Add explicit gait or stroke-cycle segmentation
- Compare hand-built features with a 1D convolutional model
- Add a labelled data-review tool for correcting ground-truth segments
- Test on smartwatch or swim-specific wearable data
