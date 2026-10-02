# Methodology notes

## Why this project exists

Raw inertial data is rarely useful by itself. A practical analysis needs to answer several questions before a model is trusted: Is the signal complete? Is the sampling rate stable? How much noise is present? Do activity labels line up with the signal? Are the features actually different across activities?

This project keeps those questions visible in the code instead of jumping directly from CSV to classifier.

## Filtering

A fourth-order Butterworth low-pass filter is applied independently to each accelerometer and gyroscope axis. The default cutoff is 5 Hz. The filter uses second-order sections and zero-phase forward/backward filtering through `scipy.signal.sosfiltfilt`.

The default is a starting point, not a universal sensor setting. A production system should choose the cutoff from the motion frequency range, device sampling frequency, and validation data.

## Windowing

The signal is divided into two-second windows with 50 percent overlap. Windows with less than 90 percent label purity are skipped so transitions do not dominate the training labels.

## Time-domain features

Each filtered axis receives mean, standard deviation, RMS, range, and median features. The same statistics are calculated for accelerometer magnitude and gyroscope magnitude.

## Frequency-domain feature

The dominant non-zero frequency is estimated from the real FFT power spectrum. This is a simple feature that can capture periodic motion such as walking without requiring a full spectral model.

## Data quality

Timestamp gaps are flagged when a positive interval is more than 2.5 times the expected sample interval. Repeated timestamps are counted separately. Extreme sensor values are flagged with a robust z-score based on median absolute deviation. Flatline rate is the fraction of consecutive steps with exactly the same value.

## Classification

Two models are compared:

1. Logistic Regression with standardized features and balanced class weights.
2. Random Forest with balanced class weights and limited tree depth.

Macro F1 is used to select the stronger baseline because it gives equal weight to each class.

## Evaluation limitation

The selected UCI dataset is useful for sequential IMU analysis but does not expose participant identity in the listed variables. The current baseline therefore uses a stratified split of analysis windows. That can overstate generalization if neighbouring windows are highly similar.

A stronger follow-up would use a dataset with subject IDs and leave-one-subject-out or group-based validation.
