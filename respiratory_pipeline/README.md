# SDEs on Respiratory Manifolds

This repository implements stochastic differential equation (SDE) models for respiratory dynamics using two signal representations:

1. ECG-derived respiration (EDR) from QRS morphology.
2. Direct respiratory airflow using a Takens delay embedding.

Both pipelines use modality-specific state construction and share common code for local geometry estimation, SDE simulation, and respiratory evaluation.

## Pipeline overview

### ECG-derived respiration

ECG
-> QRS detection
-> QRS morphology vectors
-> local morphology normalization
-> one-beat delay embedding
-> PCA denoising
-> local drift/diffusion estimation
-> SDE simulation
-> EDR reconstruction
-> respiratory evaluation

Key settings:

- QRS morphology dimension: 91
- delay-state dimension: 182
- intrinsic geometry dimension: 2
- PCA rank: selected automatically using an elbow criterion
- diffusion strengths: 0, 0.5, 1

### Direct airflow

airflow
-> resample and low-pass filter
-> signal-quality and respiratory-quality screening
-> local normalization
-> Takens delay embedding
-> local drift/diffusion estimation
-> SDE simulation
-> airflow reconstruction
-> respiratory evaluation

Key settings:

- sampling rate: 10 Hz
- low-pass cutoff: 1.5 Hz
- Takens embedding dimension: 5
- Takens delay: 0.5 s
- intrinsic geometry dimension: 3
- diffusion strengths: 0, 0.5, 1

Respiratory evaluation is segment-aware so that excluded regions are not treated as contiguous observations.

## Repository structure

respiratory_pipeline/
    airflow.py
        Airflow preprocessing, screening, normalization, and Takens embedding.

    airflow_runner.py
        End-to-end direct-airflow experiment for one subject.

    ecg.py
        ECG preprocessing and MATLAB QRS-detection interface.

    edr.py
        QRS morphology processing, delay embedding, and EDR reconstruction.

    denoising.py
        PCA denoising with automatic rank selection.

    geometry.py
        Shared local drift, diffusion, tangent-space, and local time-step estimation.

    simulation.py
        Shared SDE simulation.

    respiratory_quality.py
        Respiratory preprocessing, signal-quality checks, and RQI calculation.

    respiratory_metrics.py
        Breathing-rate, spectral, RQI, and distributional metrics.

    screening.py
        Construction of training masks.

    edr_runner.py
        End-to-end EDR experiment for one subject.

scripts/
    run_edr_subject.py
        Command-line entry point for the EDR pipeline.

    run_airflow_subject.py
        Command-line entry point for the direct-airflow pipeline.

jobs/
    edr_cohort.job
        Sun Grid Engine array job for the EDR cohort.

    airflow_cohort.job
        Sun Grid Engine array job for the airflow cohort.

matlab/ecg/
    MATLAB implementation used for QRS detection.

tests/
    test_respiratory_metrics.py
        Respiratory metric validation tests.

    validate_clean_pipeline.py
        End-to-end pipeline validation checks.

## Running one subject

### ECG-derived respiration

python -m scripts.run_edr_subject \
    /path/to/subject.mat \
    --simulation-seconds 300 \
    --output-root outputs/edr

### Direct airflow

python -m scripts.run_airflow_subject \
    /path/to/subject.mat \
    --simulation-seconds 300 \
    --output-root outputs/airflow

## Cohort execution

The cohort scripts are configured for Sun Grid Engine.

qsub -t 1-N jobs/edr_cohort.job
qsub -t 1-N jobs/airflow_cohort.job

Replace N with the number of subject MAT files.

## Outputs

Each subject directory contains a summary CSV and saved simulation outputs.

Typical structure:

outputs/
    edr_cohort_final/
        SUBJECT_ID/
            summary.csv

    airflow_cohort_final/
        SUBJECT_ID/
            summary.csv
            airflow_states.npz
            simulation_diffusion_0.npz
            simulation_diffusion_0.5.npz
            simulation_diffusion_1.npz

The summary files contain respiratory-rate statistics, respiratory quality, spectral metrics, and comparison metrics for each diffusion strength.

## Computational considerations

The EDR pipeline is substantially more computationally expensive than the airflow pipeline because geometry is estimated in a 182-dimensional ambient space over tens of thousands of states per subject.

The airflow pipeline operates on 5-dimensional Takens states and is correspondingly faster.

## Dependencies

Install the Python dependencies with:

pip install -r requirements.txt

The EDR pipeline additionally requires MATLAB for QRS detection.

## Notes

The code under respiratory_pipeline/ and scripts/ defines the current analysis workflow.
