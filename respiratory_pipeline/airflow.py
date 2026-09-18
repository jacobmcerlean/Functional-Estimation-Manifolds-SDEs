from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import uniform_filter1d

from .respiratory_quality import (
    preprocess_reference_airflow,
    compute_rqi_mask,
)
from .screening import (
    prepare_apnea_screening,
    select_training_mask,
)


def load_airflow(mat_path):
    """
    Load the raw airflow signal and sampling frequency from a VGH MAT file.
    """
    mat_path = Path(mat_path)

    data = loadmat(
        mat_path,
        variable_names=["VGH"],
    )

    airflow_struct = (
        data["VGH"][0, 0]["airflow"][0, 0]
    )

    airflow = np.asarray(
        airflow_struct["signal"]
    ).squeeze().astype(float)

    fs = float(
        np.asarray(
            airflow_struct["fs"]
        ).squeeze()
    )

    return airflow, fs


DEFAULT_FS = 10.0
DEFAULT_LOWPASS_HZ = 1.5

DEFAULT_RQI_BLOCK_SECONDS = 120.0
DEFAULT_RQI_THRESHOLD = 0.5

DEFAULT_LOCAL_Z_SECONDS = 10.0

DEFAULT_EMBEDDING_DIM = 5
DEFAULT_TAU_SECONDS = 0.5


def local_zscore(
    signal,
    fs,
    window_seconds=DEFAULT_LOCAL_Z_SECONDS,
):
    """
    Apply local mean and variance normalization using a
    centered moving window.
    """
    x = np.asarray(
        signal,
        dtype=float,
    )

    window_samples = int(
        round(
            window_seconds * fs
        )
    )

    if window_samples < 1:
        raise ValueError(
            "Normalization window must contain at least one sample."
        )

    local_mean = uniform_filter1d(
        x,
        size=window_samples,
        mode="nearest",
    )

    local_second_moment = uniform_filter1d(
        x * x,
        size=window_samples,
        mode="nearest",
    )

    local_variance = np.maximum(
        local_second_moment
        - local_mean * local_mean,
        1e-8,
    )

    return (
        x - local_mean
    ) / np.sqrt(
        local_variance
    )


def build_masked_takens(
    signal,
    sample_mask,
    fs,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    tau_seconds=DEFAULT_TAU_SECONDS,
):
    """
    Construct a delay embedding from samples belonging to the
    selected training set.

    A state is retained only when every delayed coordinate is
    contained in the training mask. Transitions are retained
    only between consecutive samples on the original grid.
    """
    x = np.asarray(
        signal,
        dtype=float,
    )

    sample_mask = np.asarray(
        sample_mask,
        dtype=bool,
    )

    if len(sample_mask) != len(x):
        raise ValueError(
            "sample_mask must have the same length as signal."
        )

    tau_samples = int(
        round(
            tau_seconds * fs
        )
    )

    if tau_samples < 1:
        raise ValueError(
            "Takens delay must be at least one sample."
        )

    maximum_lag = (
        embedding_dim - 1
    ) * tau_samples

    state_indices = np.arange(
        maximum_lag,
        len(x),
    )

    X = np.column_stack(
        [
            x[
                state_indices
                - coordinate
                * tau_samples
            ]
            for coordinate in range(
                embedding_dim
            )
        ]
    )

    keep = np.ones(
        len(state_indices),
        dtype=bool,
    )

    for coordinate in range(
        embedding_dim
    ):
        keep &= sample_mask[
            state_indices
            - coordinate
            * tau_samples
        ]

    X = X[
        keep
    ]

    retained_indices = (
        state_indices[
            keep
        ]
    )

    transition_mask = (
        np.diff(
            retained_indices
        ) == 1
    )

    return {
        "X": X,
        "transition_mask": transition_mask,
        "retained_indices": retained_indices,
        "embedding_dim": int(
            embedding_dim
        ),
        "tau_seconds": float(
            tau_seconds
        ),
        "tau_samples": int(
            tau_samples
        ),
    }


def prepare_airflow(
    mat_path,
    *,
    mode="non_apnea",
    apnea_label="apnea_1a",
    target_fs=DEFAULT_FS,
    lowpass_hz=DEFAULT_LOWPASS_HZ,
    rqi_threshold=DEFAULT_RQI_THRESHOLD,
    rqi_block_seconds=DEFAULT_RQI_BLOCK_SECONDS,
    local_z_seconds=DEFAULT_LOCAL_Z_SECONDS,
    embedding_dim=DEFAULT_EMBEDDING_DIM,
    tau_seconds=DEFAULT_TAU_SECONDS,
):
    """
    Prepare the airflow trajectory for SDE estimation.

    The default training set consists of QC-passing,
    high-RQI, non-apnea samples. An apnea-only training set
    remains available as an optional analysis mode.
    """
    airflow_raw, raw_fs = load_airflow(
        mat_path
    )

    airflow, fs = (
        preprocess_reference_airflow(
            airflow_raw,
            raw_fs,
            target_fs=target_fs,
            lowpass_hz=lowpass_hz,
        )
    )

    # RQI is calculated for all blocks. The experimental
    # threshold is applied when the training masks are built.
    block_result = compute_rqi_mask(
        airflow,
        fs,
        segment_seconds=rqi_block_seconds,
        rqi_threshold=0.0,
    )

    screening = prepare_apnea_screening(
        mat_path,
        block_rqi=(
            block_result[
                "block_rqi"
            ]
        ),
        qc_pass_blocks=(
            block_result[
                "qc_pass"
            ]
        ),
        n_samples=len(
            airflow
        ),
        fs=fs,
        rqi_threshold=rqi_threshold,
        apnea_label=apnea_label,
        block_seconds=rqi_block_seconds,
    )

    training_mask = select_training_mask(
        screening,
        mode,
    )

    airflow_normalized = local_zscore(
        airflow,
        fs,
        window_seconds=local_z_seconds,
    )

    embedding = build_masked_takens(
        airflow_normalized,
        training_mask,
        fs,
        embedding_dim=embedding_dim,
        tau_seconds=tau_seconds,
    )

    return {
        "raw_airflow": airflow_raw,
        "raw_fs": float(
            raw_fs
        ),
        "airflow": airflow,
        "airflow_normalized": airflow_normalized,
        "fs": float(
            fs
        ),
        "block_rqi": (
            block_result[
                "block_rqi"
            ]
        ),
        "qc_pass_blocks": (
            block_result[
                "qc_pass"
            ]
        ),
        "screening": screening,
        "training_mask": training_mask,
        "mode": mode,
        **embedding,
    }
