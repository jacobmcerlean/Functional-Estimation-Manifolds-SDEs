from pathlib import Path
import csv

import numpy as np

from .airflow import prepare_airflow
from .ecg import preprocess_ecg_matlab
from .edr import (
    local_detrend_qrs,
    build_delay_embedding,
    observed_edr_from_qrs,
    simulated_edr_from_delay_path,
)
from .denoising import pca_denoise
from .geometry import estimate_geometry
from .respiratory_metrics import (
    summarize_respiratory_signal,
    compare_respiratory_signals,
)
from .simulation import (
    simulate_delay_sde,
)


DEFAULT_INTRINSIC_DIM = 2
DEFAULT_DIFFUSION_STRENGTHS = (0.0, 0.5, 1.0)

DEFAULT_NUM_NEIGHBORS = 300
DEFAULT_SIMULATION_SECONDS = 7200.0
DEFAULT_SEED = 42


def _slice_regular_signal(
    signal,
    fs,
    start_seconds=None,
    stop_seconds=None,
):
    """
    Slice a regularly sampled signal using recording-relative times.
    """
    x = np.asarray(
        signal,
        dtype=float,
    )

    start_index = 0
    stop_index = len(x)

    if start_seconds is not None:
        start_index = max(
            0,
            int(round(start_seconds * fs)),
        )

    if stop_seconds is not None:
        stop_index = min(
            len(x),
            int(round(stop_seconds * fs)),
        )

    if stop_index <= start_index:
        raise ValueError(
            "Selected analysis interval is empty."
        )

    return x[start_index:stop_index]


def _select_qrs_interval(
    states,
    beat_times,
    start_seconds=None,
    stop_seconds=None,
):
    """
    Select QRS states belonging to the requested recording interval.
    """
    states = np.asarray(
        states,
        dtype=float,
    )

    beat_times = np.asarray(
        beat_times,
        dtype=float,
    )

    keep = np.ones(
        len(beat_times),
        dtype=bool,
    )

    if start_seconds is not None:
        keep &= (
            beat_times >= start_seconds
        )

    if stop_seconds is not None:
        keep &= (
            beat_times < stop_seconds
        )

    selected_states = states[
        keep
    ]

    selected_times = beat_times[
        keep
    ]

    if len(selected_states) < 20:
        raise RuntimeError(
            "Too few QRS states remain in the selected interval."
        )

    return (
        selected_states,
        selected_times,
    )


def _flatten_signal_metrics(
    prefix,
    metrics,
):
    """
    Extract scalar respiratory metrics for CSV output.
    """
    keys = (
        "mean_breathing_rate_bpm",
        "median_breathing_rate_bpm",
        "breathing_rate_sd_bpm",
        "breathing_rate_iqr_bpm",
        "dominant_frequency_hz",
        "dominant_breathing_rate_bpm",
        "spectral_centroid_hz",
        "respiratory_band_power",
        "rqi",
        "n_breathing_rate_windows",
    )

    return {
        f"{prefix}_{key}": metrics[key]
        for key in keys
    }


def run_subject(
    mat_path,
    output_dir,
    *,
    start_seconds=None,
    stop_seconds=None,
    simulation_seconds=DEFAULT_SIMULATION_SECONDS,
    num_neighbors=DEFAULT_NUM_NEIGHBORS,
    seed=DEFAULT_SEED,
):
    """
    Run the subject-level respiratory SDE workflow.

    Workflow:
        airflow preprocessing
        ECG/QRS preprocessing
        optional interval selection
        observed EDR reconstruction
        local QRS detrending
        QRS delay embedding
        one d=2 geometry estimate
        d=1 and d=2 geometry views
        local transition-time estimation
        simulations for diffusion strengths 0, 0.5, and 1
        respiratory metric evaluation
        subject-level NPZ and CSV outputs
    """
    mat_path = Path(
        mat_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    subject_id = (
        mat_path.stem
    )

    print()
    print(
        f"Subject: {subject_id}"
    )

    # ------------------------------------------------------------
    # Airflow
    # ------------------------------------------------------------

    print()
    print(
        "Preparing airflow..."
    )

    airflow = prepare_airflow(
        mat_path,
        mode="non_apnea",
    )

    airflow_fs = float(
        airflow["fs"]
    )

    observed_airflow = (
        _slice_regular_signal(
            airflow["airflow"],
            airflow_fs,
            start_seconds=(
                start_seconds
            ),
            stop_seconds=(
                stop_seconds
            ),
        )
    )

    airflow_metrics = (
        summarize_respiratory_signal(
            observed_airflow,
            fs=airflow_fs,
        )
    )

    # ------------------------------------------------------------
    # ECG / QRS
    # ------------------------------------------------------------

    print(
        "Preparing ECG/QRS with reference MATLAB preprocessing..."
    )

    ecg = preprocess_ecg_matlab(
        mat_path
    )

    states = np.asarray(
        ecg["states"],
        dtype=float,
    )

    beat_times = (
        np.asarray(
            ecg["R"],
            dtype=float,
        )
        / float(
            ecg["fs"]
        )
    )

    (
        states,
        beat_times,
    ) = _select_qrs_interval(
        states,
        beat_times,
        start_seconds=start_seconds,
        stop_seconds=stop_seconds,
    )

    print(
        "QRS beats:",
        len(states),
    )

    # ------------------------------------------------------------
    # Observed EDR
    # ------------------------------------------------------------

    print(
        "Constructing observed EDR..."
    )

    observed_edr = (
        observed_edr_from_qrs(
            states,
            beat_times,
        )
    )

    observed_edr_metrics = (
        summarize_respiratory_signal(
            observed_edr[
                "signal"
            ],
            fs=observed_edr[
                "fs"
            ],
        )
    )

    # ------------------------------------------------------------
    # Local QRS detrending and delay embedding
    # ------------------------------------------------------------

    print(
        "Building QRS delay embedding..."
    )

    detrended = (
        local_detrend_qrs(
            states,
            beat_times,
        )
    )

    delay = (
        build_delay_embedding(
            detrended[
                "states"
            ],
            beat_times,
        )
    )

    X_raw = np.asarray(
        delay["X"],
        dtype=float,
    )

    print(
        "Denoising delay-state cloud with PCA..."
    )

    denoising = pca_denoise(
        X_raw
    )

    X = denoising["X"]

    print(
        "Selected PCA rank:",
        denoising["rank"],
    )

    print(
        "Cumulative variance at selected rank:",
        denoising["cumulative_variance"][
            denoising["rank"] - 1
        ],
    )

    transition_dt = np.asarray(
        delay[
            "transition_dt"
        ],
        dtype=float,
    )

    transition_mask = np.asarray(
        delay[
            "transition_mask"
        ],
        dtype=bool,
    )

    n_valid_transitions = int(
        np.sum(
            transition_mask
        )
    )

    if n_valid_transitions < 3:
        raise RuntimeError(
            "Too few valid QRS transitions."
        )

    actual_neighbors = min(
        int(
            num_neighbors
        ),
        n_valid_transitions - 1,
    )

    print(
        "Delay states:",
        X.shape,
    )

    print(
        "Valid transitions:",
        n_valid_transitions,
    )

    print(
        "Neighbors:",
        actual_neighbors,
    )

    # ------------------------------------------------------------
    # Geometry: estimate once at d=2
    # ------------------------------------------------------------

    print(
        "Estimating d=2 geometry..."
    )

    geometry_d2 = (
        estimate_geometry(
            X,
            transition_mask,
            transition_dt,
            intrinsic_dim=2,
            num_neighbors=(
                actual_neighbors
            ),
        )
    )

    n_valid_geometry = int(
        np.sum(
            geometry_d2[
                "valid_geometry"
            ]
        )
    )

    if n_valid_geometry == 0:
        raise RuntimeError(
            "No valid local geometry was estimated."
        )

    print(
        "Valid geometry states:",
        n_valid_geometry,
    )

    # ------------------------------------------------------------
    # Local integration times
    # ------------------------------------------------------------

    timing = {
        "local_dt": geometry_d2[
            "local_dt"
        ],
        "lower_bound": geometry_d2[
            "dt_lower_bound"
        ],
        "upper_bound": geometry_d2[
            "dt_upper_bound"
        ],
        "global_median": geometry_d2[
            "global_median_dt"
        ],
    }

    print(
        "Median dt:",
        timing[
            "global_median"
        ],
    )

    # ------------------------------------------------------------
    # Save observed subject data
    # ------------------------------------------------------------

    np.savez_compressed(
        output_dir
        / "observed.npz",
        subject_id=subject_id,
        start_seconds=(
            np.nan
            if start_seconds is None
            else float(
                start_seconds
            )
        ),
        stop_seconds=(
            np.nan
            if stop_seconds is None
            else float(
                stop_seconds
            )
        ),
        airflow=observed_airflow,
        airflow_fs=airflow_fs,
        qrs_states=states,
        beat_times=beat_times,
        detrended_qrs=(
            detrended[
                "states"
            ]
        ),
        local_qrs_scale=(
            detrended[
                "local_scale"
            ]
        ),
        delay_states=X,
        delay_state_times=(
            delay[
                "state_times"
            ]
        ),
        transition_dt=(
            transition_dt
        ),
        transition_mask=(
            transition_mask
        ),
        observed_edr_time=(
            observed_edr[
                "time"
            ]
        ),
        observed_edr_signal=(
            observed_edr[
                "signal"
            ]
        ),
        observed_edr_fs=(
            observed_edr[
                "fs"
            ]
        ),
        local_dt=(
            timing[
                "local_dt"
            ]
        ),
    )

    # ------------------------------------------------------------
    # Subject summary
    # ------------------------------------------------------------

    summary_rows = []

    subject_common = {
        "subject_id": (
            subject_id
        ),
        "start_seconds": (
            np.nan
            if start_seconds is None
            else float(
                start_seconds
            )
        ),
        "stop_seconds": (
            np.nan
            if stop_seconds is None
            else float(
                stop_seconds
            )
        ),
        "n_qrs_beats": int(
            len(states)
        ),
        "n_delay_states": int(
            len(X)
        ),
        "n_valid_transitions": (
            n_valid_transitions
        ),
        "n_valid_geometry": (
            n_valid_geometry
        ),
        "num_neighbors": int(
            actual_neighbors
        ),
        "transition_dt_median": (
            timing[
                "global_median"
            ]
        ),
        "transition_dt_p05": (
            timing[
                "lower_bound"
            ]
        ),
        "transition_dt_p95": (
            timing[
                "upper_bound"
            ]
        ),
        **_flatten_signal_metrics(
            "airflow",
            airflow_metrics,
        ),
        **_flatten_signal_metrics(
            "observed_edr",
            observed_edr_metrics,
        ),
    }

    # ------------------------------------------------------------
    # Simulations
    # ------------------------------------------------------------

    intrinsic_dim = 2
    geometry = geometry_d2

    for diffusion_strength in (
        DEFAULT_DIFFUSION_STRENGTHS
    ):
        print()
        print(
            "Simulation:",
            f"d={intrinsic_dim}",
            f"diffusion={diffusion_strength:g}",
        )

        simulation = (
            simulate_delay_sde(
                X,
                geometry,
                timing[
                    "local_dt"
                ],
                simulation_seconds=(
                    simulation_seconds
                ),
                diffusion_strength=(
                    diffusion_strength
                ),
                seed=seed,
            )
        )

        simulated_edr = (
            simulated_edr_from_delay_path(
                simulation,
                observed_edr[
                    "pca"
                ],
            )
        )

        comparison = (
            compare_respiratory_signals(
                observed_edr[
                    "signal"
                ],
                simulated_edr[
                    "signal"
                ],
                fs=observed_edr[
                    "fs"
                ],
            )
        )

        simulated_metrics = (
            comparison[
                "simulated"
            ]
        )

        diffusion_tag = (
            str(
                diffusion_strength
            )
            .replace(
                ".",
                "p",
            )
        )

        simulation_name = (
            f"d{intrinsic_dim}"
            f"_diff{diffusion_tag}"
        )

        np.savez_compressed(
            output_dir
            / (
                simulation_name
                + ".npz"
            ),
            path=(
                simulation[
                    "path"
                ]
            ),
            time=(
                simulation[
                    "time"
                ]
            ),
            used_dt=(
                simulation[
                    "used_dt"
                ]
            ),
            basepoint_indices=(
                simulation[
                    "basepoint_indices"
                ]
            ),
            initial_index=(
                simulation[
                    "initial_index"
                ]
            ),
            intrinsic_dim=(
                intrinsic_dim
            ),
            diffusion_strength=(
                diffusion_strength
            ),
            simulated_edr_time=(
                simulated_edr[
                    "time"
                ]
            ),
            simulated_edr_signal=(
                simulated_edr[
                    "signal"
                ]
            ),
        )

        row = dict(
            subject_common
        )

        row.update(
            {
                "intrinsic_dim": int(
                    intrinsic_dim
                ),
                "diffusion_strength": float(
                    diffusion_strength
                ),
                "simulation_seconds_requested": float(
                    simulation_seconds
                ),
                "simulation_seconds_actual": float(
                    simulation[
                        "time"
                    ][
                        -1
                    ]
                ),
                "n_simulation_steps": int(
                    len(
                        simulation[
                            "used_dt"
                        ]
                    )
                ),
                **_flatten_signal_metrics(
                    "simulated_edr",
                    simulated_metrics,
                ),
                "mean_breathing_rate_error_bpm": (
                    comparison[
                        "mean_breathing_rate_error_bpm"
                    ]
                ),
                "median_breathing_rate_error_bpm": (
                    comparison[
                        "median_breathing_rate_error_bpm"
                    ]
                ),
                "breathing_rate_sd_error_bpm": (
                    comparison[
                        "breathing_rate_sd_error_bpm"
                    ]
                ),
                "breathing_rate_wasserstein_bpm": (
                    comparison[
                        "breathing_rate_wasserstein_bpm"
                    ]
                ),
                "rqi_difference": (
                    comparison[
                        "rqi_difference"
                    ]
                ),
                "spectral_centroid_error_hz": (
                    comparison[
                        "spectral_centroid_error_hz"
                    ]
                ),
            }
        )

        summary_rows.append(
            row
        )

    # ------------------------------------------------------------
    # CSV
    # ------------------------------------------------------------

    summary_path = (
        output_dir
        / "summary.csv"
    )

    fieldnames = list(
        summary_rows[0].keys()
    )

    with summary_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print()
    print(
        "Subject complete."
    )

    print(
        "Output:",
        output_dir,
    )

    print(
        "Summary:",
        summary_path,
    )

    return {
        "subject_id": subject_id,
        "output_dir": output_dir,
        "summary_rows": summary_rows,
        "airflow": airflow,
        "ecg": ecg,
        "observed_edr": observed_edr,
        "geometry_d2": geometry_d2,
        "timing": timing,
    }
