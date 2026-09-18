from pathlib import Path
import csv

import numpy as np

from .airflow import prepare_airflow
from .geometry import estimate_geometry
from .simulation import simulate_delay_sde
from .respiratory_metrics import (
    summarize_respiratory_signal,
    summarize_segmented_respiratory_signal,
    compare_segmented_observed_to_simulated,
)


DEFAULT_INTRINSIC_DIM = 3
DEFAULT_NUM_NEIGHBORS = 200
DEFAULT_SIMULATION_SECONDS = 300.0
DEFAULT_DIFFUSION_STRENGTHS = (0.0, 0.5, 1.0)


def _flatten_summary(prefix, summary):
    return {
        f"{prefix}_{key}": value
        for key, value in summary.items()
    }


def run_airflow_subject(
    mat_path,
    *,
    output_root="outputs/airflow",
    simulation_seconds=DEFAULT_SIMULATION_SECONDS,
    intrinsic_dim=DEFAULT_INTRINSIC_DIM,
    num_neighbors=DEFAULT_NUM_NEIGHBORS,
    diffusion_strengths=DEFAULT_DIFFUSION_STRENGTHS,
    seed=42,
):
    """
    Run the direct-airflow Takens SDE experiment for one subject.

    Pipeline:
        airflow preprocessing
        -> QC/RQI/non-apnea screening
        -> 5-D Takens embedding
        -> local d=3 drift/diffusion geometry
        -> SDE simulation
        -> first Takens coordinate
        -> respiratory metrics
    """
    mat_path = Path(mat_path)
    subject_id = mat_path.stem

    subject_output = (
        Path(output_root)
        / subject_id
    )
    subject_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Subject: {subject_id}"
    )

    # --------------------------------------------------
    # Airflow front end
    # --------------------------------------------------

    prepared = prepare_airflow(
        mat_path
    )

    X = np.asarray(
        prepared["X"],
        dtype=float,
    )

    transition_mask = np.asarray(
        prepared["transition_mask"],
        dtype=bool,
    )

    fs = float(
        prepared["fs"]
    )

    retained_indices = np.asarray(
        prepared["retained_indices"],
        dtype=np.int64,
    )

    print(
        "Takens states:",
        len(X),
    )
    print(
        "Ambient dimension:",
        X.shape[1],
    )
    print(
        "Intrinsic dimension:",
        intrinsic_dim,
    )
    print(
        "Valid transitions:",
        int(
            transition_mask.sum()
        ),
        "/",
        len(transition_mask),
    )

    if len(X) < num_neighbors:
        raise RuntimeError(
            f"Only {len(X)} Takens states are available, "
            f"but num_neighbors={num_neighbors}."
        )

    # --------------------------------------------------
    # Transition timing
    #
    # Retained states remain indexed on the original
    # 10-Hz grid. Consecutive retained points therefore
    # have dt = 1/fs. Breaks are excluded by
    # transition_mask.
    # --------------------------------------------------

    transition_dt = (
        np.diff(
            retained_indices
        )
        / fs
    )

    # --------------------------------------------------
    # Shared geometry
    # --------------------------------------------------

    geometry = estimate_geometry(
        X,
        transition_mask=transition_mask,
        transition_dt=transition_dt,
        intrinsic_dim=intrinsic_dim,
        num_neighbors=num_neighbors,
    )

    local_dt = np.asarray(
        geometry["local_dt"],
        dtype=float,
    )

    print(
        "Median dt:",
        geometry[
            "global_median_dt"
        ],
    )

    # --------------------------------------------------
    # Observed respiratory reference
    #
    # Coordinate zero of each Takens state is x(t).
    # Because retained states can contain gaps, use the
    # complete normalized 10-Hz airflow for the global
    # observed respiratory summary. The SDE is trained
    # only on accepted states.
    # --------------------------------------------------

    observed_signal = X[:, 0]

    observed_summary = (
        summarize_segmented_respiratory_signal(
            observed_signal,
            transition_mask,
            fs,
        )
    )

    rows = []

    # Use the same empirical starting state for every
    # diffusion strength.
    rng = np.random.default_rng(
        seed
    )

    valid_indices = np.flatnonzero(
        geometry[
            "valid_geometry"
        ]
    )

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No states have valid geometry."
        )

    initial_index = int(
        rng.choice(
            valid_indices
        )
    )

    # --------------------------------------------------
    # Shared simulation
    # --------------------------------------------------

    for diffusion_strength in (
        diffusion_strengths
    ):
        print()
        print(
            "Diffusion strength:",
            diffusion_strength,
        )

        simulation = simulate_delay_sde(
            X,
            geometry,
            local_dt,
            simulation_seconds,
            diffusion_strength=(
                diffusion_strength
            ),
            seed=seed,
            initial_index=(
                initial_index
            ),
        )

        simulated_path = np.asarray(
            simulation["path"],
            dtype=float,
        )

        # In the Takens convention used by airflow.py,
        # coordinate zero is the current respiratory
        # sample x(t).
        simulated_signal = (
            simulated_path[:, 0]
        )

        simulated_summary = (
            summarize_respiratory_signal(
                simulated_signal,
                fs,
            )
        )

        comparison = (
            compare_segmented_observed_to_simulated(
                observed_signal,
                transition_mask,
                simulated_signal,
                fs,
            )
        )

        row = {
            "subject_id": subject_id,
            "diffusion_strength": float(
                diffusion_strength
            ),
            "simulation_seconds": float(
                simulation_seconds
            ),
            "fs": fs,
            "embedding_dim": int(
                prepared[
                    "embedding_dim"
                ]
            ),
            "tau_seconds": float(
                prepared[
                    "tau_seconds"
                ]
            ),
            "intrinsic_dim": int(
                intrinsic_dim
            ),
            "num_neighbors": int(
                num_neighbors
            ),
            "n_takens_states": int(
                len(X)
            ),
            "n_valid_transitions": int(
                transition_mask.sum()
            ),
            "n_valid_geometry": int(
                np.sum(
                    geometry[
                        "valid_geometry"
                    ]
                )
            ),
            "median_transition_dt": float(
                geometry[
                    "global_median_dt"
                ]
            ),
            "initial_index": int(
                initial_index
            ),
        }

        row.update(
            _flatten_summary(
                "observed",
                observed_summary,
            )
        )

        row.update(
            _flatten_summary(
                "simulated",
                simulated_summary,
            )
        )

        row.update(
            comparison
        )

        rows.append(
            row
        )

        np.savez_compressed(
            subject_output
            / (
                "simulation_"
                f"diffusion_{diffusion_strength:g}.npz"
            ),
            path=simulated_path,
            time=np.asarray(
                simulation["time"]
            ),
            used_dt=np.asarray(
                simulation["used_dt"]
            ),
            basepoint_indices=np.asarray(
                simulation[
                    "basepoint_indices"
                ]
            ),
            simulated_signal=(
                simulated_signal
            ),
        )

    # --------------------------------------------------
    # Save subject summary
    # --------------------------------------------------

    summary_path = (
        subject_output
        / "summary.csv"
    )

    fieldnames = list(
        rows[0].keys()
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(
            rows
        )

    np.savez_compressed(
        subject_output
        / "airflow_states.npz",
        X=X,
        retained_indices=(
            retained_indices
        ),
        transition_mask=(
            transition_mask
        ),
        transition_dt=(
            transition_dt
        ),
        airflow=np.asarray(
            prepared["airflow"]
        ),
        airflow_normalized=(
            observed_signal
        ),
        block_rqi=np.asarray(
            prepared["block_rqi"]
        ),
        qc_pass_blocks=np.asarray(
            prepared[
                "qc_pass_blocks"
            ]
        ),
        training_mask=np.asarray(
            prepared[
                "training_mask"
            ]
        ),
    )

    print()
    print(
        "Saved:",
        summary_path,
    )

    return rows

