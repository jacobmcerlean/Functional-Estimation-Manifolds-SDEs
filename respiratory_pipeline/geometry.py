import os
import numpy as np
from scipy.spatial import cKDTree


def kernel(s):
    """
    Compactly supported kernel on |s| < 3.
    """
    s = np.asarray(s, dtype=float)

    out = np.zeros_like(s)

    mask = np.abs(s) < 3.0

    z = np.abs(
        s[mask]
    ) / 3.0

    out[mask] = np.exp(
        -1.0
        / (
            1.0
            - z * z
        )
    )

    return out


def _prepare_transition_dt(
    transition_dt,
    transition_mask,
    n_states,
):
    """
    Convert a scalar or transition-wise time increment into
    an array aligned with the usable state transitions.
    """
    transition_mask = np.asarray(
        transition_mask,
        dtype=bool,
    )

    if np.isscalar(
        transition_dt
    ):
        dt_full = np.full(
            n_states - 1,
            float(
                transition_dt
            ),
            dtype=float,
        )

    else:
        dt_full = np.asarray(
            transition_dt,
            dtype=float,
        )

        if len(dt_full) != n_states - 1:
            raise ValueError(
                "transition_dt must be a scalar or have "
                "length len(X) - 1"
            )

    return dt_full


def estimate_geometry(
    X,
    transition_mask,
    transition_dt,
    intrinsic_dim,
    num_neighbors=200,
    chunk_size=2000,
):
    """
    Estimate local drift and diffusion from observed transitions.

    Parameters
    ----------
    X : ndarray, shape (N, p)
        Observed trajectory in ambient coordinates.

    transition_mask : ndarray, shape (N - 1,)
        Indicates which consecutive state pairs may be used
        for local drift and diffusion estimation.

    transition_dt : float or ndarray, shape (N - 1,)
        Elapsed time associated with each state transition.
        A scalar may be used for uniformly sampled trajectories.

    intrinsic_dim : int
        Number of leading local diffusion directions retained.

    num_neighbors : int
        Number of transition starting points used in each
        local kernel estimate.

    chunk_size : int
        Number of query states processed per tree query batch.

    Returns
    -------
    dict
        Local tangent drift, diffusion basis, diffusion
        eigenvalues, adaptive bandwidth, and validity mask.
    """
    X = np.asarray(
        X,
        dtype=np.float64,
    )

    transition_mask = np.asarray(
        transition_mask,
        dtype=bool,
    )

    n_states, ambient_dim = (
        X.shape
    )

    if len(
        transition_mask
    ) != n_states - 1:
        raise ValueError(
            "transition_mask must have length len(X) - 1"
        )

    if (
        intrinsic_dim < 1
        or intrinsic_dim > ambient_dim
    ):
        raise ValueError(
            "intrinsic_dim must be between "
            f"1 and {ambient_dim}"
        )

    dt_full = _prepare_transition_dt(
        transition_dt,
        transition_mask,
        n_states,
    )

    X_prev = X[:-1]
    X_next = X[1:]

    increments = (
        X_next
        - X_prev
    )

    finite_transition = (
        np.all(
            np.isfinite(
                X_prev
            ),
            axis=1,
        )
        & np.all(
            np.isfinite(
                X_next
            ),
            axis=1,
        )
        & np.all(
            np.isfinite(
                increments
            ),
            axis=1,
        )
        & np.isfinite(
            dt_full
        )
        & (
            dt_full > 0
        )
    )

    usable = (
        transition_mask
        & finite_transition
    )

    X_prev_usable = np.ascontiguousarray(
        X_prev[
            usable
        ]
    )

    increments_usable = np.ascontiguousarray(
        increments[
            usable
        ]
    )

    dt_usable = np.asarray(
        dt_full[
            usable
        ],
        dtype=np.float64,
    )

    n_usable = len(
        X_prev_usable
    )

    print(
        "Usable transitions:",
        n_usable,
    )

    if n_usable < num_neighbors:
        raise ValueError(
            f"Only {n_usable} usable transitions "
            f"but num_neighbors={num_neighbors}"
        )

    transition_tree = cKDTree(
        X_prev_usable
    )

    dt_lower_bound = float(
        np.percentile(
            dt_usable,
            5,
        )
    )

    dt_upper_bound = float(
        np.percentile(
            dt_usable,
            95,
        )
    )

    global_median_dt = float(
        np.median(
            dt_usable
        )
    )

    local_transition_dt = np.full(
        n_states,
        global_median_dt,
        dtype=float,
    )

    drift_tangent = np.full(
        (
            n_states,
            ambient_dim,
        ),
        np.nan,
    )

    eigenvectors_all = np.full(
        (
            n_states,
            ambient_dim,
            intrinsic_dim,
        ),
        np.nan,
    )

    sqrt_eigenvalues = np.full(
        (
            n_states,
            intrinsic_dim,
        ),
        np.nan,
    )

    bandwidth = np.full(
        n_states,
        np.nan,
    )

    valid_geometry = np.zeros(
        n_states,
        dtype=bool,
    )

    print()
    print(
        "Estimating local drift/diffusion..."
    )

    for start in range(
        0,
        n_states,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_states,
        )

        query_points = X[
            start:stop
        ]

        workers = int(
            os.environ.get(
                "NSLOTS",
                "-1",
            )
        )

        distances, indices = (
            transition_tree.query(
                query_points,
                k=num_neighbors,
                workers=workers,
            )
        )

        kth_distance = distances[
            :,
            -1,
        ]

        valid_query = (
            np.isfinite(
                kth_distance
            )
            & (
                kth_distance > 0
            )
        )

        for local_i in np.flatnonzero(
            valid_query
        ):
            global_i = (
                start
                + local_i
            )

            dk = kth_distance[
                local_i
            ]

            scaled_distance = (
                3.0
                * distances[
                    local_i
                ]
                / dk
            )

            weights = kernel(
                scaled_distance
            )

            denominator = np.sum(
                weights
            )

            if denominator <= 0:
                continue

            neighbor_indices = indices[
                local_i
            ]

            local_increments = (
                increments_usable[
                    neighbor_indices
                ]
            )

            local_dt = dt_usable[
                neighbor_indices
            ]

            local_transition_dt[
                global_i
            ] = np.clip(
                np.median(
                    local_dt
                ),
                dt_lower_bound,
                dt_upper_bound,
            )

            # Drift is estimated from transition-wise rates.
            local_rates = (
                local_increments
                / local_dt[
                    :,
                    None,
                ]
            )

            drift = (
                np.sum(
                    weights[
                        :,
                        None,
                    ]
                    * local_rates,
                    axis=0,
                )
                / denominator
            )

            # Diffusion is estimated from increment outer
            # products normalized by elapsed transition time.
            scaled_increments = (
                local_increments
                / np.sqrt(
                    local_dt[
                        :,
                        None,
                    ]
                )
            )

            diffusion = (
                np.einsum(
                    "i,ij,ik->jk",
                    weights,
                    scaled_increments,
                    scaled_increments,
                    optimize=True,
                )
                / denominator
            )

            diffusion = (
                0.5
                * (
                    diffusion
                    + diffusion.T
                )
            )

            eigenvalues, eigenvectors = (
                np.linalg.eigh(
                    diffusion
                )
            )

            order = np.argsort(
                eigenvalues
            )[::-1]

            eigenvalues = eigenvalues[
                order
            ]

            eigenvectors = eigenvectors[
                :,
                order,
            ]

            leading_eigenvalues = (
                np.clip(
                    eigenvalues[
                        :intrinsic_dim
                    ],
                    0.0,
                    None,
                )
            )

            U = eigenvectors[
                :,
                :intrinsic_dim,
            ]

            # Restrict the drift to the estimated local
            # tangent/diffusion subspace.
            projected_drift = (
                U
                @ (
                    U.T
                    @ drift
                )
            )

            drift_tangent[
                global_i
            ] = projected_drift

            eigenvectors_all[
                global_i
            ] = U

            sqrt_eigenvalues[
                global_i
            ] = np.sqrt(
                leading_eigenvalues
            )

            bandwidth[
                global_i
            ] = (
                dk / 3.0
            )

            valid_geometry[
                global_i
            ] = True

        print(
            f"\r  {stop:,} / {n_states:,}",
            end="",
            flush=True,
        )

    print()

    print(
        "Valid geometry:",
        valid_geometry.sum(),
        "/",
        n_states,
    )

    return {
        "drift_tangent": drift_tangent,
        "U": eigenvectors_all,
        "sqrt_eigenvalues": sqrt_eigenvalues,
        "bandwidth": bandwidth,
        "valid_geometry": valid_geometry,
        "local_dt": local_transition_dt,
        "dt_lower_bound": dt_lower_bound,
        "dt_upper_bound": dt_upper_bound,
        "global_median_dt": global_median_dt,
    }
