import numpy as np
from scipy.spatial import cKDTree


def derive_geometry_view(
    full_geometry,
    intrinsic_dim,
):
    """
    Restrict a geometry estimate to its leading intrinsic directions.

    This permits d=1 and d=2 simulations to share one expensive d=2
    geometry estimate.
    """
    U = np.asarray(
        full_geometry["U"][
            :,
            :,
            :intrinsic_dim,
        ],
        dtype=float,
    )

    sqrt_eigenvalues = np.asarray(
        full_geometry[
            "sqrt_eigenvalues"
        ][
            :,
            :intrinsic_dim,
        ],
        dtype=float,
    )

    full_drift = np.asarray(
        full_geometry[
            "drift_tangent"
        ],
        dtype=float,
    )

    drift_coordinates = np.einsum(
        "nij,ni->nj",
        U,
        full_drift,
    )

    projected_drift = np.einsum(
        "nij,nj->ni",
        U,
        drift_coordinates,
    )

    geometry = dict(
        full_geometry
    )

    geometry["U"] = U
    geometry[
        "sqrt_eigenvalues"
    ] = sqrt_eigenvalues
    geometry[
        "drift_tangent"
    ] = projected_drift

    return geometry


def simulate_delay_sde(
    X,
    geometry,
    local_dt,
    simulation_seconds,
    *,
    diffusion_strength=1.0,
    seed=42,
    initial_index=None,
):
    """
    Simulate the delay-state SDE using local observed transition intervals.

    At each step, the nearest empirical state with valid geometry defines
    the drift, diffusion, local time step, and affine tangent space.
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    local_dt = np.asarray(
        local_dt,
        dtype=float,
    )

    valid_geometry = np.asarray(
        geometry["valid_geometry"],
        dtype=bool,
    )

    valid_indices = np.flatnonzero(
        valid_geometry
    )

    if len(valid_indices) == 0:
        raise RuntimeError(
            "No states have valid estimated geometry."
        )

    valid_tree = cKDTree(
        X[valid_indices]
    )

    rng = np.random.default_rng(
        seed
    )

    if initial_index is None:
        initial_index = int(
            rng.choice(
                valid_indices
            )
        )

    if not valid_geometry[
        initial_index
    ]:
        raise ValueError(
            "initial_index must have valid geometry."
        )

    current = X[
        initial_index
    ].copy()

    path = [
        current.copy()
    ]

    times = [0.0]
    used_dt = []
    basepoint_indices = []

    while times[-1] < simulation_seconds:
        _, local_index = valid_tree.query(
            current,
            k=1,
        )

        basepoint_index = int(
            valid_indices[
                int(local_index)
            ]
        )

        basepoint = X[
            basepoint_index
        ]

        dt = float(
            local_dt[
                basepoint_index
            ]
        )

        drift = np.asarray(
            geometry[
                "drift_tangent"
            ][
                basepoint_index
            ],
            dtype=float,
        )

        U = np.asarray(
            geometry["U"][
                basepoint_index
            ],
            dtype=float,
        )

        sqrt_eigenvalues = np.asarray(
            geometry[
                "sqrt_eigenvalues"
            ][
                basepoint_index
            ],
            dtype=float,
        )

        noise = rng.normal(
            size=U.shape[1]
        )

        diffusion_increment = (
            float(diffusion_strength)
            * (
                U
                @ (
                    sqrt_eigenvalues
                    * noise
                )
            )
            * np.sqrt(dt)
        )

        proposal = (
            current
            + drift * dt
            + diffusion_increment
        )

        displacement = (
            proposal - basepoint
        )

        current = (
            basepoint
            + U
            @ (
                U.T
                @ displacement
            )
        )

        path.append(
            current.copy()
        )

        times.append(
            times[-1] + dt
        )

        used_dt.append(dt)
        basepoint_indices.append(
            basepoint_index
        )

    return {
        "path": np.asarray(
            path,
            dtype=float,
        ),
        "time": np.asarray(
            times,
            dtype=float,
        ),
        "used_dt": np.asarray(
            used_dt,
            dtype=float,
        ),
        "basepoint_indices": np.asarray(
            basepoint_indices,
            dtype=int,
        ),
        "initial_index": int(
            initial_index
        ),
        "diffusion_strength": float(
            diffusion_strength
        ),
    }
