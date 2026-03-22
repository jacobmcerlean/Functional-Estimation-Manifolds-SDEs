import numpy as np
import math

# ============================================================
#  DRIFT ON THE SPHERE
# ============================================================

def drift_on_sphere(x, y, z):
    """
    Drift vector field on the unit sphere.
    Ensures tangency: v · x = 0.
    """
    return np.array([y, -x, 0.0])


# ============================================================
#  TANGENT VECTOR SAMPLING
# ============================================================

def sample_tangent(x, y, z):
    """
    Sample a random tangent vector at (x,y,z) on S^2
    with chi-square radial distribution.
    """
    t = np.random.normal(0, 1, 3)

    # project to tangent space
    n = np.array([x, y, z])
    t = t - np.dot(t, n) * n

    # normalize
    t = t / np.linalg.norm(t)

    # chi-square radial scaling
    t = t * np.sqrt(np.random.chisquare(2))

    return t

#============================================================
#  SDE STEP
# ============================================================


def sphere_step(curr_point, Delta):
    """
    Single retraction step on S^2 using the same
    scheme as pi_ret_SDE_sampling.
    """
    sampled_tangent_vec = sample_tangent(
        curr_point[0], curr_point[1], curr_point[2]
    )

    drift = drift_on_sphere(
        curr_point[0], curr_point[1], curr_point[2]
    )

    process_differential = (
        np.sqrt(Delta) * sampled_tangent_vec
        + Delta * drift
    )

    next_point = curr_point + process_differential
    next_point = next_point / np.linalg.norm(next_point)

    return next_point
# ============================================================
#  SDE SIMULATION ON S^2 (RETRACTION METHOD)
# ============================================================
def pi_ret_SDE_sampling(N, Delta, x0):
    """
    Simulate an SDE on the unit sphere using a retraction-based scheme.

    Parameters
    ----------
    N : int
        Number of time steps.
    Delta : float
        Time step size.
    x0 : (3,) array
        Initial point on S^2.

    Returns
    -------
    SDE : (N+1, 3) array
        Trajectory on the sphere.
    """
    SDE = np.zeros((N + 1, 3))
    SDE[0] = x0

    curr_point = x0

    next_mark = 10

    for i in range(N):
        sampled_tangent_vec = sample_tangent(
            curr_point[0], curr_point[1], curr_point[2]
        )

        drift = drift_on_sphere(
            curr_point[0], curr_point[1], curr_point[2]
        )

        process_differential = (
            np.sqrt(Delta) * sampled_tangent_vec
            + Delta * drift
        )

        next_point = curr_point + process_differential
        next_point = next_point / np.linalg.norm(next_point)

        SDE[i + 1] = next_point
        curr_point = next_point

        pct = 100 * (i + 1) // N
        if pct >= next_mark:
           print(f"{next_mark}% completed", flush=True)
           next_mark += 10

    return SDE
# ============================================================
#  TRUE TANGENT PROJECTION MATRICES (LOCAL CHARTS)
# ============================================================

def true_tangent_space_project_near_001(x, y, z):
    """
    Tangent-space projection near north pole (0,0,1).
    """
    T_x = np.array([1.0, 0.0, -x / z])
    T_x = T_x / np.linalg.norm(T_x)

    T_y = np.array([0.0, 1.0, -y / z])
    T_y = T_y - np.dot(T_x, T_y) * T_x
    T_y = T_y / np.linalg.norm(T_y)

    Q_p = np.array([T_x, T_y])
    P_p = Q_p.T @ Q_p
    return P_p


def true_tangent_space_project_near_100(x, y, z):
    """
    Tangent-space projection near (1,0,0).
    """
    T_y = np.array([-y / x, 1.0, 0.0])
    T_y = T_y / np.linalg.norm(T_y)

    T_z = np.array([-z / x, 0.0, 1.0])
    T_z = T_z - np.dot(T_z, T_y) * T_y
    T_z = T_z / np.linalg.norm(T_z)

    Q_x = np.array([T_y, T_z])
    P_x = Q_x.T @ Q_x
    return P_x


# ============================================================
#  ORTHONORMAL FRAME NEAR (1,0,0)
# ============================================================

def construct_onf_near_100(x, y, z):
    """
    Construct an orthonormal frame near (1,0,0) on S^2.
    u1, u2 span the tangent space.
    """
    u1 = np.array([-y / x, 1.0, 0.0])
    u2 = np.array([-z / x, 0.0, 1.0])

    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 - np.dot(u1, u2) * u1
    u2 = u2 / np.linalg.norm(u2)

    u3 = np.cross(u1, u2)

    return u1, u2, u3
