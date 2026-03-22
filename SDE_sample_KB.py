import numpy as np
import math

# ============================================================
#  DRIFT ON THE SQUARE
# ============================================================

def drift_on_square(u, v):
    """
    Drift field on the intrinsic square coordinates.
    """
    return np.array([
        1+.5*np.cos(u/2)*np.sin(v),
	.5*np.sin(2*v)
    ])


# ============================================================
#  SDE SIMULATION ON THE SQUARE
# ============================================================

def SDE_on_square(N, Delta, x0):
    """
    Simulate an SDE on the square with unit diffusion
    and prescribed drift.

    Parameters
    ----------
    N : int
        Number of time steps.
    Delta : float
        Time step size.
    x0 : (2,) array
        Initial point.

    Returns
    -------
    SDE : (N+1, 2) array
        Simulated trajectory.
    """
    SDE = np.zeros((N + 1, 2))
    SDE[0] = x0
    curr_point = x0

    for i in range(N):
        dW = np.random.randn(2) / np.sqrt(2)
        drift = drift_on_square(curr_point[0], curr_point[1])
        increment = np.sqrt(2 * Delta) * dW + Delta * drift
        curr_point = curr_point + increment
        SDE[i + 1] = curr_point

    return SDE


# ============================================================
#  KLEIN BOTTLE EMBEDDING
# ============================================================

def embed_klein_bottle(time_series, a, r):
    """
    Embed intrinsic coordinates (u, v) into R^4 Klein bottle.
    """
    KB_embed = np.zeros((len(time_series), 4))

    u = time_series[:, 0]
    v = time_series[:, 1]

    KB_embed[:, 0] = np.cos(u) * (a + r * np.cos(v))
    KB_embed[:, 1] = np.sin(u) * (a + r * np.cos(v))
    KB_embed[:, 2] = r * np.cos(u / 2) * np.sin(v)
    KB_embed[:, 3] = r * np.sin(u / 2) * np.sin(v)

    return KB_embed


# ============================================================
#  DIFFERENTIAL GEOMETRY HELPERS
# ============================================================

def klein_jacobian(u, v, a, r):
    Ju = np.array([
        -np.sin(u) * (a + r * np.cos(v)),
         np.cos(u) * (a + r * np.cos(v)),
        -0.5 * r * np.sin(u / 2) * np.sin(v),
         0.5 * r * np.cos(u / 2) * np.sin(v)
    ])

    Jv = np.array([
        -r * np.cos(u) * np.sin(v),
        -r * np.sin(u) * np.sin(v),
         r * np.cos(u / 2) * np.cos(v),
         r * np.sin(u / 2) * np.cos(v)
    ])

    return np.column_stack((Ju, Jv))  # shape (4, 2)


def pullback_to_square(w, u, v, a, r, eps=1e-10):
    J = klein_jacobian(u, v, a, r)
    G = J.T @ J + eps * np.eye(2)
    return np.linalg.solve(G, J.T @ w)


def true_diffusion(u, v, a, r):
    """
    True diffusion matrix in R^4 induced by the embedding.
    """
    J = klein_jacobian(u, v, a, r)
    return J @ J.T


def klein_ito_drift(u, v, a, r):
    """
    (1/2)(∂_uu Φ + ∂_vv Φ) for the Klein bottle embedding.
    """
    term = np.zeros(4)

    # ∂_uu Φ
    term[0] += -np.cos(u) * (a + r * np.cos(v))
    term[1] += -np.sin(u) * (a + r * np.cos(v))
    term[2] += -0.25 * r * np.cos(u / 2) * np.sin(v)
    term[3] += -0.25 * r * np.sin(u / 2) * np.sin(v)

    # ∂_vv Φ
    term[0] += -r * np.cos(u) * np.cos(v)
    term[1] += -r * np.sin(u) * np.cos(v)
    term[2] += -r * np.cos(u / 2) * np.sin(v)
    term[3] += -r * np.sin(u / 2) * np.sin(v)

    return 0.5 * term


def true_ambient_drift(u, v, a, r):
    """
    True ambient drift in R^4 from Ito's lemma.
    """
    J = klein_jacobian(u, v, a, r)
    b = drift_on_square(u, v)
    return J @ b + klein_ito_drift(u, v, a, r)
