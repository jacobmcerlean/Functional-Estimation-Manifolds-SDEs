# observed_ellipsoid.py

import numpy as np


# ============================================================
#                GEOMETRY: SPHERE → ELLIPSOID
# ============================================================

def sphere_to_ellipsoid(x, ecc):
    """
    Linear map from unit sphere to ellipsoid.
    ecc = (a1, a2, a3)
    """
    A = np.diag(ecc)
    return A @ x


def ellipsoid_to_sphere(x, ecc):
    Ainv = np.diag(1.0 / np.asarray(ecc))
    y = Ainv @ x
    return y / np.linalg.norm(y)


def true_tangent_project_ellipsoid(x, ecc):
    """
    Euclidean orthogonal projection onto tangent space
    of ellipsoid at x.

    Ellipsoid defined by x^T A^{-2} x = 1.
    """
    ecc = np.asarray(ecc, dtype=float)
    inv_sq = 1.0 / (ecc ** 2)

    n = inv_sq * x
    return np.eye(3) - np.outer(n, n) / (n @ n)


# ============================================================
#                 SAMPLING HELPER
# ============================================================

def sample_spherical_cap(num_samples, threshold):
    """
    Sample uniformly from sphere with condition x > threshold.
    """
    pts = []
    while len(pts) < num_samples:
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        if x > threshold:
            pts.append(np.array([x, y, z]))

    return np.asarray(pts)
