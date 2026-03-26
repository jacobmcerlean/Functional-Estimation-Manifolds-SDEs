import numpy as np

# ============================================================
#  KERNEL FUNCTION
# ============================================================

def kernel(s):
    """
    Compactly supported smooth kernel on [-3, 3].
    """
    s = np.asarray(s)
    norms = np.abs(s)

    out = np.zeros_like(norms, dtype=float)
    mask = norms < 3
    out[mask] = np.exp(-1.0 / (1.0 - (norms[mask] / 3.0) ** 2))

    return out


# ===========================================================
# EUCLIDEAN KERNEL ESTIMATOR
# ============================================================
def euclidean_kernel_estimate_vec(base_point, time_series, Delta, num_neighbors):
    """
    Estimate the ambient drift and diffusion matrix at a single base point
    using a compactly supported Euclidean kernel.

    Parameters
    ----------
    base_point : array-like, shape (d,)
        Spatial location where the estimator is evaluated.
    time_series : array-like, shape (T, d)
        Observed trajectory in ambient coordinates.
    Delta : float
        Time step between successive observations.
    num_neighbors : int
        Number of nearest neighbors used to define the adaptive bandwidth.

    Returns
    -------
    drift_est : array, shape (d,)
        Kernel estimate of the ambient drift at base_point.
    diff_est : array, shape (d, d)
        Kernel estimate of the ambient diffusion matrix at base_point.
    den : float
        Total kernel weight.
    h : float
        Adaptive bandwidth, chosen so that the k-th nearest point lies at 3h.
    """

    # Consecutive observations and their increments.
    # X_diff contains the observed one-step increments.
    X_prev = time_series[:-1]
    X_next = time_series[1:]
    X_diff = X_next - X_prev

    # Squared Euclidean distance from each observation X_prev[t]
    # to the evaluation point base_point.
    diff = X_prev - base_point
    dist2 = np.sum(diff * diff, axis=1)

    N = dist2.shape[0]
    k = num_neighbors

    if k <= 0 or k > N:
        raise ValueError("num_neighbors must satisfy 1 <= k <= N")

    # Squared distance to the k-th nearest neighbor.
    # This sets the local spatial scale adaptively.
    dk2 = np.partition(dist2, k - 1)[k - 1]

    # If the k-th neighbor is exactly at the base point, the local scale collapses.
    if dk2 == 0.0:
        return None, None, 0.0, 0.0

    # Bandwidth h is chosen so that the kernel support radius 3h
    # reaches the k-th nearest neighbor.
    h = np.sqrt(dk2) / 3.0

    # Rescaled distances passed into the compactly supported kernel.
    # Since kernel(s) is supported on |s| < 3, points beyond the k-th neighbor
    # automatically receive zero weight.
    distances_over_h = 3.0 * np.sqrt(dist2 / dk2)
    K = kernel(distances_over_h)

    # Total kernel mass.
    den = np.sum(K)

    # If all weights vanish, no local estimate can be formed.
    if den == 0.0:
        return None, None, 0.0, 0.0

    # Numerator for the drift estimator:
    # weighted average of one-step increments.
    drift_num = np.sum(K[:, None] * X_diff, axis=0)

    # Numerator for the diffusion estimator:
    # weighted average of increment outer products.
    diff_num = np.einsum('i,ij,ik->jk', K, X_diff, X_diff)

    # Divide by Delta to convert increments / increment products
    # into drift / diffusion-scale quantities.
    drift_est = drift_num / (Delta * den)
    diff_est = diff_num / (Delta * den)

    return drift_est, diff_est, den, h
