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

    X_prev = time_series[:-1]
    X_next = time_series[1:]
    X_diff = X_next - X_prev

    diff = X_prev - base_point
    dist2 = np.sum(diff * diff, axis=1)

    N = dist2.shape[0]
    k = num_neighbors

    if k <= 0 or k > N:
        raise ValueError("num_neighbors must satisfy 1 <= k <= N")

    dk2 = np.partition(dist2, k-1)[k-1]

    if dk2 == 0.0:
        return None, None, 0.0, 0.0

    h = np.sqrt(dk2) / 3.0

    distances_over_h = 3.0 * np.sqrt(dist2 / dk2)
    K = kernel(distances_over_h)

    den = np.sum(K)

    if den == 0.0:
        return None, None, 0.0, 0.0

    drift_num = np.sum(K[:, None] * X_diff, axis=0)
    diff_num = np.einsum('i,ij,ik->jk', K, X_diff, X_diff)

    drift_est = drift_num / (Delta * den)
    diff_est = diff_num / (Delta * den)

    return drift_est, diff_est, den, h
