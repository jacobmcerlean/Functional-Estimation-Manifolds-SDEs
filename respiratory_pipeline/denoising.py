import numpy as np
from sklearn.decomposition import PCA


def _elbow_dimension(cumulative_variance):
    """
    Select the knee of a cumulative explained-variance curve using
    maximum distance from the straight line joining its endpoints.
    """
    y = np.asarray(
        cumulative_variance,
        dtype=float,
    )

    x = np.arange(
        1,
        len(y) + 1,
        dtype=float,
    )

    x_norm = (
        x - x[0]
    ) / (
        x[-1] - x[0]
    )

    y_norm = (
        y - y[0]
    ) / (
        y[-1] - y[0]
    )

    chord = x_norm

    distance = (
        y_norm - chord
    )

    return int(
        np.argmax(distance)
        + 1
    )


def pca_denoise(
    X,
):
    """
    Denoise delay states with global PCA.

    The retained PCA rank is selected from the elbow of the cumulative
    explained-variance curve. States are reconstructed in the original
    ambient dimension so downstream geometry code is unchanged.
    """
    X = np.asarray(
        X,
        dtype=float,
    )

    if X.ndim != 2:
        raise ValueError(
            "X must be a two-dimensional array."
        )

    pca = PCA()
    scores = pca.fit_transform(
        X
    )

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    rank = _elbow_dimension(
        cumulative_variance
    )

    X_denoised = (
        scores[:, :rank]
        @ pca.components_[:rank]
        + pca.mean_
    )

    return {
        "X": X_denoised,
        "rank": rank,
        "explained_variance_ratio": (
            pca.explained_variance_ratio_
        ),
        "cumulative_variance": (
            cumulative_variance
        ),
    }
