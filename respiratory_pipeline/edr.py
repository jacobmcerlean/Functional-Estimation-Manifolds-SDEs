import numpy as np
from scipy.interpolate import PchipInterpolator
from sklearn.decomposition import PCA


DEFAULT_EDR_FS = 10.0
DEFAULT_LOCAL_DETREND_SECONDS = 30.0
QRS_STATE_DIM = 91


def standardize(signal):
    """
    Standardize a scalar signal to zero mean and unit variance.
    """
    x = np.asarray(signal, dtype=float)

    scale = float(np.std(x))

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    return (x - np.mean(x)) / scale


def local_detrend_qrs(
    states,
    beat_times,
    window_seconds=DEFAULT_LOCAL_DETREND_SECONDS,
):
    """
    Remove slowly varying QRS morphology using a centered temporal window.

    Each beat is centered using the local mean QRS morphology and divided
    by one scalar RMS value computed over the same local window.
    """
    states = np.asarray(states, dtype=float)
    beat_times = np.asarray(beat_times, dtype=float)

    if states.ndim != 2:
        raise ValueError("states must be a two-dimensional array.")

    if len(states) != len(beat_times):
        raise ValueError(
            "states and beat_times must contain the same number of beats."
        )

    half_window = window_seconds / 2.0

    detrended_states = np.empty_like(states)
    local_scale = np.empty(len(states), dtype=float)

    left = 0
    right = 0

    for i, time_value in enumerate(beat_times):
        while (
            left < len(beat_times)
            and beat_times[left] < time_value - half_window
        ):
            left += 1

        if right < i:
            right = i

        while (
            right < len(beat_times)
            and beat_times[right] <= time_value + half_window
        ):
            right += 1

        local_states = states[left:right]
        local_mean = np.mean(local_states, axis=0)

        centered_local = local_states - local_mean

        scale = float(
            np.sqrt(
                np.mean(centered_local ** 2)
            )
        )

        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        detrended_states[i] = (
            states[i] - local_mean
        ) / scale

        local_scale[i] = scale

    return {
        "states": detrended_states,
        "local_scale": local_scale,
    }


def build_delay_embedding(
    detrended_states,
    beat_times,
):
    """
    Construct ECG delay states X_i = [q_i, q_{i-1}].
    """
    qrs = np.asarray(
        detrended_states,
        dtype=float,
    )

    beat_times = np.asarray(
        beat_times,
        dtype=float,
    )

    if len(qrs) != len(beat_times):
        raise ValueError(
            "detrended_states and beat_times must have equal length."
        )

    if len(qrs) < 3:
        raise ValueError(
            "At least three QRS states are required."
        )

    X = np.concatenate(
        (
            qrs[1:],
            qrs[:-1],
        ),
        axis=1,
    )

    state_times = beat_times[1:]
    transition_dt = np.diff(state_times)

    max_transition_gap_seconds = 5.0

    transition_mask = (
        np.isfinite(transition_dt)
        & (transition_dt > 0)
        & (
            transition_dt
            <= max_transition_gap_seconds
        )
    )

    return {
        "X": X,
        "state_times": state_times,
        "transition_dt": transition_dt,
        "transition_mask": transition_mask,
    }


def fit_qrs_pca(
    states,
    n_components=5,
):
    """
    Fit the observed-QRS PCA basis used for EDR reconstruction.

    QRS morphology is centered globally and scaled by one scalar RMS.
    """
    states = np.asarray(states, dtype=float)

    mean_state = np.mean(
        states,
        axis=0,
    )

    centered_states = (
        states - mean_state
    )

    scale = float(
        np.sqrt(
            np.mean(centered_states ** 2)
        )
    )

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    normalized_states = (
        centered_states / scale
    )

    pca = PCA(
        n_components=n_components
    )

    scores = pca.fit_transform(
        normalized_states
    )

    return {
        "pca": pca,
        "scores": scores,
        "mean_state": mean_state,
        "scale": scale,
    }


def pchip_to_regular_grid(
    sample_times,
    sample_values,
    fs=DEFAULT_EDR_FS,
    standardize_output=True,
):
    """
    Interpolate irregular beat-level respiratory values onto a regular grid.

    No respiratory-band filter is applied here. This keeps the reconstructed
    signal suitable for downstream RQI and other respiratory metrics.
    """
    sample_times = np.asarray(
        sample_times,
        dtype=float,
    )

    sample_values = np.asarray(
        sample_values,
        dtype=float,
    )

    if len(sample_times) != len(sample_values):
        raise ValueError(
            "sample_times and sample_values must have equal length."
        )

    if len(sample_times) < 2:
        raise ValueError(
            "At least two samples are required for interpolation."
        )

    relative_time = (
        sample_times - sample_times[0]
    )

    strictly_increasing = np.concatenate(
        (
            [True],
            np.diff(relative_time) > 0,
        )
    )

    relative_time = relative_time[
        strictly_increasing
    ]

    sample_values = sample_values[
        strictly_increasing
    ]

    regular_time = np.arange(
        0.0,
        relative_time[-1],
        1.0 / fs,
    )

    interpolator = PchipInterpolator(
        relative_time,
        sample_values,
        extrapolate=False,
    )

    signal = interpolator(
        regular_time
    )

    valid = np.isfinite(signal)

    regular_time = regular_time[valid]
    signal = signal[valid]

    if standardize_output:
        signal = standardize(signal)

    return {
        "time": regular_time,
        "signal": signal,
        "fs": float(fs),
    }


def observed_edr_from_qrs(
    states,
    beat_times,
    fs=DEFAULT_EDR_FS,
    n_components=5,
):
    """
    Derive observed EDR from the leading QRS morphology principal component.
    """
    pca_result = fit_qrs_pca(
        states,
        n_components=n_components,
    )

    regular = pchip_to_regular_grid(
        beat_times,
        pca_result["scores"][:, 0],
        fs=fs,
    )

    return {
        **regular,
        "pca": pca_result["pca"],
        "mean_state": pca_result["mean_state"],
        "scale": pca_result["scale"],
    }


def simulated_edr_from_delay_path(
    simulation,
    qrs_pca,
    fs=DEFAULT_EDR_FS,
    qrs_dim=QRS_STATE_DIM,
):
    """
    Recover EDR from the current-QRS component of a simulated delay path.
    """
    path = np.asarray(
        simulation["path"],
        dtype=float,
    )

    simulation_time = np.asarray(
        simulation["time"],
        dtype=float,
    )

    simulated_qrs = path[
        :,
        :qrs_dim,
    ]

    qrs_scores = qrs_pca.transform(
        simulated_qrs
    )

    return pchip_to_regular_grid(
        simulation_time,
        qrs_scores[:, 0],
        fs=fs,
    )
