import numpy as np
from scipy.signal import butter, detrend, filtfilt
from scipy.stats import wasserstein_distance

from .respiratory_quality import (
    RESPIRATORY_FS,
    RESPIRATORY_LOW_HZ,
    RESPIRATORY_HIGH_HZ,
)


DEFAULT_WINDOW_SECONDS = 60.0
DEFAULT_STEP_SECONDS = 30.0
RQI_BLOCK_SECONDS = 120.0


def _respiratory_filter(
    signal,
    fs=RESPIRATORY_FS,
):
    """Bandpass filter used by the historical RQI calculation."""
    x = np.asarray(
        signal,
        dtype=float,
    )

    b, a = butter(
        3,
        [
            RESPIRATORY_LOW_HZ,
            RESPIRATORY_HIGH_HZ,
        ],
        btype="bandpass",
        fs=fs,
    )

    return filtfilt(
        b,
        a,
        x,
    )


def _spectral_rqi(
    block,
    fs,
):
    """Historical spectral respiratory quality index."""
    x = np.asarray(
        block,
        dtype=float,
    )

    finite = np.isfinite(x)

    if np.sum(finite) < 0.9 * len(x):
        return np.nan

    if not np.all(finite):
        good = np.flatnonzero(
            finite
        )

        x = np.interp(
            np.arange(len(x)),
            good,
            x[good],
        )

    x = detrend(
        x,
        type="linear",
    )

    if len(x) > 8:
        x = x[3:-4]

    n = len(x)

    if n < 32:
        return np.nan

    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2

    frequencies = np.fft.rfftfreq(
        n,
        d=1.0 / fs,
    )

    respiratory = (
        (frequencies >= RESPIRATORY_LOW_HZ)
        & (frequencies <= RESPIRATORY_HIGH_HZ)
    )

    indices = np.flatnonzero(
        respiratory
    )

    if len(indices) < 3:
        return np.nan

    band_power = power[
        indices
    ]

    total = np.sum(
        band_power
    )

    if (
        not np.isfinite(total)
        or total <= 0
    ):
        return np.nan

    local_peak = int(
        np.argmax(
            band_power
        )
    )

    lo = max(
        0,
        local_peak - 1,
    )

    hi = min(
        len(band_power),
        local_peak + 2,
    )

    peak_power = np.sum(
        band_power[
            lo:hi
        ]
    )

    return float(
        peak_power / total
    )


def calculate_rqi(
    signal,
    fs=RESPIRATORY_FS,
):
    """
    Historical pilot RQI.

    RQI is calculated in non-overlapping 120-second blocks and
    summarized by the median across finite block values.
    """
    x = np.asarray(
        signal,
        dtype=float,
    )

    x = _respiratory_filter(
        x,
        fs=fs,
    )

    block_length = int(
        round(
            RQI_BLOCK_SECONDS
            * fs
        )
    )

    values = []

    for start in range(
        0,
        len(x) - block_length + 1,
        block_length,
    ):
        block = x[
            start:
            start + block_length
        ]

        values.append(
            _spectral_rqi(
                block,
                fs,
            )
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    finite = values[
        np.isfinite(values)
    ]

    if len(finite) == 0:
        return (
            np.nan,
            np.nan,
            values,
        )

    return (
        float(np.median(finite)),
        float(np.mean(finite)),
        values,
    )


def _prepare_signal(
    signal,
    fs,
):
    """
    Validate a regularly sampled respiratory signal.
    """
    x = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    if not np.isclose(
        fs,
        RESPIRATORY_FS,
    ):
        raise ValueError(
            f"Respiratory metrics expect {RESPIRATORY_FS:g} Hz input."
        )

    if len(x) < 2:
        raise ValueError(
            "Respiratory signal is too short."
        )

    return x


def dominant_respiratory_frequency(
    signal,
    fs=RESPIRATORY_FS,
):
    """
    Estimate dominant respiratory frequency from the FFT within
    the respiratory band.
    """
    x = _prepare_signal(
        signal,
        fs,
    )

    finite = np.isfinite(x)

    if np.sum(finite) < 20:
        return np.nan

    x = x[finite]
    x = detrend(
        x,
        type="linear",
    )

    frequencies = np.fft.rfftfreq(
        len(x),
        d=1.0 / fs,
    )

    power = (
        np.abs(
            np.fft.rfft(x)
        ) ** 2
    )

    respiratory_mask = (
        (frequencies >= RESPIRATORY_LOW_HZ)
        & (frequencies <= RESPIRATORY_HIGH_HZ)
    )

    if not np.any(
        respiratory_mask
    ):
        return np.nan

    respiratory_frequencies = frequencies[
        respiratory_mask
    ]

    respiratory_power = power[
        respiratory_mask
    ]

    if (
        not np.any(
            np.isfinite(
                respiratory_power
            )
        )
        or np.nansum(
            respiratory_power
        ) <= 0
    ):
        return np.nan

    peak_index = int(
        np.nanargmax(
            respiratory_power
        )
    )

    return float(
        respiratory_frequencies[
            peak_index
        ]
    )


def spectral_centroid(
    signal,
    fs=RESPIRATORY_FS,
):
    """
    Compute the power-weighted spectral centroid within the
    respiratory band.
    """
    x = _prepare_signal(
        signal,
        fs,
    )

    finite = np.isfinite(x)

    if np.sum(finite) < 20:
        return np.nan

    x = detrend(
        x[finite],
        type="linear",
    )

    frequencies = np.fft.rfftfreq(
        len(x),
        d=1.0 / fs,
    )

    power = (
        np.abs(
            np.fft.rfft(x)
        ) ** 2
    )

    mask = (
        (frequencies >= RESPIRATORY_LOW_HZ)
        & (frequencies <= RESPIRATORY_HIGH_HZ)
    )

    band_power = power[
        mask
    ]

    band_frequencies = frequencies[
        mask
    ]

    total_power = np.sum(
        band_power
    )

    if (
        not np.isfinite(
            total_power
        )
        or total_power <= 0
    ):
        return np.nan

    return float(
        np.sum(
            band_frequencies
            * band_power
        )
        / total_power
    )


def respiratory_band_power(
    signal,
    fs=RESPIRATORY_FS,
):
    """
    Compute total FFT power within the respiratory frequency band.
    """
    x = _prepare_signal(
        signal,
        fs,
    )

    finite = np.isfinite(x)

    if np.sum(finite) < 20:
        return np.nan

    x = detrend(
        x[finite],
        type="linear",
    )

    frequencies = np.fft.rfftfreq(
        len(x),
        d=1.0 / fs,
    )

    power = (
        np.abs(
            np.fft.rfft(x)
        ) ** 2
        / len(x)
    )

    mask = (
        (frequencies >= RESPIRATORY_LOW_HZ)
        & (frequencies <= RESPIRATORY_HIGH_HZ)
    )

    return float(
        np.sum(
            power[
                mask
            ]
        )
    )


def windowed_breathing_rate(
    signal,
    fs=RESPIRATORY_FS,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    step_seconds=DEFAULT_STEP_SECONDS,
):
    """
    Estimate breathing rate in overlapping fixed-duration windows.

    Each window's breathing rate is 60 times its dominant respiratory
    frequency, giving units of breaths per minute.
    """
    x = _prepare_signal(
        signal,
        fs,
    )

    window_samples = int(
        round(
            window_seconds
            * fs
        )
    )

    step_samples = int(
        round(
            step_seconds
            * fs
        )
    )

    if window_samples < 20:
        raise ValueError(
            "window_seconds is too short."
        )

    if step_samples < 1:
        raise ValueError(
            "step_seconds must contain at least one sample."
        )

    if len(x) < window_samples:
        return {
            "time": np.empty(
                0,
                dtype=float,
            ),
            "breathing_rate_bpm": np.empty(
                0,
                dtype=float,
            ),
        }

    starts = np.arange(
        0,
        len(x) - window_samples + 1,
        step_samples,
    )

    times = np.empty(
        len(starts),
        dtype=float,
    )

    breathing_rate = np.full(
        len(starts),
        np.nan,
        dtype=float,
    )

    for i, start in enumerate(
        starts
    ):
        stop = (
            start
            + window_samples
        )

        segment = x[
            start:stop
        ]

        frequency = (
            dominant_respiratory_frequency(
                segment,
                fs=fs,
            )
        )

        if np.isfinite(
            frequency
        ):
            breathing_rate[
                i
            ] = (
                60.0
                * frequency
            )

        times[
            i
        ] = (
            start
            + 0.5
            * window_samples
        ) / fs

    return {
        "time": times,
        "breathing_rate_bpm": breathing_rate,
    }


def summarize_respiratory_signal(
    signal,
    fs=RESPIRATORY_FS,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    step_seconds=DEFAULT_STEP_SECONDS,
):
    """
    Compute summary statistics for a regular respiratory signal.
    """
    x = _prepare_signal(
        signal,
        fs,
    )

    rates = windowed_breathing_rate(
        x,
        fs=fs,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )

    br = rates[
        "breathing_rate_bpm"
    ]

    finite_br = br[
        np.isfinite(
            br
        )
    ]

    if len(finite_br) == 0:
        mean_br = np.nan
        median_br = np.nan
        std_br = np.nan
        iqr_br = np.nan
    else:
        mean_br = float(
            np.mean(
                finite_br
            )
        )

        median_br = float(
            np.median(
                finite_br
            )
        )

        std_br = float(
            np.std(
                finite_br
            )
        )

        q25, q75 = np.percentile(
            finite_br,
            [25, 75],
        )

        iqr_br = float(
            q75 - q25
        )

    dominant_frequency = (
        dominant_respiratory_frequency(
            x,
            fs=fs,
        )
    )

    centroid = spectral_centroid(
        x,
        fs=fs,
    )

    return {
        "mean_breathing_rate_bpm": mean_br,
        "median_breathing_rate_bpm": median_br,
        "breathing_rate_sd_bpm": std_br,
        "breathing_rate_iqr_bpm": iqr_br,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_breathing_rate_bpm": (
            60.0
            * dominant_frequency
            if np.isfinite(
                dominant_frequency
            )
            else np.nan
        ),
        "spectral_centroid_hz": centroid,
        "respiratory_band_power": (
            respiratory_band_power(
                x,
                fs=fs,
            )
        ),
        "rqi": calculate_rqi(
            x,
            fs=fs,
        )[0],
        "n_breathing_rate_windows": int(
            len(
                finite_br
            )
        ),
        "breathing_rate_time": rates[
            "time"
        ],
        "breathing_rate_bpm": br,
    }


def compare_respiratory_signals(
    observed,
    simulated,
    fs=RESPIRATORY_FS,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    step_seconds=DEFAULT_STEP_SECONDS,
):
    """
    Compare observed and simulated respiratory statistics.

    Pointwise phase agreement is intentionally not used because an
    independently simulated stochastic trajectory is not expected to
    remain phase synchronized with the observed recording.
    """
    observed_metrics = (
        summarize_respiratory_signal(
            observed,
            fs=fs,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
    )

    simulated_metrics = (
        summarize_respiratory_signal(
            simulated,
            fs=fs,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
    )

    observed_br = observed_metrics[
        "breathing_rate_bpm"
    ]

    simulated_br = simulated_metrics[
        "breathing_rate_bpm"
    ]

    observed_br = observed_br[
        np.isfinite(
            observed_br
        )
    ]

    simulated_br = simulated_br[
        np.isfinite(
            simulated_br
        )
    ]

    if (
        len(observed_br) > 0
        and len(simulated_br) > 0
    ):
        br_wasserstein = float(
            wasserstein_distance(
                observed_br,
                simulated_br,
            )
        )
    else:
        br_wasserstein = np.nan

    return {
        "observed": observed_metrics,
        "simulated": simulated_metrics,
        "mean_breathing_rate_error_bpm": abs(
            observed_metrics[
                "mean_breathing_rate_bpm"
            ]
            - simulated_metrics[
                "mean_breathing_rate_bpm"
            ]
        ),
        "median_breathing_rate_error_bpm": abs(
            observed_metrics[
                "median_breathing_rate_bpm"
            ]
            - simulated_metrics[
                "median_breathing_rate_bpm"
            ]
        ),
        "breathing_rate_sd_error_bpm": abs(
            observed_metrics[
                "breathing_rate_sd_bpm"
            ]
            - simulated_metrics[
                "breathing_rate_sd_bpm"
            ]
        ),
        "breathing_rate_wasserstein_bpm": (
            br_wasserstein
        ),
        "rqi_difference": (
            simulated_metrics[
                "rqi"
            ]
            - observed_metrics[
                "rqi"
            ]
        ),
        "spectral_centroid_error_hz": abs(
            observed_metrics[
                "spectral_centroid_hz"
            ]
            - simulated_metrics[
                "spectral_centroid_hz"
            ]
        ),
    }


def _contiguous_segments_from_transition_mask(
    signal,
    transition_mask,
):
    """
    Split a retained regularly sampled signal into contiguous runs.

    transition_mask[i] is True when signal[i] and signal[i + 1]
    were consecutive in the original recording.
    """
    x = np.asarray(
        signal,
        dtype=float,
    ).reshape(-1)

    transition_mask = np.asarray(
        transition_mask,
        dtype=bool,
    ).reshape(-1)

    if len(transition_mask) != len(x) - 1:
        raise ValueError(
            "transition_mask must have length len(signal) - 1."
        )

    breaks = np.flatnonzero(
        ~transition_mask
    ) + 1

    boundaries = np.concatenate(
        (
            [0],
            breaks,
            [len(x)],
        )
    )

    return [
        x[start:stop]
        for start, stop in zip(
            boundaries[:-1],
            boundaries[1:],
        )
        if stop > start
    ]


def summarize_segmented_respiratory_signal(
    signal,
    transition_mask,
    fs=RESPIRATORY_FS,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    step_seconds=DEFAULT_STEP_SECONDS,
):
    """
    Summarize a respiratory signal containing discontinuous retained runs.

    No respiratory-rate, spectral, or RQI calculation is allowed to cross
    a rejected gap.
    """
    segments = _contiguous_segments_from_transition_mask(
        signal,
        transition_mask,
    )

    window_rates = []
    rqi_values = []

    dominant_frequencies = []
    spectral_centroids = []
    band_powers = []
    spectral_weights = []

    for segment in segments:
        if len(segment) < 20:
            continue

        rates = windowed_breathing_rate(
            segment,
            fs=fs,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )

        br = np.asarray(
            rates["breathing_rate_bpm"],
            dtype=float,
        )

        finite_br = br[
            np.isfinite(br)
        ]

        if len(finite_br):
            window_rates.append(
                finite_br
            )

        rqi_result = calculate_rqi(
            segment,
            fs=fs,
        )

        segment_rqi_values = np.asarray(
            rqi_result[2],
            dtype=float,
        )

        segment_rqi_values = (
            segment_rqi_values[
                np.isfinite(
                    segment_rqi_values
                )
            ]
        )

        if len(segment_rqi_values):
            rqi_values.append(
                segment_rqi_values
            )

        dominant = (
            dominant_respiratory_frequency(
                segment,
                fs=fs,
            )
        )

        centroid = spectral_centroid(
            segment,
            fs=fs,
        )

        band_power = respiratory_band_power(
            segment,
            fs=fs,
        )

        weight = float(
            len(segment)
        )

        if np.isfinite(dominant):
            dominant_frequencies.append(
                dominant
            )

        if (
            np.isfinite(centroid)
            and np.isfinite(band_power)
        ):
            spectral_centroids.append(
                centroid
            )
            band_powers.append(
                band_power
            )
            spectral_weights.append(
                weight
            )

    if window_rates:
        br = np.concatenate(
            window_rates
        )
    else:
        br = np.empty(
            0,
            dtype=float,
        )

    if len(br):
        mean_br = float(
            np.mean(br)
        )
        median_br = float(
            np.median(br)
        )
        std_br = float(
            np.std(br)
        )
        q25, q75 = np.percentile(
            br,
            [25, 75],
        )
        iqr_br = float(
            q75 - q25
        )
    else:
        mean_br = np.nan
        median_br = np.nan
        std_br = np.nan
        iqr_br = np.nan

    if rqi_values:
        all_rqi = np.concatenate(
            rqi_values
        )
        rqi = float(
            np.median(
                all_rqi
            )
        )
    else:
        rqi = np.nan

    if spectral_weights:
        spectral_weights = np.asarray(
            spectral_weights,
            dtype=float,
        )

        centroid = float(
            np.average(
                spectral_centroids,
                weights=spectral_weights,
            )
        )

        band_power = float(
            np.average(
                band_powers,
                weights=spectral_weights,
            )
        )
    else:
        centroid = np.nan
        band_power = np.nan

    if dominant_frequencies:
        dominant_frequency = float(
            np.median(
                dominant_frequencies
            )
        )
    else:
        dominant_frequency = np.nan

    return {
        "mean_breathing_rate_bpm": mean_br,
        "median_breathing_rate_bpm": median_br,
        "breathing_rate_sd_bpm": std_br,
        "breathing_rate_iqr_bpm": iqr_br,
        "dominant_frequency_hz": dominant_frequency,
        "dominant_breathing_rate_bpm": (
            60.0 * dominant_frequency
            if np.isfinite(
                dominant_frequency
            )
            else np.nan
        ),
        "spectral_centroid_hz": centroid,
        "respiratory_band_power": band_power,
        "rqi": rqi,
        "n_breathing_rate_windows": int(
            len(br)
        ),
        "breathing_rate_time": np.arange(
            len(br),
            dtype=float,
        ),
        "breathing_rate_bpm": br,
        "n_contiguous_segments": int(
            len(segments)
        ),
    }


def compare_segmented_observed_to_simulated(
    observed,
    observed_transition_mask,
    simulated,
    fs=RESPIRATORY_FS,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    step_seconds=DEFAULT_STEP_SECONDS,
):
    """
    Compare discontinuous observed training data with one continuous
    simulated respiratory trajectory.
    """
    observed_metrics = (
        summarize_segmented_respiratory_signal(
            observed,
            observed_transition_mask,
            fs=fs,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
    )

    simulated_metrics = (
        summarize_respiratory_signal(
            simulated,
            fs=fs,
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
    )

    observed_br = np.asarray(
        observed_metrics[
            "breathing_rate_bpm"
        ],
        dtype=float,
    )

    simulated_br = np.asarray(
        simulated_metrics[
            "breathing_rate_bpm"
        ],
        dtype=float,
    )

    observed_br = observed_br[
        np.isfinite(
            observed_br
        )
    ]

    simulated_br = simulated_br[
        np.isfinite(
            simulated_br
        )
    ]

    if (
        len(observed_br)
        and len(simulated_br)
    ):
        br_wasserstein = float(
            wasserstein_distance(
                observed_br,
                simulated_br,
            )
        )
    else:
        br_wasserstein = np.nan

    return {
        "observed": observed_metrics,
        "simulated": simulated_metrics,
        "mean_breathing_rate_error_bpm": abs(
            observed_metrics[
                "mean_breathing_rate_bpm"
            ]
            - simulated_metrics[
                "mean_breathing_rate_bpm"
            ]
        ),
        "median_breathing_rate_error_bpm": abs(
            observed_metrics[
                "median_breathing_rate_bpm"
            ]
            - simulated_metrics[
                "median_breathing_rate_bpm"
            ]
        ),
        "breathing_rate_sd_error_bpm": abs(
            observed_metrics[
                "breathing_rate_sd_bpm"
            ]
            - simulated_metrics[
                "breathing_rate_sd_bpm"
            ]
        ),
        "breathing_rate_wasserstein_bpm": (
            br_wasserstein
        ),
        "rqi_difference": (
            simulated_metrics["rqi"]
            - observed_metrics["rqi"]
        ),
        "spectral_centroid_error_hz": abs(
            observed_metrics[
                "spectral_centroid_hz"
            ]
            - simulated_metrics[
                "spectral_centroid_hz"
            ]
        ),
    }
