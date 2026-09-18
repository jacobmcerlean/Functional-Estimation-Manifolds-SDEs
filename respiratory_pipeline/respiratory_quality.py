from fractions import Fraction

import numpy as np
from scipy.signal import butter, detrend, filtfilt, resample_poly


RESPIRATORY_FS = 10.0
RESPIRATORY_LOW_HZ = 0.1
RESPIRATORY_HIGH_HZ = 0.75
AIRFLOW_LOWPASS_HZ = 1.5


def preprocess_reference_airflow(
    airflow,
    fs,
    target_fs=RESPIRATORY_FS,
    lowpass_hz=AIRFLOW_LOWPASS_HZ,
):
    """
    Interpolate missing airflow samples, resample to the respiratory
    analysis grid, and apply the validated 1.5 Hz low-pass filter.
    """
    x = np.asarray(airflow, dtype=float).reshape(-1)

    finite = np.isfinite(x)

    if not np.any(finite):
        raise ValueError("Airflow contains no finite samples.")

    if not np.all(finite):
        x = np.interp(
            np.arange(len(x)),
            np.flatnonzero(finite),
            x[finite],
        )

    ratio = Fraction(
        float(target_fs) / float(fs)
    ).limit_denominator(10000)

    x = resample_poly(
        x,
        up=ratio.numerator,
        down=ratio.denominator,
    )

    fs_out = (
        float(fs)
        * ratio.numerator
        / ratio.denominator
    )

    b, a = butter(
        4,
        lowpass_hz,
        btype="lowpass",
        fs=fs_out,
    )

    x = filtfilt(b, a, x)

    return x, fs_out


def get_rqi_total_signal(
    signal,
    input_fs=RESPIRATORY_FS,
):
    """
    Compute respiratory quality as the fraction of respiratory-band
    spectral power concentrated in the three bins surrounding the
    dominant respiratory frequency.
    """
    x = np.asarray(signal, dtype=float).reshape(-1)
    x = x[np.isfinite(x)]

    if len(x) < 20:
        return np.nan

    if not np.isclose(input_fs, RESPIRATORY_FS):
        raise ValueError("RQI input must be at 10 Hz.")

    x = detrend(x, type="linear")

    fs = float(input_fs)
    n = len(x)
    duration = n / fs

    b, a = butter(
        3,
        [RESPIRATORY_LOW_HZ, RESPIRATORY_HIGH_HZ],
        btype="bandpass",
        fs=fs,
    )

    # Match the reference MATLAB signal(4:end-4) operation.
    x_trim = x[3:-4]

    if len(x_trim) < 20:
        return np.nan

    filtered = filtfilt(b, a, x_trim)

    spectrum = np.fft.fft(filtered)
    power = np.abs(spectrum) ** 2 / n

    frequencies = (
        np.arange(len(filtered), dtype=float)
        / duration
    )

    respiratory_mask = (
        (frequencies >= RESPIRATORY_LOW_HZ)
        & (frequencies <= RESPIRATORY_HIGH_HZ)
    )

    indices = np.flatnonzero(respiratory_mask)

    if len(indices) < 3:
        return np.nan

    respiratory_power = power[indices]
    total_area = np.sum(respiratory_power)

    if not np.isfinite(total_area) or total_area <= 0:
        return np.nan

    peak_index = int(
        indices[
            np.argmax(respiratory_power)
        ]
    )

    if peak_index <= 0 or peak_index >= len(power) - 1:
        return np.nan

    peak_area = (
        power[peak_index - 1]
        + power[peak_index]
        + power[peak_index + 1]
    )

    return float(peak_area / total_area)


def compute_block_qc(
    segment,
    *,
    minimum_std=1.0,
    maximum_flat_fraction=0.05,
    maximum_clip_fraction=0.01,
    clip_low=-95.0,
    clip_high=95.0,
    flat_tolerance=1e-3,
):
    """
    Compute block-level airflow quality-control statistics.
    """
    x = np.asarray(segment, dtype=float).reshape(-1)

    if len(x) == 0 or not np.all(np.isfinite(x)):
        return {
            "pass": False,
            "std": np.nan,
            "range": np.nan,
            "flat_fraction": np.nan,
            "clip_fraction": np.nan,
        }

    std = float(np.std(x))
    signal_range = float(np.ptp(x))

    if len(x) > 1:
        flat_fraction = float(
            np.mean(
                np.abs(np.diff(x))
                < flat_tolerance
            )
        )
    else:
        flat_fraction = 1.0

    clip_fraction = float(
        np.mean(x <= clip_low)
        + np.mean(x >= clip_high)
    )

    passed = (
        std > minimum_std
        and flat_fraction < maximum_flat_fraction
        and clip_fraction < maximum_clip_fraction
    )

    return {
        "pass": bool(passed),
        "std": std,
        "range": signal_range,
        "flat_fraction": flat_fraction,
        "clip_fraction": clip_fraction,
    }


def compute_rqi_mask(
    airflow_10hz,
    fs,
    segment_seconds=120.0,
    rqi_threshold=0.25,
    minimum_std=1.0,
    maximum_flat_fraction=0.05,
    maximum_clip_fraction=0.01,
    clip_low=-95.0,
    clip_high=95.0,
    flat_tolerance=1e-3,
):
    """
    Compute block-level RQI and airflow QC masks.
    """
    x = np.asarray(
        airflow_10hz,
        dtype=float,
    ).reshape(-1)

    if not np.isclose(fs, RESPIRATORY_FS):
        raise ValueError("RQI/QC input must be at 10 Hz.")

    samples_per_segment = int(
        round(segment_seconds * fs)
    )

    n_segments = len(x) // samples_per_segment

    block_rqi = np.full(
        n_segments,
        np.nan,
        dtype=float,
    )

    qc_pass = np.zeros(
        n_segments,
        dtype=bool,
    )

    block_std = np.full(n_segments, np.nan)
    block_range = np.full(n_segments, np.nan)
    block_flat_fraction = np.full(n_segments, np.nan)
    block_clip_fraction = np.full(n_segments, np.nan)

    for i in range(n_segments):
        start = i * samples_per_segment
        stop = start + samples_per_segment

        segment = x[start:stop]

        block_rqi[i] = get_rqi_total_signal(
            segment,
            input_fs=fs,
        )

        qc = compute_block_qc(
            segment,
            minimum_std=minimum_std,
            maximum_flat_fraction=maximum_flat_fraction,
            maximum_clip_fraction=maximum_clip_fraction,
            clip_low=clip_low,
            clip_high=clip_high,
            flat_tolerance=flat_tolerance,
        )

        qc_pass[i] = qc["pass"]
        block_std[i] = qc["std"]
        block_range[i] = qc["range"]
        block_flat_fraction[i] = qc["flat_fraction"]
        block_clip_fraction[i] = qc["clip_fraction"]

    rqi_pass = (
        np.isfinite(block_rqi)
        & (block_rqi >= float(rqi_threshold))
    )

    return {
        "block_rqi": block_rqi,
        "rqi_pass": rqi_pass,
        "qc_pass": qc_pass,
        "keep_segment": rqi_pass & qc_pass,
        "threshold": float(rqi_threshold),
        "samples_per_segment": samples_per_segment,
        "block_std": block_std,
        "block_range": block_range,
        "block_flat_fraction": block_flat_fraction,
        "block_clip_fraction": block_clip_fraction,
    }
