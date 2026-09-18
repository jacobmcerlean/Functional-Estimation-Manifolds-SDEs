import numpy as np

from scipy.io import loadmat
from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    resample_poly,
)
from scipy.ndimage import (
    median_filter,
    uniform_filter1d,
)


TARGET_ECG_FS = 1000.0

BEAT_SAMPLES_BEFORE_R = 30
BEAT_SAMPLES_AFTER_R = 60
BEAT_DIMENSION = 91


def load_vgh_channel(
    mat_path,
    channel_name,
):
    """
    Load a signal and sampling frequency from a VGH channel.
    """
    data = loadmat(
        mat_path,
        variable_names=["VGH"],
    )

    vgh = data[
        "VGH"
    ][0, 0]

    if channel_name not in vgh.dtype.names:
        raise KeyError(
            f"{channel_name!r} not found in VGH. "
            f"Available fields: {vgh.dtype.names}"
        )

    channel = vgh[
        channel_name
    ][0, 0]

    signal = np.asarray(
        channel[
            "signal"
        ]
    ).squeeze().astype(float)

    fs = float(
        np.asarray(
            channel[
                "fs"
            ]
        ).squeeze()
    )

    return signal, fs


def find_ecg_field(
    mat_path,
):
    """
    Identify the ECG channel from the VGH field names.
    """
    data = loadmat(
        mat_path,
        variable_names=["VGH"],
    )

    vgh = data[
        "VGH"
    ][0, 0]

    names = list(
        vgh.dtype.names
    )

    preferred = [
        "ecg",
        "ECG",
        "ecg1",
        "ECG1",
        "ekg",
        "EKG",
    ]

    for name in preferred:
        if name in names:
            return name

    for name in names:
        lower_name = name.lower()

        if (
            "ecg" in lower_name
            or "ekg" in lower_name
        ):
            return name

    raise KeyError(
        "No ECG channel identified. "
        f"Available fields: {names}"
    )


def interpolate_nonfinite(
    signal,
):
    """
    Replace non-finite samples by linear interpolation.
    """
    x = np.asarray(
        signal,
        dtype=float,
    ).copy()

    finite = np.isfinite(
        x
    )

    if not np.any(
        finite
    ):
        raise ValueError(
            "Signal contains no finite samples."
        )

    if np.all(
        finite
    ):
        return x

    indices = np.arange(
        len(x)
    )

    x[
        ~finite
    ] = np.interp(
        indices[
            ~finite
        ],
        indices[
            finite
        ],
        x[
            finite
        ],
    )

    return x


def resample_ecg(
    ecg,
    fs,
    target_fs=TARGET_ECG_FS,
):
    """
    Resample ECG to the 1000 Hz grid used for QRS morphology
    extraction.
    """
    ecg = interpolate_nonfinite(
        ecg
    )

    if np.isclose(
        fs,
        target_fs,
    ):
        return (
            ecg.copy(),
            float(
                target_fs
            ),
        )

    original_fs = int(
        round(
            fs
        )
    )

    target_fs_int = int(
        round(
            target_fs
        )
    )

    divisor = np.gcd(
        original_fs,
        target_fs_int,
    )

    up = (
        target_fs_int
        // divisor
    )

    down = (
        original_fs
        // divisor
    )

    resampled = resample_poly(
        ecg,
        up=up,
        down=down,
    )

    return (
        resampled,
        float(
            target_fs
        ),
    )


def filter_ecg(
    ecg,
    fs=TARGET_ECG_FS,
):
    """
    Low-pass filter ECG at 40 Hz and remove baseline variation
    with a 200-sample moving median.
    """
    b, a = butter(
        3,
        40.0,
        btype="lowpass",
        fs=fs,
    )

    filtered = filtfilt(
        b,
        a,
        ecg,
    )

    baseline = median_filter(
        filtered,
        size=200,
        mode="nearest",
    )

    return (
        filtered - baseline
    )


def qrs_detector_amplitude(
    ecg,
    fs=TARGET_ECG_FS,
):
    """
    Generate QRS candidates from band-limited signal energy.
    """
    b, a = butter(
        3,
        [
            5.0,
            25.0,
        ],
        btype="bandpass",
        fs=fs,
    )

    qrs_signal = filtfilt(
        b,
        a,
        ecg,
    )

    energy = (
        qrs_signal ** 2
    )

    integration_samples = int(
        round(
            0.080 * fs
        )
    )

    energy = uniform_filter1d(
        energy,
        size=integration_samples,
        mode="nearest",
    )

    minimum_distance = int(
        round(
            0.30 * fs
        )
    )

    prominence = (
        0.5
        * np.std(
            energy
        )
    )

    peaks, _ = find_peaks(
        energy,
        distance=minimum_distance,
        prominence=prominence,
    )

    return peaks


def qrs_detector_derivative(
    ecg,
    fs=TARGET_ECG_FS,
):
    """
    Generate a complementary QRS candidate set from
    derivative energy.
    """
    derivative = np.diff(
        ecg,
        prepend=ecg[0],
    )

    energy = (
        derivative ** 2
    )

    integration_samples = int(
        round(
            0.120 * fs
        )
    )

    energy = uniform_filter1d(
        energy,
        size=integration_samples,
        mode="nearest",
    )

    minimum_distance = int(
        round(
            0.30 * fs
        )
    )

    threshold = (
        np.median(
            energy
        )
        + 2.0
        * np.std(
            energy
        )
    )

    peaks, _ = find_peaks(
        energy,
        distance=minimum_distance,
        height=threshold,
    )

    if len(
        peaks
    ) < 10:
        peaks, _ = find_peaks(
            energy,
            distance=minimum_distance,
            prominence=(
                0.5
                * np.std(
                    energy
                )
            ),
        )

    return peaks


def refine_to_local_maximum(
    ecg,
    peaks,
    radius_samples=40,
):
    """
    Refine each QRS candidate to the local ECG maximum.
    """
    refined = []

    for peak in np.asarray(
        peaks,
        dtype=int,
    ):
        start = max(
            0,
            peak - radius_samples,
        )

        stop = min(
            len(ecg),
            peak
            + radius_samples
            + 1,
        )

        if stop <= start:
            continue

        local_offset = int(
            np.argmax(
                ecg[
                    start:stop
                ]
            )
        )

        refined.append(
            start
            + local_offset
        )

    if not refined:
        return np.array(
            [],
            dtype=int,
        )

    return np.unique(
        np.asarray(
            refined,
            dtype=int,
        )
    )


def match_peak_sets(
    peaks_a,
    peaks_b,
    tolerance_samples=100,
):
    """
    Match R-peak candidates from two complementary detectors.
    """
    peaks_a = np.asarray(
        peaks_a,
        dtype=int,
    )

    peaks_b = np.asarray(
        peaks_b,
        dtype=int,
    )

    if (
        len(peaks_a) == 0
        or len(peaks_b) == 0
    ):
        return np.array(
            [],
            dtype=int,
        )

    matched = []
    j = 0

    for peak_a in peaks_a:
        while (
            j + 1 < len(peaks_b)
            and abs(
                peaks_b[j + 1]
                - peak_a
            )
            <= abs(
                peaks_b[j]
                - peak_a
            )
        ):
            j += 1

        if abs(
            peaks_b[j]
            - peak_a
        ) <= tolerance_samples:
            matched.append(
                int(
                    round(
                        0.5
                        * (
                            peak_a
                            + peaks_b[j]
                        )
                    )
                )
            )

    if not matched:
        return np.array(
            [],
            dtype=int,
        )

    return np.unique(
        np.asarray(
            matched,
            dtype=int,
        )
    )


def detect_r_peaks(
    ecg,
    fs=TARGET_ECG_FS,
):
    """
    Detect R peaks by combining amplitude- and
    derivative-based QRS candidate sets.
    """
    amplitude_peaks = (
        qrs_detector_amplitude(
            ecg,
            fs,
        )
    )

    derivative_peaks = (
        qrs_detector_derivative(
            ecg,
            fs,
        )
    )

    amplitude_peaks = (
        refine_to_local_maximum(
            ecg,
            amplitude_peaks,
            radius_samples=40,
        )
    )

    derivative_peaks = (
        refine_to_local_maximum(
            ecg,
            derivative_peaks,
            radius_samples=40,
        )
    )

    matched = match_peak_sets(
        amplitude_peaks,
        derivative_peaks,
        tolerance_samples=100,
    )

    if len(
        matched
    ) < 10:
        matched = amplitude_peaks

    return matched


def select_ecg_polarity(
    ecg_positive,
    ecg_negative,
    r_positive,
    r_negative,
):
    """
    Select the ECG orientation with greater median R-peak
    prominence relative to baseline.
    """
    if len(
        r_positive
    ) == 0:
        raise RuntimeError(
            "No R peaks detected for positive ECG polarity."
        )

    if len(
        r_negative
    ) == 0:
        raise RuntimeError(
            "No R peaks detected for negative ECG polarity."
        )

    positive_prominence = (
        np.median(
            ecg_positive[
                r_positive
            ]
        )
        - np.median(
            ecg_positive
        )
    )

    negative_prominence = (
        np.median(
            ecg_negative[
                r_negative
            ]
        )
        - np.median(
            ecg_negative
        )
    )

    if (
        negative_prominence
        > positive_prominence
    ):
        return {
            "ecg": ecg_negative,
            "r_peaks": r_negative,
            "polarity": -1,
        }

    return {
        "ecg": ecg_positive,
        "r_peaks": r_positive,
        "polarity": 1,
    }


def find_qs_points(
    ecg,
    r_peaks,
):
    """
    Identify Q and S as local minima surrounding each
    retained R peak.

    Q is searched over the preceding 50 ms and S over the
    following 60 ms.
    """
    q_points = []
    retained_r = []
    s_points = []

    for r_peak in np.asarray(
        r_peaks,
        dtype=int,
    ):
        if (
            r_peak < 50
            or r_peak + 60
            >= len(ecg)
        ):
            continue

        q_segment = ecg[
            r_peak - 50:
            r_peak
        ]

        s_segment = ecg[
            r_peak + 1:
            r_peak + 61
        ]

        q_point = (
            r_peak
            - 50
            + int(
                np.argmin(
                    q_segment
                )
            )
        )

        s_point = (
            r_peak
            + 1
            + int(
                np.argmin(
                    s_segment
                )
            )
        )

        q_points.append(
            q_point
        )

        retained_r.append(
            r_peak
        )

        s_points.append(
            s_point
        )

    return (
        np.asarray(
            q_points,
            dtype=int,
        ),
        np.asarray(
            retained_r,
            dtype=int,
        ),
        np.asarray(
            s_points,
            dtype=int,
        ),
    )


def extract_qrs_states(
    ecg,
    r_peaks,
):
    """
    Construct the 91-dimensional ECG state trajectory.

    Each heartbeat is represented by samples R-30 through
    R+60 inclusive. No delay embedding is applied.
    """
    states = []
    retained_r = []

    for r_peak in np.asarray(
        r_peaks,
        dtype=int,
    ):
        start = (
            r_peak
            - BEAT_SAMPLES_BEFORE_R
        )

        stop = (
            r_peak
            + BEAT_SAMPLES_AFTER_R
            + 1
        )

        if (
            start < 0
            or stop > len(ecg)
        ):
            continue

        beat = ecg[
            start:stop
        ]

        if len(
            beat
        ) != BEAT_DIMENSION:
            continue

        states.append(
            beat
        )

        retained_r.append(
            r_peak
        )

    states = np.asarray(
        states,
        dtype=float,
    )

    retained_r = np.asarray(
        retained_r,
        dtype=int,
    )

    if (
        states.ndim != 2
        or states.shape[1]
        != BEAT_DIMENSION
    ):
        raise RuntimeError(
            "Failed to construct the 91-dimensional "
            "QRS trajectory."
        )

    return (
        states,
        retained_r,
    )


def preprocess_ecg(
    ecg,
    fs,
):
    """
    Preprocess ECG and construct the 91-dimensional QRS
    morphology trajectory.

    ECG is resampled to 1000 Hz, filtered, evaluated in both
    polarities, and segmented around detected R peaks.
    """
    ecg_1000, processed_fs = (
        resample_ecg(
            ecg,
            fs,
            target_fs=TARGET_ECG_FS,
        )
    )

    positive = filter_ecg(
        ecg_1000,
        processed_fs,
    )

    negative = filter_ecg(
        -ecg_1000,
        processed_fs,
    )

    r_positive = detect_r_peaks(
        positive,
        processed_fs,
    )

    r_negative = detect_r_peaks(
        negative,
        processed_fs,
    )

    margin = 150

    r_positive = r_positive[
        (r_positive > margin)
        & (
            r_positive
            < len(positive)
            - margin
        )
    ]

    r_negative = r_negative[
        (r_negative > margin)
        & (
            r_negative
            < len(negative)
            - margin
        )
    ]

    polarity_result = (
        select_ecg_polarity(
            positive,
            negative,
            r_positive,
            r_negative,
        )
    )

    processed_ecg = (
        polarity_result[
            "ecg"
        ]
    )

    r_peaks = (
        polarity_result[
            "r_peaks"
        ]
    )

    q_points, r_peaks, s_points = (
        find_qs_points(
            processed_ecg,
            r_peaks,
        )
    )

    states, retained_r = (
        extract_qrs_states(
            processed_ecg,
            r_peaks,
        )
    )

    r_lookup = {
        int(r_peak): index
        for index, r_peak in enumerate(
            r_peaks
        )
    }

    retained_indices = np.asarray(
        [
            r_lookup[
                int(r_peak)
            ]
            for r_peak in retained_r
        ],
        dtype=int,
    )

    return {
        "processed_ecg": processed_ecg,
        "fs": float(
            processed_fs
        ),
        "polarity": int(
            polarity_result[
                "polarity"
            ]
        ),
        "Q": q_points[
            retained_indices
        ],
        "R": retained_r,
        "S": s_points[
            retained_indices
        ],
        "states": states,
    }


def screen_ecg_states(
    ecg_result,
    respiratory_mask,
    respiratory_fs,
):
    """
    Select ECG states whose R-peak times fall within the
    requested respiratory training set.

    The transition mask excludes transitions across beats
    removed by respiratory screening.
    """
    states = np.asarray(
        ecg_result[
            "states"
        ],
        dtype=float,
    )

    r_peaks = np.asarray(
        ecg_result[
            "R"
        ],
        dtype=int,
    )

    beat_times = (
        r_peaks
        / ecg_result[
            "fs"
        ]
    )

    respiratory_indices = np.rint(
        beat_times
        * respiratory_fs
    ).astype(int)

    valid_time = (
        (respiratory_indices >= 0)
        & (
            respiratory_indices
            < len(
                respiratory_mask
            )
        )
    )

    keep = np.zeros(
        len(states),
        dtype=bool,
    )

    keep[
        valid_time
    ] = respiratory_mask[
        respiratory_indices[
            valid_time
        ]
    ]

    original_indices = (
        np.flatnonzero(
            keep
        )
    )

    selected_states = states[
        keep
    ]

    selected_times = beat_times[
        keep
    ]

    selected_r = r_peaks[
        keep
    ]

    transition_mask = (
        np.diff(
            original_indices
        ) == 1
    )

    return {
        "states": selected_states,
        "beat_times": selected_times,
        "r_peaks": selected_r,
        "original_indices": original_indices,
        "transition_mask": transition_mask,
    }


def normalize_ecg_states(
    states,
):
    """
    Center the QRS morphology cloud and apply a single global
    scale factor.

    A common scalar is applied to all 91 coordinates. This
    preserves relative waveform morphology and Euclidean
    neighborhood structure while standardizing overall scale.
    """
    states = np.asarray(
        states,
        dtype=float,
    )

    mean_state = np.mean(
        states,
        axis=0,
    )

    centered = (
        states
        - mean_state[
            None,
            :
        ]
    )

    scale = float(
        np.sqrt(
            np.mean(
                centered ** 2
            )
        )
    )

    if (
        not np.isfinite(
            scale
        )
        or scale <= 0
    ):
        raise ValueError(
            "Invalid ECG global scale."
        )

    X = (
        centered
        / scale
    )

    return {
        "X": X,
        "mean_state": mean_state,
        "scale": scale,
    }


def prepare_ecg(
    mat_path,
    respiratory_mask,
    respiratory_fs,
    ecg_field=None,
):
    """
    Prepare the screened 91-dimensional ECG state trajectory.

    No Takens embedding is applied to the ECG states.
    """
    if ecg_field is None:
        ecg_field = find_ecg_field(
            mat_path
        )

    ecg_raw, raw_fs = (
        load_vgh_channel(
            mat_path,
            ecg_field,
        )
    )

    preprocessing = preprocess_ecg(
        ecg_raw,
        raw_fs,
    )

    screening = screen_ecg_states(
        preprocessing,
        respiratory_mask,
        respiratory_fs,
    )

    if len(
        screening[
            "states"
        ]
    ) < 2:
        raise RuntimeError(
            "Too few ECG states remain after screening."
        )

    normalized = normalize_ecg_states(
        screening[
            "states"
        ]
    )

    return {
        "ecg_field": ecg_field,

        "raw_fs": float(
            raw_fs
        ),

        "processed_ecg": preprocessing[
            "processed_ecg"
        ],

        "fs": preprocessing[
            "fs"
        ],

        "polarity": preprocessing[
            "polarity"
        ],

        "Q": preprocessing[
            "Q"
        ],

        "R": preprocessing[
            "R"
        ],

        "S": preprocessing[
            "S"
        ],

        "states_all": preprocessing[
            "states"
        ],

        "states": screening[
            "states"
        ],

        "X": normalized[
            "X"
        ],

        "mean_state": normalized[
            "mean_state"
        ],

        "scale": normalized[
            "scale"
        ],

        "beat_times": screening[
            "beat_times"
        ],

        "r_peaks": screening[
            "r_peaks"
        ],

        "original_indices": screening[
            "original_indices"
        ],

        "transition_mask": screening[
            "transition_mask"
        ],
    }


def preprocess_ecg_matlab(
    mat_path,
):
    """
    Run the reference MATLAB ECG preprocessing pipeline and return
    QRS fiducial points and 91-sample QRS morphology vectors.
    """
    from pathlib import Path
    import subprocess
    import tempfile

    from scipy.io import loadmat

    mat_path = Path(mat_path).resolve()

    repo_root = Path(__file__).resolve().parents[1]
    matlab_dir = repo_root / "matlab" / "ecg"

    if not mat_path.exists():
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}"
        )

    if not matlab_dir.exists():
        raise FileNotFoundError(
            f"MATLAB ECG directory not found: {matlab_dir}"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = (
            Path(tmpdir)
            / "ecg_preprocessed.mat"
        )

        input_matlab = str(mat_path).replace(
            "'",
            "''",
        )

        output_matlab = str(output_path).replace(
            "'",
            "''",
        )

        matlab_command = (
            "run_preprocess_ecg_file("
            f"'{input_matlab}',"
            f"'{output_matlab}'"
            ")"
        )

        result = subprocess.run(
            [
                "matlab",
                "-batch",
                matlab_command,
            ],
            cwd=str(matlab_dir),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "MATLAB ECG preprocessing failed.\n\n"
                f"STDOUT:\n{result.stdout}\n\n"
                f"STDERR:\n{result.stderr}"
            )

        if not output_path.exists():
            raise RuntimeError(
                "MATLAB completed without producing the expected output file."
            )

        data = loadmat(
            output_path
        )

    prep_ecg = np.asarray(
        data["prep_ecg"],
        dtype=float,
    ).squeeze()

    Q = np.asarray(
        data["Q"],
        dtype=int,
    ).squeeze()

    R = np.asarray(
        data["R"],
        dtype=int,
    ).squeeze()

    S = np.asarray(
        data["S"],
        dtype=int,
    ).squeeze()

    states = np.asarray(
        data["B"],
        dtype=float,
    )

    fs = float(
        np.asarray(
            data["fs"]
        ).squeeze()
    )

    # Convert MATLAB one-based sample indices to Python zero-based indices.
    Q = Q - 1
    R = R - 1
    S = S - 1

    return {
        "processed_ecg": prep_ecg,
        "fs": fs,
        "Q": Q,
        "R": R,
        "S": S,
        "states": states,
    }
