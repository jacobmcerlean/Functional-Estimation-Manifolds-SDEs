import numpy as np

from respiratory_pipeline.respiratory_metrics import (
    summarize_respiratory_signal,
    compare_respiratory_signals,
)


FS = 10.0
DURATION_SECONDS = 300.0

time = np.arange(
    0.0,
    DURATION_SECONDS,
    1.0 / FS,
)

# 0.25 Hz = 15 breaths/minute.
observed = np.sin(
    2.0
    * np.pi
    * 0.25
    * time
)

# 0.30 Hz = 18 breaths/minute.
simulated = np.sin(
    2.0
    * np.pi
    * 0.30
    * time
)


observed_metrics = (
    summarize_respiratory_signal(
        observed,
        fs=FS,
    )
)

comparison = (
    compare_respiratory_signals(
        observed,
        simulated,
        fs=FS,
    )
)


print(
    "Observed mean BR:",
    observed_metrics[
        "mean_breathing_rate_bpm"
    ],
)

print(
    "Observed dominant BR:",
    observed_metrics[
        "dominant_breathing_rate_bpm"
    ],
)

print(
    "Mean BR error:",
    comparison[
        "mean_breathing_rate_error_bpm"
    ],
)

print(
    "BR Wasserstein distance:",
    comparison[
        "breathing_rate_wasserstein_bpm"
    ],
)
