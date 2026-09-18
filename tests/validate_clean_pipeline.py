from pathlib import Path

import numpy as np

from respiratory_pipeline.airflow import prepare_airflow
from respiratory_pipeline.ecg import prepare_ecg
from respiratory_pipeline.edr import (
    local_detrend_qrs,
    build_delay_embedding,
    observed_edr_from_qrs,
)
from respiratory_pipeline.geometry import estimate_geometry
from respiratory_pipeline.simulation import (
    derive_geometry_view,
    simulate_delay_sde,
)


MAT_PATH = Path("TVGH25Data/20241020052.mat")


print("1. Airflow")
airflow = prepare_airflow(
    MAT_PATH,
    mode="non_apnea",
)

print("   airflow samples:", len(airflow["airflow"]))
print("   airflow states:", airflow["X"].shape)
print("   airflow transitions:", int(np.sum(airflow["transition_mask"])))


print("\n2. ECG / QRS")
ecg = prepare_ecg(
    MAT_PATH,
    np.ones(
        len(airflow["airflow"]),
        dtype=bool,
    ),
    airflow["fs"],
)

states = np.asarray(
    ecg["states"],
    dtype=float,
)

beat_times = (
    np.asarray(
        ecg["R"],
        dtype=float,
    )
    / float(ecg["fs"])
)

print("   QRS states:", states.shape)
print("   beats:", len(beat_times))


print("\n3. Observed EDR")
observed_edr = observed_edr_from_qrs(
    states,
    beat_times,
)

print("   EDR samples:", len(observed_edr["signal"]))
print("   EDR fs:", observed_edr["fs"])


print("\n4. Local QRS detrending")
detrended = local_detrend_qrs(
    states,
    beat_times,
)

qrs_detrended = detrended["states"]

print("   detrended states:", qrs_detrended.shape)


print("\n5. Delay embedding")
delay = build_delay_embedding(
    qrs_detrended,
    beat_times,
)

X = delay["X"]
transition_dt = delay["transition_dt"]
transition_mask = delay["transition_mask"]

print("   delay states:", X.shape)
print("   valid transitions:", int(np.sum(transition_mask)))
print(
    "   median transition dt:",
    float(np.median(transition_dt[transition_mask])),
)


print("\n6. Geometry")
num_neighbors = min(
    300,
    int(np.sum(transition_mask)) - 1,
)

geometry_d2 = estimate_geometry(
    X,
    transition_mask,
    transition_dt,
    intrinsic_dim=2,
    num_neighbors=num_neighbors,
)

print(
    "   valid geometry states:",
    int(np.sum(geometry_d2["valid_geometry"])),
)


print("\n7. Local transition times")
timing = {
    "local_dt": geometry_d2["local_dt"],
    "lower_bound": geometry_d2["dt_lower_bound"],
    "upper_bound": geometry_d2["dt_upper_bound"],
    "global_median": geometry_d2["global_median_dt"],
}

print("   global median dt:", timing["global_median"])
print("   dt lower bound:", timing["lower_bound"])
print("   dt upper bound:", timing["upper_bound"])


print("\n8. Short d=1 deterministic simulation")
geometry_d1 = derive_geometry_view(
    geometry_d2,
    intrinsic_dim=1,
)

simulation = simulate_delay_sde(
    X,
    geometry_d1,
    timing["local_dt"],
    simulation_seconds=30.0,
    diffusion_strength=0.0,
    seed=42,
)

print("   path:", simulation["path"].shape)
print("   duration:", simulation["time"][-1])
print("   steps:", len(simulation["used_dt"]))


print("\nCLEAN PIPELINE VALIDATION PASSED")
