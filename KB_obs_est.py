#!/usr/bin/env python3

import os
import argparse
import numpy as np
import time
import glob

from kernel_estimators import euclidean_kernel_estimate_vec
from SDE_sample_KB import (
    embed_klein_bottle,
    pullback_to_square,
    true_diffusion,
    true_ambient_drift,
)

def log_stage(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def _pick_latest(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))

def square_grid_points(grid_side: int, u_min: float, u_max: float, v_min: float, v_max: float):
    xs = np.linspace(u_min, u_max, grid_side, endpoint=False)
    ys = np.linspace(v_min, v_max, grid_side, endpoint=False)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel()])

def load_index(traj_dir: str):
    idx_candidates = glob.glob(os.path.join(traj_dir, "index_sub*.npz"))
    if not idx_candidates:
        raise FileNotFoundError(f"Missing index_sub*.npz in traj_dir: {traj_dir}")
    idx_path = _pick_latest(sorted(idx_candidates))
    idx = np.load(idx_path, allow_pickle=True)
    return idx, idx_path

def resolve_single_kb_traj_file(traj_dir: str, idx):
    if "traj_files" in idx.files:
        tf = idx["traj_files"]
        if isinstance(tf, np.ndarray) and tf.dtype == object:
            tf = list(tf.tolist())
        elif isinstance(tf, np.ndarray):
            tf = list(tf)
        else:
            tf = [str(tf)]
    else:
        tf = []

    cand = []
    for p in tf:
        p = str(p)
        if "kb_traj_sim" not in p:
            continue
        if not os.path.isabs(p):
            p = os.path.join(traj_dir, p)
        if os.path.isfile(p):
            cand.append(p)

    if cand:
        return cand[0]

    fallbacks = sorted(glob.glob(os.path.join(traj_dir, "kb_traj_sim*_sub*.npy")))
    if fallbacks:
        return _pick_latest(fallbacks)

    raise FileNotFoundError(
        f"No KB ambient trajectory found. Expected kb_traj_sim*_sub*.npy in {traj_dir} "
        f"(or a matching entry in index traj_files)."
    )

def run_experiment(args):
    idx, idx_path = load_index(args.traj_dir)
    traj_path = resolve_single_kb_traj_file(args.traj_dir, idx)

    N_full = int(np.asarray(idx["N"]).item()) if "N" in idx.files else int(args.N)
    Delta_full = float(np.asarray(idx["Delta"]).item()) if "Delta" in idx.files else float(args.Delta)
    T_full = float(np.asarray(idx["T"]).item()) if "T" in idx.files else (N_full * Delta_full)

    subsample = int(np.asarray(idx["subsample"]).item()) if "subsample" in idx.files else 1
    Delta_eff = Delta_full * subsample

    X_KB = np.load(traj_path, mmap_mode="r")
    if X_KB.ndim != 2 or X_KB.shape[1] != 4:
        raise ValueError(f"Expected KB ambient trajectory shape (N_eff+1,4). Got {X_KB.shape} from {traj_path}")

    N_eff = int(X_KB.shape[0] - 1)
    T_eff = Delta_eff * N_eff

    log_stage(
        f"Stage B | traj_dir={args.traj_dir} | index={idx_path} | traj={traj_path} | "
        f"N_full={N_full} Delta_full={Delta_full:g} subsample={subsample} | "
        f"N_eff={N_eff} Delta_eff={Delta_eff:g} | T={T_full:g} (T_eff={T_eff:g})"
    )

    X_eff = np.asarray(X_KB, dtype=(np.float32 if args.traj_dtype == "float32" else np.float64))
    log_stage(f"Stage B | embedded traj_eff in NPZ | shape={X_eff.shape} dtype={X_eff.dtype}")

    base_points = square_grid_points(args.grid_side, args.u_min, args.u_max, args.v_min, args.v_max)
    num_basepoints = base_points.shape[0]
    num_neighbors = int(args.num_neighbors)

    log_stage(f"Starting experiment | basepoints={num_basepoints} | k={num_neighbors}")

    P_true = np.zeros((num_basepoints, 4, 4))
    mu_true = np.zeros((num_basepoints, 4))

    for i in range(num_basepoints):
        u, v = base_points[i]
        Sigma = true_diffusion(u, v, args.a, args.r)
        U, _, _ = np.linalg.svd(Sigma)
        U2 = U[:, :2]
        P_true[i] = U2 @ U2.T
        mu_true[i] = true_ambient_drift(u, v, args.a, args.r)

    mu_proj_TT = np.full((num_basepoints, 4), np.nan)
    mu_proj_TE = np.full((num_basepoints, 4), np.nan)
    mu_proj_EE = np.full((num_basepoints, 4), np.nan)

    mu_proj_TT_sq = np.full((num_basepoints, 2), np.nan)
    mu_proj_TE_sq = np.full((num_basepoints, 2), np.nan)
    mu_proj_EE_sq = np.full((num_basepoints, 2), np.nan)

    occ_den = np.full(num_basepoints, np.nan)
    mu_hat_store = np.full((num_basepoints, 4), np.nan)
    sigma_hat_store = np.full((num_basepoints, 4, 4), np.nan)
    h_store = np.full(num_basepoints, np.nan)

    nan_counter = 0
    t0 = time.time()

    for i in range(num_basepoints):
        u, v = base_points[i]
        x0_KB = embed_klein_bottle(np.array([[u, v]]), args.a, args.r)[0]

        mu_hat, Sigma_hat, density, h = euclidean_kernel_estimate_vec(
            x0_KB,
            X_eff,
            Delta_eff,
            num_neighbors=num_neighbors
        )

        if mu_hat is None or Sigma_hat is None:
            occ_den[i] = float(density) if density is not None else np.nan
            h_store[i] = float(h) if h is not None else np.nan
            nan_counter += 1
        else:
            occ_den[i] = float(density)
            h_store[i] = float(h)
            mu_hat_store[i] = mu_hat

            Sigma_hat = 0.5 * (Sigma_hat + Sigma_hat.T)
            Sigma_hat += 1e-8 * np.eye(4)
            sigma_hat_store[i] = Sigma_hat

            try:
                U_hat, _, _ = np.linalg.svd(Sigma_hat)
            except np.linalg.LinAlgError:
                nan_counter += 1
                continue

            U2_hat = U_hat[:, :2]
            P_est = U2_hat @ U2_hat.T

            mu_TT = P_true[i] @ mu_true[i]
            mu_TE = P_true[i] @ mu_hat
            mu_EE = P_est @ mu_hat

            mu_proj_TT[i] = mu_TT
            mu_proj_TE[i] = mu_TE
            mu_proj_EE[i] = mu_EE

            mu_proj_TT_sq[i] = pullback_to_square(mu_TT, u, v, args.a, args.r)
            mu_proj_TE_sq[i] = pullback_to_square(mu_TE, u, v, args.a, args.r)
            mu_proj_EE_sq[i] = pullback_to_square(mu_EE, u, v, args.a, args.r)

        if (i + 1) % max(1, num_basepoints // 10) == 0 or (i + 1) == num_basepoints:
            elapsed = (time.time() - t0) / 60
            log_stage(f"Completed {i+1}/{num_basepoints} basepoints | elapsed={elapsed:.2f} min")

    total_minutes = (time.time() - t0) / 60
    log_stage(f"Stage B done | total={total_minutes:.2f} min | nan_skips={nan_counter}")

    return {
        "base_points": base_points,
        "u_min": args.u_min,
        "u_max": args.u_max,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "mu_proj_TT": mu_proj_TT[None, :, :],
        "mu_proj_TE": mu_proj_TE[None, :, :],
        "mu_proj_EE": mu_proj_EE[None, :, :],
        "mu_proj_TT_square": mu_proj_TT_sq[None, :, :],
        "mu_proj_TE_square": mu_proj_TE_sq[None, :, :],
        "mu_proj_EE_square": mu_proj_EE_sq[None, :, :],
        "occ_den": occ_den[None, :],
        "sigma_hat": sigma_hat_store[None, :, :, :],
        "mu_hat": mu_hat_store[None, :, :],
        "h_used": h_store[None, :],
        "seed": args.seed,
        "N_full": N_full,
        "N_eff": N_eff,
        "Delta_full": Delta_full,
        "Delta_eff": Delta_eff,
        "T": T_full,
        "subsample": subsample,
        "a": args.a,
        "r": args.r,
        "num_neighbors": num_neighbors,
        "num_sims": 1,
        "traj_dir": args.traj_dir,
        "index_path": idx_path,
        "traj_path": traj_path,
        "traj_eff": X_eff,
        "traj_eff_shape": np.asarray(X_eff.shape, dtype=np.int64),
        "traj_eff_dtype": str(X_eff.dtype),
    }

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--Delta", type=float, default=0.0)

    parser.add_argument("--grid_side", type=int, default=8)
    parser.add_argument("--a", type=float, default=2.0)
    parser.add_argument("--r", type=float, default=1.0)
    parser.add_argument("--u_min", type=float, default=0.0)
    parser.add_argument("--u_max", type=float, default=2*np.pi)
    parser.add_argument("--v_min", type=float, default=0.0)
    parser.add_argument("--v_max", type=float, default=2*np.pi)

    parser.add_argument("--num_neighbors", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--traj_dtype", type=str, default="float32", choices=["float32", "float64"])

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = run_experiment(args)

    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.output_dir, f"obsKB_stageB_seed{args.seed}_{ts}.npz")
    np.savez_compressed(path, **results)
    log_stage(f"Saved to {path}")

if __name__ == "__main__":
    main()
