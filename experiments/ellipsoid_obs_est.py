#!/usr/bin/env python3
from kernel_estimators import euclidean_kernel_estimate_vec
import os
import argparse
import numpy as np
import time
import glob

from observed_ellipsoid import (
    sphere_to_ellipsoid,
    true_tangent_project_ellipsoid,
    sample_spherical_cap
)

def log_stage(msg: str) -> None:
    # Timestamped logger for long-running experiment stages.
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def ecc_to_tag(ecc):
    # Convert one eccentricity triple into a compact filename tag.
    a1, a2, a3 = [float(x) for x in ecc]
    return f"{a1:g}-{a2:g}-{a3:g}"

def _pick_latest(paths):
    # From a list of candidate files, keep only existing files
    # and return the most recently modified one.
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return None
    return max(paths, key=lambda p: os.path.getmtime(p))

def find_traj_path(traj_dir: str, ecc):
    # Find the most recent Stage A index file, then resolve the
    # ellipsoid trajectory corresponding to the requested eccentricity.
    idx_candidates = glob.glob(os.path.join(traj_dir, "index_sub*.npz"))
    if len(idx_candidates) == 0:
        raise FileNotFoundError(f"Missing index_sub*.npz in traj_dir: {traj_dir}")

    idx_path = _pick_latest(sorted(idx_candidates))
    idx = np.load(idx_path, allow_pickle=True)

    tag = ecc_to_tag(ecc)

    # Recover the subsampling factor from the index when available.
    if "subsample" in idx.files:
        s = int(np.asarray(idx["subsample"]).item())
    else:
	s = 1

    candidates = []

    # Prefer explicit trajectory filenames stored in the index.
    if "traj_files" in idx.files:
        traj_files = idx["traj_files"]
        if isinstance(traj_files, np.ndarray) and traj_files.dtype == object:
            traj_files = list(traj_files.tolist())
        elif isinstance(traj_files, np.ndarray):
            traj_files = list(traj_files)
        else:
            traj_files = [str(traj_files)]

        for p in traj_files:
            sp = str(p)
            if (f"traj_ecc{tag}_" in sp) and (f"_sub{s}.npy" in sp):
                candidates.append(sp)

    # If the index does not resolve the file directly, fall back to the
    # expected filename pattern in traj_dir.
    if len(candidates) == 0:
        p = os.path.join(traj_dir, f"traj_ecc{tag}_sub{s}.npy")
        if os.path.isfile(p):
            return p, idx, idx_path
        raise FileNotFoundError(f"No trajectory found for ecc={tag} sub={s} in {traj_dir}")

    p = candidates[0]
    if not os.path.isabs(p):
        p = os.path.join(traj_dir, p)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Index points to missing file: {p}")

    return p, idx, idx_path

def run_experiment(args):
    # Resolve the trajectory file for the requested ellipsoid and load
    # metadata from the Stage A index file.
    ecc = np.asarray(args.eccentricities, dtype=float)

    traj_path, idx, idx_path = find_traj_path(args.traj_dir, ecc)

    # Prefer values recorded in the index; fall back to CLI defaults only if needed.
    if "N" in idx.files:
        N_full = int(np.asarray(idx["N"]).item())
    else:
	N_full = int(args.N)

    if "Delta" in idx.files:
        Delta_full = float(np.asarray(idx["Delta"]).item())
    else:
	Delta_full = float(args.Delta)

    if "subsample" in idx.files:
        subsample = int(np.asarray(idx["subsample"]).item())
    else:
	subsample = 1

    log_stage(f"Stage B | loading trajectory for ecc={ecc} from {traj_path}")
    log_stage(f"Stage B | index={idx_path}")

    # Load the down-sampled ellipsoid trajectory from Stage A.
    X = np.load(traj_path, mmap_mode="r")
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Expected trajectory shape (N_eff+1,3). Got {X.shape}")

    # Effective values refer to the subsampled trajectory that is actually used.
    N_eff = int(X.shape[0] - 1)
    Delta_eff = Delta_full * subsample
    T_full = Delta_full * N_full
    T_eff = Delta_eff * N_eff

    num_neighbors = int(args.num_neighbors)
    B = int(args.num_samples)

    log_stage(
	f"Trajectory loaded (mmap) | shape={X.shape} dtype={X.dtype} | "
        f"N_full={N_full} subsample={subsample} N_eff={N_eff} | "
        f"Delta_full={Delta_full:g} Delta_eff={Delta_eff:g} | "
        f"T={T_full:g} (T_eff={T_eff:g})"
    )

    # Seed RNG for reproducible base-point sampling on the spherical cap.
    np.random.seed(args.seed)

    # Sample base points on the sphere, then push them forward to the ellipsoid.
    sampled_sphere = sample_spherical_cap(
        args.num_samples,
        args.cap_threshold
    )
    sampled_points = np.array([sphere_to_ellipsoid(p, ecc) for p in sampled_sphere], dtype=float)

    # Storage for:
    # mu1 = raw ambient drift estimate
    # mu2 = estimated tangent projection of the ambient drift estimate
    # mu3 = true tangent projection of the ambient drift estimate
    mu1 = np.full((B, 3), np.nan)
    mu2 = np.full((B, 3), np.nan)
    mu3 = np.full((B, 3), np.nan)

    # Store estimated diffusion matrices, occupation weights, and chosen bandwidths.
    sigma_hats = np.full((B, 3, 3), np.nan)
    occupation = np.full(B, np.nan)
    h_store = np.full(B, np.nan)

    t0 = time.time()
    log_stage(f"Stage B | basepoint estimation | B={B} k={num_neighbors}")

    # Main estimation loop over sampled ellipsoid base points.
    for i, point in enumerate(sampled_points):
        mu_hat, sigma_hat, den, h = euclidean_kernel_estimate_vec(
            point,
            X,
            Delta_eff,
            num_neighbors=num_neighbors
        )

        # If the local estimator degenerates, leave NaNs in the outputs.
	if mu_hat is None or sigma_hat is None:
            continue

        occupation[i] = den
        h_store[i] = h
        mu1[i] = mu_hat
        sigma_hats[i] = sigma_hat

        # Symmetrize the diffusion estimate and add a small ridge for SVD stability.
        sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)
        sigma_hat += 1e-10 * np.eye(3)

        try:
            # Estimated tangent space comes from the top two singular directions.
            U, S, _ = np.linalg.svd(sigma_hat)
        except np.linalg.LinAlgError:
            continue

        idxs = np.argsort(S)[::-1]
        U2 = U[:, idxs[:2]]
        P_est = U2 @ U2.T
        mu2[i] = P_est @ mu_hat

        # True tangent projection on the ellipsoid at the current base point.
        P_true = true_tangent_project_ellipsoid(point, ecc)
        mu3[i] = P_true @ mu_hat

        # Progress logging over base points.
        if (i + 1) % max(1, B // 10) == 0 or (i + 1) == B:
            elapsed = (time.time() - t0) / 60
            log_stage(f"Completed {i+1}/{B} basepoints | elapsed={elapsed:.2f} min")

    log_stage(f"Stage B done | total={(time.time()-t0)/60:.2f} min")

    # Add a leading singleton axis for compatibility with downstream code
    # expecting a simulation dimension.
    mu1_all = mu1[None, :, :]
    mu2_all = mu2[None, :, :]
    mu3_all = mu3[None, :, :]
    sigma_all = sigma_hats[None, :, :, :]
    occupation_all = occupation[None, :]
    h_all = h_store[None, :]

    # Package all outputs needed for later analysis and figure generation.
    return {
	"sampled_points": sampled_points,
        "mu1_hats": mu1_all,
        "mu2_hats": mu2_all,
        "mu3_hats": mu3_all,
        "sigma_hats": sigma_all,
        "occupation": occupation_all,
        "h_used": h_all,
        "eccentricities": ecc,
        "N_full": N_full,
        "N_eff": N_eff,
        "Delta_full": Delta_full,
        "Delta_eff": Delta_eff,
        "T": T_full,
        "subsample": subsample,
        "num_sims": 1,
        "num_neighbors": num_neighbors,
        "seed": args.seed,
        "traj_path": traj_path,
        "traj_dir": args.traj_dir,
        "index_path": idx_path,
    }

def main():
    # Parse command-line arguments for trajectory input, ellipsoid choice,
    # kernel-estimation settings, and output destination.
    parser = argparse.ArgumentParser()

    parser.add_argument("--traj_dir", type=str, required=True)
    parser.add_argument("--eccentricities", nargs=3, type=float, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--cap_threshold", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    # Fallback values, used only if the Stage A index file does not contain them.
    parser.add_argument("--N", type=int, default=0)
    parser.add_argument("--Delta", type=float, default=0.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Run Stage B and save results as a compressed NPZ.
    results = run_experiment(args)

    ecc = np.asarray(args.eccentricities, dtype=float)
    tag = f"ecc{ecc_to_tag(ecc)}"
    filename = f"obsEll_exp2_stageB_seed{args.seed}_k{args.num_neighbors}_{tag}.npz"
    path = os.path.join(args.output_dir, filename)

    log_stage("STAGE: saving NPZ")
    np.savez_compressed(path, **results)
    log_stage(f"STAGE: saved {path}")

if __name__ == "__main__":
    main()
