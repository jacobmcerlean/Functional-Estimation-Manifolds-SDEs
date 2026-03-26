#!/usr/bin/env python3
from kernel_estimators import euclidean_kernel_estimate_vec
from SDE_sample_Sphere import pi_ret_SDE_sampling
import numpy as np
import argparse
import os
import time
from multiprocessing import Pool

from observed_ellipsoid import (
    sphere_to_ellipsoid,
    true_tangent_project_ellipsoid
)

# ------------------------------------------------------------
# Single simulation (worker)
# ------------------------------------------------------------

def run_single_sim(args_tuple):
    # Each worker handles one independent simulation / estimation replicate.
    # The tuple carries the simulation index together with the shared CLI args.
    sim_idx, args = args_tuple

    # Use a simulation-specific seed so parallel workers remain reproducible
    # while still producing independent trajectories.
    np.random.seed(args.seed + sim_idx)

    N = args.N
    Delta = float(args.Delta)  # <-- UPDATED: user-supplied
    num_neighbors = args.num_neighbors
    ecc = np.array(args.eccentricities, dtype=float)

    # The estimator is evaluated at a single fixed base point:
    # the north pole on the sphere, pushed forward to the ellipsoid.
    base_sphere = np.array([0.0, 0.0, 1.0])
    base_point = sphere_to_ellipsoid(base_sphere, ecc)

    # True tangent projector at the base point, used to benchmark
    # the estimated tangent projection.
    P_true = true_tangent_project_ellipsoid(base_point, ecc)

    # --- Sample initial condition ---
    # Start each trajectory from an independent random point on the sphere.
    y0 = np.random.normal(size=3)
    y0 /= np.linalg.norm(y0)

    # --- Simulate on sphere ---
    # Simulate the latent sphere-valued diffusion.
    Y = pi_ret_SDE_sampling(N, Delta, y0)

    # --- Pushforward trajectory ---
    # Map the sphere trajectory into the observed ellipsoid coordinates.
    X = np.array([sphere_to_ellipsoid(y, ecc) for y in Y])

    # --- Kernel estimate ---
    # Estimate ambient drift and diffusion at the fixed ellipsoid base point
    # using the observed ambient trajectory.
    mu_hat, sigma_hat, den, h = euclidean_kernel_estimate_vec(
        base_point,
        X,
	Delta,
	num_neighbors=num_neighbors
    )

    # Safety guard
    # If the local estimator degenerates (for example zero kernel mass),
    # return NaNs so the caller can keep array shapes consistent.
    if mu_hat is None:
        return (
            sim_idx,
            np.full(3, np.nan),
            np.full((3, 3), np.nan),
            np.full(3, np.nan),
            np.full(3, np.nan),
            0.0,
            0.0
        )

    # --- Estimated tangent projector ---
    # Symmetrize the estimated diffusion and add a tiny ridge so SVD is stable.
    sigma_hat = 0.5 * (sigma_hat + sigma_hat.T)
    sigma_hat += 1e-10 * np.eye(3)

    # The top two singular directions give the estimated tangent space
    # on the 2-dimensional ellipsoid.
    U, S, _ = np.linalg.svd(sigma_hat)
    idx = np.argsort(S)[::-1]
    U2 = U[:, idx[:2]]
    P_est = U2 @ U2.T

    # Project the estimated ambient drift using:
    # (i) the estimated tangent projector, and
    # (ii) the true tangent projector.
    # This lets the experiment separate tangent-space estimation error
    # from ambient drift-estimation error.
    mu_proj_est = P_est @ mu_hat
    mu_proj_true = P_true @ mu_hat

    return (
	sim_idx,
        mu_hat,
        sigma_hat,
        mu_proj_est,
        mu_proj_true,
        den,
	h
    )


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------

def run_experiment(args):

    # Global experiment settings shared across all Monte Carlo replicates.
    N = args.N
    Delta = float(args.Delta)  # <-- UPDATED: used in logging + saved
    num_sims = args.num_sims
    ecc = np.array(args.eccentricities, dtype=float)

    # Preallocate storage for all simulation outputs.
    # Each row corresponds to one independent Monte Carlo replicate.
    results = {
        "mu_euclidean": np.zeros((num_sims, 3)),
        "sigma_hat": np.zeros((num_sims, 3, 3)),
        "mu_proj_estimated": np.zeros((num_sims, 3)),
        "mu_proj_true": np.zeros((num_sims, 3)),
        "occupation_density": np.zeros(num_sims),
        "h_used": np.zeros(num_sims),
        "eccentricities": ecc,
        "N": N,
        "Delta": Delta,                 # <-- UPDATED: save Delta
        "num_neighbors": args.num_neighbors,
        "seed": args.seed,
    }

    print(
	f"Starting experiment | N={N} | Delta={Delta:g} | sims={num_sims} | "
        f"k={args.num_neighbors} | ecc={ecc}",
        flush=True
    )

    t0 = time.time()

    # Use the cluster-provided NSLOTS environment variable when available;
    # otherwise default to a single worker.
    n_cores = int(os.environ.get("NSLOTS", 1))
    print(f"Using {n_cores} worker processes", flush=True)

    with Pool(processes=n_cores) as pool:

        # Distribute simulations across workers and collect results as they finish.
        # imap_unordered improves throughput because slow simulations do not block
        # later completed tasks from being returned.
        for count, output in enumerate(
            pool.imap_unordered(
                run_single_sim,
                [(i, args) for i in range(num_sims)]
            ),
            start=1
        ):

            sim_idx, mu_hat, sigma_hat, mu_proj_est, mu_proj_true, den, h = output

            # Write each worker's outputs into the correct slot in the results arrays.
            results["mu_euclidean"][sim_idx] = mu_hat
            results["sigma_hat"][sim_idx] = sigma_hat
            results["mu_proj_estimated"][sim_idx] = mu_proj_est
            results["mu_proj_true"][sim_idx] = mu_proj_true
            results["occupation_density"][sim_idx] = den
            results["h_used"][sim_idx] = h

            # Simple progress / ETA reporting.
            elapsed = time.time() - t0
            avg = elapsed / count
            eta = avg * (num_sims - count)

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"Completed {count}/{num_sims} | "
                f"elapsed={elapsed/60:.2f} min | "
                f"ETA={eta/60:.2f} min",
                flush=True
            )

    total_time = (time.time() - t0) / 60.0
    print(f"Finished all sims in {total_time:.2f} minutes", flush=True)

    return results


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def main():

    # Command-line interface for the Monte Carlo normality / estimator experiment.
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--Delta", type=float, required=True)  # <-- UPDATED: user-supplied Delta
    parser.add_argument("--num_sims", type=int, default=500)
    parser.add_argument("--eccentricities", nargs=3, type=float, required=True)
    parser.add_argument("--num_neighbors", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    # Basic validation of the time step.
    if not (np.isfinite(args.Delta) and args.Delta > 0):
        raise ValueError(f"--Delta must be positive and finite; got {args.Delta}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run the experiment and save all Monte Carlo outputs to one compressed NPZ.
    results = run_experiment(args)

    filename = f"obsEll_exp1_seed{args.seed}_k{args.num_neighbors}_N{args.N}_Delta{args.Delta:g}.npz"
    path = os.path.join(args.output_dir, filename)

    np.savez_compressed(path, **results)

    print(f"Saved to {path}", flush=True)


if __name__ == "__main__":
    main()
