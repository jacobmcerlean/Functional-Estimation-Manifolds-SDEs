#!/usr/bin/env python3
import os
import argparse
import numpy as np
import time

from SDE_sample_Sphere import sphere_step


def run_experiment(args):
    np.random.seed(args.seed)

    checkpoints = np.array(sorted(args.checkpoints), dtype=int)
    N_max = int(checkpoints[-1])

    n_theta = int(args.n_theta)
    n_phi = int(args.n_phi)

    Delta = float(args.Delta)

    theta_edges = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    phi_edges = np.linspace(0.0, np.pi, n_phi + 1)

    dtheta = 2.0 * np.pi / n_theta
    dphi = np.pi / n_phi

    # area weights for (theta,phi) bins on S^2: sin(phi) dtheta dphi
    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    sin_phi = np.sin(phi_centers)
    area = np.outer(np.ones(n_theta), sin_phi) * dtheta * dphi  # (n_theta, n_phi)

    H = np.zeros((n_theta, n_phi), dtype=np.float64)
    densities = np.zeros((len(checkpoints), n_theta, n_phi), dtype=np.float64)

    y = np.array([0.0, 0.0, 1.0], dtype=float)

    checkpoint_index = 0
    start_time = time.time()
    progress_interval = max(1, N_max // 1000)

    for step in range(1, N_max + 1):

        # Use the same integrator
        y = sphere_step(y, Delta)

        # spherical coordinates
        theta = np.arctan2(y[1], y[0])
        if theta < 0:
            theta += 2.0 * np.pi

        # clamp for numerical safety: arccos input in [-1,1]
        z = float(np.clip(y[2], -1.0, 1.0))
        phi = np.arccos(z)

        i = int(theta / (2.0 * np.pi) * n_theta)
        j = int(phi / np.pi * n_phi)

        if i == n_theta:
            i = n_theta - 1
        if j == n_phi:
            j = n_phi - 1

        H[i, j] += 1.0

        if checkpoint_index < len(checkpoints) and step == checkpoints[checkpoint_index]:
            print(f"Checkpoint at N = {step}", flush=True)
            densities[checkpoint_index] = H / (step * area)
            checkpoint_index += 1

        if step % progress_interval == 0 or step == N_max:
            elapsed = time.time() - start_time
            rate = step / elapsed if elapsed > 0 else 0.0
            remaining = (N_max - step) / rate if rate > 0 else 0.0

            print(
                f"[{step:,}/{N_max:,}] "
                f"{100*step/N_max:5.2f}% "
                f"Elapsed: {elapsed/60:6.1f} min "
                f"ETA: {remaining/60:6.1f} min",
                flush=True
            )

    return {
        "densities": densities,
        "theta_edges": theta_edges,
        "phi_edges": phi_edges,
        "checkpoints": checkpoints,
        "n_theta": n_theta,
        "n_phi": n_phi,
        "seed": args.seed,
        "N_max": N_max,
        "Delta": Delta,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoints", nargs="+", type=int, required=True)

    parser.add_argument("--n_theta", type=int, default=120)
    parser.add_argument("--n_phi", type=int, default=60)

    # --- UPDATED: user-supplied Delta ---
    parser.add_argument("--Delta", type=float, required=True)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = run_experiment(args)

    path = os.path.join(args.output_dir, "obsSphere_exp3_density.npz")
    np.savez_compressed(path, **results)
    print(f"Saved to {path}", flush=True)


if __name__ == "__main__":
    main()
