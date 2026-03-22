#!/usr/bin/env python3
# observed_KB_experiment_3.py
import os
import argparse
import numpy as np
import time

# Use existing KB codebase
from SDE_sample_KB import drift_on_square  


def wrap_angle(x):
    return np.mod(x, 2.0 * np.pi)


def run_experiment(args):
    np.random.seed(args.seed)

    N_max = int(args.N_max)
    n_u = int(args.n_u)
    n_v = int(args.n_v)
    Delta = float(args.Delta)   

    # initial condition (u,v)
    x = np.array([args.u0, args.v0], dtype=float)

    # Histogram edges
    u_edges = np.linspace(0.0, 2.0 * np.pi, n_u + 1)
    v_edges = np.linspace(0.0, 2.0 * np.pi, n_v + 1)

    du = u_edges[1] - u_edges[0]
    dv = v_edges[1] - v_edges[0]

    # Histogram accumulator
    H = np.zeros((n_u, n_v), dtype=np.float64)

    checkpoints = np.array(args.checkpoints, dtype=int)
    checkpoints = checkpoints[checkpoints <= N_max]
    checkpoints = np.unique(checkpoints)

    densities = np.zeros((len(checkpoints), n_u, n_v), dtype=np.float64)
    checkpoint_index = 0

    start_time = time.time()
    progress_interval = max(1, N_max // 1000)

    for step in range(1, N_max + 1):
        # Euler-Maruyama on (u,v) with unit diffusion 
        dW = np.random.randn(2) / np.sqrt(2.0)
        drift = drift_on_square(x[0], x[1])  
        x = x + np.sqrt(2.0 * Delta) * dW + Delta * drift

        # Wrap to fundamental domain
        u = wrap_angle(x[0])
        v = wrap_angle(x[1])

        # Update histogram bin
        i = int(u / (2.0 * np.pi) * n_u)
        j = int(v / (2.0 * np.pi) * n_v)

        # Edge safety
        if i == n_u:
            i = n_u - 1
        if j == n_v:
            j = n_v - 1

        H[i, j] += 1.0

        # Checkpoint density
        if checkpoint_index < len(checkpoints) and step == checkpoints[checkpoint_index]:
            print(f"Checkpoint at N = {step}", flush=True)
            densities[checkpoint_index] = H / (step * du * dv)
            checkpoint_index += 1

        # Progress logging
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
        "u_edges": u_edges,
        "v_edges": v_edges,
        "checkpoints": checkpoints,
        "n_u": n_u,
        "n_v": n_v,
        "seed": int(args.seed),
        "N_max": N_max,
        "Delta": Delta,
        "u0": float(args.u0),
        "v0": float(args.v0),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--N_max", type=int, required=True)
    parser.add_argument("--Delta", type=float, required=True)  # <-- NEW (required)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)

    parser.add_argument("--n_u", type=int, default=120)
    parser.add_argument("--n_v", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)

    # optional IC overrides
    parser.add_argument("--u0", type=float, default=np.pi)
    parser.add_argument("--v0", type=float, default=0.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    results = run_experiment(args)

    path = os.path.join(args.output_dir, "obsKB_exp3_density.npz")
    np.savez_compressed(path, **results)
    print(f"Saved to {path}", flush=True)


if __name__ == "__main__":
    main()
