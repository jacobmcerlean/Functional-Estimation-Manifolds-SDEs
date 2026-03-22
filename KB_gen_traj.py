#!/usr/bin/env python3
import os
import argparse
import numpy as np
import time

from SDE_sample_KB import (
    SDE_on_square,
    embed_klein_bottle,
)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--Delta", type=float, required=True)
    p.add_argument("--num_sims", type=int, default=200)
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--r", type=float, default=1.0)
    p.add_argument("--u_min", type=float, default=0.0)
    p.add_argument("--u_max", type=float, default=2*np.pi)
    p.add_argument("--v_min", type=float, default=0.0)
    p.add_argument("--v_max", type=float, default=2*np.pi)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--subsample", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    trajdir = os.path.join(args.output_dir, "traj")
    os.makedirs(trajdir, exist_ok=True)

    N_full = int(args.N)
    Delta_full = float(args.Delta)
    T_full = N_full * Delta_full

    s = int(args.subsample)
    if s < 1:
        raise ValueError("--subsample must be >= 1")

    dtype = np.float32 if args.dtype == "float32" else np.float64

    np.random.seed(args.seed)

    traj_files = []
    square_files = []
    x0_list = np.zeros((args.num_sims, 2), dtype=np.float64)

    start = time.time()
    print(f"Stage A | N={N_full} Delta={Delta_full:g} T={T_full:g} sims={args.num_sims} subsample={s} a={args.a:g} r={args.r:g}", flush=True)

    for sim in range(args.num_sims):
        x0 = np.array([
            np.random.uniform(args.u_min, args.u_max),
            np.random.uniform(args.v_min, args.v_max),
        ], dtype=float)
        x0_list[sim] = x0

        t0 = time.time()
        X_square_full = SDE_on_square(N_full, Delta_full, x0)
        X_kb_full = embed_klein_bottle(X_square_full, args.a, args.r)

        X_square = X_square_full[::s].astype(dtype, copy=False)
        X_kb = X_kb_full[::s].astype(dtype, copy=False)

        tag = f"sim{sim:04d}_sub{s}"
        sq_path = os.path.join(trajdir, f"kb_square_{tag}.npy")
        kb_path = os.path.join(trajdir, f"kb_traj_{tag}.npy")

        np.save(sq_path, X_square)
        np.save(kb_path, X_kb)

        square_files.append(os.path.basename(sq_path))
        traj_files.append(os.path.basename(kb_path))

        if (sim + 1) % max(1, args.num_sims // 10) == 0 or (sim + 1) == args.num_sims:
            elapsed = (time.time() - start) / 60
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Completed {sim+1}/{args.num_sims} | elapsed={elapsed:.2f} min", flush=True)

    N_eff = int(np.load(os.path.join(trajdir, traj_files[0]), mmap_mode="r").shape[0] - 1) if traj_files else 0
    Delta_eff = Delta_full * s
    T_eff = Delta_eff * N_eff

    idx_path = os.path.join(trajdir, f"index_sub{s}.npz")
    np.savez_compressed(
        idx_path,
        N=np.int64(N_full),
        Delta=np.float64(Delta_full),
        T=np.float64(T_full),
        seed=np.int64(args.seed),
        num_sims=np.int64(args.num_sims),
        a=np.float64(args.a),
        r=np.float64(args.r),
        u_min=np.float64(args.u_min),
        u_max=np.float64(args.u_max),
        v_min=np.float64(args.v_min),
        v_max=np.float64(args.v_max),
        subsample=np.int64(s),
        N_eff=np.int64(N_eff),
        Delta_eff=np.float64(Delta_eff),
        T_eff=np.float64(T_eff),
        x0_list=x0_list,
        traj_files=np.asarray(traj_files, dtype=object),
        square_files=np.asarray(square_files, dtype=object),
        dtype=np.asarray([args.dtype], dtype=object),
    )
    print(f"Saved {idx_path}", flush=True)

if __name__ == "__main__":
    main()
