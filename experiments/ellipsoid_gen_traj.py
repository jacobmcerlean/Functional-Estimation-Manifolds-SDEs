#!/usr/bin/env python3
#!/usr/bin/env python3
from SDE_sample_Sphere import pi_ret_SDE_sampling
from observed_ellipsoid import sphere_to_ellipsoid
import numpy as np
import os, argparse, time

def parse_ecc_list(vals):
    # Parse a flat list of eccentricity values into an array of shape (m, 3),
    # where each row gives one ellipsoid scaling triple.
    vals = list(map(float, vals))
    if len(vals) % 3 != 0:
        raise ValueError("ecc_list must be a multiple of 3 floats")
    return np.asarray(vals, dtype=float).reshape(-1, 3)

def ecc_tag(ecc):
    # Turn one eccentricity triple into a compact filename tag.
    return f"{ecc[0]:g}-{ecc[1]:g}-{ecc[2]:g}"

def main():
    # Parse command-line arguments controlling the sphere simulation,
    # ellipsoid scalings, output location, storage dtype, and subsampling.
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, required=True)
    p.add_argument("--Delta", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ecc_list", nargs="+", required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32","float64"])
    p.add_argument("--subsample", type=int, default=1)
    args = p.parse_args()

    # Create output directories.
    os.makedirs(args.output_dir, exist_ok=True)
    trajdir = os.path.join(args.output_dir, "traj")
    os.makedirs(trajdir, exist_ok=True)

    # Read the list of ellipsoid eccentricity triples.
    ecc_list = parse_ecc_list(args.ecc_list)

    # Full simulation parameters before subsampling.
    N = args.N
    Delta = float(args.Delta)
    T = N * Delta

    # Storage dtype for trajectories written to disk.
    dtype = np.float32 if args.dtype == "float32" else np.float64

    # Keep every s-th sample from the full sphere trajectory.
    s = int(args.subsample)
    if s < 1:
	raise ValueError("--subsample must be >= 1")

    # Seed NumPy RNG and sample a random initial point on the sphere.
    np.random.seed(args.seed)
    y0 = np.random.normal(size=3)
    y0 /= np.linalg.norm(y0)

    # Simulate one sphere trajectory, then reuse it for all ellipsoid scalings.
    t0 = time.time()
    print(f"Simulating sphere: N={N} Delta={Delta:g} T={T:g}", flush=True)
    X_full = pi_ret_SDE_sampling(N, Delta, y0).astype(dtype, copy=False)
    print(f"Sphere done in {(time.time()-t0)/60:.2f} min", flush=True)

    # Down-sample the sphere trajectory once, before applying any ellipsoid map.
    X = X_full[::s].astype(dtype, copy=False)
    N_eff = X.shape[0] - 1

    # For each eccentricity triple, scale the sphere coordinates componentwise
    # to obtain the corresponding ellipsoid trajectory.
    traj_files = []
    for ecc in ecc_list:
        tag = ecc_tag(ecc)
        out = os.path.join(trajdir, f"traj_ecc{tag}_sub{s}.npy")
        Y = (X * ecc[None, :]).astype(dtype, copy=False)
        np.save(out, Y)
        traj_files.append(os.path.basename(out))
        print(f"Saved {out}", flush=True)

    # Also save the down-sampled sphere trajectory itself.
    sphere_path = os.path.join(trajdir, f"sphere_traj_sub{s}.npy")
    np.save(sphere_path, X)
    print(f"Saved {sphere_path}", flush=True)

    # Save an index file containing metadata and filenames needed by later stages.
    index_path = os.path.join(trajdir, f"index_sub{s}.npz")
    np.savez_compressed(
        index_path,
        N=np.int64(N),
        Delta=np.float64(Delta),
        T=np.float64(T),
        seed=np.int64(args.seed),
        y0=y0.astype(np.float64),
        ecc_list=ecc_list.astype(np.float64),
        subsample=np.int64(s),
        N_eff=np.int64(N_eff),
        traj_files=np.asarray(traj_files, dtype=object),
        sphere_file=os.path.basename(sphere_path),
        dtype=np.asarray([args.dtype], dtype=object),
    )
    print(f"Saved {index_path}", flush=True)

if __name__ == "__main__":
    main()
