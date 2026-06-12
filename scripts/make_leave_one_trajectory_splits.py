from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    raw = np.load(args.input, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    if "trajectory_id" not in data:
        raise KeyError("trajectory_id missing")

    traj = data["trajectory_id"].astype(str)
    ids = sorted(set(traj))
    n = len(traj)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("trajectories:")
    for tid in ids:
        print(" -", tid)

    for heldout in ids:
        short = heldout.split("/")[-1]
        train_idx = np.nonzero(traj != heldout)[0]
        val_idx = np.nonzero(traj == heldout)[0]

        for split_name, idx in [("train", train_idx), ("val", val_idx)]:
            out = {}
            for k, arr in data.items():
                if arr.ndim >= 1 and arr.shape[0] == n:
                    out[k] = arr[idx]
                else:
                    out[k] = arr

            out["split_name"] = np.asarray(split_name)
            out["heldout_trajectory_id"] = np.asarray(heldout)
            out["split_trajectory_ids"] = np.asarray(sorted(set(traj[idx])), dtype="U256")

            path = args.output_dir / f"loo_{short}_{split_name}.npz"
            np.savez_compressed(path, **out)
            print(f"{heldout} {split_name}: {len(idx)} -> {path}")


if __name__ == "__main__":
    main()
