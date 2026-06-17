from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def subset_npz(d, idx):
    n = d["action"].shape[0]
    out = {}

    for k in d.files:
        v = d[k]
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v

    return out


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input, allow_pickle=True)

    tids = np.asarray(d["trajectory_id"]).astype(str)
    unique_tids = np.asarray(sorted(set(tids.tolist())))

    rng = np.random.default_rng(args.seed)
    shuffled = unique_tids.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(args.val_frac * len(shuffled))))
    val_tids = set(shuffled[:n_val].tolist())
    train_tids = set(shuffled[n_val:].tolist())

    train_idx = np.asarray([i for i, t in enumerate(tids) if t in train_tids], dtype=np.int64)
    val_idx = np.asarray([i for i, t in enumerate(tids) if t in val_tids], dtype=np.int64)

    np.savez_compressed(args.out_dir / "features_train_seen.npz", **subset_npz(d, train_idx))
    np.savez_compressed(args.out_dir / "features_val_seen.npz", **subset_npz(d, val_idx))

    print("Input:", args.input)
    print("Total trajectories:", len(unique_tids))
    print("Seen-train trajectories:", len(train_tids), sorted(train_tids)[:8])
    print("Seen-val trajectories:", len(val_tids), sorted(val_tids)[:8])
    print("Seen-train samples:", len(train_idx))
    print("Seen-val samples:", len(val_idx))
    print("Overlap:", sorted(train_tids & val_tids))


if __name__ == "__main__":
    main()
