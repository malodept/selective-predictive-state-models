from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-envs", nargs="+", required=True)
    p.add_argument("--val-envs", nargs="+", required=True)
    p.add_argument("--test-envs", nargs="+", required=True)
    return p.parse_args()


def get_environments(d):
    if "environment" in d.files:
        return np.asarray(d["environment"]).astype(str)

    if "env" in d.files:
        return np.asarray(d["env"]).astype(str)

    if "trajectory_id" in d.files:
        tids = np.asarray(d["trajectory_id"]).astype(str)
        return np.asarray([x.split("/")[0] for x in tids]).astype(str)

    raise KeyError("Cannot infer environment: no environment/env/trajectory_id field found.")


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


def save_split(d, envs, split_name, split_envs, out_dir):
    mask = np.isin(envs, np.asarray(split_envs).astype(str))
    idx = np.where(mask)[0]

    out_path = out_dir / f"features_{split_name}.npz"
    out = subset_npz(d, idx)

    np.savez_compressed(out_path, **out)

    print(f"{split_name}: {len(idx)} samples -> {out_path}")
    print("  envs:", sorted(set(envs[idx].tolist())))

    if "trajectory_id" in d.files:
        tids = np.asarray(d["trajectory_id"]).astype(str)[idx]
        print("  trajectories:", len(set(tids.tolist())))

    return idx


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input, allow_pickle=True)
    envs = get_environments(d)

    print("Input:", args.input)
    print("Samples:", d["action"].shape[0])
    print("Available envs:", sorted(set(envs.tolist())))

    train_idx = save_split(d, envs, "train", args.train_envs, args.output_dir)
    val_idx = save_split(d, envs, "val", args.val_envs, args.output_dir)
    test_idx = save_split(d, envs, "test", args.test_envs, args.output_dir)

    if "trajectory_id" in d.files:
        tids = np.asarray(d["trajectory_id"]).astype(str)

        sets = {
            "features_train": set(tids[train_idx].tolist()),
            "features_val": set(tids[val_idx].tolist()),
            "features_test": set(tids[test_idx].tolist()),
        }

        print("\nLeakage check:")
        ok = True
        keys = list(sets.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                inter = sets[keys[i]] & sets[keys[j]]
                print(f"{keys[i]} vs {keys[j]}: traj_overlap={sorted(inter)[:5]}")
                ok = ok and len(inter) == 0

        print("OK:", ok)


if __name__ == "__main__":
    main()
