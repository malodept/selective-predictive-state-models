from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-envs", nargs="+", required=True)
    p.add_argument("--val-envs", nargs="+", required=True)
    p.add_argument("--test-envs", nargs="+", required=True)
    return p.parse_args()


def write_split(
    name: str,
    envs: list[str],
    data: dict[str, np.ndarray],
    env_arr: np.ndarray,
    n: int,
    output_dir: Path,
) -> Path:
    mask = np.isin(env_arr, np.asarray(envs))
    idx = np.nonzero(mask)[0]

    out = {}
    for k, arr in data.items():
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[k] = arr[idx]
        else:
            out[k] = arr

    out["split_name"] = np.asarray(name)
    out["split_environments"] = np.asarray(envs, dtype="U128")

    path = output_dir / f"features_{name}.npz"
    np.savez_compressed(path, **out)

    split_traj = sorted(set(out["trajectory_id"].astype(str))) if "trajectory_id" in out else []
    split_env = sorted(set(out["environment"].astype(str))) if "environment" in out else []

    print(f"{name}: {len(idx)} samples -> {path}")
    print("  envs:", split_env)
    print("  trajectories:", len(split_traj))

    return path


def main() -> None:
    args = parse_args()

    raw = np.load(args.input, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}

    if "environment" not in data:
        raise KeyError("environment key missing")
    if "trajectory_id" not in data:
        raise KeyError("trajectory_id key missing")

    env_arr = data["environment"].astype(str)
    traj_arr = data["trajectory_id"].astype(str)
    n = len(env_arr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_requested = set(args.train_envs + args.val_envs + args.test_envs)
    available = set(env_arr)

    missing = sorted(all_requested - available)
    if missing:
        raise ValueError(f"Requested envs missing from dataset: {missing}")

    overlap_train_val = set(args.train_envs) & set(args.val_envs)
    overlap_train_test = set(args.train_envs) & set(args.test_envs)
    overlap_val_test = set(args.val_envs) & set(args.test_envs)
    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            "Environment overlap detected: "
            f"train/val={overlap_train_val}, train/test={overlap_train_test}, val/test={overlap_val_test}"
        )

    print("Input:", args.input)
    print("Samples:", n)
    print("Available envs:", sorted(available))
    print("Available trajectories:", len(set(traj_arr)))
    print()

    paths = []
    paths.append(write_split("train", args.train_envs, data, env_arr, n, args.output_dir))
    paths.append(write_split("val", args.val_envs, data, env_arr, n, args.output_dir))
    paths.append(write_split("test", args.test_envs, data, env_arr, n, args.output_dir))

    print("\nLeakage check:")
    splits = {}
    for path in paths:
        d = np.load(path, allow_pickle=True)
        splits[path.stem] = {
            "env": set(d["environment"].astype(str)),
            "traj": set(d["trajectory_id"].astype(str)),
        }

    names = list(splits)
    ok = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            env_overlap = splits[a]["env"] & splits[b]["env"]
            traj_overlap = splits[a]["traj"] & splits[b]["traj"]
            print(f"{a} vs {b}: env_overlap={sorted(env_overlap)} traj_overlap={sorted(traj_overlap)}")
            ok = ok and not env_overlap and not traj_overlap

    print("OK:", ok)


if __name__ == "__main__":
    main()
