from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-count", type=int, default=3)
    p.add_argument("--calib-count", type=int, default=1)
    p.add_argument("--val-count", type=int, default=1)
    return p.parse_args()


def infer_num_samples(data: dict[str, np.ndarray]) -> int:
    for key in ["z_current", "z_future", "action", "trajectory_id"]:
        if key in data and data[key].ndim >= 1:
            return int(data[key].shape[0])
    raise ValueError("Could not infer number of samples.")


def write_split(
    path: Path,
    data: dict[str, np.ndarray],
    n: int,
    idx: np.ndarray,
    split_name: str,
    split_trajectory_ids: np.ndarray,
    all_trajectory_ids: np.ndarray,
) -> None:
    out = {}

    for key, arr in data.items():
        if arr.ndim >= 1 and arr.shape[0] == n:
            out[key] = arr[idx]
        else:
            out[key] = arr

    out["split_name"] = np.asarray(split_name)
    out["split_trajectory_ids"] = split_trajectory_ids.astype("U256")
    out["all_trajectory_ids"] = all_trajectory_ids.astype("U256")

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)

    print(f"{split_name:5s}: {len(idx):6d} samples -> {path}")
    for tid in split_trajectory_ids:
        print(f"       - {tid}")


def main() -> None:
    args = parse_args()

    raw = np.load(args.input, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}

    if "trajectory_id" not in data:
        raise KeyError("Input NPZ must contain trajectory_id. Regenerate features with metadata first.")

    n = infer_num_samples(data)
    traj = data["trajectory_id"].astype(str)

    all_ids = np.asarray(sorted(set(traj)), dtype="U256")
    rng = np.random.default_rng(args.seed)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    needed = args.train_count + args.calib_count + args.val_count
    if needed >= len(shuffled):
        raise ValueError(
            f"Need at least one held-out test trajectory. "
            f"Got {len(shuffled)} trajectories but requested train+calib+val={needed}."
        )

    train_ids = shuffled[: args.train_count]
    calib_ids = shuffled[args.train_count : args.train_count + args.calib_count]
    val_ids = shuffled[
        args.train_count + args.calib_count :
        args.train_count + args.calib_count + args.val_count
    ]
    test_ids = shuffled[args.train_count + args.calib_count + args.val_count :]

    splits = {
        "train": train_ids,
        "calib": calib_ids,
        "val": val_ids,
        "test": test_ids,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input: {args.input}")
    print(f"Samples: {n}")
    print(f"Trajectories: {len(all_ids)}")
    print("Split assignment:")

    for split_name, ids in splits.items():
        mask = np.isin(traj, ids)
        idx = np.nonzero(mask)[0]
        write_split(
            path=args.output_dir / f"features_{split_name}.npz",
            data=data,
            n=n,
            idx=idx,
            split_name=split_name,
            split_trajectory_ids=ids,
            all_trajectory_ids=all_ids,
        )

    # Hard safety check: no trajectory overlap between splits.
    for a_name, a_ids in splits.items():
        for b_name, b_ids in splits.items():
            if a_name >= b_name:
                continue
            overlap = set(a_ids.astype(str)) & set(b_ids.astype(str))
            if overlap:
                raise RuntimeError(f"Trajectory leakage between {a_name} and {b_name}: {overlap}")

    print("OK: no trajectory overlap across splits.")


if __name__ == "__main__":
    main()
