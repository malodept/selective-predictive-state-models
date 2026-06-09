from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, default=None)
    parser.add_argument("--val-output", type=Path, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def infer_num_samples(data: dict[str, np.ndarray]) -> int:
    preferred_keys = [
        "z_current",
        "z_future",
        "action",
        "expected_unreliable",
        "observed_surprise",
    ]

    for key in preferred_keys:
        if key in data and data[key].ndim >= 1:
            return int(data[key].shape[0])

    for arr in data.values():
        if arr.ndim >= 1:
            return int(arr.shape[0])

    raise ValueError("Could not infer number of samples from NPZ file.")


def main() -> None:
    args = parse_args()

    input_path = args.input
    train_output = args.train_output or input_path.with_name(input_path.stem + "_train.npz")
    val_output = args.val_output or input_path.with_name(input_path.stem + "_val.npz")

    raw = np.load(input_path, allow_pickle=True)
    data = {key: raw[key] for key in raw.files}

    n = infer_num_samples(data)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_val = int(round(args.val_fraction * n))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    def write(path: Path, idx: np.ndarray) -> None:
        out = {}

        for key, arr in data.items():
            if arr.ndim >= 1 and arr.shape[0] == n:
                out[key] = arr[idx]
            else:
                out[key] = arr

        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **out)
        print(f"Wrote {len(idx)} samples to {path}")

    write(train_output, train_idx)
    write(val_output, val_idx)


if __name__ == "__main__":
    main()
