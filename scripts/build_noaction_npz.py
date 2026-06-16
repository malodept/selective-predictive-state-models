from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        src = args.input_dir / f"features_{split}.npz"
        dst = args.output_dir / f"features_{split}.npz"

        d = np.load(src, allow_pickle=True)
        out = {k: d[k] for k in d.files}

        out["action_original_7d"] = out["action"].astype(np.float32)
        out["action"] = np.zeros_like(out["action"], dtype=np.float32)
        out["action_type"] = np.asarray("zero_no_action_ablation")

        np.savez_compressed(dst, **out)
        print("wrote", dst, out["z_current"].shape, out["action"].shape)


if __name__ == "__main__":
    main()
