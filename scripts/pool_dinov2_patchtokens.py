from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--source-grid", type=int, default=16)
    p.add_argument("--target-grid", type=int, default=4)
    return p.parse_args()


def pool_tokens(z: np.ndarray, source_grid: int, target_grid: int) -> np.ndarray:
    # z: [N, source_grid*source_grid, D]
    n, t, d = z.shape
    assert t == source_grid * source_grid, (t, source_grid)

    factor = source_grid // target_grid
    assert source_grid % target_grid == 0

    z = z.reshape(n, source_grid, source_grid, d)
    z = z.reshape(n, target_grid, factor, target_grid, factor, d)
    z = z.mean(axis=(2, 4))
    z = z.reshape(n, target_grid * target_grid, d)
    return z.astype(np.float16)


def maybe_copy(src, key, out):
    if key in src.files:
        out[key] = src[key]


def main():
    args = parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input, allow_pickle=True)

    z_current = pool_tokens(d["z_current"], args.source_grid, args.target_grid)
    z_future = pool_tokens(d["z_future"], args.source_grid, args.target_grid)

    out = {
        "z_current": z_current,
        "z_future": z_future,
        "action": d["action"].astype(np.float32),
        "candidate_indices": d["candidate_indices"].astype(np.int64),
        "correct_candidate": d["correct_candidate"].astype(np.int64),
        "encoder": np.asarray(f"{str(d['encoder'])}_pooled_{args.target_grid}x{args.target_grid}"),
        "latent_shape": np.asarray(z_current.shape[1:]),
        "latent_tokens": np.asarray(z_current.shape[1]),
        "latent_dim": np.asarray(z_current.shape[2]),
        "action_dim": np.asarray(d["action"].shape[1]),
        "source_npz": np.asarray(str(args.input)),
        "source_grid": np.asarray(args.source_grid),
        "target_grid": np.asarray(args.target_grid),
    }

    for key in [
        "cls_current",
        "cls_future",
        "group_id",
        "action_id",
        "action_names",
        "state_xy",
        "future_xy",
        "blocked_mask",
        "patch_grid",
        "horizon",
        "velocity",
        "num_blocked_directions",
        "dataset_version",
        "image_size",
    ]:
        maybe_copy(d, key, out)

    np.savez_compressed(args.output, **out)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "source_shape": list(d["z_current"].shape),
        "pooled_shape": list(z_current.shape),
        "source_grid": args.source_grid,
        "target_grid": args.target_grid,
        "output_size_gb": args.output.stat().st_size / 1e9,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# DINOv2 patch-token pooling",
        "",
        f"- input: `{args.input}`",
        f"- output: `{args.output}`",
        f"- source shape: `{tuple(d['z_current'].shape)}`",
        f"- pooled shape: `{tuple(z_current.shape)}`",
        f"- source grid: `{args.source_grid}x{args.source_grid}`",
        f"- target grid: `{args.target_grid}x{args.target_grid}`",
        f"- output size GB: `{report['output_size_gb']:.4f}`",
        "",
        "## Interpretation",
        "",
        "This keeps frozen DINOv2 dense visual features but reduces the patch-token grid from 16x16 to 4x4 for stable exact-intervention training.",
    ]
    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
