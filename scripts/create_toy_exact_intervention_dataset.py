from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ACTIONS = {
    "stay": (0, 0),
    "right": (1, 0),
    "left": (-1, 0),
    "down": (0, 1),
    "up": (0, -1),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--groups", type=int, default=1000)
    p.add_argument("--image-size", type=int, default=64)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--patch-grid", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def render_state(x: float, y: float, image_size: int, radius: int) -> np.ndarray:
    img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    img[:] = np.array([22, 22, 26], dtype=np.uint8)

    # faint grid, useful to make spatial translation visible
    for u in range(0, image_size, 8):
        img[:, u:u+1] = np.array([35, 35, 42], dtype=np.uint8)
        img[u:u+1, :] = np.array([35, 35, 42], dtype=np.uint8)

    cv2.circle(img, (int(round(x)), int(round(y))), radius, (230, 80, 50), -1)
    cv2.circle(img, (int(round(x)), int(round(y))), radius + 1, (255, 180, 120), 1)
    return img


def image_to_tokens(img: np.ndarray, patch_grid: int) -> np.ndarray:
    """Return a 4x4-like token grid with simple dense image statistics.

    token dim = 8:
      mean RGB, max red, local red centroid x/y, patch center x/y.
    """
    h, w, _ = img.shape
    ph = h // patch_grid
    pw = w // patch_grid

    tokens = []
    for gy in range(patch_grid):
        for gx in range(patch_grid):
            patch = img[gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw].astype(np.float32) / 255.0
            mean_rgb = patch.mean(axis=(0, 1))
            red = patch[:, :, 0]
            max_red = np.array([red.max()], dtype=np.float32)

            yy, xx = np.meshgrid(
                np.linspace(0.0, 1.0, ph, dtype=np.float32),
                np.linspace(0.0, 1.0, pw, dtype=np.float32),
                indexing="ij",
            )
            mass = red.sum() + 1e-6
            cx = np.array([(xx * red).sum() / mass], dtype=np.float32)
            cy = np.array([(yy * red).sum() / mass], dtype=np.float32)

            pcx = np.array([(gx + 0.5) / patch_grid], dtype=np.float32)
            pcy = np.array([(gy + 0.5) / patch_grid], dtype=np.float32)

            tok = np.concatenate([mean_rgb, max_red, cx, cy, pcx, pcy]).astype(np.float32)
            tokens.append(tok)

    return np.stack(tokens).astype(np.float32)


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    action_names = list(ACTIONS.keys())
    k = len(action_names)
    n = args.groups * k

    z_current = []
    z_future = []
    action_vecs = []

    group_id = []
    action_id = []
    state_xy = []
    future_xy = []

    candidate_indices = np.zeros((args.groups, k), dtype=np.int64)

    margin = args.radius + args.step + 3
    low = margin
    high = args.image_size - margin - 1

    for g in range(args.groups):
        x = float(rng.uniform(low, high))
        y = float(rng.uniform(low, high))

        current_img = render_state(x, y, args.image_size, args.radius)
        current_tok = image_to_tokens(current_img, args.patch_grid)

        for j, name in enumerate(action_names):
            dx, dy = ACTIONS[name]
            xf = x + args.step * dx
            yf = y + args.step * dy

            future_img = render_state(xf, yf, args.image_size, args.radius)
            future_tok = image_to_tokens(future_img, args.patch_grid)

            idx = g * k + j
            candidate_indices[g, j] = idx

            z_current.append(current_tok)
            z_future.append(future_tok)
            action_vecs.append([dx, dy])

            group_id.append(g)
            action_id.append(j)
            state_xy.append([x, y])
            future_xy.append([xf, yf])

    z_current = np.stack(z_current).astype(np.float32)
    z_future = np.stack(z_future).astype(np.float32)
    action_vecs = np.asarray(action_vecs, dtype=np.float32)

    np.savez_compressed(
        args.out,
        z_current=z_current,
        z_future=z_future,
        action=action_vecs,
        candidate_indices=candidate_indices,
        correct_candidate=np.arange(k, dtype=np.int64)[None, :].repeat(args.groups, axis=0),
        group_id=np.asarray(group_id, dtype=np.int64),
        action_id=np.asarray(action_id, dtype=np.int64),
        action_names=np.asarray(action_names),
        state_xy=np.asarray(state_xy, dtype=np.float32),
        future_xy=np.asarray(future_xy, dtype=np.float32),
        encoder=np.asarray("toy_exact_intervention_patch_tokens"),
        latent_shape=np.asarray(z_current.shape[1:]),
        latent_tokens=np.asarray(z_current.shape[1]),
        latent_dim=np.asarray(z_current.shape[2]),
        action_dim=np.asarray(action_vecs.shape[1]),
        image_size=np.asarray(args.image_size),
        step=np.asarray(args.step),
        radius=np.asarray(args.radius),
        patch_grid=np.asarray(args.patch_grid),
        dataset_version=np.asarray("toy_exact_intervention_v0"),
    )

    report = {
        "out": str(args.out),
        "groups": args.groups,
        "candidates": k,
        "samples": int(n),
        "chance_top1": 1.0 / k,
        "z_current_shape": list(z_current.shape),
        "z_future_shape": list(z_future.shape),
        "action_shape": list(action_vecs.shape),
        "image_size": args.image_size,
        "step": args.step,
        "radius": args.radius,
        "patch_grid": args.patch_grid,
        "action_names": action_names,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# Toy exact-intervention dataset audit",
        "",
        f"- output: `{args.out}`",
        f"- groups: `{args.groups}`",
        f"- candidates per group: `{k}`",
        f"- samples: `{n}`",
        f"- chance top-1: `{1.0 / k:.6f}`",
        f"- latent shape: `{tuple(z_current.shape[1:])}`",
        f"- action dimension: `{action_vecs.shape[1]}`",
        f"- image size: `{args.image_size}`",
        f"- step: `{args.step}`",
        "",
        "## Interpretation",
        "",
        "Each group is an exact-intervention branch: the same synthetic state is reset and branched under five different actions.",
        "This is not meant to be a realistic benchmark. It is a protocol sanity check: the oracle should be far above chance, ideally exactly 1.0 at alpha=1.",
    ]
    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
