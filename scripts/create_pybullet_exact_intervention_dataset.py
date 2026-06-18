from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pybullet as p


ACTIONS = {
    "stay": (0.0, 0.0),
    "right": (1.0, 0.0),
    "left": (-1.0, 0.0),
    "forward": (0.0, 1.0),
    "backward": (0.0, -1.0),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--velocity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def image_to_tokens(img: np.ndarray, patch_grid: int) -> np.ndarray:
    h, w, _ = img.shape
    ph = h // patch_grid
    pw = w // patch_grid

    img_f = img.astype(np.float32) / 255.0
    tokens = []

    for gy in range(patch_grid):
        for gx in range(patch_grid):
            patch = img_f[gy * ph:(gy + 1) * ph, gx * pw:(gx + 1) * pw]

            mean_rgb = patch.mean(axis=(0, 1))
            max_rgb = patch.max(axis=(0, 1))

            r = patch[:, :, 0]
            g = patch[:, :, 1]
            b = patch[:, :, 2]

            # Red-objectness map: suppress grey background.
            obj = np.maximum(r - 0.5 * (g + b), 0.0)

            yy, xx = np.meshgrid(
                np.linspace(0.0, 1.0, ph, dtype=np.float32),
                np.linspace(0.0, 1.0, pw, dtype=np.float32),
                indexing="ij",
            )

            mass = obj.sum() + 1e-6
            cx = np.array([(xx * obj).sum() / mass], dtype=np.float32)
            cy = np.array([(yy * obj).sum() / mass], dtype=np.float32)
            obj_mass = np.array([obj.mean()], dtype=np.float32)
            obj_max = np.array([obj.max()], dtype=np.float32)

            pcx = np.array([(gx + 0.5) / patch_grid], dtype=np.float32)
            pcy = np.array([(gy + 0.5) / patch_grid], dtype=np.float32)

            tok = np.concatenate([
                mean_rgb,
                max_rgb,
                obj_mass,
                obj_max,
                cx,
                cy,
                pcx,
                pcy,
            ]).astype(np.float32)

            tokens.append(tok)

    return np.stack(tokens).astype(np.float32)


def make_world():
    cid = p.connect(p.DIRECT)
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 60.0)

    plane_col = p.createCollisionShape(p.GEOM_PLANE)
    plane_vis = p.createVisualShape(
        p.GEOM_PLANE,
        rgbaColor=[0.70, 0.70, 0.70, 1.0],
    )
    p.createMultiBody(0.0, plane_col, plane_vis)

    radius = 0.08
    sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphere_vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[0.95, 0.05, 0.02, 1.0],
    )
    sphere_id = p.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=sphere_col,
        baseVisualShapeIndex=sphere_vis,
        basePosition=[0.0, 0.0, radius],
    )

    p.changeDynamics(
        sphere_id,
        -1,
        lateralFriction=0.9,
        linearDamping=0.05,
        angularDamping=0.05,
    )

    return cid, sphere_id, radius


def render(image_size: int) -> np.ndarray:
    # Mostly top-down camera: makes action branches visually identifiable.
    view = p.computeViewMatrix(
        cameraEyePosition=[0.0, -0.20, 2.25],
        cameraTargetPosition=[0.0, 0.0, 0.0],
        cameraUpVector=[0.0, 1.0, 0.0],
    )
    proj = p.computeProjectionMatrixFOV(
        fov=55.0,
        aspect=1.0,
        nearVal=0.01,
        farVal=10.0,
    )

    _, _, rgba, _, _ = p.getCameraImage(
        width=image_size,
        height=image_size,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
    )

    arr = np.asarray(rgba, dtype=np.uint8).reshape(image_size, image_size, 4)
    return arr[:, :, :3].copy()


def draw_grid(img: np.ndarray) -> np.ndarray:
    out = img.copy()
    h, w, _ = out.shape
    step = max(1, w // 8)

    for u in range(0, w, step):
        cv2.line(out, (u, 0), (u, h - 1), (45, 45, 45), 1)
    for v in range(0, h, step):
        cv2.line(out, (0, v), (w - 1, v), (45, 45, 45), 1)

    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    _, sphere_id, radius = make_world()

    action_names = list(ACTIONS.keys())
    num_actions = len(action_names)

    z_current = []
    z_future = []
    action_vecs = []
    group_id = []
    action_id = []
    state_xy = []
    future_xy = []

    candidate_indices = np.zeros((args.groups, num_actions), dtype=np.int64)

    for group in range(args.groups):
        x = float(rng.uniform(-0.45, 0.45))
        y = float(rng.uniform(-0.45, 0.45))

        p.resetBasePositionAndOrientation(sphere_id, [x, y, radius], [0, 0, 0, 1])
        p.resetBaseVelocity(sphere_id, [0, 0, 0], [0, 0, 0])

        for _ in range(8):
            p.stepSimulation()

        current_rgb = draw_grid(render(args.image_size))
        current_tokens = image_to_tokens(current_rgb, args.patch_grid)

        saved_state = p.saveState()

        for j, name in enumerate(action_names):
            p.restoreState(saved_state)

            dx, dy = ACTIONS[name]
            p.resetBaseVelocity(
                sphere_id,
                linearVelocity=[args.velocity * dx, args.velocity * dy, 0.0],
                angularVelocity=[0.0, 0.0, 0.0],
            )

            for _ in range(args.horizon):
                p.stepSimulation()

            pos, _ = p.getBasePositionAndOrientation(sphere_id)

            future_rgb = draw_grid(render(args.image_size))
            future_tokens = image_to_tokens(future_rgb, args.patch_grid)

            idx = group * num_actions + j
            candidate_indices[group, j] = idx

            z_current.append(current_tokens)
            z_future.append(future_tokens)
            action_vecs.append([dx, dy])
            group_id.append(group)
            action_id.append(j)
            state_xy.append([x, y])
            future_xy.append([pos[0], pos[1]])

        p.removeState(saved_state)

        if group % 100 == 0:
            print(f"group={group}/{args.groups}", flush=True)

    p.disconnect()

    z_current = np.stack(z_current).astype(np.float32)
    z_future = np.stack(z_future).astype(np.float32)
    action_vecs = np.asarray(action_vecs, dtype=np.float32)

    np.savez_compressed(
        args.out,
        z_current=z_current,
        z_future=z_future,
        action=action_vecs,
        candidate_indices=candidate_indices,
        correct_candidate=np.arange(num_actions, dtype=np.int64)[None, :].repeat(args.groups, axis=0),
        group_id=np.asarray(group_id, dtype=np.int64),
        action_id=np.asarray(action_id, dtype=np.int64),
        action_names=np.asarray(action_names),
        state_xy=np.asarray(state_xy, dtype=np.float32),
        future_xy=np.asarray(future_xy, dtype=np.float32),
        encoder=np.asarray("pybullet_rgb_patch_tokens_v0"),
        latent_shape=np.asarray(z_current.shape[1:]),
        latent_tokens=np.asarray(z_current.shape[1]),
        latent_dim=np.asarray(z_current.shape[2]),
        action_dim=np.asarray(action_vecs.shape[1]),
        image_size=np.asarray(args.image_size),
        patch_grid=np.asarray(args.patch_grid),
        horizon=np.asarray(args.horizon),
        velocity=np.asarray(args.velocity),
        dataset_version=np.asarray("pybullet_exact_intervention_v0"),
    )

    report = {
        "out": str(args.out),
        "groups": int(args.groups),
        "candidates": int(num_actions),
        "samples": int(args.groups * num_actions),
        "chance_top1": float(1.0 / num_actions),
        "z_current_shape": list(z_current.shape),
        "z_future_shape": list(z_future.shape),
        "action_shape": list(action_vecs.shape),
        "image_size": int(args.image_size),
        "patch_grid": int(args.patch_grid),
        "horizon": int(args.horizon),
        "velocity": float(args.velocity),
        "action_names": action_names,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# PyBullet exact-intervention dataset audit",
        "",
        f"- output: `{args.out}`",
        f"- groups: `{args.groups}`",
        f"- candidates per group: `{num_actions}`",
        f"- samples: `{args.groups * num_actions}`",
        f"- chance top-1: `{1.0 / num_actions:.6f}`",
        f"- latent shape: `{tuple(z_current.shape[1:])}`",
        f"- action dimension: `{action_vecs.shape[1]}`",
        f"- image size: `{args.image_size}`",
        f"- patch grid: `{args.patch_grid}`",
        f"- horizon: `{args.horizon}`",
        f"- velocity: `{args.velocity}`",
        "",
        "## Interpretation",
        "",
        "Each group is generated by saving one PyBullet simulator state and restoring it under five alternative actions.",
        "This is the first physical exact-intervention benchmark after the synthetic toy sanity check.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
