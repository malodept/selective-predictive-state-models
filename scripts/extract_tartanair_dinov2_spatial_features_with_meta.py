from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


IMAGE_DIR_CANDIDATES = ["image_left", "image_right"]
POSE_FILE_CANDIDATES = ["pose_left.txt", "pose_right.txt"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-name", type=str, default="dinov2_vits14")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--hard-gap", type=int, default=3)
    p.add_argument("--mismatch-prob", type=float, default=0.15)
    p.add_argument("--max-trajectories", type=int, default=999)
    p.add_argument("--max-frames-per-trajectory", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_trajectories(root: Path):
    found = []
    for traj_dir in root.rglob("*"):
        if not traj_dir.is_dir():
            continue

        image_dir = None
        for name in IMAGE_DIR_CANDIDATES:
            c = traj_dir / name
            if c.is_dir() and any(c.glob("*.png")):
                image_dir = c
                break

        if image_dir is None:
            continue

        pose_file = None
        for name in POSE_FILE_CANDIDATES:
            c = traj_dir / name
            if c.is_file():
                pose_file = c
                break

        if pose_file is not None:
            found.append((traj_dir, image_dir, pose_file))

    return sorted(found, key=lambda x: str(x[0]))


def parse_meta(traj_dir: Path):
    parts = traj_dir.parts
    # expected: .../<env>/<difficulty>/<env>/<difficulty>/<Pxxx>
    trajectory = traj_dir.name
    difficulty = "unknown"
    environment = "unknown"

    for i, part in enumerate(parts):
        if re.fullmatch(r"P\d+", part):
            trajectory = part
            if i >= 1:
                difficulty = parts[i - 1]
            if i >= 2:
                environment = parts[i - 2]
            break

    trajectory_id = f"{environment}/{difficulty}/{trajectory}"
    return environment, difficulty, trajectory, trajectory_id


def list_images(image_dir: Path):
    return sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))


def load_poses(path: Path):
    poses = np.loadtxt(path, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


def preprocess(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@torch.no_grad()
def encode_images_spatial(images, model, tfm, device, batch_size):
    feats = []

    for start in tqdm(range(0, len(images), batch_size), desc="encoding DINOv2 spatial", leave=False):
        batch_paths = images[start:start + batch_size]
        batch = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch.append(tfm(img))

        x = torch.stack(batch, dim=0).to(device)

        out = model.forward_features(x)
        cls = out["x_norm_clstoken"]
        patches = out["x_norm_patchtokens"]

        patch_mean = patches.mean(dim=1)
        patch_std = patches.std(dim=1)

        z = torch.cat([cls, patch_mean, patch_std], dim=1)

        feats.append(z.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(feats, axis=0)


def build_transitions(features, poses, traj_dir, max_gap, hard_gap, mismatch_prob, rng):
    n = min(len(features), len(poses))
    features = features[:n]
    poses = poses[:n]

    env, difficulty, trajectory, trajectory_id = parse_meta(traj_dir)

    z_current, z_future, actions = [], [], []
    expected_unreliable, observed_surprise = [], []
    frame_i, frame_j, gaps = [], [], []
    trajectory_ids, environments, difficulties, trajectories = [], [], [], []

    for i in range(n - 1):
        max_j = min(n - 1, i + max_gap)

        for j in range(i + 1, max_j + 1):
            gap = j - i
            z_i = features[i]
            z_j = features[j].copy()
            action = (poses[j] - poses[i]).astype(np.float32)

            delta_xyz = action[:min(3, action.shape[0])]
            expected_hard = float(gap >= hard_gap or np.linalg.norm(delta_xyz) > 0.5)

            surprise = 0.0
            if rng.random() < mismatch_prob:
                random_idx = int(rng.integers(0, n))
                z_j = features[random_idx].copy()
                surprise = 1.0

            z_current.append(z_i)
            z_future.append(z_j)
            actions.append(action)
            expected_unreliable.append(expected_hard)
            observed_surprise.append(surprise)
            frame_i.append(i)
            frame_j.append(j)
            gaps.append(gap)
            trajectory_ids.append(trajectory_id)
            environments.append(env)
            difficulties.append(difficulty)
            trajectories.append(trajectory)

    return {
        "z_current": np.stack(z_current).astype(np.float32),
        "z_future": np.stack(z_future).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "expected_unreliable": np.asarray(expected_unreliable, dtype=np.float32),
        "observed_surprise": np.asarray(observed_surprise, dtype=np.float32),
        "frame_i": np.asarray(frame_i, dtype=np.int32),
        "frame_j": np.asarray(frame_j, dtype=np.int32),
        "gap": np.asarray(gaps, dtype=np.int16),
        "trajectory_id": np.asarray(trajectory_ids),
        "environment": np.asarray(environments),
        "difficulty": np.asarray(difficulties),
        "trajectory": np.asarray(trajectories),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    trajectories = find_trajectories(args.input)[:args.max_trajectories]
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found under {args.input}")

    print(f"Found {len(trajectories)} trajectories")

    print(f"Loading DINOv2 model: {args.model_name}")
    model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    model.eval().to(args.device)

    tfm = preprocess(args.image_size)
    chunks = []

    for idx, (traj_dir, image_dir, pose_file) in enumerate(trajectories, start=1):
        images = list_images(image_dir)[:args.max_frames_per_trajectory]
        poses = load_poses(pose_file)[:len(images)]

        print("\n" + "=" * 100)
        print(f"[{idx}/{len(trajectories)}] {traj_dir}")
        print(f"images={len(images)} poses={len(poses)}")

        features = encode_images_spatial(images, model, tfm, args.device, args.batch_size)

        chunks.append(
            build_transitions(
                features,
                poses,
                traj_dir,
                args.max_gap,
                args.hard_gap,
                args.mismatch_prob,
                rng,
            )
        )

    merged = {}
    for key in chunks[0].keys():
        merged[key] = np.concatenate([c[key] for c in chunks], axis=0)

    merged["encoder"] = np.asarray(args.model_name)
    merged["latent_dim"] = np.asarray(merged["z_current"].shape[1])
    merged["action_dim"] = np.asarray(merged["action"].shape[1])
    merged["source"] = np.asarray("tartanair_dinov2_spatial_cls_mean_std")
    merged["image_size"] = np.asarray(args.image_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)

    print(f"Wrote merged dataset: {args.output}")
    print(f"samples={len(merged['z_current'])}")
    print(f"latent_dim={merged['z_current'].shape[1]} action_dim={merged['action'].shape[1]}")
    print("environments=", sorted(set(str(x) for x in merged["environment"])))


if __name__ == "__main__":
    main()
