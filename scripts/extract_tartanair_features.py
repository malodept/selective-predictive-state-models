from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms


IMAGE_DIR_CANDIDATES = [
    "image_left",
    "image_right",
    "image_lcam_front",
    "image_rcam_front",
    "image_front",
]

POSE_FILE_CANDIDATES = [
    "pose_left.txt",
    "pose_right.txt",
    "pose_lcam_front.txt",
    "pose_rcam_front.txt",
    "pose.txt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", type=str, default="resnet18", choices=["resnet18"])
    parser.add_argument("--resnet-weights", type=str, default="imagenet", choices=["imagenet", "none"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-gap", type=int, default=5)
    parser.add_argument("--hard-gap", type=int, default=3)
    parser.add_argument("--mismatch-prob", type=float, default=0.15)
    parser.add_argument("--max-trajectories", type=int, default=2)
    parser.add_argument("--max-frames-per-trajectory", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_resnet18(weights: str, device: str) -> nn.Module:
    if weights == "imagenet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)

    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def find_trajectories(root: Path) -> list[tuple[Path, Path, Path]]:
    found: list[tuple[Path, Path, Path]] = []

    for traj_dir in root.rglob("*"):
        if not traj_dir.is_dir():
            continue

        image_dir = None
        for name in IMAGE_DIR_CANDIDATES:
            candidate = traj_dir / name
            if candidate.is_dir() and (
                any(candidate.glob("*.png"))
                or any(candidate.glob("*.jpg"))
                or any(candidate.glob("*.jpeg"))
            ):
                image_dir = candidate
                break

        if image_dir is None:
            continue

        pose_file = None
        for name in POSE_FILE_CANDIDATES:
            candidate = traj_dir / name
            if candidate.is_file():
                pose_file = candidate
                break

        if pose_file is None:
            continue

        found.append((traj_dir, image_dir, pose_file))

    return sorted(found, key=lambda x: str(x[0]))


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


@torch.no_grad()
def encode_images(
    images: list[Path],
    model: nn.Module,
    tfm: transforms.Compose,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    feats = []

    for start in tqdm(range(0, len(images), batch_size), desc="encoding frames", leave=False):
        batch_paths = images[start : start + batch_size]
        batch = []

        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            batch.append(tfm(img))

        x = torch.stack(batch, dim=0).to(device)
        z = model(x).detach().cpu().numpy().astype(np.float32)
        feats.append(z)

    return np.concatenate(feats, axis=0)


def build_transitions(
    features: np.ndarray,
    poses: np.ndarray,
    max_gap: int,
    hard_gap: int,
    mismatch_prob: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    n = min(len(features), len(poses))
    features = features[:n]
    poses = poses[:n]

    z_current = []
    z_future = []
    actions = []
    expected_unreliable = []
    observed_surprise = []

    for i in range(n - 1):
        max_j = min(n - 1, i + max_gap)

        for j in range(i + 1, max_j + 1):
            gap = j - i

            z_i = features[i]
            z_j = features[j].copy()

            action = (poses[j] - poses[i]).astype(np.float32)

            delta_xyz = action[: min(3, action.shape[0])]
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

    return {
        "z_current": np.stack(z_current).astype(np.float32),
        "z_future": np.stack(z_future).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "expected_unreliable": np.asarray(expected_unreliable, dtype=np.float32),
        "observed_surprise": np.asarray(observed_surprise, dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    trajectories = find_trajectories(args.input)

    if not trajectories:
        raise FileNotFoundError(
            f"No TartanAir trajectories found under {args.input}. "
            "Expected folders containing image_left/ and pose_left.txt."
        )

    trajectories = trajectories[: args.max_trajectories]

    print(f"Found {len(trajectories)} trajectories:")
    for traj_dir, image_dir, pose_file in trajectories:
        print(f"  - {traj_dir} | images={image_dir.name} | poses={pose_file.name}")

    model = build_resnet18(args.resnet_weights, args.device)
    tfm = preprocess()

    chunks = []

    for traj_dir, image_dir, pose_file in trajectories:
        images = list_images(image_dir)[: args.max_frames_per_trajectory]
        poses = load_poses(pose_file)[: len(images)]

        print(f"Processing {traj_dir}: {len(images)} images, {len(poses)} poses")

        if len(images) < 2 or len(poses) < 2:
            print(f"Skipping {traj_dir}: not enough frames/poses")
            continue

        features = encode_images(images, model, tfm, args.device)
        chunk = build_transitions(
            features=features,
            poses=poses,
            max_gap=args.max_gap,
            hard_gap=args.hard_gap,
            mismatch_prob=args.mismatch_prob,
            rng=rng,
        )
        chunks.append(chunk)

    if not chunks:
        raise RuntimeError("No usable transitions extracted.")

    merged = {}
    for key in chunks[0].keys():
        merged[key] = np.concatenate([c[key] for c in chunks], axis=0)

    merged["encoder"] = np.asarray(args.encoder)
    merged["latent_dim"] = np.asarray(merged["z_current"].shape[1])
    merged["action_dim"] = np.asarray(merged["action"].shape[1])
    merged["source"] = np.asarray("tartanair_tiny")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)

    print(f"Wrote {len(merged['z_current'])} TartanAir feature transitions to {args.output}")
    print(f"latent_dim={merged['z_current'].shape[1]} action_dim={merged['action'].shape[1]}")


if __name__ == "__main__":
    main()
