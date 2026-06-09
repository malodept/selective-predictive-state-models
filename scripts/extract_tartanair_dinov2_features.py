from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


IMAGE_DIR_CANDIDATES = ["image_left", "image_right"]
POSE_FILE_CANDIDATES = ["pose_left.txt", "pose_right.txt"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-name", type=str, default="dinov2_vits14")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--hard-gap", type=int, default=3)
    p.add_argument("--mismatch-prob", type=float, default=0.15)
    p.add_argument("--max-trajectories", type=int, default=6)
    p.add_argument("--max-frames-per-trajectory", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def find_trajectories(root: Path) -> list[tuple[Path, Path, Path]]:
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


def list_images(image_dir: Path) -> list[Path]:
    return sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


def preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


@torch.no_grad()
def encode_images(
    images: list[Path],
    model: torch.nn.Module,
    tfm: transforms.Compose,
    device: str,
    batch_size: int,
) -> np.ndarray:
    feats = []

    for start in tqdm(range(0, len(images), batch_size), desc="encoding DINOv2", leave=False):
        batch_paths = images[start : start + batch_size]
        batch = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch.append(tfm(img))

        x = torch.stack(batch, dim=0).to(device)
        z = model(x)

        if isinstance(z, dict):
            z = z.get("x_norm_clstoken", next(iter(z.values())))

        feats.append(z.detach().cpu().numpy().astype(np.float32))

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

    z_current, z_future, actions = [], [], []
    expected_unreliable, observed_surprise = [], []

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

    trajectories = find_trajectories(args.input)[: args.max_trajectories]
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found under {args.input}")

    print(f"Found {len(trajectories)} trajectories:")
    for traj_dir, image_dir, pose_file in trajectories:
        print(f"  - {traj_dir} | images={image_dir.name} | poses={pose_file.name}")

    print(f"Loading DINOv2 model: {args.model_name}")
    model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    model.eval().to(args.device)

    tfm = preprocess()
    chunks = []

    for traj_dir, image_dir, pose_file in trajectories:
        images = list_images(image_dir)[: args.max_frames_per_trajectory]
        poses = load_poses(pose_file)[: len(images)]

        print(f"Processing {traj_dir}: {len(images)} images, {len(poses)} poses")
        features = encode_images(images, model, tfm, args.device, args.batch_size)

        chunks.append(
            build_transitions(
                features,
                poses,
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
    merged["source"] = np.asarray("tartanair_dinov2")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **merged)

    print(f"Wrote {len(merged['z_current'])} DINOv2 TartanAir transitions to {args.output}")
    print(f"latent_dim={merged['z_current'].shape[1]} action_dim={merged['action'].shape[1]}")


if __name__ == "__main__":
    main()
