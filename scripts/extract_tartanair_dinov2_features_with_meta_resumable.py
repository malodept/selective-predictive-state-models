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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks-dir", type=Path, required=True)
    p.add_argument("--model-name", type=str, default="dinov2_vits14")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--hard-gap", type=int, default=3)
    p.add_argument("--mismatch-prob", type=float, default=0.15)
    p.add_argument("--max-trajectories", type=int, default=999)
    p.add_argument("--max-frames-per-trajectory", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")))


def load_poses(path: Path) -> np.ndarray:
    poses = np.loadtxt(path, dtype=np.float32)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


def find_trajectories(root: Path) -> list[tuple[Path, Path, Path]]:
    found = []
    seen = set()

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

        if pose_file is None:
            continue

        key = str(traj_dir.resolve())
        if key not in seen:
            seen.add(key)
            found.append((traj_dir, image_dir, pose_file))

    return sorted(found, key=lambda x: str(x[0]))


def infer_meta(traj_dir: Path) -> tuple[str, str, str, str]:
    parts = traj_dir.parts
    difficulty = "unknown"
    for d in ["Easy", "Hard"]:
        if d in parts:
            difficulty = d
            break

    env = "unknown"
    if difficulty in parts:
        idx = parts.index(difficulty)
        if idx > 0:
            env = parts[idx - 1]

    traj = traj_dir.name
    trajectory_id = f"{env}/{difficulty}/{traj}"
    return env, difficulty, traj, trajectory_id


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def preprocess(image_size: int) -> transforms.Compose:
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
def encode_images(
    images: list[Path],
    model: torch.nn.Module,
    tfm: transforms.Compose,
    device: str,
    batch_size: int,
    amp: bool,
) -> np.ndarray:
    feats = []

    for start in tqdm(range(0, len(images), batch_size), desc="encoding DINOv2", leave=False):
        batch_paths = images[start : start + batch_size]
        batch = []

        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            batch.append(tfm(img))

        x = torch.stack(batch, dim=0).to(device, non_blocking=True)

        if amp and device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                z = model(x)
        else:
            z = model(x)

        if isinstance(z, dict):
            z = z.get("x_norm_clstoken", next(iter(z.values())))

        feats.append(z.detach().float().cpu().numpy().astype(np.float32))

        del x, z
        if device == "cuda":
            torch.cuda.empty_cache()

    return np.concatenate(feats, axis=0)


def build_transitions(
    features: np.ndarray,
    poses: np.ndarray,
    trajectory_id: str,
    environment: str,
    difficulty: str,
    trajectory: str,
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
    frame_i, frame_j, gaps = [], [], []

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
            frame_i.append(i)
            frame_j.append(j)
            gaps.append(gap)

    m = len(z_current)

    return {
        "z_current": np.stack(z_current).astype(np.float32),
        "z_future": np.stack(z_future).astype(np.float32),
        "action": np.stack(actions).astype(np.float32),
        "expected_unreliable": np.asarray(expected_unreliable, dtype=np.float32),
        "observed_surprise": np.asarray(observed_surprise, dtype=np.float32),
        "frame_i": np.asarray(frame_i, dtype=np.int32),
        "frame_j": np.asarray(frame_j, dtype=np.int32),
        "gap": np.asarray(gaps, dtype=np.int16),
        "trajectory_id": np.asarray([trajectory_id] * m, dtype="U256"),
        "environment": np.asarray([environment] * m, dtype="U128"),
        "difficulty": np.asarray([difficulty] * m, dtype="U32"),
        "trajectory": np.asarray([trajectory] * m, dtype="U32"),
    }


def merge_chunks(chunks: list[Path], output: Path, model_name: str) -> None:
    if not chunks:
        raise FileNotFoundError("No chunk files to merge.")

    loaded = [np.load(p, allow_pickle=True) for p in chunks]
    keys = list(loaded[0].files)

    merged = {}
    for key in keys:
        arrays = [d[key] for d in loaded]
        if arrays[0].ndim >= 1:
            merged[key] = np.concatenate(arrays, axis=0)
        else:
            merged[key] = arrays[0]

    merged["encoder"] = np.asarray(model_name)
    merged["latent_dim"] = np.asarray(merged["z_current"].shape[1])
    merged["action_dim"] = np.asarray(merged["action"].shape[1])
    merged["source"] = np.asarray("tartanair_dinov2_multi_env")

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)

    print(f"Wrote merged dataset: {output}")
    print(f"samples={merged['z_current'].shape[0]}")
    print(f"latent_dim={merged['z_current'].shape[1]} action_dim={merged['action'].shape[1]}")
    print(f"trajectories={len(set(merged['trajectory_id'].astype(str)))}")
    print(f"environments={sorted(set(merged['environment'].astype(str)))}")


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    rng = np.random.default_rng(args.seed)

    trajectories = find_trajectories(args.input)[: args.max_trajectories]
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found under {args.input}")

    print(f"Found {len(trajectories)} trajectories")
    print(f"Loading DINOv2 model: {args.model_name}")

    model = torch.hub.load("facebookresearch/dinov2", args.model_name)
    model.eval().to(args.device)

    tfm = preprocess(args.image_size)
    args.chunks_dir.mkdir(parents=True, exist_ok=True)

    written_chunks = []

    for idx, (traj_dir, image_dir, pose_file) in enumerate(trajectories):
        env, difficulty, traj, trajectory_id = infer_meta(traj_dir)
        chunk_path = args.chunks_dir / f"{idx:04d}_{safe_name(trajectory_id)}.npz"

        if chunk_path.exists() and not args.force:
            print(f"[skip] {trajectory_id} -> {chunk_path}")
            written_chunks.append(chunk_path)
            continue

        images = list_images(image_dir)[: args.max_frames_per_trajectory]
        poses = load_poses(pose_file)[: len(images)]

        print("\n" + "=" * 100)
        print(f"[{idx+1}/{len(trajectories)}] {trajectory_id}")
        print(f"images={len(images)} poses={len(poses)} chunk={chunk_path}")

        features = encode_images(images, model, tfm, args.device, args.batch_size, args.amp)

        chunk = build_transitions(
            features=features,
            poses=poses,
            trajectory_id=trajectory_id,
            environment=env,
            difficulty=difficulty,
            trajectory=traj,
            max_gap=args.max_gap,
            hard_gap=args.hard_gap,
            mismatch_prob=args.mismatch_prob,
            rng=rng,
        )

        np.savez_compressed(chunk_path, **chunk)
        written_chunks.append(chunk_path)

        print(f"[wrote] {chunk_path} samples={chunk['z_current'].shape[0]}")

        del features, chunk
        if args.device == "cuda":
            torch.cuda.empty_cache()

    chunks = sorted(args.chunks_dir.glob("*.npz"))
    merge_chunks(chunks, args.output, args.model_name)


if __name__ == "__main__":
    main()
