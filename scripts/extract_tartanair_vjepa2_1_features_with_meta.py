from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VJEPA = ROOT / "external" / "vjepa2"

sys.path.insert(0, str(VJEPA))
sys.path.insert(0, str(VJEPA / "src"))

from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-npz", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunks-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--pool-grid", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-trajectories", type=int, default=0)
    p.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    p.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def find_image_dir(raw_root: Path, trajectory_id: str) -> Path:
    parts = trajectory_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"Unexpected trajectory_id={trajectory_id}")

    env, difficulty, traj = parts

    candidates = [
        raw_root / env / difficulty / env / difficulty / traj / "image_left",
        raw_root / env / difficulty / traj / "image_left",
        raw_root / env / difficulty / env / difficulty / traj / "image_left_color",
        raw_root / env / difficulty / traj / "image_left_color",
    ]

    for c in candidates:
        if c.exists():
            return c

    matches = [
        p for p in raw_root.rglob("image_left")
        if traj in str(p) and env in str(p) and difficulty in str(p)
    ]

    if not matches:
        raise FileNotFoundError(f"Could not find image_left for {trajectory_id}")

    return matches[0]


def list_images(image_dir: Path) -> List[Path]:
    exts = ["*.png", "*.jpg", "*.jpeg"]
    files = []
    for e in exts:
        files.extend(image_dir.glob(e))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return files


def load_image(path: Path, image_size: int, mean, std) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std

    return np.transpose(img, (2, 0, 1))  # C,H,W


def make_causal_clip(images: List[Path], frame_idx: int, num_frames: int, image_size: int, mean, std) -> np.ndarray:
    start = frame_idx - num_frames + 1
    idxs = [max(0, start + k) for k in range(num_frames)]
    frames = [load_image(images[i], image_size, mean, std) for i in idxs]
    arr = np.stack(frames, axis=1)  # C,T,H,W
    return arr


def load_encoder(checkpoint: Path, num_frames: int, device: str):
    encoder, _ = vjepa2_1_vit_base_384(
        pretrained=False,
        num_frames=num_frames,
    )

    ckpt = torch.load(checkpoint, map_location="cpu")
    key = "ema_encoder" if "ema_encoder" in ckpt else "target_encoder"
    sd = _clean_backbone_key(ckpt[key])
    msg = encoder.load_state_dict(sd, strict=False)
    print("load_state_dict:", msg)

    encoder = encoder.to(device).eval()
    return encoder


def pool_last_temporal_tokens(out: torch.Tensor, num_frames: int, image_size: int, pool_grid: int) -> np.ndarray:
    # out: [B, T_tokens * H_tokens * W_tokens, D]
    b, n, d = out.shape
    t_tokens = num_frames // 2
    s_tokens = image_size // 16

    expected = t_tokens * s_tokens * s_tokens
    if n != expected:
        raise ValueError(f"Unexpected token count: got {n}, expected {expected}")

    x = out.reshape(b, t_tokens, s_tokens, s_tokens, d)
    x = x[:, -1]  # B, 24, 24, D

    if s_tokens % pool_grid != 0:
        raise ValueError(f"Cannot pool {s_tokens} to grid {pool_grid}")

    block = s_tokens // pool_grid
    x = x.reshape(b, pool_grid, block, pool_grid, block, d).mean(dim=(2, 4))
    x = x.reshape(b, pool_grid * pool_grid, d)

    return x.detach().cpu().numpy().astype(np.float16)


@torch.no_grad()
def extract_trajectory_features(
    encoder,
    trajectory_id: str,
    needed_frames: np.ndarray,
    raw_root: Path,
    args,
) -> Tuple[np.ndarray, np.ndarray]:
    chunk = args.chunks_dir / f"{safe_name(trajectory_id)}.npz"

    if chunk.exists() and not args.force:
        d = np.load(chunk)
        return d["frame_indices"], d["features"]

    image_dir = find_image_dir(raw_root, trajectory_id)
    images = list_images(image_dir)

    frame_indices = np.asarray(sorted(set(int(x) for x in needed_frames)), dtype=np.int64)

    print(f"[extract] {trajectory_id}: frames={len(frame_indices)} images={len(images)} dir={image_dir}", flush=True)

    feats = []

    for start in range(0, len(frame_indices), args.batch_size):
        batch_frames = frame_indices[start:start + args.batch_size]

        clips = [
            make_causal_clip(images, int(fi), args.num_frames, args.image_size, args.mean, args.std)
            for fi in batch_frames
        ]

        x = torch.from_numpy(np.stack(clips, axis=0)).to(args.device)  # B,C,T,H,W

        if args.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = encoder(x)
        else:
            out = encoder(x)

        pooled = pool_last_temporal_tokens(out.float(), args.num_frames, args.image_size, args.pool_grid)
        feats.append(pooled)

        print(f"  {start + len(batch_frames)}/{len(frame_indices)}", flush=True)

    features = np.concatenate(feats, axis=0)

    chunk.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        chunk,
        trajectory_id=np.asarray(trajectory_id),
        frame_indices=frame_indices,
        features=features,
    )

    print(f"[wrote] {chunk} {features.shape}", flush=True)

    return frame_indices, features


def subset_source_arrays(source, keep_idx: np.ndarray) -> Dict[str, np.ndarray]:
    out = {}
    n = source["action"].shape[0]

    for k in source.files:
        if k in ["z_current", "z_future"]:
            continue

        v = source[k]
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            out[k] = v[keep_idx]
        else:
            out[k] = v

    return out


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.chunks_dir.mkdir(parents=True, exist_ok=True)

    source = np.load(args.source_npz, allow_pickle=True)

    trajectory_id = np.asarray(source["trajectory_id"]).astype(str)
    frame_i = source["frame_i"].astype(np.int64)
    frame_j = source["frame_j"].astype(np.int64)

    unique_tids = sorted(set(trajectory_id.tolist()))

    if args.max_trajectories > 0:
        keep_tids = set(unique_tids[:args.max_trajectories])
        keep_idx = np.asarray([i for i, t in enumerate(trajectory_id) if t in keep_tids], dtype=np.int64)
        unique_tids = sorted(keep_tids)
    else:
        keep_idx = np.arange(len(trajectory_id), dtype=np.int64)

    print("samples kept:", len(keep_idx))
    print("trajectories kept:", len(unique_tids))

    out = subset_source_arrays(source, keep_idx)

    traj_kept = trajectory_id[keep_idx]
    fi_kept = frame_i[keep_idx]
    fj_kept = frame_j[keep_idx]

    _, tokens, old_dim = source["z_current"].shape
    del old_dim

    z_current = np.empty((len(keep_idx), args.pool_grid * args.pool_grid, 768), dtype=np.float16)
    z_future = np.empty((len(keep_idx), args.pool_grid * args.pool_grid, 768), dtype=np.float16)

    encoder = load_encoder(args.checkpoint, args.num_frames, args.device)

    for tid in unique_tids:
        local_rows = np.where(traj_kept == tid)[0]
        needed = np.concatenate([fi_kept[local_rows], fj_kept[local_rows]])

        frame_indices, features = extract_trajectory_features(
            encoder=encoder,
            trajectory_id=tid,
            needed_frames=needed,
            raw_root=args.raw_root,
            args=args,
        )

        frame_to_pos = {int(f): p for p, f in enumerate(frame_indices.tolist())}

        for r in local_rows:
            z_current[r] = features[frame_to_pos[int(fi_kept[r])]]
            z_future[r] = features[frame_to_pos[int(fj_kept[r])]]

    out["z_current"] = z_current
    out["z_future"] = z_future
    out["encoder"] = np.asarray("vjepa2_1_vit_base_384_causal16_last_temporal_pool4")
    out["latent_dim"] = np.asarray(768)
    out["tokens"] = np.asarray(args.pool_grid * args.pool_grid)
    out["video_frames"] = np.asarray(args.num_frames)
    out["image_size"] = np.asarray(args.image_size)
    out["pool_grid"] = np.asarray(args.pool_grid)

    np.savez_compressed(args.output, **out)

    print("Wrote merged dataset:", args.output)
    print("z_current:", z_current.shape, z_current.dtype)
    print("z_future :", z_future.shape, z_future.dtype)
    print("action   :", out["action"].shape, out["action"].dtype)
    print("encoder  :", out["encoder"])


if __name__ == "__main__":
    main()
