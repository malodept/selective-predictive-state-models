from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from spsm.features.patch_encoder import patch_mean_features


def sorted_frames(seq_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in seq_dir.iterdir() if p.suffix.lower() in exts)


def build_sequence_features(frames: list[Path], args: argparse.Namespace) -> list[np.ndarray]:
    if args.encoder == "patch_mean":
        return [patch_mean_features(p, grid_size=args.grid_size) for p in frames]

    if args.encoder == "resnet18":
        from spsm.features.torchvision_encoder import (
            ResNet18FeatureExtractor,
            TorchvisionFeatureConfig,
        )

        extractor = ResNet18FeatureExtractor(
            TorchvisionFeatureConfig(
                encoder="resnet18",
                weights=args.resnet_weights,
                image_size=args.image_size,
                batch_size=args.batch_size,
                device=args.device,
            )
        )
        return list(extractor.encode_paths(frames))

    raise ValueError(f"Unknown encoder: {args.encoder}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_image_sequences")
    parser.add_argument("--output", default="outputs/image_sequence_features/features.npz")
    parser.add_argument("--encoder", choices=["patch_mean", "resnet18"], default="patch_mean")
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resnet-weights", choices=["imagenet", "random"], default="imagenet")
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--hard-gap", type=int, default=3)
    parser.add_argument("--mismatch-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(args.input)
    if not root.exists():
        raise FileNotFoundError(f"Input folder not found: {root}")
    seq_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not seq_dirs:
        raise ValueError(f"No sequence folders found under {root}")

    rng = np.random.default_rng(args.seed)
    z_current, clean_future, actions, expected = [], [], [], []
    metadata = []

    for seq_dir in tqdm(seq_dirs, desc=f"extracting {args.encoder} sequence features"):
        frames = sorted_frames(seq_dir)
        if len(frames) < 2:
            continue
        feats = build_sequence_features(frames, args)
        for i in range(len(feats) - 1):
            for gap in range(1, args.max_gap + 1):
                j = i + gap
                if j >= len(feats):
                    continue
                z_current.append(feats[i])
                clean_future.append(feats[j])
                actions.append([gap / float(args.max_gap)])
                expected.append(float(gap >= args.hard_gap))
                metadata.append((seq_dir.name, i, j, gap))

    z_current = np.stack(z_current).astype(np.float32)
    clean_future = np.stack(clean_future).astype(np.float32)
    action = np.asarray(actions, dtype=np.float32)
    expected_unreliable = np.asarray(expected, dtype=np.float32)

    n = z_current.shape[0]
    observed_surprise = rng.random(n) < args.mismatch_prob
    perm = rng.permutation(n)
    z_future = clean_future.copy()
    z_future[observed_surprise] = clean_future[perm[observed_surprise]]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        z_current=z_current,
        action=action,
        z_future=z_future.astype(np.float32),
        clean_future=clean_future,
        expected_unreliable=expected_unreliable,
        observed_surprise=observed_surprise.astype(np.float32),
        metadata=np.asarray(metadata, dtype=object),
        encoder=args.encoder,
        resnet_weights=args.resnet_weights,
    )
    print(f"Wrote {n} feature transitions to {out}")
    print(f"encoder={args.encoder} latent_dim={z_current.shape[1]} action_dim={action.shape[1]}")


if __name__ == "__main__":
    main()
