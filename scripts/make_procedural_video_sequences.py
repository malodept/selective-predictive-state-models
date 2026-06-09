from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-sequences", type=int, default=24)
    parser.add_argument("--n-frames", type=int, default=80)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_background(rng: np.random.Generator, canvas: int) -> Image.Image:
    base = rng.normal(loc=128, scale=45, size=(canvas, canvas, 3)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="RGB").filter(ImageFilter.GaussianBlur(radius=2.5))
    draw = ImageDraw.Draw(img, "RGBA")

    # Large textured regions.
    for _ in range(80):
        x0 = int(rng.integers(0, canvas))
        y0 = int(rng.integers(0, canvas))
        w = int(rng.integers(20, 160))
        h = int(rng.integers(20, 160))
        color = tuple(int(v) for v in rng.integers(30, 230, size=3)) + (int(rng.integers(30, 110)),)
        draw.rectangle([x0, y0, x0 + w, y0 + h], fill=color)

    # Line structures, like roads/buildings/edges.
    for _ in range(50):
        x0 = int(rng.integers(0, canvas))
        y0 = int(rng.integers(0, canvas))
        x1 = int(rng.integers(0, canvas))
        y1 = int(rng.integers(0, canvas))
        color = tuple(int(v) for v in rng.integers(0, 255, size=3)) + (int(rng.integers(80, 180)),)
        width = int(rng.integers(1, 6))
        draw.line([x0, y0, x1, y1], fill=color, width=width)

    return img.filter(ImageFilter.GaussianBlur(radius=0.7))


def crop_with_motion(bg: Image.Image, t: int, n_frames: int, size: int, seq_id: int) -> Image.Image:
    canvas = bg.size[0]
    max_shift = canvas - size - 1

    phase = 2.0 * math.pi * t / max(n_frames - 1, 1)
    drift_x = 0.5 + 0.35 * math.sin(phase + 0.4 * seq_id)
    drift_y = 0.5 + 0.35 * math.cos(0.8 * phase + 0.3 * seq_id)

    x = int(max_shift * drift_x)
    y = int(max_shift * drift_y)

    frame = bg.crop((x, y, x + size, y + size))

    # Small rotation to simulate camera attitude changes.
    angle = 3.0 * math.sin(phase + seq_id)
    frame = frame.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(20, 20, 20))

    return frame


def add_moving_objects(
    frame: Image.Image,
    rng: np.random.Generator,
    t: int,
    n_frames: int,
    seq_id: int,
) -> Image.Image:
    draw = ImageDraw.Draw(frame, "RGBA")
    w, h = frame.size

    # Moving circle.
    phase = t / max(n_frames - 1, 1)
    cx = int((0.1 + 0.8 * phase) * w)
    cy = int((0.5 + 0.3 * math.sin(2 * math.pi * phase + seq_id)) * h)
    r = 12 + (seq_id % 4) * 3
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(240, 80, 60, 170))

    # Occasional occluder.
    if (t + seq_id) % 17 in {0, 1, 2, 3}:
        ox = int((0.2 + 0.5 * math.sin(0.3 * t + seq_id)) * w)
        oy = int((0.2 + 0.5 * math.cos(0.2 * t)) * h)
        draw.rectangle([ox, oy, ox + 45, oy + 35], fill=(20, 20, 20, 150))

    # Brightness shift.
    arr = np.asarray(frame).astype(np.float32)
    factor = 0.85 + 0.25 * math.sin(0.12 * t + seq_id)
    arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def make_sequence(output: Path, seq_id: int, args: argparse.Namespace, rng: np.random.Generator) -> None:
    seq_dir = output / f"seq_{seq_id:04d}"
    seq_dir.mkdir(parents=True, exist_ok=True)

    canvas = args.size * 3
    bg = make_background(rng, canvas)

    for t in range(args.n_frames):
        frame = crop_with_motion(bg, t, args.n_frames, args.size, seq_id)
        frame = add_moving_objects(frame, rng, t, args.n_frames, seq_id)
        frame.save(seq_dir / f"{t:04d}.png")


def main() -> None:
    args = parse_args()

    if args.output.exists() and args.overwrite:
        for p in sorted(args.output.glob("*")):
            if p.is_dir():
                for q in p.glob("*.png"):
                    q.unlink()
                p.rmdir()

    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for seq_id in range(args.n_sequences):
        make_sequence(args.output, seq_id, args, rng)

    print(f"Wrote {args.n_sequences} procedural video-like sequences to {args.output}")


if __name__ == "__main__":
    main()
