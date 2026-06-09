from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def make_sequence(out_dir: Path, seq_id: int, n_frames: int, size: int) -> None:
    seq_dir = out_dir / f"seq_{seq_id:03d}"
    seq_dir.mkdir(parents=True, exist_ok=True)
    # Different deterministic motion per sequence.
    vx = 2 + (seq_id % 4)
    vy = 1 + ((seq_id * 2) % 5)
    radius = 8 + (seq_id % 5)
    for t in range(n_frames):
        img = Image.new("RGB", (size, size), (8, 8, 12))
        draw = ImageDraw.Draw(img)
        x = (12 + vx * t + 7 * seq_id) % (size - 2 * radius) + radius
        y = (18 + vy * t + 11 * seq_id) % (size - 2 * radius) + radius
        color = (200, 220 - 10 * (seq_id % 8), 80 + 15 * (seq_id % 10))
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        # Add a static landmark so the encoder sees background structure too.
        draw.rectangle((size - 26, 8, size - 10, 24), fill=(40, 120, 220))
        img.save(seq_dir / f"{t:04d}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sample_image_sequences")
    parser.add_argument("--n-sequences", type=int, default=24)
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=96)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for seq_id in range(args.n_sequences):
        make_sequence(out_dir, seq_id, args.n_frames, args.size)
    print(f"Wrote sample image sequences to {out_dir}")


if __name__ == "__main__":
    main()
