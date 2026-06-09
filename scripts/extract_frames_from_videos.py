from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frame sequences from videos for SPSM real-video smoke tests."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--output", type=Path, default=Path("data/real_video_sequences"))
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--max-frames-per-video", type=int, default=120)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--backend",
        choices=["auto", "opencv", "ffmpeg"],
        default="auto",
        help="Frame extraction backend. 'auto' tries OpenCV first, then ffmpeg CLI.",
    )
    return parser.parse_args()


def iter_videos(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS)


def clean_sequence_dir(seq_dir: Path, overwrite: bool) -> None:
    if seq_dir.exists() and overwrite:
        shutil.rmtree(seq_dir)
    seq_dir.mkdir(parents=True, exist_ok=True)


def resize_and_save(frame_rgb, path: Path, image_size: int) -> None:
    image = Image.fromarray(frame_rgb)
    image = image.resize((image_size, image_size), Image.BICUBIC)
    image.save(path)


def extract_with_opencv(video_path: Path, seq_dir: Path, fps: float, max_frames: int, image_size: int) -> int:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV is not installed") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps is None or source_fps <= 0:
        source_fps = fps
    stride = max(1, int(round(source_fps / fps)))

    written = 0
    frame_idx = 0
    while written < max_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            resize_and_save(frame_rgb, seq_dir / f"{written:04d}.png", image_size)
            written += 1
        frame_idx += 1

    cap.release()
    return written


def extract_with_ffmpeg(video_path: Path, seq_dir: Path, fps: float, max_frames: int, image_size: int) -> int:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg CLI is not available on PATH")

    output_pattern = seq_dir / "%04d.png"
    vf = f"fps={fps},scale={image_size}:{image_size}:force_original_aspect_ratio=increase,crop={image_size}:{image_size}"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        str(output_pattern),
    ]
    subprocess.run(cmd, check=True)
    return len(sorted(seq_dir.glob("*.png")))


def extract_video(args: argparse.Namespace, video_path: Path, seq_id: int) -> int:
    seq_dir = args.output / f"seq_{seq_id:04d}"
    clean_sequence_dir(seq_dir, overwrite=args.overwrite)

    # Avoid mixing old and new frames when overwrite is false.
    if any(seq_dir.glob("*.png")):
        return len(sorted(seq_dir.glob("*.png")))

    if args.backend in {"auto", "opencv"}:
        try:
            return extract_with_opencv(
                video_path, seq_dir, args.fps, args.max_frames_per_video, args.image_size
            )
        except Exception:
            if args.backend == "opencv":
                raise

    return extract_with_ffmpeg(video_path, seq_dir, args.fps, args.max_frames_per_video, args.image_size)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(
            f"Input video folder not found: {args.input}. Create it and put .mp4/.mov/.mkv files inside."
        )

    videos = iter_videos(args.input)
    if not videos:
        raise ValueError(f"No video files found under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    total = 0
    for seq_id, video in enumerate(tqdm(videos, desc="extracting video frames")):
        n = extract_video(args, video, seq_id)
        total += n
        print(f"{video.name}: wrote {n} frames to {args.output / f'seq_{seq_id:04d}'}")

    print(f"Done. Extracted {total} frames from {len(videos)} videos into {args.output}")


if __name__ == "__main__":
    main()
