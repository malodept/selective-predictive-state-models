from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_ENVS = [
    "japanesealley",
    "office",
    "carwelding",
    "hospital",
    "neighborhood",
    "seasidetown",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", default="theairlabcmu/tartanair")
    p.add_argument("--repo-type", default="dataset")
    p.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    p.add_argument("--difficulties", nargs="+", default=["Hard"])
    p.add_argument("--modalities", nargs="+", default=["image_left"])
    p.add_argument("--hf-dir", type=Path, default=Path("data/tartanair_hf"))
    p.add_argument("--raw-dir", type=Path, default=Path("data/tartanair_raw"))
    p.add_argument("--extract", action="store_true")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def safe_extract(zip_path: Path, out_dir: Path, force: bool = False) -> None:
    marker = out_dir / ".extracted_ok"
    if marker.exists() and not force:
        print(f"[skip extract] {zip_path} -> {out_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            name = member.filename
            if name.startswith("/") or ".." in Path(name).parts:
                raise RuntimeError(f"Unsafe zip member: {name}")
        z.extractall(out_dir)

    marker.write_text(str(zip_path) + "\n")


def main() -> None:
    args = parse_args()

    args.hf_dir.mkdir(parents=True, exist_ok=True)
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []

    for env in args.envs:
        for difficulty in args.difficulties:
            for modality in args.modalities:
                filename = f"{env}/{difficulty}/{modality}.zip"
                print("\n" + "=" * 100)
                print(f"[download] {args.repo_id} :: {filename}")

                local_path = Path(
                    hf_hub_download(
                        repo_id=args.repo_id,
                        repo_type=args.repo_type,
                        filename=filename,
                        local_dir=args.hf_dir,
                    )
                )

                downloaded.append(local_path)
                print(f"[ok] {local_path} size_gb={local_path.stat().st_size / 1e9:.3f}")

                if args.extract:
                    # This mirrors your current layout:
                    # data/tartanair_raw/<env>/<difficulty>/<zip contents...>
                    extract_root = args.raw_dir / env / difficulty
                    safe_extract(local_path, extract_root, force=args.force)

    print("\nDownloaded files:")
    for p in downloaded:
        print(" -", p)


if __name__ == "__main__":
    main()
