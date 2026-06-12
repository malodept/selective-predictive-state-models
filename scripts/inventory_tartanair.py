from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


IMAGE_DIR_CANDIDATES = ["image_left", "image_right"]
POSE_FILE_CANDIDATES = ["pose_left.txt", "pose_right.txt"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=Path("reports/tables/protocol/tartanair_inventory.csv"))
    p.add_argument("--out-md", type=Path, default=Path("reports/tables/protocol/tartanair_inventory.md"))
    return p.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )


def infer_env_difficulty_traj(traj_dir: Path):
    parts = traj_dir.parts

    difficulty = None
    for d in ["Easy", "Hard"]:
        if d in parts:
            difficulty = d
            break

    traj = traj_dir.name

    # Try to infer environment as the token before first Easy/Hard if possible.
    env = None
    if difficulty is not None and difficulty in parts:
        idx = parts.index(difficulty)
        if idx > 0:
            env = parts[idx - 1]

    return env or "unknown", difficulty or "unknown", traj


def df_to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table writer, avoids optional pandas dependency 'tabulate'."""
    cols = list(df.columns)
    rows = []

    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")

    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            text = str(v)
            text = text.replace("|", "\\|")
            vals.append(text)
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join(rows)


def main():
    args = parse_args()

    rows = []
    seen = set()

    for root in args.roots:
        if not root.exists():
            print(f"[WARN] missing root: {root}")
            continue

        for traj_dir in root.rglob("*"):
            if not traj_dir.is_dir():
                continue

            image_dir = None
            image_side = None
            for name in IMAGE_DIR_CANDIDATES:
                c = traj_dir / name
                if c.is_dir() and len(list_images(c)) > 0:
                    image_dir = c
                    image_side = name
                    break

            if image_dir is None:
                continue

            pose_file = None
            pose_side = None
            for name in POSE_FILE_CANDIDATES:
                c = traj_dir / name
                if c.is_file():
                    pose_file = c
                    pose_side = name
                    break

            if pose_file is None:
                continue

            key = str(traj_dir.resolve())
            if key in seen:
                continue
            seen.add(key)

            images = list_images(image_dir)
            try:
                n_pose_lines = sum(1 for _ in pose_file.open("r"))
            except Exception:
                n_pose_lines = -1

            env, difficulty, traj = infer_env_difficulty_traj(traj_dir)

            rows.append(
                {
                    "root": str(root),
                    "trajectory_dir": str(traj_dir),
                    "environment": env,
                    "difficulty": difficulty,
                    "trajectory": traj,
                    "image_side": image_side,
                    "pose_side": pose_side,
                    "n_images": len(images),
                    "n_pose_lines": n_pose_lines,
                    "usable_frames": min(len(images), n_pose_lines) if n_pose_lines >= 0 else len(images),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No TartanAir trajectories found.")
        return

    df = df.sort_values(["environment", "difficulty", "trajectory_dir"]).reset_index(drop=True)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    summary = (
        df.groupby(["environment", "difficulty"], dropna=False)
        .agg(
            trajectories=("trajectory_dir", "count"),
            total_usable_frames=("usable_frames", "sum"),
            min_frames=("usable_frames", "min"),
            max_frames=("usable_frames", "max"),
        )
        .reset_index()
        .sort_values(["environment", "difficulty"])
    )

    with args.out_md.open("w") as f:
        f.write("# TartanAir inventory\n\n")
        f.write("## Summary by environment and difficulty\n\n")
        f.write(df_to_markdown(summary))
        f.write("\n\n## Trajectories\n\n")
        f.write(df_to_markdown(df[[
            "environment",
            "difficulty",
            "trajectory",
            "usable_frames",
            "trajectory_dir",
        ]]))
        f.write("\n")

    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
