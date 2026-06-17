from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("data/tartanair_raw"))
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    q = np.quantile(x, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    return {
        "min": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "max": float(q[6]),
        "mean": float(np.mean(x)),
    }


def as_str_array(x):
    return np.asarray(x).astype(str)


def resolve_pose_path(raw_root: Path, trajectory_id: str) -> Path:
    # Expected trajectory_id format: "environment/Hard/P000"
    parts = trajectory_id.split("/")

    candidates = []
    if len(parts) >= 3:
        env, diff, traj = parts[0], parts[1], parts[2]
        candidates.extend([
            raw_root / env / diff / env / diff / traj / "pose_left.txt",
            raw_root / env / diff / traj / "pose_left.txt",
            raw_root / trajectory_id / "pose_left.txt",
        ])
    else:
        candidates.append(raw_root / trajectory_id / "pose_left.txt")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find pose_left.txt for trajectory_id={trajectory_id}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_pose_cache(raw_root: Path, trajectory_ids: np.ndarray) -> dict[str, np.ndarray]:
    cache = {}
    for tid in sorted(set(trajectory_ids.tolist())):
        path = resolve_pose_path(raw_root, tid)
        arr = np.loadtxt(path).astype(np.float32)

        if arr.ndim == 1:
            arr = arr[None, :]

        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError(f"Bad pose file shape for {path}: {arr.shape}")

        cache[tid] = arr

    return cache


def safe_pose(cache: dict[str, np.ndarray], tid: str, frame: int) -> np.ndarray:
    arr = cache[tid]
    idx = int(frame)

    if idx < 0:
        idx = 0
    if idx >= len(arr):
        idx = len(arr) - 1

    return arr[idx]


def quat_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q1.astype(np.float64)
    q2 = q2.astype(np.float64)

    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)

    # q and -q represent the same rotation.
    return float(1.0 - abs(np.dot(q1, q2)))


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data, allow_pickle=True)
    groups = np.load(args.groups, allow_pickle=True)

    candidate_indices = groups["candidate_indices"].astype(np.int64)
    n_groups, n_candidates = candidate_indices.shape

    trajectory_id = as_str_array(data["trajectory_id"])
    frame_i = np.asarray(data["frame_i"]).astype(int)
    frame_j = np.asarray(data["frame_j"]).astype(int)

    pose_cache = load_pose_cache(args.raw_root, trajectory_id)

    pos_dists = []
    vel_dists = []
    quat_dists = []

    for row in candidate_indices:
        anchor = int(row[0])
        anchor_tid = trajectory_id[anchor]

        anchor_pose_i = safe_pose(pose_cache, anchor_tid, frame_i[anchor])
        anchor_pose_j = safe_pose(pose_cache, anchor_tid, frame_j[anchor])

        anchor_pos = anchor_pose_i[:3]
        anchor_vel = anchor_pose_j[:3] - anchor_pose_i[:3]
        anchor_quat = anchor_pose_i[3:7] if anchor_pose_i.shape[0] >= 7 else None

        for cand in row[1:]:
            cand = int(cand)
            cand_tid = trajectory_id[cand]

            cand_pose_i = safe_pose(pose_cache, cand_tid, frame_i[cand])
            cand_pose_j = safe_pose(pose_cache, cand_tid, frame_j[cand])

            cand_pos = cand_pose_i[:3]
            cand_vel = cand_pose_j[:3] - cand_pose_i[:3]

            pos_dists.append(float(np.linalg.norm(anchor_pos - cand_pos)))
            vel_dists.append(float(np.linalg.norm(anchor_vel - cand_vel)))

            if anchor_quat is not None and cand_pose_i.shape[0] >= 7:
                cand_quat = cand_pose_i[3:7]
                quat_dists.append(quat_distance(anchor_quat, cand_quat))

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "n_groups": int(n_groups),
        "candidates": int(n_candidates),
        "pose_position_distance": qstats(pos_dists),
        "pose_velocity_distance": qstats(vel_dists),
        "quat_dot_distance": qstats(quat_dists),
    }

    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md_path = args.out.with_suffix(".md")
    lines = [
        "# Pose audit for matched-state groups",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- group count: `{n_groups}`",
        f"- candidates: `{n_candidates}`",
        "",
        "| metric | mean | p05 | p25 | median | p75 | p95 | max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, name in [
        ("pose_position_distance", "position distance"),
        ("pose_velocity_distance", "velocity distance"),
        ("quat_dot_distance", "quaternion dot distance"),
    ]:
        q = report[key]
        if q is None:
            continue

        lines.append(
            f"| {name} | {q['mean']:.6f} | {q['p05']:.6f} | {q['p25']:.6f} | "
            f"{q['median']:.6f} | {q['p75']:.6f} | {q['p95']:.6f} | {q['max']:.6f} |"
        )

    md_path.write_text("\n".join(lines) + "\n")

    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()
