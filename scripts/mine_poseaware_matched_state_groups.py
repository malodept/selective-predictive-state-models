from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, default=Path("data/tartanair_raw"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)

    p.add_argument("--groups", type=int, default=8000)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--pool-size", type=int, default=50000)
    p.add_argument("--per-anchor-pool", type=int, default=5000)

    p.add_argument("--max-pos-dist", type=float, default=5.0)
    p.add_argument("--max-vel-dist", type=float, default=1.0)
    p.add_argument("--max-quat-dist", type=float, default=0.15)
    p.add_argument("--min-action-dist", type=float, default=0.25)
    p.add_argument("--min-future-dist", type=float, default=0.02)

    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def as_str_array(x):
    return np.asarray(x).astype(str)


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


def resolve_pose_path(raw_root: Path, trajectory_id: str) -> Path:
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

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"No pose_left.txt for {trajectory_id}")


def load_pose_cache(raw_root: Path, trajectory_ids: np.ndarray):
    cache = {}
    for tid in sorted(set(trajectory_ids.tolist())):
        p = resolve_pose_path(raw_root, tid)
        arr = np.loadtxt(p).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        cache[tid] = arr
    return cache


def safe_pose(cache, tid, frame):
    arr = cache[tid]
    idx = int(frame)
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


def quat_dist(q1, q2):
    q1 = q1.astype(np.float64)
    q2 = q2.astype(np.float64)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    q2 = q2 / (np.linalg.norm(q2) + 1e-12)
    return float(1.0 - abs(np.dot(q1, q2)))


def l2_standardize(x):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def subset_rows(arr, idx):
    return arr[idx]


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    action = d["action"].astype(np.float32)

    traj = as_str_array(d["trajectory_id"])
    env = as_str_array(d["environment"]) if "environment" in d.files else np.asarray([""] * len(action))
    frame_i = np.asarray(d["frame_i"]).astype(int)
    frame_j = np.asarray(d["frame_j"]).astype(int)

    n = len(action)
    cache = load_pose_cache(args.raw_root, traj)

    pos = np.zeros((n, 3), dtype=np.float32)
    vel = np.zeros((n, 3), dtype=np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        p0 = safe_pose(cache, traj[i], frame_i[i])
        p1 = safe_pose(cache, traj[i], frame_j[i])

        pos[i] = p0[:3]
        vel[i] = p1[:3] - p0[:3]

        if len(p0) >= 7:
            q = p0[3:7]
            quat[i] = q / (np.linalg.norm(q) + 1e-8)
        else:
            quat[i] = np.array([1, 0, 0, 0], dtype=np.float32)

    future = y.mean(axis=1)
    future_std = l2_standardize(future)
    action_std = l2_standardize(action)

    pool_size = min(args.pool_size, n)
    anchor_order = rng.permutation(n)

    group_anchor = []
    group_candidates = []

    latent_distances = []
    action_distances = []
    future_distances = []
    pos_distances = []
    vel_distances = []
    quat_distances = []
    same_traj = []
    same_env = []

    # Pre-index by environment for speed and to avoid cross-map pose comparisons.
    env_to_indices = {}
    for e in sorted(set(env.tolist())):
        idx = np.where(env == e)[0]
        if len(idx) > pool_size:
            idx = rng.choice(idx, size=pool_size, replace=False)
        env_to_indices[e] = idx

    for ii, anchor in enumerate(anchor_order):
        if ii % 1000 == 0:
            print(f"scan anchors={ii}/{len(anchor_order)} groups={len(group_anchor)}", flush=True)

        if len(group_anchor) >= args.groups:
            break

        e = env[anchor]
        pool = env_to_indices[e]

        # different trajectory, same environment
        valid = traj[pool] != traj[anchor]
        cand_pool = pool[valid]

        if len(cand_pool) < args.candidates - 1:
            continue

        if len(cand_pool) > args.per_anchor_pool:
            cand_pool = rng.choice(cand_pool, size=args.per_anchor_pool, replace=False)

        pd = np.linalg.norm(pos[cand_pool] - pos[anchor], axis=1)
        vd = np.linalg.norm(vel[cand_pool] - vel[anchor], axis=1)
        qd = np.asarray([quat_dist(quat[anchor], quat[j]) for j in cand_pool], dtype=np.float32)
        ad = np.linalg.norm(action_std[cand_pool] - action_std[anchor], axis=1)
        fd = np.linalg.norm(future_std[cand_pool] - future_std[anchor], axis=1)

        keep = (
            (pd <= args.max_pos_dist)
            & (vd <= args.max_vel_dist)
            & (qd <= args.max_quat_dist)
            & (ad >= args.min_action_dist)
            & (fd >= args.min_future_dist)
        )

        cand_pool = cand_pool[keep]
        pd = pd[keep]
        vd = vd[keep]
        qd = qd[keep]
        ad = ad[keep]
        fd = fd[keep]

        need = args.candidates - 1
        if len(cand_pool) < need:
            continue

        # Prefer closest pose, then closest velocity.
        score = pd + 0.5 * vd + 2.0 * qd
        order = np.argsort(score)[:need]
        chosen = cand_pool[order]

        candidates = np.concatenate([[anchor], chosen]).astype(np.int64)

        group_anchor.append(anchor)
        group_candidates.append(candidates)

        zmean = z.mean(axis=1)
        ld = np.linalg.norm(zmean[candidates] - zmean[anchor], axis=1)

        latent_distances.append(ld.astype(np.float32))
        action_distances.append(np.concatenate([[0.0], ad[order]]).astype(np.float32))
        future_distances.append(np.concatenate([[0.0], fd[order]]).astype(np.float32))
        pos_distances.append(np.concatenate([[0.0], pd[order]]).astype(np.float32))
        vel_distances.append(np.concatenate([[0.0], vd[order]]).astype(np.float32))
        quat_distances.append(np.concatenate([[0.0], qd[order]]).astype(np.float32))
        same_traj.append((traj[candidates] == traj[anchor]).astype(np.int8))
        same_env.append((env[candidates] == env[anchor]).astype(np.int8))

    if len(group_anchor) == 0:
        raise RuntimeError("No pose-aware groups mined. Relax thresholds.")

    group_anchor = np.asarray(group_anchor, dtype=np.int64)
    group_candidates = np.stack(group_candidates).astype(np.int64)

    latent_distances = np.stack(latent_distances)
    action_distances = np.stack(action_distances)
    future_distances = np.stack(future_distances)
    pos_distances = np.stack(pos_distances)
    vel_distances = np.stack(vel_distances)
    quat_distances = np.stack(quat_distances)
    same_traj = np.stack(same_traj)
    same_env = np.stack(same_env)

    np.savez_compressed(
        args.out,
        anchor_index=group_anchor,
        candidate_indices=group_candidates,
        correct_candidate=np.zeros(len(group_anchor), dtype=np.int64),
        latent_distances=latent_distances,
        action_distances=action_distances,
        future_distances=future_distances,
        pose_position_distances=pos_distances,
        pose_velocity_distances=vel_distances,
        pose_quat_distances=quat_distances,
        same_trajectory=same_traj,
        same_environment=same_env,
        anchor_environment=env[group_anchor],
        anchor_trajectory=traj[group_anchor],
        candidate_environments=env[group_candidates],
        candidate_trajectories=traj[group_candidates],
        frame_i=frame_i[group_candidates],
        frame_j=frame_j[group_candidates],
        source_npz=str(args.data),
        mining_version="tartanair_silver_poseaware_v0",
        candidates=args.candidates,
        groups=len(group_anchor),
        max_pos_dist=args.max_pos_dist,
        max_vel_dist=args.max_vel_dist,
        max_quat_dist=args.max_quat_dist,
        min_action_dist=args.min_action_dist,
        min_future_dist=args.min_future_dist,
    )

    off = np.ones_like(pos_distances, dtype=bool)
    off[:, 0] = False

    report = {
        "data": str(args.data),
        "out": str(args.out),
        "groups": int(len(group_anchor)),
        "candidates": int(args.candidates),
        "chance_top1": float(1.0 / args.candidates),
        "max_pos_dist": float(args.max_pos_dist),
        "max_vel_dist": float(args.max_vel_dist),
        "max_quat_dist": float(args.max_quat_dist),
        "min_action_dist": float(args.min_action_dist),
        "min_future_dist": float(args.min_future_dist),
        "same_trajectory_fraction_offdiag": float(same_traj[off].mean()),
        "same_environment_fraction_offdiag": float(same_env[off].mean()),
        "unique_anchor_trajectories": int(len(set(traj[group_anchor].tolist()))),
        "unique_anchor_environments": int(len(set(env[group_anchor].tolist()))),
        "position_distance_offdiag": qstats(pos_distances[off]),
        "velocity_distance_offdiag": qstats(vel_distances[off]),
        "quat_distance_offdiag": qstats(quat_distances[off]),
        "latent_distance_offdiag": qstats(latent_distances[off]),
        "action_distance_offdiag": qstats(action_distances[off]),
        "future_distance_offdiag": qstats(future_distances[off]),
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# TartanAir silver pose-aware group audit",
        "",
        f"- data: `{args.data}`",
        f"- output: `{args.out}`",
        f"- groups: `{len(group_anchor)}`",
        f"- candidates per group: `{args.candidates}`",
        f"- chance top-1: `{1.0 / args.candidates:.6f}`",
        f"- max position distance: `{args.max_pos_dist}`",
        f"- max velocity distance: `{args.max_vel_dist}`",
        f"- max quaternion distance: `{args.max_quat_dist}`",
        f"- min action distance: `{args.min_action_dist}`",
        f"- min future distance: `{args.min_future_dist}`",
        "",
        "## Quality-control summary",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| same trajectory fraction offdiag | {report['same_trajectory_fraction_offdiag']:.6f} |",
        f"| same environment fraction offdiag | {report['same_environment_fraction_offdiag']:.6f} |",
        f"| unique anchor trajectories | {report['unique_anchor_trajectories']} |",
        f"| unique anchor environments | {report['unique_anchor_environments']} |",
        "",
        "## Off-diagonal distributions",
        "",
        "| metric | mean | p05 | p25 | median | p75 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, name in [
        ("position_distance_offdiag", "position distance"),
        ("velocity_distance_offdiag", "velocity distance"),
        ("quat_distance_offdiag", "quaternion distance"),
        ("latent_distance_offdiag", "latent distance"),
        ("action_distance_offdiag", "action distance"),
        ("future_distance_offdiag", "future distance"),
    ]:
        q = report[key]
        lines.append(
            f"| {name} | {q['mean']:.6f} | {q['p05']:.6f} | {q['p25']:.6f} | "
            f"{q['median']:.6f} | {q['p75']:.6f} | {q['p95']:.6f} |"
        )

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
