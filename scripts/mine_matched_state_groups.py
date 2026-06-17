from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)

    p.add_argument("--groups", type=int, default=8000)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--pool-size", type=int, default=30000)
    p.add_argument("--state-proj-dim", type=int, default=64)
    p.add_argument("--nearest-pool", type=int, default=256)

    p.add_argument("--min-action-dist", type=float, default=0.25)
    p.add_argument("--min-future-dist", type=float, default=0.02)
    p.add_argument("--allow-same-trajectory", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def as_str_array(x):
    return np.asarray(x).astype(str)


def l2_normalize(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def quantiles(x):
    if len(x) == 0:
        return {}
    q = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
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


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    action = d["action"].astype(np.float32)

    n = z.shape[0]
    tokens = z.shape[1]
    dim = z.shape[2]

    traj = as_str_array(d["trajectory_id"]) if "trajectory_id" in d.files else np.asarray([""] * n)
    env = as_str_array(d["environment"]) if "environment" in d.files else np.asarray([""] * n)

    frame_i = np.asarray(d["frame_i"]) if "frame_i" in d.files else np.arange(n)
    frame_j = np.asarray(d["frame_j"]) if "frame_j" in d.files else np.arange(n)

    # Latent state descriptor: mean-pool tokens, standardize, random project, L2 normalize.
    state = z.mean(axis=1)
    future = y.mean(axis=1)

    state_mu = state.mean(axis=0, keepdims=True)
    state_sd = state.std(axis=0, keepdims=True) + 1e-6
    state_std = (state - state_mu) / state_sd

    future_mu = future.mean(axis=0, keepdims=True)
    future_sd = future.std(axis=0, keepdims=True) + 1e-6
    future_std = (future - future_mu) / future_sd

    act_mu = action.mean(axis=0, keepdims=True)
    act_sd = action.std(axis=0, keepdims=True) + 1e-6
    action_std = (action - act_mu) / act_sd

    proj_dim = min(args.state_proj_dim, state_std.shape[1])
    rp = rng.normal(size=(state_std.shape[1], proj_dim)).astype(np.float32)
    rp /= np.sqrt(proj_dim)

    state_desc = l2_normalize(state_std @ rp)

    pool_size = min(args.pool_size, n)
    pool_idx = rng.choice(n, size=pool_size, replace=False)
    pool_desc = state_desc[pool_idx]

    anchor_order = rng.permutation(n)

    group_anchor = []
    group_candidates = []
    correct_candidate = []

    group_latent_dist = []
    group_action_dist = []
    group_future_dist = []
    group_same_traj = []
    group_same_env = []

    for anchor in anchor_order:
        if len(group_anchor) >= args.groups:
            break

        a_desc = state_desc[anchor : anchor + 1]
        dist = np.sum((pool_desc - a_desc) ** 2, axis=1)

        nn_local = np.argpartition(dist, kth=min(args.nearest_pool, len(dist) - 1))[: args.nearest_pool]
        cand_pool = pool_idx[nn_local]
        cand_dist = dist[nn_local]

        # Remove anchor itself.
        valid = cand_pool != anchor

        if not args.allow_same_trajectory:
            valid &= traj[cand_pool] != traj[anchor]

        # Require action difference.
        ad = np.linalg.norm(action_std[cand_pool] - action_std[anchor], axis=1)
        valid &= ad >= args.min_action_dist

        # Require future separation.
        fd = np.linalg.norm(future_std[cand_pool] - future_std[anchor], axis=1)
        valid &= fd >= args.min_future_dist

        cand_pool = cand_pool[valid]
        cand_dist = cand_dist[valid]
        ad = ad[valid]
        fd = fd[valid]

        need = args.candidates - 1
        if len(cand_pool) < need:
            continue

        order = np.argsort(cand_dist)
        chosen = cand_pool[order[:need]]

        candidates = np.concatenate([[anchor], chosen]).astype(np.int64)

        group_anchor.append(anchor)
        group_candidates.append(candidates)
        correct_candidate.append(0)

        group_latent_dist.append(
            np.concatenate([[0.0], np.sqrt(cand_dist[order[:need]])]).astype(np.float32)
        )
        group_action_dist.append(
            np.concatenate([[0.0], ad[order[:need]]]).astype(np.float32)
        )
        group_future_dist.append(
            np.concatenate([[0.0], fd[order[:need]]]).astype(np.float32)
        )
        group_same_traj.append((traj[candidates] == traj[anchor]).astype(np.int8))
        group_same_env.append((env[candidates] == env[anchor]).astype(np.int8))

    if len(group_anchor) == 0:
        raise RuntimeError("No groups mined. Relax thresholds.")

    group_anchor = np.asarray(group_anchor, dtype=np.int64)
    group_candidates = np.stack(group_candidates).astype(np.int64)
    correct_candidate = np.asarray(correct_candidate, dtype=np.int64)

    group_latent_dist = np.stack(group_latent_dist)
    group_action_dist = np.stack(group_action_dist)
    group_future_dist = np.stack(group_future_dist)
    group_same_traj = np.stack(group_same_traj)
    group_same_env = np.stack(group_same_env)

    np.savez_compressed(
        args.out,
        anchor_index=group_anchor,
        candidate_indices=group_candidates,
        correct_candidate=correct_candidate,
        latent_distances=group_latent_dist,
        action_distances=group_action_dist,
        future_distances=group_future_dist,
        same_trajectory=group_same_traj,
        same_environment=group_same_env,
        anchor_environment=env[group_anchor],
        anchor_trajectory=traj[group_anchor],
        candidate_environments=env[group_candidates],
        candidate_trajectories=traj[group_candidates],
        frame_i=frame_i[group_candidates],
        frame_j=frame_j[group_candidates],
        source_npz=str(args.data),
        candidates=args.candidates,
        groups=len(group_anchor),
        mining_version="tartanair_silver_latent_only_v0",
        min_action_dist=args.min_action_dist,
        min_future_dist=args.min_future_dist,
        allow_same_trajectory=args.allow_same_trajectory,
    )

    offdiag = np.ones_like(group_latent_dist, dtype=bool)
    offdiag[:, 0] = False

    report = {
        "data": str(args.data),
        "out": str(args.out),
        "n_samples": int(n),
        "tokens": int(tokens),
        "dim": int(dim),
        "groups": int(len(group_anchor)),
        "candidates": int(args.candidates),
        "chance_top1": float(1.0 / args.candidates),
        "pool_size": int(pool_size),
        "state_proj_dim": int(proj_dim),
        "nearest_pool": int(args.nearest_pool),
        "min_action_dist": float(args.min_action_dist),
        "min_future_dist": float(args.min_future_dist),
        "allow_same_trajectory": bool(args.allow_same_trajectory),
        "latent_distance_offdiag": quantiles(group_latent_dist[offdiag]),
        "action_distance_offdiag": quantiles(group_action_dist[offdiag]),
        "future_distance_offdiag": quantiles(group_future_dist[offdiag]),
        "same_trajectory_fraction_offdiag": float(group_same_traj[offdiag].mean()),
        "same_environment_fraction_offdiag": float(group_same_env[offdiag].mean()),
        "unique_anchor_trajectories": int(len(set(traj[group_anchor].tolist()))),
        "unique_anchor_environments": int(len(set(env[group_anchor].tolist()))),
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# TartanAir silver matched-state group audit",
        "",
        f"- data: `{args.data}`",
        f"- output: `{args.out}`",
        f"- mining version: `tartanair_silver_latent_only_v0`",
        f"- samples: `{n}`",
        f"- groups: `{len(group_anchor)}`",
        f"- candidates per group: `{args.candidates}`",
        f"- chance top-1: `{1.0 / args.candidates:.6f}`",
        f"- pool size: `{pool_size}`",
        f"- state projection dim: `{proj_dim}`",
        f"- nearest pool: `{args.nearest_pool}`",
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
        "## Off-diagonal candidate distributions",
        "",
        "| metric | mean | p05 | p25 | median | p75 | p95 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, name in [
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

    print(args.out)
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
