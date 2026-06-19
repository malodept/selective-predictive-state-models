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
    p.add_argument("--groups", type=int, default=1000)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--max-pos-dist", type=float, default=0.15)
    p.add_argument("--prefer-opposite-blocked", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def qstats(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"mean": None, "p05": None, "median": None, "p95": None}
    return {
        "mean": float(x.mean()),
        "p05": float(np.quantile(x, 0.05)),
        "median": float(np.median(x)),
        "p95": float(np.quantile(x, 0.95)),
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)

    action_id = d["action_id"].astype(np.int64)
    group_id = d["group_id"].astype(np.int64)
    state_xy = d["state_xy"].astype(np.float32)
    blocked = d["blocked_mask"].astype(bool)
    action = d["action"].astype(np.float32)

    z0 = d["z_current"].astype(np.float32).reshape(len(action_id), -1)
    z1 = d["z_future"].astype(np.float32).reshape(len(action_id), -1)

    n = len(action_id)
    all_indices = np.arange(n)

    anchor_indices = []
    candidate_indices = []

    pos_dists = []
    future_dists = []
    blocked_mismatch = []

    # Exclude stay anchors because it has no blocked/unblocked contrast.
    movable = all_indices[action_id != 0]
    rng.shuffle(movable)

    for anchor in movable:
        aid = action_id[anchor]

        same_action = all_indices[action_id == aid]
        same_action = same_action[group_id[same_action] != group_id[anchor]]

        if len(same_action) < args.candidates - 1:
            continue

        dist_xy = np.linalg.norm(state_xy[same_action] - state_xy[anchor], axis=1)
        close = same_action[dist_xy <= args.max_pos_dist]

        if len(close) < args.candidates - 1:
            continue

        if args.prefer_opposite_blocked:
            opp = close[blocked[close] != blocked[anchor]]
            same = close[blocked[close] == blocked[anchor]]

            chosen = []
            if len(opp) > 0:
                order = np.argsort(np.linalg.norm(state_xy[opp] - state_xy[anchor], axis=1))
                chosen.extend(opp[order].tolist())

            if len(chosen) < args.candidates - 1 and len(same) > 0:
                order = np.argsort(np.linalg.norm(state_xy[same] - state_xy[anchor], axis=1))
                chosen.extend(same[order].tolist())

            chosen = chosen[: args.candidates - 1]
        else:
            order = np.argsort(np.linalg.norm(state_xy[close] - state_xy[anchor], axis=1))
            chosen = close[order[: args.candidates - 1]].tolist()

        if len(chosen) != args.candidates - 1:
            continue

        cand = [int(anchor)] + [int(x) for x in chosen]

        anchor_indices.append(int(anchor))
        candidate_indices.append(cand)

        pd = np.linalg.norm(state_xy[np.asarray(cand[1:])] - state_xy[anchor], axis=1)
        fd = np.linalg.norm(z1[np.asarray(cand[1:])] - z1[anchor], axis=1)
        bm = blocked[np.asarray(cand[1:])] != blocked[anchor]

        pos_dists.extend(pd.tolist())
        future_dists.extend(fd.tolist())
        blocked_mismatch.extend(bm.astype(np.float32).tolist())

        if len(anchor_indices) >= args.groups:
            break

    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    correct = np.zeros(len(anchor_indices), dtype=np.int64)

    if len(anchor_indices) == 0:
        raise RuntimeError("No groups mined. Relax --max-pos-dist or reduce --groups.")

    np.savez_compressed(
        args.out,
        anchor_indices=anchor_indices,
        candidate_indices=candidate_indices,
        correct_candidate=correct,
        action=action[anchor_indices],
        action_id=action_id[anchor_indices],
        anchor_blocked=blocked[anchor_indices].astype(np.int64),
        candidate_blocked=blocked[candidate_indices].astype(np.int64),
        anchor_state_xy=state_xy[anchor_indices],
        candidate_state_xy=state_xy[candidate_indices],
        source_npz=np.asarray(str(args.data)),
        mining_version=np.asarray("same_action_hard_negatives_v0"),
        candidates=np.asarray(args.candidates),
        max_pos_dist=np.asarray(args.max_pos_dist),
    )

    mismatch = np.asarray(blocked_mismatch, dtype=np.float32)

    report = {
        "data": str(args.data),
        "out": str(args.out),
        "groups_requested": int(args.groups),
        "groups_mined": int(len(anchor_indices)),
        "candidates": int(args.candidates),
        "chance_top1": float(1.0 / args.candidates),
        "max_pos_dist": float(args.max_pos_dist),
        "prefer_opposite_blocked": bool(args.prefer_opposite_blocked),
        "pos_dist": qstats(pos_dists),
        "future_dist": qstats(future_dists),
        "blocked_mismatch_fraction": float(mismatch.mean()) if len(mismatch) else None,
        "anchor_blocked_fraction": float(blocked[anchor_indices].mean()),
        "action_counts": {
            str(aid): int((action_id[anchor_indices] == aid).sum())
            for aid in sorted(set(action_id[anchor_indices].tolist()))
        },
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# Same-action hard-negative group mining",
        "",
        f"- data: `{args.data}`",
        f"- output: `{args.out}`",
        f"- groups requested: `{args.groups}`",
        f"- groups mined: `{len(anchor_indices)}`",
        f"- candidates per group: `{args.candidates}`",
        f"- chance top-1: `{1.0 / args.candidates:.6f}`",
        f"- max position distance: `{args.max_pos_dist}`",
        f"- prefer opposite blocked: `{args.prefer_opposite_blocked}`",
        f"- anchor blocked fraction: `{report['anchor_blocked_fraction']:.6f}`",
        f"- offdiag blocked-mismatch fraction: `{report['blocked_mismatch_fraction']:.6f}`",
        "",
        "## Off-diagonal distributions",
        "",
        "| quantity | mean | p05 | median | p95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| start position distance | {report['pos_dist']['mean']:.6f} | {report['pos_dist']['p05']:.6f} | {report['pos_dist']['median']:.6f} | {report['pos_dist']['p95']:.6f} |",
        f"| DINOv2 future distance | {report['future_dist']['mean']:.6f} | {report['future_dist']['p05']:.6f} | {report['future_dist']['median']:.6f} | {report['future_dist']['p95']:.6f} |",
        "",
        "## Interpretation",
        "",
        "All candidates in a group share the same action. Therefore an action-only model should not be able to identify the correct future.",
        "The groups are hard because off-diagonal candidates are selected from nearby starting positions and preferably opposite blocked/unblocked conditions.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
