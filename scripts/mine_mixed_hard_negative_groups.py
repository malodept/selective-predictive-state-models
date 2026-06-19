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
    p.add_argument("--same-state-negatives", type=int, default=2)
    p.add_argument("--max-pos-dist", type=float, default=0.15)
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
    delta = z1 - z0

    all_idx = np.arange(len(action_id))
    movable = all_idx[action_id != 0]
    rng.shuffle(movable)

    same_state_needed = int(args.same_state_negatives)
    same_action_needed = int(args.candidates - 1 - same_state_needed)
    assert same_action_needed >= 1

    anchor_indices = []
    candidate_indices = []
    negative_type = []

    pos_dists = []
    future_delta_dists = []
    blocked_mismatch = []

    for anchor in movable:
        aid = action_id[anchor]
        gid = group_id[anchor]

        same_state = all_idx[(group_id == gid) & (action_id != aid)]
        if len(same_state) < same_state_needed:
            continue
        same_state_chosen = rng.choice(same_state, size=same_state_needed, replace=False).tolist()

        same_action = all_idx[(action_id == aid) & (group_id != gid)]
        dist_xy = np.linalg.norm(state_xy[same_action] - state_xy[anchor], axis=1)
        close = same_action[dist_xy <= args.max_pos_dist]

        if len(close) < same_action_needed:
            continue

        opp = close[blocked[close] != blocked[anchor]]
        same = close[blocked[close] == blocked[anchor]]

        chosen_same_action = []
        if len(opp) > 0:
            order = np.argsort(np.linalg.norm(state_xy[opp] - state_xy[anchor], axis=1))
            chosen_same_action.extend(opp[order].tolist())

        if len(chosen_same_action) < same_action_needed and len(same) > 0:
            order = np.argsort(np.linalg.norm(state_xy[same] - state_xy[anchor], axis=1))
            chosen_same_action.extend(same[order].tolist())

        chosen_same_action = chosen_same_action[:same_action_needed]
        if len(chosen_same_action) < same_action_needed:
            continue

        cand = [int(anchor)] + [int(x) for x in same_state_chosen] + [int(x) for x in chosen_same_action]
        types = ["positive"] + ["same_state_diff_action"] * same_state_needed + ["same_action_diff_state"] * same_action_needed

        anchor_indices.append(int(anchor))
        candidate_indices.append(cand)
        negative_type.append(types)

        for j in cand[1:]:
            pos_dists.append(float(np.linalg.norm(state_xy[j] - state_xy[anchor])))
            future_delta_dists.append(float(np.linalg.norm(delta[j] - delta[anchor])))
            blocked_mismatch.append(float(blocked[j] != blocked[anchor]))

        if len(anchor_indices) >= args.groups:
            break

    anchor_indices = np.asarray(anchor_indices, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    negative_type = np.asarray(negative_type)

    if len(anchor_indices) == 0:
        raise RuntimeError("No mixed hard-negative groups mined.")

    np.savez_compressed(
        args.out,
        anchor_indices=anchor_indices,
        candidate_indices=candidate_indices,
        correct_candidate=np.zeros(len(anchor_indices), dtype=np.int64),
        negative_type=negative_type,
        action=action[anchor_indices],
        action_id=action_id[anchor_indices],
        anchor_blocked=blocked[anchor_indices].astype(np.int64),
        candidate_blocked=blocked[candidate_indices].astype(np.int64),
        anchor_state_xy=state_xy[anchor_indices],
        candidate_state_xy=state_xy[candidate_indices],
        source_npz=np.asarray(str(args.data)),
        mining_version=np.asarray("mixed_hard_negatives_v0"),
        candidates=np.asarray(args.candidates),
        same_state_negatives=np.asarray(same_state_needed),
        same_action_negatives=np.asarray(same_action_needed),
        max_pos_dist=np.asarray(args.max_pos_dist),
    )

    report = {
        "data": str(args.data),
        "out": str(args.out),
        "groups_requested": int(args.groups),
        "groups_mined": int(len(anchor_indices)),
        "candidates": int(args.candidates),
        "chance_top1": float(1.0 / args.candidates),
        "same_state_diff_action_negatives": int(same_state_needed),
        "same_action_diff_state_negatives": int(same_action_needed),
        "max_pos_dist": float(args.max_pos_dist),
        "anchor_blocked_fraction": float(blocked[anchor_indices].mean()),
        "offdiag_blocked_mismatch_fraction": float(np.mean(blocked_mismatch)),
        "offdiag_pos_dist": qstats(pos_dists),
        "offdiag_delta_dist": qstats(future_delta_dists),
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# Mixed hard-negative group mining",
        "",
        f"- data: `{args.data}`",
        f"- output: `{args.out}`",
        f"- groups requested: `{args.groups}`",
        f"- groups mined: `{len(anchor_indices)}`",
        f"- candidates per group: `{args.candidates}`",
        f"- chance top-1: `{1.0 / args.candidates:.6f}`",
        f"- same-state different-action negatives: `{same_state_needed}`",
        f"- same-action different-state negatives: `{same_action_needed}`",
        f"- max position distance: `{args.max_pos_dist}`",
        f"- anchor blocked fraction: `{report['anchor_blocked_fraction']:.6f}`",
        f"- offdiag blocked-mismatch fraction: `{report['offdiag_blocked_mismatch_fraction']:.6f}`",
        "",
        "## Off-diagonal distributions",
        "",
        "| quantity | mean | p05 | median | p95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| start position distance | {report['offdiag_pos_dist']['mean']:.6f} | {report['offdiag_pos_dist']['p05']:.6f} | {report['offdiag_pos_dist']['median']:.6f} | {report['offdiag_pos_dist']['p95']:.6f} |",
        f"| DINOv2 delta distance | {report['offdiag_delta_dist']['mean']:.6f} | {report['offdiag_delta_dist']['p05']:.6f} | {report['offdiag_delta_dist']['median']:.6f} | {report['offdiag_delta_dist']['p95']:.6f} |",
        "",
        "## Interpretation",
        "",
        "Each group mixes same-state/different-action negatives and same-action/different-state negatives.",
        "This is designed so that action-only and state-only models should both fail, while a full state-action model should succeed.",
    ]

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
