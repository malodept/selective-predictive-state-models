from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


NEG_CORRECT = "correct"
NEG_SAME_STATE = "same_state_diff_action"
NEG_SAME_ACTION = "same_action_diff_state"


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return {"mean": None, "p05": None, "median": None, "p95": None}
    return {
        "mean": float(x.mean()),
        "p05": float(np.quantile(x, 0.05)),
        "median": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
    }


def split_states(states, seed, train_frac=0.8, val_frac=0.1):
    rng = np.random.default_rng(seed)
    states = np.asarray(states, dtype=np.int64)
    perm = rng.permutation(states)

    n = len(perm)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    return {
        "train": set(map(int, perm[:n_train])),
        "val": set(map(int, perm[n_train:n_train + n_val])),
        "test": set(map(int, perm[n_train + n_val:])),
    }


def mine_split(
    *,
    split_name,
    split_states_set,
    row_by_state_action,
    rows_by_split_action,
    state_xy_by_state,
    group_id,
    action_id,
    action,
    blocked,
    n_groups,
    same_state_negatives,
    same_action_negatives,
    max_pos_dist,
    prefer_block_mismatch,
    rng,
):
    split_rows = []
    for (s, a), r in row_by_state_action.items():
        if s in split_states_set:
            split_rows.append(r)
    split_rows = np.asarray(split_rows, dtype=np.int64)

    anchors_out = []
    candidates_out = []
    correct_out = []
    negative_type_out = []
    action_out = []
    action_id_out = []
    anchor_blocked_out = []
    candidate_blocked_out = []
    anchor_xy_out = []
    candidate_xy_out = []
    split_out = []

    start_dists = []
    blocked_mismatch = []

    max_attempts = max(10000, n_groups * 100)
    attempts = 0

    while len(anchors_out) < n_groups and attempts < max_attempts:
        attempts += 1

        anchor = int(rng.choice(split_rows))
        s = int(group_id[anchor])
        a = int(action_id[anchor])

        # Same state, different actions.
        same_state_pool = []
        for (ss, aa), rr in row_by_state_action.items():
            if ss == s and aa != a:
                same_state_pool.append(rr)

        if len(same_state_pool) < same_state_negatives:
            continue

        same_state_chosen = rng.choice(
            np.asarray(same_state_pool, dtype=np.int64),
            size=same_state_negatives,
            replace=False,
        )

        # Same action, different state, same split.
        pool = rows_by_split_action[(split_name, a)]
        pool = pool[group_id[pool] != s]

        if len(pool) < same_action_negatives:
            continue

        anchor_xy = state_xy_by_state[s]
        pool_xy = np.asarray([state_xy_by_state[int(group_id[r])] for r in pool], dtype=np.float32)
        dpos = np.linalg.norm(pool_xy - anchor_xy[None], axis=1)
        close_mask = dpos <= max_pos_dist

        close_pool = pool[close_mask]
        close_dpos = dpos[close_mask]

        if len(close_pool) < same_action_negatives:
            continue

        if prefer_block_mismatch:
            mismatch_mask = blocked[close_pool] != blocked[anchor]
            preferred_pool = close_pool[mismatch_mask]
            preferred_dpos = close_dpos[mismatch_mask]
            if len(preferred_pool) >= same_action_negatives:
                order = np.argsort(preferred_dpos)
                top = preferred_pool[order[: min(len(order), 100)]]
            else:
                order = np.argsort(close_dpos)
                top = close_pool[order[: min(len(order), 100)]]
        else:
            order = np.argsort(close_dpos)
            top = close_pool[order[: min(len(order), 100)]]

        if len(top) < same_action_negatives:
            continue

        same_action_chosen = rng.choice(top, size=same_action_negatives, replace=False)

        cand = np.concatenate([
            np.asarray([anchor], dtype=np.int64),
            same_state_chosen.astype(np.int64),
            same_action_chosen.astype(np.int64),
        ])

        if len(cand) != 1 + same_state_negatives + same_action_negatives:
            continue

        neg_types = [NEG_CORRECT]
        neg_types += [NEG_SAME_STATE] * same_state_negatives
        neg_types += [NEG_SAME_ACTION] * same_action_negatives

        anchors_out.append(anchor)
        candidates_out.append(cand)
        correct_out.append(0)
        negative_type_out.append(neg_types)
        action_out.append(action[anchor])
        action_id_out.append(a)
        anchor_blocked_out.append(blocked[anchor])
        candidate_blocked_out.append(blocked[cand])
        anchor_xy_out.append(anchor_xy)
        candidate_xy_out.append(np.asarray([state_xy_by_state[int(group_id[r])] for r in cand], dtype=np.float32))
        split_out.append(split_name)

        for r in cand[1:]:
            start_dists.append(float(np.linalg.norm(state_xy_by_state[int(group_id[r])] - anchor_xy)))
            blocked_mismatch.append(float(blocked[r] != blocked[anchor]))

    return {
        "anchor_indices": np.asarray(anchors_out, dtype=np.int64),
        "candidate_indices": np.asarray(candidates_out, dtype=np.int64),
        "correct_candidate": np.asarray(correct_out, dtype=np.int64),
        "negative_type": np.asarray(negative_type_out),
        "action": np.asarray(action_out, dtype=np.float32),
        "action_id": np.asarray(action_id_out, dtype=np.int64),
        "anchor_blocked": np.asarray(anchor_blocked_out, dtype=np.int64),
        "candidate_blocked": np.asarray(candidate_blocked_out, dtype=np.int64),
        "anchor_state_xy": np.asarray(anchor_xy_out, dtype=np.float32),
        "candidate_state_xy": np.asarray(candidate_xy_out, dtype=np.float32),
        "split": np.asarray(split_out),
        "start_dists": np.asarray(start_dists, dtype=np.float32),
        "blocked_mismatch": np.asarray(blocked_mismatch, dtype=np.float32),
        "attempts": attempts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-groups", type=int, default=8000)
    parser.add_argument("--val-groups", type=int, default=1000)
    parser.add_argument("--test-groups", type=int, default=1000)
    parser.add_argument("--same-state-negatives", type=int, default=2)
    parser.add_argument("--same-action-negatives", type=int, default=2)
    parser.add_argument("--max-pos-dist", type=float, default=0.15)
    parser.add_argument("--prefer-block-mismatch", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    d = np.load(args.data, allow_pickle=True)
    group_id = d["group_id"].astype(np.int64)
    action_id = d["action_id"].astype(np.int64)
    action = d["action"].astype(np.float32)
    blocked = d["blocked_mask"].astype(np.int64)
    state_xy = d["state_xy"].astype(np.float32)

    states = np.unique(group_id)
    state_splits = split_states(states, args.seed)

    # One representative state_xy per simulator state.
    state_xy_by_state = {}
    for s in states:
        idx = np.where(group_id == s)[0][0]
        state_xy_by_state[int(s)] = state_xy[idx]

    row_by_state_action = {}
    for r, (s, a) in enumerate(zip(group_id, action_id)):
        row_by_state_action[(int(s), int(a))] = int(r)

    rows_by_split_action = {}
    for split_name, split_state_set in state_splits.items():
        for a in np.unique(action_id):
            rows = [
                r for (s, aa), r in row_by_state_action.items()
                if aa == int(a) and s in split_state_set
            ]
            rows_by_split_action[(split_name, int(a))] = np.asarray(rows, dtype=np.int64)

    wanted = {
        "train": args.train_groups,
        "val": args.val_groups,
        "test": args.test_groups,
    }

    mined = {}
    for split_name in ["train", "val", "test"]:
        print(f"mining split={split_name} states={len(state_splits[split_name])} target={wanted[split_name]}", flush=True)
        mined[split_name] = mine_split(
            split_name=split_name,
            split_states_set=state_splits[split_name],
            row_by_state_action=row_by_state_action,
            rows_by_split_action=rows_by_split_action,
            state_xy_by_state=state_xy_by_state,
            group_id=group_id,
            action_id=action_id,
            action=action,
            blocked=blocked,
            n_groups=wanted[split_name],
            same_state_negatives=args.same_state_negatives,
            same_action_negatives=args.same_action_negatives,
            max_pos_dist=args.max_pos_dist,
            prefer_block_mismatch=args.prefer_block_mismatch,
            rng=rng,
        )
        print(f"  mined={len(mined[split_name]['anchor_indices'])} attempts={mined[split_name]['attempts']}", flush=True)

    def cat(key):
        return np.concatenate([mined["train"][key], mined["val"][key], mined["test"][key]], axis=0)

    anchor_indices = cat("anchor_indices")
    candidate_indices = cat("candidate_indices")
    split = cat("split")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        anchor_indices=anchor_indices,
        candidate_indices=candidate_indices,
        correct_candidate=cat("correct_candidate"),
        negative_type=cat("negative_type"),
        action=cat("action"),
        action_id=cat("action_id"),
        anchor_blocked=cat("anchor_blocked"),
        candidate_blocked=cat("candidate_blocked"),
        anchor_state_xy=cat("anchor_state_xy"),
        candidate_state_xy=cat("candidate_state_xy"),
        split=split,
        source_npz=str(args.data),
        mining_version="state_disjoint_mixed_hard_v0",
        candidates=np.asarray(candidate_indices.shape[1], dtype=np.int64),
        same_state_negatives=np.asarray(args.same_state_negatives, dtype=np.int64),
        same_action_negatives=np.asarray(args.same_action_negatives, dtype=np.int64),
        max_pos_dist=np.asarray(args.max_pos_dist, dtype=np.float64),
        prefer_block_mismatch=np.asarray(bool(args.prefer_block_mismatch)),
        seed=np.asarray(args.seed, dtype=np.int64),
    )

    # Verify state disjointness from the actual output.
    state_sets = {}
    for name in ["train", "val", "test"]:
        mask = split == name
        rows = np.unique(candidate_indices[mask].reshape(-1))
        state_sets[name] = set(map(int, group_id[rows]))

    overlaps = {
        "train_val": len(state_sets["train"] & state_sets["val"]),
        "train_test": len(state_sets["train"] & state_sets["test"]),
        "val_test": len(state_sets["val"] & state_sets["test"]),
    }

    all_dists = np.concatenate([mined[k]["start_dists"] for k in ["train", "val", "test"]])
    all_bm = np.concatenate([mined[k]["blocked_mismatch"] for k in ["train", "val", "test"]])

    report = {
        "data": str(args.data),
        "output": str(args.out),
        "groups_total": int(len(anchor_indices)),
        "groups_by_split": {k: int(len(mined[k]["anchor_indices"])) for k in ["train", "val", "test"]},
        "states_by_split": {k: int(len(state_splits[k])) for k in ["train", "val", "test"]},
        "state_overlaps": overlaps,
        "candidates_per_group": int(candidate_indices.shape[1]),
        "chance_top1": float(1.0 / candidate_indices.shape[1]),
        "same_state_negatives": int(args.same_state_negatives),
        "same_action_negatives": int(args.same_action_negatives),
        "max_pos_dist": float(args.max_pos_dist),
        "prefer_block_mismatch": bool(args.prefer_block_mismatch),
        "start_position_distance": summarize(all_dists),
        "blocked_mismatch_fraction": float(all_bm.mean()) if len(all_bm) else None,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# State-disjoint mixed hard-negative group mining",
        "",
        f"- data: `{args.data}`",
        f"- output: `{args.out}`",
        f"- groups total: `{report['groups_total']}`",
        f"- groups by split: `{report['groups_by_split']}`",
        f"- states by split: `{report['states_by_split']}`",
        f"- candidates per group: `{report['candidates_per_group']}`",
        f"- chance top-1: `{report['chance_top1']:.6f}`",
        f"- same-state different-action negatives: `{args.same_state_negatives}`",
        f"- same-action different-state negatives: `{args.same_action_negatives}`",
        f"- max position distance: `{args.max_pos_dist}`",
        f"- prefer blocked mismatch: `{bool(args.prefer_block_mismatch)}`",
        "",
        "## State-overlap verification",
        "",
        "| split pair | state overlap |",
        "| --- | ---: |",
        f"| train/val | {overlaps['train_val']} |",
        f"| train/test | {overlaps['train_test']} |",
        f"| val/test | {overlaps['val_test']} |",
        "",
        "## Off-diagonal diagnostics",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| mean start-position distance | {report['start_position_distance']['mean']:.6f} |",
        f"| median start-position distance | {report['start_position_distance']['median']:.6f} |",
        f"| p95 start-position distance | {report['start_position_distance']['p95']:.6f} |",
        f"| blocked-mismatch fraction | {report['blocked_mismatch_fraction']:.6f} |",
        "",
        "## Interpretation",
        "",
        "This group file enforces state-disjoint train/validation/test splits before hard-negative mining.",
        "It is the stronger generalization protocol: no simulator state used by candidate futures in train appears in validation or test.",
    ]
    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
