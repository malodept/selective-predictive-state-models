from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_group_split(n_groups: int, seed: int, train_frac: float = 0.8, val_frac: float = 0.1):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)

    n_train = int(train_frac * n_groups)
    n_val = int(val_frac * n_groups)

    return {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:],
    }


def n_overlap(a, b) -> int:
    return len(set(map(int, a)) & set(map(int, b)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=True)
    groups = np.load(args.groups, allow_pickle=True)

    anchor_indices = groups["anchor_indices"].astype(np.int64)
    candidate_indices = groups["candidate_indices"].astype(np.int64)

    state_id = data["group_id"].astype(np.int64)

    split = make_group_split(
        n_groups=len(anchor_indices),
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    split_stats = {}
    sample_sets = {}
    anchor_sample_sets = {}
    candidate_sample_sets = {}
    state_sets = {}
    anchor_state_sets = {}
    candidate_state_sets = {}

    for name, group_ids in split.items():
        anchors = anchor_indices[group_ids]
        candidates = candidate_indices[group_ids].reshape(-1)

        all_samples = np.concatenate([anchors, candidates])

        anchor_states = state_id[anchors]
        candidate_states = state_id[candidates]
        all_states = np.concatenate([anchor_states, candidate_states])

        sample_sets[name] = set(map(int, all_samples))
        anchor_sample_sets[name] = set(map(int, anchors))
        candidate_sample_sets[name] = set(map(int, candidates))

        state_sets[name] = set(map(int, all_states))
        anchor_state_sets[name] = set(map(int, anchor_states))
        candidate_state_sets[name] = set(map(int, candidate_states))

        split_stats[name] = {
            "groups": int(len(group_ids)),
            "samples_used": int(len(sample_sets[name])),
            "anchor_samples": int(len(anchor_sample_sets[name])),
            "candidate_samples": int(len(candidate_sample_sets[name])),
            "states_used": int(len(state_sets[name])),
            "anchor_states": int(len(anchor_state_sets[name])),
            "candidate_states": int(len(candidate_state_sets[name])),
        }

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]

    overlaps = {}
    for a, b in pairs:
        key = f"{a}_{b}"
        overlaps[key] = {
            "sample_overlap": n_overlap(sample_sets[a], sample_sets[b]),
            "anchor_sample_overlap": n_overlap(anchor_sample_sets[a], anchor_sample_sets[b]),
            "candidate_sample_overlap": n_overlap(candidate_sample_sets[a], candidate_sample_sets[b]),
            "state_overlap": n_overlap(state_sets[a], state_sets[b]),
            "anchor_state_overlap": n_overlap(anchor_state_sets[a], anchor_state_sets[b]),
            "candidate_state_overlap": n_overlap(candidate_state_sets[a], candidate_state_sets[b]),
        }

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "seed": args.seed,
        "n_mined_groups": int(len(anchor_indices)),
        "split_stats": split_stats,
        "overlaps": overlaps,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")

    lines = [
        "# Mixed hard split leakage audit",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- split seed: `{args.seed}`",
        f"- mined groups: `{len(anchor_indices)}`",
        "",
        "## Split content",
        "",
        "| split | groups | samples used | states used | anchor states | candidate states |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name in ["train", "val", "test"]:
        s = split_stats[name]
        lines.append(
            f"| {name} | {s['groups']} | {s['samples_used']} | {s['states_used']} | "
            f"{s['anchor_states']} | {s['candidate_states']} |"
        )

    lines += [
        "",
        "## Overlap between splits",
        "",
        "| split pair | sample overlap | anchor sample overlap | candidate sample overlap | state overlap | anchor state overlap | candidate state overlap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for a, b in pairs:
        o = overlaps[f"{a}_{b}"]
        lines.append(
            f"| {a}/{b} | {o['sample_overlap']} | {o['anchor_sample_overlap']} | "
            f"{o['candidate_sample_overlap']} | {o['state_overlap']} | "
            f"{o['anchor_state_overlap']} | {o['candidate_state_overlap']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This audit checks whether the random mined-group split reuses exact-intervention samples or simulator states across train/validation/test.",
        "For the strongest generalization claim, the ideal protocol is state-disjoint: no simulator state should appear in more than one split.",
        "If state overlap is high, the current result remains a strong protocol/architecture result, but the next benchmark should enforce state-disjoint mining.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
