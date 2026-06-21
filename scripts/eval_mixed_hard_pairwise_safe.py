from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_mixed_hard_delta_transformer import DeltaTransformer


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-groups", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    data = np.load(args.data, allow_pickle=True)
    groups = np.load(args.groups, allow_pickle=True)

    z = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    a = data["action"].astype(np.float32)

    anchor = groups["anchor_indices"].astype(np.int64)
    cand = groups["candidate_indices"].astype(np.int64)
    negative_type = groups["negative_type"].astype(str)

    if "split" in groups.files:
        test_groups = np.where(groups["split"].astype(str) == "test")[0]
        split_source = "explicit"
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(anchor))
        n_train = int(0.8 * len(anchor))
        n_val = int(0.1 * len(anchor))
        test_groups = perm[n_train + n_val:]
        split_source = "random_fallback"

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]

    model = DeltaTransformer(
        token_dim=int(checkpoint["token_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        tokens=int(checkpoint["tokens"]),
        model_dim=int(ckpt_args["model_dim"]),
        heads=int(ckpt_args["heads"]),
        layers=int(ckpt_args["layers"]),
        dropout=float(ckpt_args["dropout"]),
        mode=ckpt_args["mode"],
    ).to(args.device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    delta = y - z

    all_distances = []
    all_types = []

    for start in range(0, len(test_groups), args.batch_groups):
        group_ids = test_groups[start:start + args.batch_groups]
        anchor_rows = anchor[group_ids]
        candidate_rows = cand[group_ids]

        z_current = torch.from_numpy(z[anchor_rows]).float().to(args.device)
        action = torch.from_numpy(a[anchor_rows]).float().to(args.device)

        pred_delta = model(z_current, action).cpu().numpy()
        candidate_delta = delta[candidate_rows]

        distances = ((pred_delta[:, None] - candidate_delta) ** 2).mean(axis=(2, 3))

        all_distances.append(distances)
        all_types.append(negative_type[group_ids])

    distances = np.concatenate(all_distances, axis=0)
    types = np.concatenate(all_types, axis=0)

    predicted = distances.argmin(axis=1)
    correct = predicted == 0

    rank_matrix = np.argsort(np.argsort(distances, axis=1), axis=1)
    correct_ranks = rank_matrix[:, 0] + 1

    rows = []
    for neg_type in sorted(set(types.reshape(-1))):
        if neg_type == "correct":
            continue

        margins = []
        wins = []

        for i in range(distances.shape[0]):
            js = np.where(types[i] == neg_type)[0]
            for j in js:
                margin = float(distances[i, j] - distances[i, 0])
                margins.append(margin)
                wins.append(float(margin > 0.0))

        rows.append({
            "negative_type": neg_type,
            "pairwise_accuracy": float(np.mean(wins)),
            "mean_margin": float(np.mean(margins)),
            "count": int(len(wins)),
        })

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "checkpoint": str(args.checkpoint),
        "mode": ckpt_args["mode"],
        "split_source": split_source,
        "test_groups": int(len(test_groups)),
        "top1": float(correct.mean()),
        "mean_rank": float(correct_ranks.mean()),
        "pairwise_rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        f"# Safe mixed hard pairwise audit: {ckpt_args['mode']}",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- mode: `{ckpt_args['mode']}`",
        f"- split source: `{split_source}`",
        f"- test groups: `{len(test_groups)}`",
        f"- top-1: `{correct.mean():.6f}`",
        f"- mean rank: `{correct_ranks.mean():.6f}`",
        "",
        "| negative type | pairwise accuracy | mean margin | count |",
        "| --- | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            f"| {row['negative_type']} | {row['pairwise_accuracy']:.6f} | "
            f"{row['mean_margin']:.6f} | {row['count']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This safe audit computes pairwise accuracy directly from the same distance matrix used for top-1 ranking.",
        "Therefore, if top-1 is near-perfect, pairwise accuracy against each negative type should also be near-perfect.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
