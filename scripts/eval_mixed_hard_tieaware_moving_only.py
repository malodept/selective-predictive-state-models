from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_mixed_hard_delta_transformer import DeltaTransformer


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-groups", type=int, default=128)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    d = np.load(args.data, allow_pickle=True)
    g = np.load(args.groups, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)
    neg_type = g["negative_type"].astype(str)
    action_id = g["action_id"].astype(np.int64)
    action_names = d["action_names"].astype(str)

    if "split" in g.files:
        test_groups = np.where(g["split"].astype(str) == "test")[0]
        split_source = "explicit"
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(anchor))
        n_train = int(0.8 * len(anchor))
        n_val = int(0.1 * len(anchor))
        test_groups = perm[n_train + n_val:]
        split_source = "random_fallback"

    stay_ids = set(np.where(action_names == "stay")[0].tolist())
    moving_mask = np.array([action_id[i] not in stay_ids for i in test_groups], dtype=bool)
    test_groups = test_groups[moving_mask]

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ca = ckpt["args"]

    model = DeltaTransformer(
        token_dim=int(ckpt["token_dim"]),
        action_dim=int(ckpt["action_dim"]),
        tokens=int(ckpt["tokens"]),
        model_dim=int(ca["model_dim"]),
        heads=int(ca["heads"]),
        layers=int(ca["layers"]),
        dropout=float(ca["dropout"]),
        mode=ca["mode"],
    ).to(args.device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    delta = y - z

    all_dist = []
    all_type = []

    for start in range(0, len(test_groups), args.batch_groups):
        gids = test_groups[start:start + args.batch_groups]
        anch = anchor[gids]
        c = cand[gids]

        zt = torch.from_numpy(z[anch]).float().to(args.device)
        at = torch.from_numpy(a[anch]).float().to(args.device)

        pred_delta = model(zt, at).cpu().numpy()
        candidate_delta = delta[c]
        dist = ((pred_delta[:, None] - candidate_delta) ** 2).mean(axis=(2, 3))

        all_dist.append(dist)
        all_type.append(neg_type[gids])

    dist = np.concatenate(all_dist, axis=0)
    typ = np.concatenate(all_type, axis=0)

    d0 = dist[:, 0]
    min_dist = dist.min(axis=1)

    biased_top1 = float((dist.argmin(axis=1) == 0).mean())
    strict_top1 = float((d0 < np.min(dist[:, 1:], axis=1) - args.eps).mean())

    tieaware_scores = []
    tie_counts = []
    for i in range(dist.shape[0]):
        tied = np.where(np.abs(dist[i] - min_dist[i]) <= args.eps)[0]
        tie_counts.append(len(tied))
        if 0 in tied:
            tieaware_scores.append(1.0 / len(tied))
        else:
            tieaware_scores.append(0.0)

    pair_rows = []
    for t in sorted(set(typ.reshape(-1))):
        if t == "correct":
            continue

        margins = []
        for i in range(dist.shape[0]):
            js = np.where(typ[i] == t)[0]
            for j in js:
                margins.append(float(dist[i, j] - dist[i, 0]))

        margins = np.asarray(margins, dtype=np.float64)

        pair_rows.append({
            "negative_type": t,
            "strict_win_frac": float((margins > args.eps).mean()),
            "tie_frac": float((np.abs(margins) <= args.eps).mean()),
            "loss_frac": float((margins < -args.eps).mean()),
            "nonloss_frac": float((margins >= -args.eps).mean()),
            "mean_margin": float(margins.mean()),
            "count": int(len(margins)),
        })

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "checkpoint": str(args.checkpoint),
        "mode": ca["mode"],
        "split_source": split_source,
        "test_groups_moving_only": int(len(test_groups)),
        "excluded_action": "stay",
        "eps": float(args.eps),
        "biased_argmin_top1": biased_top1,
        "strict_top1": strict_top1,
        "tieaware_top1": float(np.mean(tieaware_scores)),
        "mean_tie_count_at_min": float(np.mean(tie_counts)),
        "pairwise_rows": pair_rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        f"# Moving-only tie-aware mixed hard evaluation: {ca['mode']}",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- split source: `{split_source}`",
        f"- excluded action: `stay`",
        f"- moving-only test groups: `{len(test_groups)}`",
        f"- epsilon: `{args.eps}`",
        "",
        "## Global metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| biased argmin top-1 | {biased_top1:.6f} |",
        f"| strict top-1 | {strict_top1:.6f} |",
        f"| tie-aware top-1 | {np.mean(tieaware_scores):.6f} |",
        f"| mean tie count at minimum | {np.mean(tie_counts):.6f} |",
        "",
        "## Pairwise margin decomposition",
        "",
        "| negative type | strict win | tie | loss | non-loss | mean margin | count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in pair_rows:
        lines.append(
            f"| {r['negative_type']} | {r['strict_win_frac']:.6f} | "
            f"{r['tie_frac']:.6f} | {r['loss_frac']:.6f} | "
            f"{r['nonloss_frac']:.6f} | {r['mean_margin']:.6f} | {r['count']} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "This evaluation removes `stay` anchors, which are non-identifiable in latent-displacement space because their true delta is exactly zero across states.",
        "It reports performance on the identifiable moving-action subset.",
    ]

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
