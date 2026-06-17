from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    return p.parse_args()


def rank_metrics(pred, futures):
    # pred: [G, K, T, D]
    # futures: [G, K, T, D]
    mse = ((pred[:, :, None] - futures[:, None, :]) ** 2).mean(axis=(3, 4))

    top1 = []
    ranks = []
    margins = []
    positive = []

    for g in range(mse.shape[0]):
        for k in range(mse.shape[1]):
            order = np.argsort(mse[g, k])
            rank = int(np.where(order == k)[0][0]) + 1
            ranks.append(rank)
            top1.append(rank == 1)

            off = np.delete(mse[g, k], k)
            margin = float(off.min() - mse[g, k, k])
            margins.append(margin)
            positive.append(margin > 0)

    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(ranks)),
        "positive_margin_frac": float(np.mean(positive)),
        "mean_margin": float(np.mean(margins)),
    }


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    g = np.load(args.groups, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    cand = g["candidate_indices"].astype(np.int64)

    z_anchor = z[cand[:, 0]]
    z_cand = z[cand]
    y_cand = y[cand]
    delta_cand = y_cand - z_cand

    rows = []

    for alpha in args.alphas:
        pred = z_anchor[:, None] + alpha * delta_cand
        m = rank_metrics(pred, y_cand)
        m["alpha"] = float(alpha)
        rows.append(m)

    out_json = args.out_dir / f"delta_oracle_{args.name}.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        f"# Delta-transfer oracle: {args.name}",
        "",
        "This oracle applies each candidate's true latent displacement to the anchor state:",
        "",
        "`prediction_k = anchor_current + alpha * (candidate_future_k - candidate_current_k)`",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- candidate groups: `{cand.shape[0]}`",
        f"- candidates: `{cand.shape[1]}`",
        f"- chance top-1: `{1.0 / cand.shape[1]:.6f}`",
        "",
        "| alpha | top-1 | mean rank | positive margin frac | mean margin |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['alpha']:.2f} | {r['top1']:.6f} | {r['mean_rank']:.6f} | "
            f"{r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} |"
        )

    out_md = args.out_dir / f"delta_oracle_{args.name}.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
