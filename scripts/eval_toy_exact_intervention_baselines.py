from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--name", default="toy_exact_v0")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rank_metrics(pred: np.ndarray, futures: np.ndarray):
    # pred: [G, K, T, D]
    # futures: [G, K, T, D]
    mse = ((pred[:, :, None] - futures[:, None, :]) ** 2).mean(axis=(3, 4))

    top1 = []
    ranks = []
    margins = []

    for g in range(mse.shape[0]):
        for k in range(mse.shape[1]):
            order = np.argsort(mse[g, k])
            rank = int(np.where(order == k)[0][0]) + 1
            ranks.append(rank)
            top1.append(rank == 1)

            off = np.delete(mse[g, k], k)
            margins.append(float(off.min() - mse[g, k, k]))

    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(ranks)),
        "positive_margin_frac": float(np.mean(np.asarray(margins) > 0)),
        "mean_margin": float(np.mean(margins)),
    }


def deranged_permutation(k: int, rng: np.random.Generator):
    if k == 1:
        return np.arange(k)
    while True:
        p = rng.permutation(k)
        if np.all(p != np.arange(k)):
            return p


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    cand = d["candidate_indices"].astype(np.int64)

    z_anchor = z[cand[:, 0]]
    y_cand = y[cand]
    z_cand = z[cand]

    G, K = cand.shape

    rows = []

    # Identity baseline.
    pred_identity = np.repeat(z_anchor[:, None, :, :], K, axis=1)
    m = rank_metrics(pred_identity, y_cand)
    m.update({"model": "identity", "alpha": 0.0})
    rows.append(m)

    # Exact delta-transfer oracle:
    # since all candidates branch from the same state, candidate_current == anchor_current.
    # At alpha=1, this should reproduce every candidate future exactly.
    delta = y_cand - z_cand
    for alpha in args.alphas:
        pred = z_anchor[:, None, :, :] + alpha * delta
        m = rank_metrics(pred, y_cand)
        m.update({"model": "delta_oracle_original", "alpha": float(alpha)})
        rows.append(m)

    # Shuffled / deranged oracle: action-future assignment is deliberately wrong.
    pred_shuf = np.zeros_like(y_cand)
    for g in range(G):
        perm = deranged_permutation(K, rng)
        pred_shuf[g] = y_cand[g, perm]
    m = rank_metrics(pred_shuf, y_cand)
    m.update({"model": "delta_oracle_deranged", "alpha": 1.0})
    rows.append(m)

    # Random candidate future assignment with possible fixed points.
    pred_random = np.zeros_like(y_cand)
    for g in range(G):
        perm = rng.permutation(K)
        pred_random[g] = y_cand[g, perm]
    m = rank_metrics(pred_random, y_cand)
    m.update({"model": "delta_oracle_random_shuffle", "alpha": 1.0})
    rows.append(m)

    out_json = args.out_dir / f"{args.name}_baseline_rows.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        f"# Toy exact-intervention baseline evaluation: {args.name}",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{G}`",
        f"- candidates: `{K}`",
        f"- chance top-1: `{1.0 / K:.6f}`",
        "",
        "| model | alpha | top-1 | mean rank | positive margin frac | mean margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['model']} | {r['alpha']:.2f} | {r['top1']:.6f} | "
            f"{r['mean_rank']:.6f} | {r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} |"
        )

    lines += [
        "",
        "## Expected outcome",
        "",
        "The identity baseline should be near chance. The original delta oracle should reach top-1 = 1.0 at alpha = 1.",
        "The deranged oracle should fail because the action-future correspondence is deliberately wrong.",
        "This verifies that the candidate-matching metric can detect true exact-intervention grounding when the protocol is identifiable.",
    ]

    out_md = args.out_dir / f"{args.name}_baseline_eval.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
