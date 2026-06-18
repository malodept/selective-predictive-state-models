from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def rank_metrics(pred: np.ndarray, futures: np.ndarray):
    # pred:    [G, K, T, D]
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
            margin = float(off.min() - mse[g, k, k])
            margins.append(margin)

    margins = np.asarray(margins)

    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(ranks)),
        "positive_margin_frac": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
    }


def deranged_permutation(k: int, rng: np.random.Generator):
    if k <= 1:
        return np.arange(k)

    while True:
        perm = rng.permutation(k)
        if np.all(perm != np.arange(k)):
            return perm


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    cand = d["candidate_indices"].astype(np.int64)

    G, K = cand.shape

    z_anchor = z[cand[:, 0]]
    z_cand = z[cand]
    y_cand = y[cand]

    rows = []

    # Identity baseline.
    pred_identity = np.repeat(z_anchor[:, None, :, :], K, axis=1)
    m = rank_metrics(pred_identity, y_cand)
    m.update({"model": "identity", "alpha": 0.0})
    rows.append(m)

    # Delta-transfer oracle:
    # prediction_k = anchor_current + alpha * (candidate_future_k - candidate_current_k)
    delta = y_cand - z_cand

    for alpha in args.alphas:
        pred = z_anchor[:, None, :, :] + alpha * delta
        m = rank_metrics(pred, y_cand)
        m.update({"model": "delta_oracle_original", "alpha": float(alpha)})
        rows.append(m)

    # Deranged oracle: deliberately wrong action/future correspondence.
    pred_deranged = np.zeros_like(y_cand)
    for g in range(G):
        perm = deranged_permutation(K, rng)
        pred_deranged[g] = y_cand[g, perm]

    m = rank_metrics(pred_deranged, y_cand)
    m.update({"model": "delta_oracle_deranged", "alpha": 1.0})
    rows.append(m)

    # Random shuffle: expected top-1 around chance.
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
        f"# Exact-intervention baseline evaluation: {args.name}",
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
        "A valid exact-intervention protocol should make the original delta oracle clearly outperform identity and random/shuffled baselines.",
        "If the oracle is weak, the representation or rendering does not make the intervention branches identifiable enough.",
    ]

    out_md = args.out_dir / f"{args.name}_baseline_eval.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
