from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_patchtoken_action_transformer import (
    load_npz,
    TokenActionTransformer,
    predict_delta,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_model(checkpoint, data, device):
    ckpt = torch.load(checkpoint, map_location=device)

    z = data["z_current"]
    action = data["action"]
    _, tokens, dim = z.shape
    action_dim = action.shape[1]

    model = TokenActionTransformer(
        dim=dim,
        action_dim=action_dim,
        tokens=tokens,
        layers=ckpt.get("args", {}).get("layers", 3),
        heads=ckpt.get("args", {}).get("heads", 6),
        action_hidden=ckpt.get("args", {}).get("action_hidden", 256),
        dropout=ckpt.get("args", {}).get("dropout", 0.05),
    ).to(device)

    key = "model" if "model" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[key])
    model.eval()

    stats = ckpt["stats"]
    stats_np = {}
    for k, v in stats.items():
        if torch.is_tensor(v):
            stats_np[k] = v.detach().cpu().numpy()
        else:
            stats_np[k] = np.asarray(v)

    return model, stats_np


def rank_metrics(pred, futures):
    # pred: [G, K, T, D]
    # futures: [G, K, T, D]
    mse = ((pred[:, :, None] - futures[:, None, :]) ** 2).mean(axis=(3, 4))
    # mse[g, predicted_action_index, future_index]
    diag = np.diagonal(mse, axis1=1, axis2=2)
    ranks = []
    top1 = []

    margins = []
    positive = []

    for g in range(mse.shape[0]):
        for k in range(mse.shape[1]):
            order = np.argsort(mse[g, k])
            rank = int(np.where(order == k)[0][0]) + 1
            ranks.append(rank)
            top1.append(1 if rank == 1 else 0)

            off = np.delete(mse[g, k], k)
            margin = float(off.min() - mse[g, k, k])
            margins.append(margin)
            positive.append(1 if margin > 0 else 0)

    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(ranks)),
        "positive_margin_frac": float(np.mean(positive)),
        "mean_margin": float(np.mean(margins)),
        "mean_diag_mse": float(np.mean(diag)),
        "mean_best_offdiag_mse": float(np.mean([np.delete(mse[g, k], k).min() for g in range(mse.shape[0]) for k in range(mse.shape[1])])),
    }


def predict_for_group_actions(model, data, stats_np, candidate_indices, batch_size, device, mode, seed):
    rng = np.random.default_rng(seed)

    z_all = data["z_current"].astype(np.float32)
    y_all = data["z_future"].astype(np.float32)
    a_all = data["action"].astype(np.float32)

    G, K = candidate_indices.shape

    anchor_idx = candidate_indices[:, 0]
    z_anchor = z_all[anchor_idx]

    futures = y_all[candidate_indices]

    if mode == "original":
        actions = a_all[candidate_indices]
    elif mode == "zero":
        actions = np.zeros_like(a_all[candidate_indices])
    elif mode == "within_group_shuffle":
        actions = a_all[candidate_indices].copy()
        for g in range(G):
            perm = rng.permutation(K)
            actions[g] = actions[g, perm]
    elif mode == "within_group_reverse":
        actions = a_all[candidate_indices][:, ::-1].copy()
    else:
        raise ValueError(mode)

    preds = []

    for k in range(K):
        tmp = {
            "z_current": z_anchor,
            "z_future": futures[:, k],
            "action": actions[:, k],
        }
        delta = predict_delta(model, tmp, stats_np, batch_size, device)
        preds.append(z_anchor + delta)

    pred = np.stack(preds, axis=1)
    return pred, futures


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    data = load_npz(args.data)
    groups = np.load(args.groups, allow_pickle=True)
    candidate_indices = groups["candidate_indices"].astype(np.int64)

    model, stats_np = load_model(args.checkpoint, data, args.device)

    rows = []
    for mode in ["original", "zero", "within_group_shuffle", "within_group_reverse"]:
        pred_base, futures = predict_for_group_actions(
            model=model,
            data=data,
            stats_np=stats_np,
            candidate_indices=candidate_indices,
            batch_size=args.batch_size,
            device=args.device,
            mode=mode,
            seed=args.seed,
        )

        z_anchor = data["z_current"].astype(np.float32)[candidate_indices[:, 0]]

        for alpha in args.alphas:
            pred = z_anchor[:, None] + alpha * (pred_base - z_anchor[:, None])
            m = rank_metrics(pred, futures)
            m.update({"mode": mode, "alpha": float(alpha)})
            rows.append(m)

    out_json = args.out_dir / f"matched_state_eval_{args.name}.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")

    lines = [
        f"# Matched-state group evaluation: {args.name}",
        "",
        f"- data: `{args.data}`",
        f"- groups: `{args.groups}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- candidate groups: `{candidate_indices.shape[0]}`",
        f"- candidates: `{candidate_indices.shape[1]}`",
        f"- chance top-1: `{1.0 / candidate_indices.shape[1]:.6f}`",
        "",
        "| mode | alpha | top-1 | mean rank | positive margin frac | mean margin | diag MSE | best offdiag MSE |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['alpha']:.2f} | {r['top1']:.6f} | {r['mean_rank']:.6f} | "
            f"{r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} | "
            f"{r['mean_diag_mse']:.6f} | {r['mean_best_offdiag_mse']:.6f} |"
        )

    out_md = args.out_dir / f"matched_state_eval_{args.name}.md"
    out_md.write_text("\n".join(lines) + "\n")

    print(out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
