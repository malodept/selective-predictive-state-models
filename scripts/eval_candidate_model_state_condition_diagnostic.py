from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        mods = []
        d = input_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout)]
            d = hidden
        mods.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-groups", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def build_condition(z_cond: torch.Tensor, action: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return torch.cat([z_cond, action], dim=-1)
    if mode == "action_only":
        return action
    if mode == "state_only":
        return z_cond
    if mode == "no_context":
        return torch.ones((z_cond.shape[0], 1), device=z_cond.device, dtype=z_cond.dtype)
    raise ValueError(mode)


def rank_metrics(pred: np.ndarray, futures: np.ndarray):
    mse = ((pred[:, :, None] - futures[:, None, :]) ** 2).mean(axis=3)

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

    margins = np.asarray(margins)

    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(ranks)),
        "positive_margin_frac": float(np.mean(margins > 0)),
        "mean_margin": float(np.mean(margins)),
    }


@torch.no_grad()
def eval_variant(model, z_flat, y_flat, action, cand, groups, mode, variant, batch_groups, device, seed):
    rng = np.random.default_rng(seed)
    preds = []
    futs = []

    for start in range(0, len(groups), batch_groups):
        group_ids = groups[start:start + batch_groups]
        cand_b = cand[group_ids]
        G, K = cand_b.shape

        z_anchor = z_flat[cand_b[:, 0]]
        y_cand = y_flat[cand_b]
        a_cand = action[cand_b]

        z_base = np.repeat(z_anchor[:, None, :], K, axis=1).reshape(G * K, -1)
        z_cond = z_base.copy()
        a_eval = a_cand.copy()

        if variant == "original":
            pass
        elif variant == "cond_state_shuffle":
            perm = rng.permutation(G)
            z_shuf = z_anchor[perm]
            z_cond = np.repeat(z_shuf[:, None, :], K, axis=1).reshape(G * K, -1)
        elif variant == "cond_state_zero":
            z_cond = np.zeros_like(z_cond)
        elif variant == "cond_state_group_mean":
            mean_state = z_anchor.mean(axis=0, keepdims=True)
            z_cond = np.repeat(mean_state, G * K, axis=0)
        elif variant == "action_zero":
            a_eval = np.zeros_like(a_eval)
        elif variant == "action_shuffle":
            for g in range(G):
                a_eval[g] = a_eval[g, rng.permutation(K)]
        else:
            raise ValueError(variant)

        a_flat = a_eval.reshape(G * K, -1)

        z_base_t = torch.from_numpy(z_base).float().to(device)
        z_cond_t = torch.from_numpy(z_cond).float().to(device)
        a_t = torch.from_numpy(a_flat).float().to(device)

        cond = build_condition(z_cond_t, a_t, mode)
        delta = model(cond)
        pred = z_base_t + delta

        preds.append(pred.cpu().numpy().reshape(G, K, -1))
        futs.append(y_cand.reshape(G, K, -1))

    pred = np.concatenate(preds, axis=0)
    fut = np.concatenate(futs, axis=0)
    return rank_metrics(pred, fut)


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    action = d["action"].astype(np.float32)
    cand = d["candidate_indices"].astype(np.int64)

    z_flat = z.reshape(len(z), -1)
    y_flat = y.reshape(len(y), -1)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]
    mode = ckpt_args["mode"]

    model = MLP(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        hidden=int(ckpt_args["hidden"]),
        layers=int(ckpt_args["layers"]),
        dropout=float(ckpt_args["dropout"]),
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_groups = cand.shape[0]
    rng = np.random.default_rng(int(ckpt_args.get("seed", args.seed)))
    group_perm = rng.permutation(n_groups)

    n_train = int(float(ckpt_args.get("train_frac", 0.8)) * n_groups)
    n_val = int(float(ckpt_args.get("val_frac", 0.1)) * n_groups)
    test_groups = group_perm[n_train + n_val:]

    variants = [
        "original",
        "cond_state_shuffle",
        "cond_state_zero",
        "cond_state_group_mean",
        "action_zero",
        "action_shuffle",
    ]

    rows = []
    for variant in variants:
        m = eval_variant(
            model=model,
            z_flat=z_flat,
            y_flat=y_flat,
            action=action,
            cand=cand,
            groups=test_groups,
            mode=mode,
            variant=variant,
            batch_groups=args.batch_groups,
            device=args.device,
            seed=args.seed,
        )
        m.update({"variant": variant})
        rows.append(m)

    report = {
        "data": str(args.data),
        "checkpoint": str(args.checkpoint),
        "mode": mode,
        "test_groups": int(len(test_groups)),
        "chance_top1": float(1.0 / cand.shape[1]),
        "rows": rows,
    }

    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        "# Candidate model state-condition diagnostic",
        "",
        f"- data: `{args.data}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- mode: `{mode}`",
        f"- test groups: `{len(test_groups)}`",
        f"- chance top-1: `{1.0 / cand.shape[1]:.6f}`",
        "",
        "| variant | top-1 | mean rank | positive margin frac | mean margin |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['variant']} | {r['top1']:.6f} | {r['mean_rank']:.6f} | "
            f"{r['positive_margin_frac']:.6f} | {r['mean_margin']:.6f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "`cond_state_shuffle` and `cond_state_zero` perturb only the state given to the predictor, while keeping the residual base state fixed.",
        "If these variants stay close to `original`, the predictor mostly ignores the state condition and uses an action-template shortcut.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
