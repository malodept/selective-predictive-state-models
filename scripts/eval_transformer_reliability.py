from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DeltaTransformer(nn.Module):
    def __init__(self, token_dim, action_dim, tokens, model_dim, heads, layers, dropout, mode):
        super().__init__()
        self.mode = mode
        self.in_proj = nn.Linear(token_dim, model_dim)
        self.action_proj = nn.Linear(action_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, tokens, model_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=4 * model_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.norm = nn.LayerNorm(model_dim)
        self.out_proj = nn.Linear(model_dim, token_dim)

    def forward(self, z, a):
        if self.mode in ["full", "state_only"]:
            h = self.in_proj(z)
        else:
            h = torch.zeros(z.shape[0], z.shape[1], self.pos.shape[-1], device=z.device)

        if self.mode in ["full", "action_only"]:
            h = h + self.action_proj(a).unsqueeze(1)

        h = h + self.pos
        h = self.encoder(h)
        h = self.norm(h)
        return self.out_proj(h)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def calibration_bins(conf, correct, bins=10):
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & (conf < hi if i < bins - 1 else conf <= hi)
        if mask.sum() == 0:
            rows.append((lo, hi, 0, None, None))
            continue

        acc = float(correct[mask].mean())
        c = float(conf[mask].mean())
        frac = float(mask.mean())
        ece += frac * abs(acc - c)
        rows.append((lo, hi, int(mask.sum()), acc, c))

    return rows, float(ece)


def selective_curve(conf, correct):
    rows = []
    order = np.argsort(-conf)

    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        n = max(1, int(round(cov * len(conf))))
        idx = order[:n]
        rows.append({
            "coverage": float(cov),
            "kept": int(n),
            "accuracy": float(correct[idx].mean()),
            "mean_confidence": float(conf[idx].mean()),
        })

    return rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--batch-groups", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.data, allow_pickle=True)
    g = np.load(args.groups, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)
    delta = y - z

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)

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

    n_groups = cand.shape[0]
    rng = np.random.default_rng(int(ca.get("seed", 0)))
    perm = rng.permutation(n_groups)
    n_train = int(float(ca.get("train_frac", 0.8)) * n_groups)
    n_val = int(float(ca.get("val_frac", 0.1)) * n_groups)
    test_groups = perm[n_train + n_val:]

    all_dist = []

    for s in range(0, len(test_groups), args.batch_groups):
        gids = test_groups[s:s + args.batch_groups]
        anch = anchor[gids]
        c = cand[gids]

        zt = torch.from_numpy(z[anch]).float().to(args.device)
        at = torch.from_numpy(a[anch]).float().to(args.device)

        pred = model(zt, at).cpu().numpy()
        dist = ((pred[:, None] - delta[c]) ** 2).mean(axis=(2, 3))
        all_dist.append(dist)

    dist = np.concatenate(all_dist, axis=0)
    logits = -dist / args.temperature
    prob = softmax(logits, axis=1)

    pred_idx = prob.argmax(axis=1)
    correct = pred_idx == 0
    conf = prob.max(axis=1)

    sorted_dist = np.sort(dist, axis=1)
    margin = sorted_dist[:, 1] - sorted_dist[:, 0]

    bins, ece = calibration_bins(conf, correct, bins=10)
    curve = selective_curve(conf, correct)

    report = {
        "data": str(args.data),
        "groups": str(args.groups),
        "checkpoint": str(args.checkpoint),
        "mode": ca["mode"],
        "test_groups": int(len(test_groups)),
        "top1": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "ece_10_bins": float(ece),
        "mean_margin": float(margin.mean()),
        "calibration_bins": [
            {"lo": lo, "hi": hi, "count": count, "accuracy": acc, "confidence": c}
            for lo, hi, count, acc, c in bins
        ],
        "selective_curve": curve,
    }

    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        "# Transformer reliability audit",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- mode: `{ca['mode']}`",
        f"- test groups: `{len(test_groups)}`",
        f"- top-1 accuracy: `{correct.mean():.6f}`",
        f"- mean confidence: `{conf.mean():.6f}`",
        f"- ECE 10 bins: `{ece:.6f}`",
        f"- mean distance margin: `{margin.mean():.6f}`",
        "",
        "## Selective prediction curve",
        "",
        "| coverage | kept groups | accuracy | mean confidence |",
        "| ---: | ---: | ---: | ---: |",
    ]

    for r in curve:
        lines.append(
            f"| {r['coverage']:.2f} | {r['kept']} | {r['accuracy']:.6f} | {r['mean_confidence']:.6f} |"
        )

    lines += [
        "",
        "## Calibration bins",
        "",
        "| confidence bin | count | accuracy | mean confidence |",
        "| --- | ---: | ---: | ---: |",
    ]

    for lo, hi, count, acc, c in bins:
        acc_s = "NA" if acc is None else f"{acc:.6f}"
        c_s = "NA" if c is None else f"{c:.6f}"
        lines.append(f"| [{lo:.1f}, {hi:.1f}] | {count} | {acc_s} | {c_s} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "This audit checks whether the model confidence is informative about correctness.",
        "If high-confidence subsets have higher accuracy, the model can support selective prediction or selective compute.",
    ]

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
