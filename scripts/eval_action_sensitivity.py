from __future__ import annotations

import argparse
import sys
from collections import defaultdict
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


def load_model(checkpoint: Path, data, device: str):
    ckpt = torch.load(checkpoint, map_location=device)
    args = ckpt["args"]

    _, tokens, dim = data["z_current"].shape
    action_dim = data["action"].shape[1]

    model = TokenActionTransformer(
        dim=dim,
        action_dim=action_dim,
        tokens=tokens,
        layers=int(args["layers"]),
        heads=int(args["heads"]),
        action_hidden=int(args["action_hidden"]),
        dropout=float(args["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = {k: v.cpu().numpy() for k, v in ckpt["stats"].items()}
    return model, stats


def build_groups(data, candidates: int, max_groups: int, seed: int):
    buckets = defaultdict(list)
    tids = np.asarray(data["trajectory_id"]).astype(str)
    frame_i = data["frame_i"].astype(int)

    for idx, (tid, fi) in enumerate(zip(tids, frame_i)):
        buckets[(tid, int(fi))].append(idx)

    groups = []
    for _, idxs in buckets.items():
        idxs = sorted(idxs, key=lambda u: int(data["gap"][u]))
        if len(idxs) >= candidates:
            groups.append(idxs[:candidates])

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    if max_groups > 0:
        groups = groups[:max_groups]

    return np.asarray(groups, dtype=np.int64)


def subset_data(data, groups):
    flat = groups.reshape(-1)
    out = {}
    n = data["action"].shape[0]

    for k, v in data.items():
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            out[k] = v[flat]
        else:
            out[k] = v

    slices = []
    pos = 0
    for _ in range(groups.shape[0]):
        slices.append((pos, pos + groups.shape[1]))
        pos += groups.shape[1]

    return out, slices


def pairwise_mse(x):
    diff = x[:, None, :, :] - x[None, :, :, :]
    dist = np.mean(diff ** 2, axis=(2, 3))
    k = dist.shape[0]
    mask = ~np.eye(k, dtype=bool)
    return float(dist[mask].mean())


def summarize_spread(pred, true, slices):
    pred_spreads = []
    true_spreads = []
    ratios = []

    for a, b in slices:
        ps = pairwise_mse(pred[a:b])
        ts = pairwise_mse(true[a:b])
        pred_spreads.append(ps)
        true_spreads.append(ts)
        ratios.append(ps / (ts + 1e-12))

    return {
        "pred_spread": float(np.mean(pred_spreads)),
        "true_spread": float(np.mean(true_spreads)),
        "spread_ratio": float(np.mean(ratios)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-groups", type=int, default=8000)
    p.add_argument("--candidates", type=int, default=5)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    data = load_npz(args.test)
    groups = build_groups(data, args.candidates, args.max_groups, args.seed)
    sub, slices = subset_data(data, groups)

    model, stats = load_model(args.checkpoint, sub, args.device)

    z0 = sub["z_current"].astype(np.float32)
    true = sub["z_future"].astype(np.float32)

    rows = []

    for mode in ["original", "zero"]:
        d = dict(sub)

        if mode == "zero":
            d["action"] = np.zeros_like(d["action"], dtype=np.float32)

        delta = predict_delta(model, d, stats, args.batch_size, args.device)
        pred = z0 + delta

        r = summarize_spread(pred, true, slices)
        r["mode"] = mode
        rows.append(r)

    lines = [
        "# Action sensitivity diagnostic",
        "",
        f"Checkpoint: `{args.name}`.",
        "",
        "For each current frame, this measures how much predictions vary across different candidate actions compared with how much the true futures vary.",
        "",
        "| checkpoint | inference action | prediction pairwise spread | true future pairwise spread | spread ratio |",
        "| --- | --- | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {args.name} | {r['mode']} | {r['pred_spread']:.6f} | "
            f"{r['true_spread']:.6f} | {r['spread_ratio']:.6f} |"
        )

    out = args.out_dir / f"action_sensitivity_{args.name}.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
