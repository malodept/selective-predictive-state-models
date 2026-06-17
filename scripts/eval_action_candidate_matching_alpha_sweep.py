from __future__ import annotations

import argparse
import json
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


def intervene_actions(actions, slices, mode, seed):
    out = actions.copy()
    rng = np.random.default_rng(seed)

    if mode == "original":
        return out

    if mode == "zero":
        return np.zeros_like(out)

    if mode == "within_group_shuffle":
        for a, b in slices:
            perm = rng.permutation(b - a)
            out[a:b] = out[a:b][perm]
        return out

    raise ValueError(mode)


def mse_matrix(pred, true):
    diff = pred[:, None, :, :] - true[None, :, :, :]
    return np.mean(diff ** 2, axis=(2, 3))


def evaluate(pred, true, slices):
    top1 = []
    margins = []
    diag_mse = []
    best_offdiag = []
    predicted_rank = []

    for a, b in slices:
        dist = mse_matrix(pred[a:b], true[a:b])
        k = b - a

        for r in range(k):
            order = np.argsort(dist[r])
            top1.append(float(order[0] == r))

            rank = int(np.where(order == r)[0][0]) + 1
            predicted_rank.append(rank)

            diag = float(dist[r, r])
            off = np.delete(dist[r], r)
            best_off = float(np.min(off))

            diag_mse.append(diag)
            best_offdiag.append(best_off)
            margins.append(best_off - diag)

    return {
        "top1": float(np.mean(top1)),
        "mean_rank": float(np.mean(predicted_rank)),
        "diag_mse": float(np.mean(diag_mse)),
        "best_offdiag_mse": float(np.mean(best_offdiag)),
        "margin": float(np.mean(margins)),
        "positive_margin_frac": float(np.mean(np.asarray(margins) > 0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
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

    data = load_npz(args.data)
    groups = build_groups(data, args.candidates, args.max_groups, args.seed)
    sub, slices = subset_data(data, groups)

    model, stats = load_model(args.checkpoint, sub, args.device)

    metrics_path = args.checkpoint.parent / "metrics.json"
    if metrics_path.exists():
        m = json.loads(metrics_path.read_text())
        calibrated_alpha = float(m["val"]["global_alpha"])
    else:
        calibrated_alpha = float("nan")

    z0 = sub["z_current"].astype(np.float32)
    true = sub["z_future"].astype(np.float32)
    actions = sub["action"].astype(np.float32)

    alpha_values = [0.0, 0.25, calibrated_alpha, 0.5, 0.75, 1.0]
    alpha_values = sorted(set([round(float(a), 6) for a in alpha_values if np.isfinite(a)]))

    rows = []

    for mode in ["original", "zero", "within_group_shuffle"]:
        d = dict(sub)
        d["action"] = intervene_actions(actions, slices, mode, args.seed)

        print(f"Predicting mode={mode}", flush=True)
        delta = predict_delta(model, d, stats, args.batch_size, args.device)

        for alpha in alpha_values:
            pred = z0 + alpha * delta
            r = evaluate(pred, true, slices)
            r["mode"] = mode
            r["alpha"] = alpha
            rows.append(r)

    lines = [
        "# Action candidate matching alpha sweep",
        "",
        f"Checkpoint: `{args.name}`.",
        "",
        "This evaluates candidate matching with different residual scaling coefficients. `alpha=1` tests the raw model displacement, while the calibrated alpha is optimized for global MSE.",
        "",
        f"Groups: `{len(slices)}`. Candidates: `{args.candidates}`. Calibrated alpha: `{calibrated_alpha:.6f}`.",
        "",
        "| inference action | alpha | top-1 accuracy | mean correct rank | diag MSE | best offdiag MSE | margin best offdiag - diag | positive margin frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['alpha']:.6f} | {r['top1']:.6f} | {r['mean_rank']:.6f} | "
            f"{r['diag_mse']:.6f} | {r['best_offdiag_mse']:.6f} | "
            f"{r['margin']:.6f} | {r['positive_margin_frac']:.6f} |"
        )

    out = args.out_dir / f"candidate_matching_alpha_sweep_{args.name}.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
