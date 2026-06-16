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


def build_groups(data, max_groups: int, min_candidates: int, seed: int):
    buckets = defaultdict(list)

    tids = np.asarray(data["trajectory_id"]).astype(str)
    frame_i = data["frame_i"].astype(int)

    for idx, (tid, fi) in enumerate(zip(tids, frame_i)):
        buckets[(tid, int(fi))].append(idx)

    groups = []
    for _, idxs in buckets.items():
        idxs = sorted(idxs, key=lambda u: int(data["gap"][u]))
        if len(idxs) >= min_candidates:
            groups.append(idxs[:min_candidates])

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    if max_groups > 0:
        groups = groups[:max_groups]

    flat = np.array([i for g in groups for i in g], dtype=np.int64)
    slices = []
    pos = 0
    for g in groups:
        k = len(g)
        slices.append((pos, pos + k))
        pos += k

    return flat, slices


def make_mode_actions(actions, slices, mode: str, seed: int):
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

    if mode == "within_group_reverse":
        for a, b in slices:
            out[a:b] = out[a:b][::-1]
        return out

    raise ValueError(mode)


def mse_matrix(pred, true):
    # pred: [K, T, D], true: [K, T, D]
    diff = pred[:, None, :, :] - true[None, :, :, :]
    return np.mean(diff ** 2, axis=(2, 3))


def evaluate_matching(pred, true, slices):
    top1 = []
    diag_mse = []
    best_offdiag_mse = []
    margins = []

    for a, b in slices:
        p = pred[a:b]
        t = true[a:b]
        k = b - a

        dist = mse_matrix(p, t)

        for r in range(k):
            correct = r
            order = np.argsort(dist[r])
            top1.append(float(order[0] == correct))

            diag = float(dist[r, correct])
            off = np.delete(dist[r], correct)
            best_off = float(np.min(off))

            diag_mse.append(diag)
            best_offdiag_mse.append(best_off)
            margins.append(best_off - diag)

    return {
        "top1_accuracy": float(np.mean(top1)),
        "mean_diag_mse": float(np.mean(diag_mse)),
        "mean_best_offdiag_mse": float(np.mean(best_offdiag_mse)),
        "mean_margin_best_offdiag_minus_diag": float(np.mean(margins)),
        "positive_margin_frac": float(np.mean(np.array(margins) > 0.0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
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
    model, stats = load_model(args.checkpoint, data, args.device)

    metrics = json.loads((args.checkpoint.parent / "metrics.json").read_text())
    alpha = float(metrics["val"]["global_alpha"])

    flat, slices = build_groups(
        data=data,
        max_groups=args.max_groups,
        min_candidates=args.candidates,
        seed=args.seed,
    )

    base = {}
    n_samples = data["action"].shape[0]

    for k in data.keys():
        v = data[k]

        # Per-sample arrays have first dimension equal to n_samples.
        # Scalar metadata arrays such as encoder/latent_dim/action_dim must be copied unchanged.
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n_samples:
            base[k] = v[flat]
        else:
            base[k] = v

    true = base["z_future"].astype(np.float32)
    z0 = base["z_current"].astype(np.float32)
    original_actions = base["action"].astype(np.float32)

    rows = []

    for mode in ["original", "zero", "within_group_shuffle", "within_group_reverse"]:
        print(f"Evaluating {mode}", flush=True)

        dmode = dict(base)
        dmode["action"] = make_mode_actions(original_actions, slices, mode, args.seed)

        delta = predict_delta(model, dmode, stats, args.batch_size, args.device)
        pred = z0 + alpha * delta

        r = evaluate_matching(pred, true, slices)
        r["mode"] = mode
        rows.append(r)

    lines = [
        "# Action candidate matching diagnostic",
        "",
        "For each current frame, the model predicts multiple candidate futures using different actions from the same state. The prediction should be closest to the matching future if the action is used.",
        "",
        f"Groups: `{len(slices)}`. Candidates per group: `{args.candidates}`. Alpha: `{alpha:.6f}`.",
        "",
        "| inference action | top-1 candidate accuracy | mean diag MSE | mean best offdiag MSE | margin best offdiag - diag | positive margin frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['top1_accuracy']:.6f} | "
            f"{r['mean_diag_mse']:.6f} | {r['mean_best_offdiag_mse']:.6f} | "
            f"{r['mean_margin_best_offdiag_minus_diag']:.6f} | "
            f"{r['positive_margin_frac']:.6f} |"
        )

    path = args.out_dir / "action_candidate_matching_diagnostic.md"
    path.write_text("\n".join(lines) + "\n")

    print(path)
    print(path.read_text())


if __name__ == "__main__":
    main()
