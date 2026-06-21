from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_mixed_hard_delta_transformer import DeltaTransformer


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def calibration_bins(conf: np.ndarray, target: np.ndarray, n_bins: int = 10):
    rows = []
    ece = 0.0
    n = len(conf)

    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        if b == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append({
                "bin": f"[{lo:.1f},{hi:.1f}]",
                "count": 0,
                "mean_confidence": None,
                "mean_target": None,
                "gap": None,
            })
            continue

        mc = float(conf[mask].mean())
        mt = float(target[mask].mean())
        gap = abs(mc - mt)
        ece += (count / n) * gap

        rows.append({
            "bin": f"[{lo:.1f},{hi:.1f}]",
            "count": count,
            "mean_confidence": mc,
            "mean_target": mt,
            "gap": float(gap),
        })

    return float(ece), rows


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--groups", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-groups", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    data = np.load(args.data, allow_pickle=True)
    groups = np.load(args.groups, allow_pickle=True)

    z = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    a = data["action"].astype(np.float32)

    anchor = groups["anchor_indices"].astype(np.int64)
    cand = groups["candidate_indices"].astype(np.int64)
    action_id = groups["action_id"].astype(np.int64)
    action_names = data["action_names"].astype(str)

    if "split" in groups.files:
        test_groups = np.where(groups["split"].astype(str) == "test")[0]
        split_source = "explicit"
    else:
        raise ValueError("This diagnostic expects an explicit state-disjoint split.")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt["args"]

    model = DeltaTransformer(
        token_dim=int(ckpt["token_dim"]),
        action_dim=int(ckpt["action_dim"]),
        tokens=int(ckpt["tokens"]),
        model_dim=int(ckpt_args["model_dim"]),
        heads=int(ckpt_args["heads"]),
        layers=int(ckpt_args["layers"]),
        dropout=float(ckpt_args["dropout"]),
        mode=ckpt_args["mode"],
    ).to(args.device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    delta = y - z

    all_dist = []
    all_action_ids = []

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
        all_action_ids.append(action_id[gids])

    dist = np.concatenate(all_dist, axis=0)
    action_ids = np.concatenate(all_action_ids, axis=0)

    logits = -dist / args.temperature
    probs = softmax_np(logits, axis=1)

    pred = dist.argmin(axis=1)
    confidence = probs.max(axis=1)

    d0 = dist[:, 0]
    min_dist = dist.min(axis=1)
    strict_correct = (d0 < np.min(dist[:, 1:], axis=1) - args.eps).astype(float)

    tieaware_target = []
    tie_count = []
    correct_tied = []

    for i in range(dist.shape[0]):
        tied = np.where(np.abs(dist[i] - min_dist[i]) <= args.eps)[0]
        tie_count.append(len(tied))
        if 0 in tied:
            tieaware_target.append(1.0 / len(tied))
            correct_tied.append(float(len(tied) > 1))
        else:
            tieaware_target.append(0.0)
            correct_tied.append(0.0)

    tieaware_target = np.asarray(tieaware_target, dtype=float)
    tie_count = np.asarray(tie_count, dtype=float)
    correct_tied = np.asarray(correct_tied, dtype=float)

    biased_correct = (pred == 0).astype(float)

    ece_tieaware, bins_tieaware = calibration_bins(confidence, tieaware_target, n_bins=10)
    ece_strict, bins_strict = calibration_bins(confidence, strict_correct, n_bins=10)

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    order = np.argsort(-confidence)

    selective_rows = []
    stay_id = int(np.where(action_names == "stay")[0][0]) if "stay" in action_names else -1

    for cov in coverages:
        k = max(1, int(round(cov * len(order))))
        keep = order[:k]
        selective_rows.append({
            "coverage": float(cov),
            "kept": int(k),
            "mean_confidence": float(confidence[keep].mean()),
            "biased_top1": float(biased_correct[keep].mean()),
            "strict_top1": float(strict_correct[keep].mean()),
            "tieaware_top1": float(tieaware_target[keep].mean()),
            "stay_fraction": float((action_ids[keep] == stay_id).mean()) if stay_id >= 0 else None,
            "mean_tie_count": float(tie_count[keep].mean()),
        })

    action_rows = []
    for aid in sorted(set(action_ids.tolist())):
        mask = action_ids == aid
        action_rows.append({
            "action": str(action_names[aid]),
            "count": int(mask.sum()),
            "mean_confidence": float(confidence[mask].mean()),
            "biased_top1": float(biased_correct[mask].mean()),
            "strict_top1": float(strict_correct[mask].mean()),
            "tieaware_top1": float(tieaware_target[mask].mean()),
            "correct_tied_frac": float(correct_tied[mask].mean()),
            "mean_tie_count": float(tie_count[mask].mean()),
        })

    report = {
        "checkpoint": str(args.checkpoint),
        "mode": ckpt_args["mode"],
        "split_source": split_source,
        "test_groups": int(len(test_groups)),
        "temperature": float(args.temperature),
        "eps": float(args.eps),
        "mean_confidence": float(confidence.mean()),
        "biased_top1": float(biased_correct.mean()),
        "strict_top1": float(strict_correct.mean()),
        "tieaware_top1": float(tieaware_target.mean()),
        "correct_tied_frac": float(correct_tied.mean()),
        "mean_tie_count": float(tie_count.mean()),
        "ece_tieaware": ece_tieaware,
        "ece_strict": ece_strict,
        "selective_rows": selective_rows,
        "action_rows": action_rows,
        "calibration_bins_tieaware": bins_tieaware,
        "calibration_bins_strict": bins_strict,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    md = args.out.with_suffix(".md")
    lines = [
        f"# State-disjoint confidence diagnostic: {ckpt_args['mode']}",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- split source: `{split_source}`",
        f"- test groups: `{len(test_groups)}`",
        f"- temperature: `{args.temperature}`",
        "",
        "## Global metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mean confidence | {confidence.mean():.6f} |",
        f"| biased top-1 | {biased_correct.mean():.6f} |",
        f"| strict top-1 | {strict_correct.mean():.6f} |",
        f"| tie-aware top-1 | {tieaware_target.mean():.6f} |",
        f"| correct tied with another candidate | {correct_tied.mean():.6f} |",
        f"| mean tie count | {tie_count.mean():.6f} |",
        f"| ECE vs tie-aware target | {ece_tieaware:.6f} |",
        f"| ECE vs strict target | {ece_strict:.6f} |",
        "",
        "## Selective prediction curve",
        "",
        "| coverage | kept | confidence | biased top-1 | strict top-1 | tie-aware top-1 | stay fraction | mean tie count |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in selective_rows:
        lines.append(
            f"| {r['coverage']:.2f} | {r['kept']} | {r['mean_confidence']:.6f} | "
            f"{r['biased_top1']:.6f} | {r['strict_top1']:.6f} | {r['tieaware_top1']:.6f} | "
            f"{r['stay_fraction']:.6f} | {r['mean_tie_count']:.6f} |"
        )

    lines += [
        "",
        "## Action breakdown",
        "",
        "| action | count | confidence | biased top-1 | strict top-1 | tie-aware top-1 | tied correct frac | mean tie count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in action_rows:
        lines.append(
            f"| {r['action']} | {r['count']} | {r['mean_confidence']:.6f} | "
            f"{r['biased_top1']:.6f} | {r['strict_top1']:.6f} | {r['tieaware_top1']:.6f} | "
            f"{r['correct_tied_frac']:.6f} | {r['mean_tie_count']:.6f} |"
        )

    lines += [
        "",
        "## Calibration bins against tie-aware target",
        "",
        "| confidence bin | count | mean confidence | mean target | gap |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for r in bins_tieaware:
        mc = "NA" if r["mean_confidence"] is None else f"{r['mean_confidence']:.6f}"
        mt = "NA" if r["mean_target"] is None else f"{r['mean_target']:.6f}"
        gap = "NA" if r["gap"] is None else f"{r['gap']:.6f}"
        lines.append(f"| {r['bin']} | {r['count']} | {mc} | {mt} | {gap} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "This diagnostic asks whether model confidence tracks strict and tie-aware correctness.",
        "A useful reliability signal should assign lower confidence to non-identifiable `stay` ties and should improve accuracy as coverage decreases.",
    ]

    md.write_text("\n".join(lines) + "\n")
    print(md)
    print(md.read_text())


if __name__ == "__main__":
    main()
