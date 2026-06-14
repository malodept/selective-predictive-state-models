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

from scripts.eval_conditional_residual_shrinkage import load_npz
from scripts.plot_latent_geometry_diagnostics import predict_delta_directional


def per_sample_mse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean((a - b) ** 2, axis=1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return dot / (na * nb + 1e-12)


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    sorted_x = x[order]

    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = average_ranks(s)
    rank_sum_pos = float(ranks[y].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def summarize_group(name: str, mask: np.ndarray, signals: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    row = {"group": name, "n": int(mask.sum())}

    for key, x in signals.items():
        xm = x[mask]
        row[f"{key}_mean"] = float(np.mean(xm)) if len(xm) else float("nan")
        row[f"{key}_median"] = float(np.median(xm)) if len(xm) else float("nan")

    return row


def fmt(x):
    if isinstance(x, int):
        return str(x)
    if isinstance(x, str):
        return x
    if np.isnan(x):
        return "nan"
    return f"{x:.6f}"


def write_group_table(path: Path, rows: list[dict]):
    keys = [
        "group", "n",
        "gain_mean", "gain_median",
        "gap_mean", "gap_median",
        "action_norm_mean", "action_norm_median",
        "pred_norm_mean", "pred_norm_median",
        "true_delta_norm_mean", "true_delta_norm_median",
        "cosine_mean", "cosine_median",
        "identity_error_mean", "residual_error_mean",
    ]

    lines = [
        "# Positive-gain profile for directional residual",
        "",
        "This diagnostic compares samples where the calibrated residual improves over identity (`gain > 0`) with samples where it hurts.",
        "",
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in keys[1:]]) + " |",
    ]

    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k, float("nan"))) for k in keys) + " |")

    path.write_text("\n".join(lines) + "\n")


def write_auc_table(path: Path, auc_rows: list[dict]):
    lines = [
        "# Deployable and oracle signals for positive-gain detection",
        "",
        "Label: `gain > 0`, where `gain = error(identity) - error(calibrated residual)`.",
        "",
        "| signal | availability | AUROC | best-direction AUROC | best direction |",
        "| --- | --- | ---: | ---: | --- |",
    ]

    for r in auc_rows:
        lines.append(
            f"| {r['signal']} | {r['availability']} | {r['auc']:.6f} | "
            f"{r['best_auc']:.6f} | {r['best_direction']} |"
        )

    path.write_text("\n".join(lines) + "\n")


def write_quantile_table(path: Path, signals: dict[str, np.ndarray], gain: np.ndarray):
    lines = [
        "# Gain by deployable signal quantiles",
        "",
        "| signal | bin | n | signal low | signal high | mean gain | median gain | positive frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for name in ["gap", "action_norm", "pred_norm"]:
        x = signals[name]

        if name == "gap":
            bins = sorted(set(x.astype(int).tolist()))
            for b in bins:
                m = x == b
                lines.append(
                    f"| {name} | {b} | {int(m.sum())} | {float(b):.6f} | {float(b):.6f} | "
                    f"{float(np.mean(gain[m])):.6f} | {float(np.median(gain[m])):.6f} | "
                    f"{float(np.mean(gain[m] > 0)):.6f} |"
                )
        else:
            qs = np.quantile(x, np.linspace(0, 1, 6))
            for b in range(5):
                if b < 4:
                    m = (x >= qs[b]) & (x < qs[b + 1])
                else:
                    m = (x >= qs[b]) & (x <= qs[b + 1])
                lines.append(
                    f"| {name} | {b} | {int(m.sum())} | {float(qs[b]):.6f} | {float(qs[b + 1]):.6f} | "
                    f"{float(np.mean(gain[m])):.6f} | {float(np.median(gain[m])):.6f} | "
                    f"{float(np.mean(gain[m] > 0)):.6f} |"
                )

    path.write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    data = load_npz(args.test)
    z0 = data["z_current"].astype(np.float32)
    z1 = data["z_future"].astype(np.float32)
    action = data["action"].astype(np.float32)
    gap = data["gap"].astype(np.float32)

    print("Predicting residuals...")
    delta = predict_delta_directional(args.checkpoint, data, args.batch_size, args.device)

    metrics = json.loads((args.checkpoint.parent / "metrics.json").read_text())
    alpha = float(metrics["val"]["global_alpha"])

    pred = z0 + alpha * delta
    true_delta = z1 - z0

    identity_error = per_sample_mse(z0, z1)
    residual_error = per_sample_mse(pred, z1)
    gain = identity_error - residual_error

    signals = {
        "gain": gain,
        "gap": gap,
        "action_norm": np.linalg.norm(action, axis=1),
        "pred_norm": np.linalg.norm(delta, axis=1),
        "true_delta_norm": np.linalg.norm(true_delta, axis=1),
        "cosine": cosine(delta, true_delta),
        "identity_error": identity_error,
        "residual_error": residual_error,
    }

    positive = gain > 0

    group_rows = [
        summarize_group("ALL", np.ones_like(positive, dtype=bool), signals),
        summarize_group("gain>0", positive, signals),
        summarize_group("gain<=0", ~positive, signals),
    ]

    write_group_table(args.out_dir / "positive_gain_profile.md", group_rows)

    auc_rows = []
    auc_specs = [
        ("gap", "deployable"),
        ("action_norm", "deployable"),
        ("pred_norm", "deployable"),
        ("true_delta_norm", "oracle-only"),
        ("cosine", "oracle-only"),
        ("identity_error", "oracle-only"),
    ]

    for name, availability in auc_specs:
        s = signals[name]
        auc = auc_score(positive, s)
        best_auc = max(auc, 1.0 - auc)
        best_direction = "higher => positive gain" if auc >= 0.5 else "lower => positive gain"
        auc_rows.append({
            "signal": name,
            "availability": availability,
            "auc": auc,
            "best_auc": best_auc,
            "best_direction": best_direction,
        })

    write_auc_table(args.out_dir / "positive_gain_signal_auroc.md", auc_rows)
    write_quantile_table(args.out_dir / "gain_by_signal_quantiles.md", signals, gain)

    print(args.out_dir / "positive_gain_profile.md")
    print((args.out_dir / "positive_gain_profile.md").read_text())
    print(args.out_dir / "positive_gain_signal_auroc.md")
    print((args.out_dir / "positive_gain_signal_auroc.md").read_text())
    print(args.out_dir / "gain_by_signal_quantiles.md")
    print((args.out_dir / "gain_by_signal_quantiles.md").read_text())


if __name__ == "__main__":
    main()
