from __future__ import annotations

import argparse
import json
import math
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


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def compute_split(data, delta, alpha: float):
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    action = data["action"].astype(np.float32)
    gap = data["gap"].astype(np.float32)

    pred = z0 + alpha * delta

    identity_err = per_sample_mse(z0, y)
    residual_err = per_sample_mse(pred, y)
    gain = identity_err - residual_err

    action_norm = np.linalg.norm(action, axis=1)
    pred_norm = np.linalg.norm(delta, axis=1)

    return {
        "identity_err": identity_err,
        "residual_err": residual_err,
        "gain": gain,
        "gap": gap,
        "action_norm": action_norm,
        "pred_norm": pred_norm,
    }


def apply_policy(split, signal_name: str, threshold: float, direction: str):
    signal = split[signal_name]
    if direction == ">=":
        selected = signal >= threshold
    elif direction == "<=":
        selected = signal <= threshold
    else:
        raise ValueError(direction)

    err = np.where(selected, split["residual_err"], split["identity_err"])
    identity = split["identity_err"]

    return {
        "error": float(np.mean(err)),
        "identity_error": float(np.mean(identity)),
        "gain": float(np.mean(identity) - np.mean(err)),
        "selected": float(np.mean(selected)),
        "positive_gain_selected": float(np.mean(split["gain"][selected] > 0.0)) if selected.any() else 0.0,
        "selected_mean_gain": float(np.mean(split["gain"][selected])) if selected.any() else 0.0,
    }


def candidate_thresholds(x: np.ndarray):
    qs = np.linspace(0.0, 1.0, 51)
    vals = np.unique(np.quantile(x, qs))
    return vals.tolist()


def choose_threshold_on_val(split, signal_name: str):
    best = None

    for direction in [">=", "<="]:
        for thr in candidate_thresholds(split[signal_name]):
            r = apply_policy(split, signal_name, float(thr), direction)
            row = {
                "signal": signal_name,
                "threshold": float(thr),
                "direction": direction,
                **r,
            }
            if best is None or row["gain"] > best["gain"]:
                best = row

    return best


def format_row(row):
    return (
        f"| {row['policy']} | {row['signal']} | {row['direction']} | {row['threshold']:.6f} | "
        f"{row['val_gain']:.6f} | {row['test_gain']:.6f} | "
        f"{row['test_error']:.6f} | {row['test_selected']:.6f} | "
        f"{row['test_selected_mean_gain']:.6f} | {row['test_positive_gain_selected']:.6f} |"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    val_data = load_npz(args.val)
    test_data = load_npz(args.test)

    print("Predicting validation residuals...")
    val_delta = predict_delta_directional(args.checkpoint, val_data, args.batch_size, args.device)

    print("Predicting test residuals...")
    test_delta = predict_delta_directional(args.checkpoint, test_data, args.batch_size, args.device)

    metrics = json.loads((args.checkpoint.parent / "metrics.json").read_text())
    alpha = float(metrics["val"]["global_alpha"])

    val = compute_split(val_data, val_delta, alpha)
    test = compute_split(test_data, test_delta, alpha)

    rows = []

    # Always residual baseline
    always_val = {
        "gain": float(np.mean(val["gain"])),
    }
    always_test_err = float(np.mean(test["residual_err"]))
    always_test_gain = float(np.mean(test["identity_err"]) - always_test_err)

    rows.append({
        "policy": "always_residual",
        "signal": "none",
        "direction": "-",
        "threshold": 0.0,
        "val_gain": always_val["gain"],
        "test_gain": always_test_gain,
        "test_error": always_test_err,
        "test_selected": 1.0,
        "test_selected_mean_gain": float(np.mean(test["gain"])),
        "test_positive_gain_selected": float(np.mean(test["gain"] > 0.0)),
    })

    for signal in ["gap", "action_norm", "pred_norm"]:
        best_val = choose_threshold_on_val(val, signal)
        test_eval = apply_policy(test, signal, best_val["threshold"], best_val["direction"])

        rows.append({
            "policy": "val_selected_threshold",
            "signal": signal,
            "direction": best_val["direction"],
            "threshold": best_val["threshold"],
            "val_gain": best_val["gain"],
            "test_gain": test_eval["gain"],
            "test_error": test_eval["error"],
            "test_selected": test_eval["selected"],
            "test_selected_mean_gain": test_eval["selected_mean_gain"],
            "test_positive_gain_selected": test_eval["positive_gain_selected"],
        })

    # Oracle upper bound on test: apply residual exactly when sample gain is positive.
    oracle_selected = test["gain"] > 0.0
    oracle_err = np.where(oracle_selected, test["residual_err"], test["identity_err"])
    rows.append({
        "policy": "test_oracle_upper_bound",
        "signal": "gain",
        "direction": ">0",
        "threshold": 0.0,
        "val_gain": float("nan"),
        "test_gain": float(np.mean(test["identity_err"]) - np.mean(oracle_err)),
        "test_error": float(np.mean(oracle_err)),
        "test_selected": float(np.mean(oracle_selected)),
        "test_selected_mean_gain": float(np.mean(test["gain"][oracle_selected])),
        "test_positive_gain_selected": 1.0,
    })

    lines = [
        "# Directional selective residual thresholding",
        "",
        f"Residual alpha is selected on validation: `{alpha:.6f}`.",
        "",
        "Policies choose whether to use `z_current + alpha * delta_hat` or fall back to identity.",
        "",
        "| policy | signal | direction | threshold | val gain | test gain | test error | test selected | selected mean gain | selected positive frac |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(format_row(row))

    out = args.out_dir / "directional_selective_thresholds.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
