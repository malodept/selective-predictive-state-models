from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_conditional_residual_shrinkage import (
    load_npz,
    torch_stats_to_numpy,
    normalized_input,
    build_model,
    predict_delta,
    oracle_alpha,
    mean_mse,
)
from scripts.eval_ensemble_uncertainty_shrinkage import best_alpha_closed_form


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint-template", type=str, required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--model", choices=["cheap_residual", "expensive_residual"], default="cheap_residual")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def per_sample_mse(a, b):
    return np.mean((a - b) ** 2, axis=1)


def compute_geometry(data, delta_hat, alpha_global):
    z0 = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)
    true_delta = y - z0

    pred_global = z0 + alpha_global * delta_hat

    true_norm = np.linalg.norm(true_delta, axis=1)
    pred_norm = np.linalg.norm(delta_hat, axis=1)

    dot = np.sum(delta_hat * true_delta, axis=1)
    cosine = dot / (pred_norm * true_norm + 1e-12)
    norm_ratio = np.where(true_norm > 1e-3, pred_norm / true_norm, np.nan)

    alpha_star = oracle_alpha(delta_hat, true_delta)

    action = data["action"].astype(np.float32)
    action_norm = np.linalg.norm(action, axis=1)

    if "gap" in data:
        gap = data["gap"].astype(np.int32)
    else:
        gap = np.zeros(len(z0), dtype=np.int32)

    return pd.DataFrame({
        "identity_error": per_sample_mse(z0, y),
        "global_error": per_sample_mse(pred_global, y),
        "improvement_global": per_sample_mse(z0, y) - per_sample_mse(pred_global, y),
        "true_delta_norm": true_norm,
        "pred_delta_norm": pred_norm,
        "norm_ratio": norm_ratio,
        "cosine": cosine,
        "alpha_oracle": alpha_star,
        "action_norm": action_norm,
        "gap": gap,
    })


def summarize_group(split, group_name, df):
    return {
        "split": split,
        "group": group_name,
        "n": len(df),
        "identity_error": df["identity_error"].mean(),
        "global_error": df["global_error"].mean(),
        "improvement_global": df["improvement_global"].mean(),
        "cosine_mean": df["cosine"].mean(),
        "cosine_median": df["cosine"].median(),
        "cosine_positive_frac": float((df["cosine"] > 0).mean()),
        "true_delta_norm": float(df["true_delta_norm"].median()),
        "pred_delta_norm": float(df["pred_delta_norm"].median()),
        "norm_ratio": float(np.nanmedian(df["norm_ratio"])),
        "alpha_oracle_mean": df["alpha_oracle"].mean(),
        "alpha_oracle_std": df["alpha_oracle"].std(),
        "action_norm": df["action_norm"].mean(),
    }


def write_md(path, rows):
    lines = [
        "# Residual geometry diagnostic",
        "",
        "This diagnostic decomposes residual prediction into direction alignment and magnitude calibration.",
        "",
        "| split | group | n | improvement global | cosine mean | cosine positive frac | true Δ norm | pred Δ norm | norm ratio | oracle α mean | action norm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['split']} | {r['group']} | {r['n']} | "
            f"{r['improvement_global']:.6f} | {r['cosine_mean']:.6f} | {r['cosine_positive_frac']:.6f} | "
            f"{r['true_delta_norm']:.6f} | {r['pred_delta_norm']:.6f} | {r['norm_ratio']:.6f} | "
            f"{r['alpha_oracle_mean']:.6f} | {r['action_norm']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    val = load_npz(args.val)
    test = load_npz(args.test)

    deltas = {"val": [], "test": []}

    for seed in args.seeds:
        print(f"Loading seed {seed}")
        ckpt = torch.load(Path(args.checkpoint_template.format(seed=seed)), map_location=args.device)
        stats = torch_stats_to_numpy(ckpt["stats"])
        model = build_model(ckpt, args.model, args.device)

        deltas["val"].append(predict_delta(model, normalized_input(val, stats), stats, args.batch_size, args.device))
        deltas["test"].append(predict_delta(model, normalized_input(test, stats), stats, args.batch_size, args.device))

    delta_val = np.mean(np.stack(deltas["val"], axis=0), axis=0).astype(np.float32)
    delta_test = np.mean(np.stack(deltas["test"], axis=0), axis=0).astype(np.float32)

    alpha_global = best_alpha_closed_form(
        val["z_current"].astype(np.float32),
        val["z_future"].astype(np.float32),
        delta_val,
    )

    frames = {
        "val": compute_geometry(val, delta_val, alpha_global),
        "test": compute_geometry(test, delta_test, alpha_global),
    }

    rows = []

    for split, df in frames.items():
        rows.append(summarize_group(split, "ALL", df))

        for gap, g in df.groupby("gap"):
            rows.append(summarize_group(split, f"gap={gap}", g))

        df2 = df.copy()
        df2["action_bin"] = pd.qcut(df2["action_norm"].rank(method="first"), q=5, labels=False)
        for b, g in df2.groupby("action_bin"):
            rows.append(summarize_group(split, f"action_bin={int(b)}", g))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = args.out_dir / "residual_geometry_summary.csv"
    out_md = args.out_dir / "residual_geometry_summary.md"

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_md(out_md, rows)

    print("alpha_global:", alpha_global)
    print(out_md)
    print(out_md.read_text())


if __name__ == "__main__":
    main()
