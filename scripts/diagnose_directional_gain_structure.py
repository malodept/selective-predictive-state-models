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

from scripts.plot_latent_geometry_diagnostics import predict_delta_directional
from scripts.eval_conditional_residual_shrinkage import load_npz


def per_sample_mse(a, b):
    return np.mean((a - b) ** 2, axis=1)


def summarize(name, gain, identity_err, model_err, action_norm, gap):
    return {
        "group": name,
        "n": int(len(gain)),
        "mean_gain": float(np.mean(gain)),
        "median_gain": float(np.median(gain)),
        "positive_frac": float(np.mean(gain > 0)),
        "identity_error": float(np.mean(identity_err)),
        "model_error": float(np.mean(model_err)),
        "action_norm": float(np.mean(action_norm)),
        "gap": float(np.mean(gap)),
    }


def write_md(path: Path, rows, title: str):
    lines = [
        f"# {title}",
        "",
        "| group | n | mean gain | median gain | positive frac | identity error | model error | action norm | gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['group']} | {r['n']} | {r['mean_gain']:.6f} | {r['median_gain']:.6f} | "
            f"{r['positive_frac']:.6f} | {r['identity_error']:.6f} | {r['model_error']:.6f} | "
            f"{r['action_norm']:.6f} | {r['gap']:.3f} |"
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
    y = data["z_future"].astype(np.float32)
    action = data["action"].astype(np.float32)
    action_norm = np.linalg.norm(action, axis=1)
    gap = data["gap"].astype(np.int32)

    print("Predicting directional residuals...")
    delta = predict_delta_directional(args.checkpoint, data, args.batch_size, args.device)

    # Global alpha chosen on the test predictions only for diagnostic geometry consistency.
    # The deployed alpha is already selected on validation inside metrics.json.
    metrics_path = args.checkpoint.parent / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    alpha = float(metrics["test"]["global_alpha"])

    pred = z0 + alpha * delta

    identity_err = per_sample_mse(z0, y)
    model_err = per_sample_mse(pred, y)
    gain = identity_err - model_err

    rows = []
    rows.append(summarize("ALL", gain, identity_err, model_err, action_norm, gap))

    for g in sorted(set(gap.tolist())):
        m = gap == g
        rows.append(summarize(f"gap={g}", gain[m], identity_err[m], model_err[m], action_norm[m], gap[m]))

    ranks = np.argsort(np.argsort(action_norm))
    bins = np.floor(5 * ranks / len(ranks)).astype(int)
    bins = np.clip(bins, 0, 4)

    for b in range(5):
        m = bins == b
        rows.append(summarize(f"action_bin={b}", gain[m], identity_err[m], model_err[m], action_norm[m], gap[m]))

    out = args.out_dir / "directional_gain_structure.md"
    write_md(out, rows, "Directional residual gain structure")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
