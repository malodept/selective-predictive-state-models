from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_real_selective_refinement_bestval import (
    MLP,
    load_feature_npz,
    per_sample_mse,
    predict,
    reliability_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.json"
    ckpt_path = run_dir / "bestval_checkpoint.pt"

    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    metrics = json.loads(metrics_path.read_text())
    model_info = metrics["model"]

    val_x, val_y, val_info = load_feature_npz(metrics["val"])

    input_dim = int(model_info["input_dim"])
    latent_dim = int(model_info["latent_dim"])
    dropout = float(model_info.get("dropout", 0.05))

    cheap = MLP(
        input_dim,
        latent_dim,
        int(model_info["cheap_hidden"]),
        int(model_info["cheap_layers"]),
        dropout,
    ).to(args.device)

    expensive = MLP(
        input_dim,
        latent_dim,
        int(model_info["expensive_hidden"]),
        int(model_info["expensive_layers"]),
        dropout,
    ).to(args.device)

    reliability = MLP(
        input_dim,
        1,
        int(model_info["reliability_hidden"]),
        2,
        dropout,
    ).to(args.device)

    ckpt = torch.load(ckpt_path, map_location=args.device)
    cheap.load_state_dict(ckpt["cheap"])
    expensive.load_state_dict(ckpt["expensive"])
    reliability.load_state_dict(ckpt["reliability"])

    cheap_pred = predict(cheap, val_x, args.batch_size, args.device)
    expensive_pred = predict(expensive, val_x, args.batch_size, args.device)
    scores = reliability_scores(reliability, val_x, args.batch_size, args.device)

    cheap_errors = per_sample_mse(cheap_pred, val_y)
    expensive_errors = per_sample_mse(expensive_pred, val_y)

    hard_threshold = float(metrics["labeling"]["hard_threshold"])
    hard_labels = (cheap_errors >= hard_threshold).astype(np.int64)

    action_dim = int(val_info["action_dim"])
    if action_dim > 0:
        action = val_x[:, -action_dim:].astype(np.float32)
        action_norm = np.linalg.norm(action, axis=1).astype(np.float32)
    else:
        action = np.zeros((val_x.shape[0], 0), dtype=np.float32)
        action_norm = np.zeros((val_x.shape[0],), dtype=np.float32)

    out = run_dir / "eval_arrays.npz"
    np.savez_compressed(
        out,
        val_x=val_x.astype(np.float32),
        val_y=val_y.astype(np.float32),
        cheap_pred=cheap_pred.astype(np.float32),
        expensive_pred=expensive_pred.astype(np.float32),
        reliability_scores=scores.astype(np.float32),
        cheap_errors=cheap_errors.astype(np.float32),
        expensive_errors=expensive_errors.astype(np.float32),
        hard_labels=hard_labels.astype(np.int64),
        action=action.astype(np.float32),
        action_norm=action_norm.astype(np.float32),
    )

    print(f"Wrote {out}")
    print(f"samples={len(val_y)}")
    print(f"cheap_error={cheap_errors.mean():.6f}")
    print(f"expensive_error={expensive_errors.mean():.6f}")
    print(f"score_mean={scores.mean():.6f}")
    print(f"action_norm_mean={action_norm.mean():.6f}")


if __name__ == "__main__":
    main()
