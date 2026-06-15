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

from scripts.train_patchtoken_action_transformer import (
    load_npz,
    TokenActionTransformer,
    predict_delta,
    evaluate_alpha,
    eval_with_alpha,
)


def intervention_data(data, mode: str, seed: int):
    out = {k: data[k] for k in data.keys()}

    action = data["action"].astype(np.float32).copy()

    if mode == "original":
        new_action = action
    elif mode == "zero":
        new_action = np.zeros_like(action)
    elif mode == "shuffled":
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(action))
        new_action = action[perm]
    else:
        raise ValueError(mode)

    out["action"] = new_action
    return out


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    val = load_npz(args.val)
    test = load_npz(args.test)

    model, stats = load_model(args.checkpoint, val, args.device)

    rows = []

    # First compute alpha on original validation actions.
    val_original = intervention_data(val, "original", args.seed)
    val_delta_original = predict_delta(model, val_original, stats, args.batch_size, args.device)
    alpha_original, _, _, _ = evaluate_alpha(
        val_original["z_current"].astype(np.float32),
        val_original["z_future"].astype(np.float32),
        val_delta_original,
    )

    for mode in ["original", "zero", "shuffled"]:
        print(f"Evaluating mode={mode}")

        val_m = intervention_data(val, mode, args.seed)
        test_m = intervention_data(test, mode, args.seed)

        val_delta = predict_delta(model, val_m, stats, args.batch_size, args.device)
        test_delta = predict_delta(model, test_m, stats, args.batch_size, args.device)

        # Metric A: fixed alpha from original validation actions.
        val_fixed = eval_with_alpha(
            val_m["z_current"].astype(np.float32),
            val_m["z_future"].astype(np.float32),
            val_delta,
            alpha_original,
        )
        test_fixed = eval_with_alpha(
            test_m["z_current"].astype(np.float32),
            test_m["z_future"].astype(np.float32),
            test_delta,
            alpha_original,
        )

        # Metric B: alpha recalibrated on validation under this intervention.
        alpha_mode, _, _, _ = evaluate_alpha(
            val_m["z_current"].astype(np.float32),
            val_m["z_future"].astype(np.float32),
            val_delta,
        )
        test_recal = eval_with_alpha(
            test_m["z_current"].astype(np.float32),
            test_m["z_future"].astype(np.float32),
            test_delta,
            alpha_mode,
        )

        rows.append({
            "mode": mode,
            "alpha_original": alpha_original,
            "alpha_mode": alpha_mode,
            "fixed_gain": test_fixed["global_improvement_vs_identity"],
            "fixed_error": test_fixed["global_error"],
            "fixed_cosine": test_fixed["cosine_mean"],
            "recal_gain": test_recal["global_improvement_vs_identity"],
            "recal_error": test_recal["global_error"],
            "recal_cosine": test_recal["cosine_mean"],
            "positive_frac": test_recal["cosine_positive_frac"],
        })

    lines = [
        "# Action intervention diagnostic",
        "",
        "A patch-token Transformer trained with the original 7D action is evaluated under action interventions at inference time.",
        "",
        f"Original validation alpha: `{alpha_original:.6f}`.",
        "",
        "| inference action | alpha original | alpha recalibrated | fixed-alpha gain | fixed-alpha error | fixed-alpha cosine | recalibrated gain | recalibrated error | recalibrated cosine | positive cosine frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['mode']} | {r['alpha_original']:.6f} | {r['alpha_mode']:.6f} | "
            f"{r['fixed_gain']:.6f} | {r['fixed_error']:.6f} | {r['fixed_cosine']:.6f} | "
            f"{r['recal_gain']:.6f} | {r['recal_error']:.6f} | {r['recal_cosine']:.6f} | "
            f"{r['positive_frac']:.6f} |"
        )

    path = args.out_dir / "action_intervention_diagnostic.md"
    path.write_text("\n".join(lines) + "\n")

    print(path)
    print(path.read_text())


if __name__ == "__main__":
    main()
