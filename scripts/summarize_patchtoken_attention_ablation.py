from __future__ import annotations

import json
import math
from pathlib import Path


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pm(xs):
    return f"{mean(xs):.6f} ± {std(xs):.6f}"


configs = {
    "patch-token MLP, no token attention": [
        Path("outputs/envsplit_dinov2_patchtokens_mlp_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_mlp_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_mlp_mse_seed2"),
    ],
    "patch-token Transformer, token attention": [
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed2"),
    ],
}

out_dir = Path("reports/tables/protocol/patchtoken_transformer")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Patch-token attention ablation",
    "",
    "Both models receive the same 16 DINOv2 patch tokens and the same 7D action. The MLP predicts each token independently, while the Transformer allows token-token attention.",
    "",
    "| model | seeds | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | global alpha | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

summary = {}

for name, paths in configs.items():
    vals = []

    for path in paths:
        m = json.loads((path / "metrics.json").read_text())
        r = m["test"]
        vals.append({
            "identity": r["identity_error"],
            "global_error": r["global_error"],
            "gain": r["global_improvement_vs_identity"],
            "relative": 100.0 * r["global_improvement_vs_identity"] / r["identity_error"],
            "cosine": r["cosine_mean"],
            "positive": r["cosine_positive_frac"],
            "alpha": r["global_alpha"],
            "pred_norm": r["pred_delta_norm_median"],
        })

    summary[name] = vals

    lines.append(
        f"| {name} | {len(vals)} | "
        f"{pm([v['identity'] for v in vals])} | "
        f"{pm([v['global_error'] for v in vals])} | "
        f"{pm([v['gain'] for v in vals])} | "
        f"{pm([v['relative'] for v in vals])}% | "
        f"{pm([v['cosine'] for v in vals])} | "
        f"{pm([v['positive'] for v in vals])} | "
        f"{pm([v['alpha'] for v in vals])} | "
        f"{pm([v['pred_norm'] for v in vals])} |"
    )

mlp_gain = mean([v["gain"] for v in summary["patch-token MLP, no token attention"]])
tr_gain = mean([v["gain"] for v in summary["patch-token Transformer, token attention"]])
ratio = tr_gain / mlp_gain

lines += [
    "",
    "## Interpretation",
    "",
    f"- Mean Transformer gain is `{tr_gain:.6f}`, compared with `{mlp_gain:.6f}` for the token-wise MLP.",
    f"- The Transformer achieves approximately `{ratio:.2f}x` the calibrated gain of the no-attention token-wise baseline.",
    "- This supports the conclusion that the improvement is not merely due to exposing more DINOv2 patch information; token-token interaction is important for OOD latent dynamics.",
]

path = out_dir / "patchtoken_attention_ablation_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
