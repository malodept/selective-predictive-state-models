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
    "naive pose-difference action 7D": [
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed2"),
    ],
    "relative SE(3) action 6D": [
        Path("outputs/envsplit_dinov2_patchtokens_transformer_se3_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_se3_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_se3_mse_seed2"),
    ],
    "no action": [
        Path("outputs/envsplit_dinov2_patchtokens_transformer_noaction_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_noaction_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_noaction_mse_seed2"),
    ],
}

out_dir = Path("reports/tables/protocol/patchtoken_transformer")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Action-necessity ablation for patch-token dynamics",
    "",
    "All models use the same patch-token Transformer and the same DINOv2 4x4 token states. Only the action input changes.",
    "",
    "| action input | seeds | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | global alpha | pred Δ norm med | best epoch |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            "best_epoch": float(m.get("best_epoch", 0)),
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
        f"{pm([v['pred_norm'] for v in vals])} | "
        f"{pm([v['best_epoch'] for v in vals])} |"
    )

naive_gain = mean([v["gain"] for v in summary["naive pose-difference action 7D"]])
se3_gain = mean([v["gain"] for v in summary["relative SE(3) action 6D"]])
noaction_gain = mean([v["gain"] for v in summary["no action"]])

lines += [
    "",
    "## Interpretation",
    "",
    f"- Mean gain with naive 7D action: `{naive_gain:.6f}`.",
    f"- Mean gain with relative SE(3) action: `{se3_gain:.6f}`.",
    f"- Mean gain with no action: `{noaction_gain:.6f}`.",
    "- Under this short-horizon protocol, removing the action does not hurt performance.",
    "- The current result therefore supports strong patch-token latent dynamics, but not yet a strong action-conditioned world-model claim.",
    "- A stronger action-conditioned protocol should introduce counterfactual or ambiguous futures where the same current observation can lead to different next states depending on the action.",
]

path = out_dir / "action_necessity_ablation_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
