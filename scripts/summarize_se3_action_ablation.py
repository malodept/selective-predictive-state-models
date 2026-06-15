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
    "naive pose difference 7D": [
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed2"),
    ],
    "relative SE(3) 6D": [
        Path("outputs/envsplit_dinov2_patchtokens_transformer_se3_mse_seed0"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_se3_mse_seed1"),
        Path("outputs/envsplit_dinov2_patchtokens_transformer_se3_mse_seed2"),
    ],
}

out_dir = Path("reports/tables/protocol/patchtoken_transformer")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Action representation ablation: naive pose difference vs relative SE(3)",
    "",
    "Both models use the same patch-token Transformer architecture and the same DINOv2 4x4 patch-token states. Only the action representation changes.",
    "",
    "| action representation | seeds | action dim | identity error | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | global alpha | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

summary = {}

for name, paths in configs.items():
    vals = []
    for path in paths:
        m = json.loads((path / "metrics.json").read_text())
        r = m["test"]
        args = m.get("args", {})
        vals.append({
            "action_dim": args.get("action_dim", None),
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
    action_dim = "7" if "naive" in name else "6"

    lines.append(
        f"| {name} | {len(vals)} | {action_dim} | "
        f"{pm([v['identity'] for v in vals])} | "
        f"{pm([v['global_error'] for v in vals])} | "
        f"{pm([v['gain'] for v in vals])} | "
        f"{pm([v['relative'] for v in vals])}% | "
        f"{pm([v['cosine'] for v in vals])} | "
        f"{pm([v['positive'] for v in vals])} | "
        f"{pm([v['alpha'] for v in vals])} | "
        f"{pm([v['pred_norm'] for v in vals])} |"
    )

naive_gain = mean([v["gain"] for v in summary["naive pose difference 7D"]])
se3_gain = mean([v["gain"] for v in summary["relative SE(3) 6D"]])

lines += [
    "",
    "## Interpretation",
    "",
    f"- Mean naive-action gain: `{naive_gain:.6f}`.",
    f"- Mean SE(3)-action gain: `{se3_gain:.6f}`.",
    "- Relative SE(3) is geometrically cleaner and slightly improves the multiseed mean, but the effect is small.",
    "- This suggests that the current model is not yet strongly exploiting the full geometry of camera motion.",
]

path = out_dir / "se3_action_ablation_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
