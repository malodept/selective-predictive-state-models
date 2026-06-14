from __future__ import annotations

import json
import math
from pathlib import Path


def load_metrics(path: Path):
    return json.loads((path / "metrics.json").read_text())


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
    "spatial MSE-only": [
        Path("outputs/envsplit_dinov2_spatial_mse_residual_seed0"),
        Path("outputs/envsplit_dinov2_spatial_mse_residual_seed1"),
        Path("outputs/envsplit_dinov2_spatial_mse_residual_seed2"),
    ],
    "spatial directional λcos=0.01": [
        Path("outputs/envsplit_dinov2_spatial_directional_residual_seed0_lcos001_lnorm0001"),
        Path("outputs/envsplit_dinov2_spatial_directional_residual_seed1_lcos001_lnorm0001"),
        Path("outputs/envsplit_dinov2_spatial_directional_residual_seed2_lcos001_lnorm0001"),
    ],
}

out_dir = Path("reports/tables/protocol/spatial_dinov2_latents")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Spatial DINOv2 latent multiseed summary",
    "",
    "This table evaluates DINOv2 spatial latents `[CLS, patch_mean, patch_std]` under the same environment-held-out protocol.",
    "",
    "| method | seeds | identity error | raw error | global alpha | global error | improvement vs identity | relative improvement | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for name, dirs in configs.items():
    metrics = [load_metrics(d)["test"] for d in dirs]

    identity = [m["identity_error"] for m in metrics]
    raw = [m["raw_error"] for m in metrics]
    alpha = [m["global_alpha"] for m in metrics]
    global_error = [m["global_error"] for m in metrics]
    improvement = [m["global_improvement_vs_identity"] for m in metrics]
    relative = [100.0 * m["global_improvement_vs_identity"] / m["identity_error"] for m in metrics]
    cosine = [m["cosine_mean"] for m in metrics]
    positive = [m["cosine_positive_frac"] for m in metrics]
    true_norm = [m["true_delta_norm_median"] for m in metrics]
    pred_norm = [m["pred_delta_norm_median"] for m in metrics]

    lines.append(
        f"| {name} | {len(metrics)} | "
        f"{pm(identity)} | {pm(raw)} | {pm(alpha)} | {pm(global_error)} | "
        f"{pm(improvement)} | {pm(relative)}% | {pm(cosine)} | {pm(positive)} | "
        f"{pm(true_norm)} | {pm(pred_norm)} |"
    )

path = out_dir / "spatial_dinov2_multiseed_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
