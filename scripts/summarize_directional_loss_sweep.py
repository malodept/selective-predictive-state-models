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


def load_metrics(path: Path):
    return json.loads((path / "metrics.json").read_text())


configs = {
    "lambda_cos=0.01": [
        Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos001_lnorm0001"),
        Path("outputs/envsplit_dinov2_directional_residual_seed1_lcos001_lnorm0001"),
        Path("outputs/envsplit_dinov2_directional_residual_seed2_lcos001_lnorm0001"),
    ],
    "lambda_cos=0.05": [
        Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos005_lnorm0001"),
        Path("outputs/envsplit_dinov2_directional_residual_seed1_lcos005_lnorm0001"),
        Path("outputs/envsplit_dinov2_directional_residual_seed2_lcos005_lnorm0001"),
    ],
}

seed0_sweep = [
    ("lambda_cos=0.01", Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos001_lnorm0001")),
    ("lambda_cos=0.02", Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos002_lnorm0001")),
    ("lambda_cos=0.05", Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos005_lnorm0001")),
    ("lambda_cos=0.10", Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos010_lnorm0001")),
]

out_dir = Path("reports/tables/protocol/directional_loss_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Directional loss sweep summary",
    "",
    "The directional residual loss is `MSE + lambda_cos * cosine_loss + 0.001 * lognorm_loss`.",
    "",
    "## Multiseed comparison",
    "",
    "| config | seeds | test global error | test improvement vs identity | test cosine mean | test cosine positive frac | test global alpha | test pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for name, dirs in configs.items():
    ms = [load_metrics(d) for d in dirs]
    rows = [m["test"] for m in ms]
    lines.append(
        f"| {name} | {len(rows)} | "
        f"{pm([r['global_error'] for r in rows])} | "
        f"{pm([r['global_improvement_vs_identity'] for r in rows])} | "
        f"{pm([r['cosine_mean'] for r in rows])} | "
        f"{pm([r['cosine_positive_frac'] for r in rows])} | "
        f"{pm([r['global_alpha'] for r in rows])} | "
        f"{pm([r['pred_delta_norm_median'] for r in rows])} |"
    )

lines += [
    "",
    "## Seed-0 lambda sweep",
    "",
    "| config | test global error | test improvement vs identity | test cosine mean | test cosine positive frac | test global alpha | test pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for name, d in seed0_sweep:
    r = load_metrics(d)["test"]
    lines.append(
        f"| {name} | "
        f"{r['global_error']:.6f} | "
        f"{r['global_improvement_vs_identity']:.6f} | "
        f"{r['cosine_mean']:.6f} | "
        f"{r['cosine_positive_frac']:.6f} | "
        f"{r['global_alpha']:.6f} | "
        f"{r['pred_delta_norm_median']:.6f} |"
    )

path = out_dir / "directional_loss_sweep_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
