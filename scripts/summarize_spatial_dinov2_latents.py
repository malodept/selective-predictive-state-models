from __future__ import annotations

import json
from pathlib import Path


def load(path: Path):
    return json.loads((path / "metrics.json").read_text())


runs = {
    "CLS directional λcos=0.01": Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos001_lnorm0001"),
    "spatial MSE-only": Path("outputs/envsplit_dinov2_spatial_mse_residual_seed0"),
    "spatial directional λcos=0.01": Path("outputs/envsplit_dinov2_spatial_directional_residual_seed0_lcos001_lnorm0001"),
}

rows = []

for name, path in runs.items():
    m = load(path)
    r = m["test"]
    rows.append({
        "method": name,
        "identity_error": r["identity_error"],
        "raw_error": r["raw_error"],
        "global_alpha": r["global_alpha"],
        "global_error": r["global_error"],
        "improvement": r["global_improvement_vs_identity"],
        "relative_improvement_pct": 100.0 * r["global_improvement_vs_identity"] / r["identity_error"],
        "cosine_mean": r["cosine_mean"],
        "positive_frac": r["cosine_positive_frac"],
        "true_norm": r["true_delta_norm_median"],
        "pred_norm": r["pred_delta_norm_median"],
    })

lines = [
    "# Spatial DINOv2 latent diagnostic",
    "",
    "This diagnostic compares the original DINOv2 CLS latent with a spatial DINOv2 latent built as `[CLS, patch_mean, patch_std]`.",
    "",
    "| method | identity error | raw error | global alpha | global error | improvement vs identity | relative improvement | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for r in rows:
    lines.append(
        f"| {r['method']} | "
        f"{r['identity_error']:.6f} | {r['raw_error']:.6f} | {r['global_alpha']:.6f} | "
        f"{r['global_error']:.6f} | {r['improvement']:.6f} | "
        f"{r['relative_improvement_pct']:.3f}% | {r['cosine_mean']:.6f} | "
        f"{r['positive_frac']:.6f} | {r['true_norm']:.6f} | {r['pred_norm']:.6f} |"
    )

path = Path("reports/tables/protocol/spatial_dinov2_latents/spatial_dinov2_latent_summary.md")
path.write_text("\n".join(lines) + "\n")
print(path)
print(path.read_text())
