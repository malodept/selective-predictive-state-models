from __future__ import annotations

import json
from pathlib import Path


runs = {
    "CLS directional λcos=0.01": Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos001_lnorm0001"),
    "spatial stats MSE-only": Path("outputs/envsplit_dinov2_spatial_mse_residual_seed0"),
    "spatial stats directional λcos=0.01": Path("outputs/envsplit_dinov2_spatial_directional_residual_seed0_lcos001_lnorm0001"),
    "patch-grid MSE-only": Path("outputs/envsplit_dinov2_patchgrid_mse_residual_seed0"),
    "patch-grid directional λcos=0.01": Path("outputs/envsplit_dinov2_patchgrid_directional_residual_seed0_lcos001_lnorm0001"),
}

rows = []

for name, path in runs.items():
    m = json.loads((path / "metrics.json").read_text())
    r = m["test"]

    rows.append({
        "method": name,
        "identity": r["identity_error"],
        "raw": r["raw_error"],
        "alpha": r["global_alpha"],
        "global_error": r["global_error"],
        "gain": r["global_improvement_vs_identity"],
        "relative": 100.0 * r["global_improvement_vs_identity"] / r["identity_error"],
        "cosine": r["cosine_mean"],
        "positive": r["cosine_positive_frac"],
        "true_norm": r["true_delta_norm_median"],
        "pred_norm": r["pred_delta_norm_median"],
    })

out_dir = Path("reports/tables/protocol/patchgrid_dinov2_latents")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Patch-grid DINOv2 latent pilot",
    "",
    "This pilot compares the previous CLS and spatial-statistics DINOv2 latents against a projected patch-grid latent.",
    "",
    "Patch-grid latent: `[CLS, patch_mean, patch_std, random_projection(pool_4x4(patch_tokens))]`.",
    "",
    "| method | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for r in rows:
    lines.append(
        f"| {r['method']} | "
        f"{r['identity']:.6f} | {r['raw']:.6f} | {r['alpha']:.6f} | "
        f"{r['global_error']:.6f} | {r['gain']:.6f} | {r['relative']:.3f}% | "
        f"{r['cosine']:.6f} | {r['positive']:.6f} | "
        f"{r['true_norm']:.6f} | {r['pred_norm']:.6f} |"
    )

lines += [
    "",
    "## Interpretation",
    "",
    "- The patch-grid representation improves over identity after calibration.",
    "- Directional loss strongly improves patch-grid dynamics compared with MSE-only.",
    "- However, projected patch-grid latents do not outperform spatial-statistics latents in relative calibrated gain.",
    "- This suggests that preserving patch-level information is not sufficient if the downstream predictor collapses it into a global vector; a true patch-token or attention-based dynamics model is likely required.",
]

path = out_dir / "patchgrid_dinov2_pilot_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
