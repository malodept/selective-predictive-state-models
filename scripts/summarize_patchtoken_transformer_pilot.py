from __future__ import annotations

import json
from pathlib import Path

runs = {
    "patch-token transformer MSE-only": Path("outputs/envsplit_dinov2_patchtokens_transformer_mse_seed0"),
    "patch-token transformer directional": Path("outputs/envsplit_dinov2_patchtokens_transformer_seed0_lcos001_lnorm0001"),
    "patch-grid vector MSE-only": Path("outputs/envsplit_dinov2_patchgrid_mse_residual_seed0"),
    "patch-grid vector directional": Path("outputs/envsplit_dinov2_patchgrid_directional_residual_seed0_lcos001_lnorm0001"),
    "spatial stats directional": Path("outputs/envsplit_dinov2_spatial_directional_residual_seed0_lcos001_lnorm0001"),
    "CLS directional": Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos001_lnorm0001"),
}

out_dir = Path("reports/tables/protocol/patchtoken_transformer")
out_dir.mkdir(parents=True, exist_ok=True)

lines = [
    "# Patch-token Action Transformer pilot",
    "",
    "This pilot compares global-vector DINOv2 dynamics with a token-preserving action-conditioned transformer.",
    "",
    "| method | identity error | raw error | global alpha | global error | gain vs identity | relative gain | cosine mean | positive cosine frac | true Δ norm med | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for name, path in runs.items():
    m = json.loads((path / "metrics.json").read_text())
    r = m["test"]
    rel = 100.0 * r["global_improvement_vs_identity"] / r["identity_error"]

    lines.append(
        f"| {name} | "
        f"{r['identity_error']:.6f} | {r['raw_error']:.6f} | {r['global_alpha']:.6f} | "
        f"{r['global_error']:.6f} | {r['global_improvement_vs_identity']:.6f} | "
        f"{rel:.3f}% | {r['cosine_mean']:.6f} | {r['cosine_positive_frac']:.6f} | "
        f"{r['true_delta_norm_median']:.6f} | {r['pred_delta_norm_median']:.6f} |"
    )

lines += [
    "",
    "## Interpretation",
    "",
    "- The patch-token transformer is substantially stronger than vectorized patch-grid dynamics.",
    "- MSE-only slightly outperforms the directional variant in this pilot.",
    "- The main effect therefore comes from preserving token structure and using attention, not from the auxiliary directional loss.",
    "- The high positive cosine fraction suggests that the transformer predicts latent displacements in the correct half-space for nearly all test samples.",
]

path = out_dir / "patchtoken_transformer_pilot_summary.md"
path.write_text("\n".join(lines) + "\n")

print(path)
print(path.read_text())
