from pathlib import Path

out_dir = Path("reports/tables/protocol/patchtoken_transformer")
out = out_dir / "action_grounding_diagnostics_summary.md"

text = """# Action-grounding diagnostics summary

This summary aggregates the diagnostics testing whether the patch-token Transformer genuinely uses the action input.

## Main findings

| diagnostic | result | interpretation |
| --- | --- | --- |
| No-action ablation | no-action is comparable to action-conditioned models | Global prediction does not require action under the current short-horizon protocol. |
| Action intervention | zero/shuffled actions barely change global prediction | The MSE-trained model is almost action-insensitive. |
| Candidate objective | improves global prediction and action sensitivity | The objective changes the model geometry, but does not solve OOD action grounding. |
| Candidate matching at calibrated alpha | top-1 remains near chance | MSE calibration shrinks predictions and hides weak action signal. |
| Candidate matching at alpha=1, train | original action reaches top-1 ≈ 0.231 vs 0.200 for zero/shuffle | Weak in-distribution action-grounding exists. |
| Candidate matching at alpha=1, test | original action remains ≈ 0.201 | The weak action-grounding does not generalize OOD. |

## Interpretation

The current strongest positive result is not an action-conditioned world model yet. It is a token-preserving latent dynamics model: preserving DINOv2 patch tokens and allowing token-token attention substantially improves OOD latent prediction.

The action-grounding diagnostics show a more nuanced picture. The candidate objective creates some action sensitivity and a weak in-distribution matching signal, but this signal does not transfer to the held-out environment. Therefore, the current protocol mostly measures short-horizon latent flow rather than robust action-conditioned prediction.

## Consequence

The next stage should not simply increase model size. It should modify either:

1. the protocol, by constructing futures where the same current state can lead to different outcomes depending on action;
2. the representation, by moving from image-level DINOv2 features to video/action-aware encoders;
3. the action model, by using stronger SE(3)-aware conditioning and geometric inductive bias.

The current DINOv2 patch-token branch should be reported as a strong representation/architecture result, while the action-grounding limitations should be stated explicitly.
"""

out.write_text(text)
print(out)
print(out.read_text())
