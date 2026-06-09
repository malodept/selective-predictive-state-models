# Real-video ResNet18 smoke test

This experiment replaces generated image sequences with real video frames while keeping the same SPSM pipeline:

```text
video files -> extracted frames -> frozen ResNet18 features -> predictive latent model -> reliability -> selective compute diagnostics
```

The purpose is not to claim final performance. It checks whether the SPSM training and evaluation pipeline remains valid when the latent states are extracted from real temporal observations.

## Expected use

Place a few `.mp4`, `.mov`, `.mkv`, or `.avi` files in:

```text
data/raw_videos/
```

Then run:

```bash
bash scripts/run_real_video_resnet18_smoke.sh
```

On the ESA VM through Apptainer, run each command in the script with:

```bash
apptainer exec /shared/projects/phisat2/containers/phisat2.sif <command>
```

## What to inspect

- `retrieval`: whether predicted future latents retrieve the correct future state.
- `expected_learned_auroc`: whether predictable hard transitions are anticipated by the reliability head.
- `observed_residual_auroc`: whether realized prediction error detects post-observation surprises.
- `utility_vs_compute`: whether selective refinement creates a useful compute/error tradeoff.

If retrieval remains very low and prediction error is nearly zero, the representation is likely too non-discriminative for this smoke setup. The next step should be either a stronger feature extractor or a better real sequence dataset.
