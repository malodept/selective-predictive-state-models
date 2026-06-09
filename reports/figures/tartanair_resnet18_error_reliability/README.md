# Experiment figures

This folder contains diagnostic figures for a selective predictive-state run.

Key metrics:

- Retrieval R@1: 0.067
- Retrieval R@5: 0.315
- Expected learned AUROC: 0.760
- Observed residual AUROC: 0.955
- Best policy: τ=0.50
- Best utility: -0.258
- Best mean compute: 1.571
- Best mean error: 0.195

Interpretation: the learned reliability score anticipates predictable hard transitions,
while residual error detects post-observation surprises. The adaptive policy uses the
pre-observation reliability score to decide when optional refinement computation is worth paying for.
