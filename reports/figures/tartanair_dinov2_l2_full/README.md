# Experiment figures

This folder contains diagnostic figures for a selective predictive-state run.

Key metrics:

- Retrieval R@1: 0.000
- Retrieval R@5: 0.001
- Expected learned AUROC: 0.991
- Observed residual AUROC: 0.491
- Best policy: cheap-only
- Best utility: -0.046
- Best mean compute: 1.000
- Best mean error: 0.006

Interpretation: the learned reliability score anticipates predictable hard transitions,
while residual error detects post-observation surprises. The adaptive policy uses the
pre-observation reliability score to decide when optional refinement computation is worth paying for.
