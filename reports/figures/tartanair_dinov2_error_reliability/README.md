# Experiment figures

This folder contains diagnostic figures for a selective predictive-state run.

Key metrics:

- Retrieval R@1: 0.138
- Retrieval R@5: 0.576
- Expected learned AUROC: 0.767
- Observed residual AUROC: 0.985
- Best policy: all-expensive
- Best utility: -0.676
- Best mean compute: 4.000
- Best mean error: 0.516

Interpretation: the learned reliability score anticipates predictable hard transitions,
while residual error detects post-observation surprises. The adaptive policy uses the
pre-observation reliability score to decide when optional refinement computation is worth paying for.
