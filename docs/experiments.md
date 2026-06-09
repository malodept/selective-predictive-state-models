# Experiment registry

## W1-MRE: synthetic future latent prediction

Purpose: prove that the repo trains, evaluates retrieval, estimates surprise, and exports a compute-utility curve.

Inputs:
- synthetic latent current state `z_t`
- optional action vector `a_t`
- future latent target `z_t+delta`
- binary surprise label

Metrics:
- R@1, R@5, R@10
- AUROC, AUPRC, Brier score
- mean valid transition MSE
- utility-vs-compute table
- parameter count and latency

## W2-SELECTOR: adaptive selector prototype

Purpose: compare static cheap, static expensive, and adaptive policies.

Main ablations:
- residual-based reliability vs learned reliability
- selector inputs: residual only vs learned reliability vs reliability + budget
- thresholds and compute costs

## Public data baseline

Purpose: replace synthetic latents with public trajectory data and frozen teacher features.

Status: planned.
