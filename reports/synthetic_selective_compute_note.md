# Synthetic Selective Compute Baseline

## Purpose

This experiment is a controlled proof of concept for **Selective Predictive State Models**. It does not aim to be a real visual benchmark. Its goal is to verify that the core mechanism is internally coherent before moving to visual features and public datasets.

The synthetic environment separates two notions that are often conflated:

- **Expected unreliability**: a predictable transition difficulty available before the future is observed.
- **Observed surprise**: a post-observation mismatch between the predicted future state and the future state that actually arrives.

## Model

For each transition, the model receives a latent state \(z_t\) and an action \(a_t\). A predictor produces a cheap future-state estimate \(\hat z_{t+1}\). A reliability head receives \(z_t\), \(a_t\), \(\hat z_{t+1}\), and simple action-derived features, then predicts whether the cheap prediction is likely to be unreliable.

A selector uses this predicted unreliability to decide whether to keep the cheap prediction or activate an optional refinement module. The refinement module is represented as a controlled proxy: it reduces error much more on hard transitions than on easy transitions.

## Metrics

- **Future retrieval** tests whether predicted future latents retrieve the correct future latent among candidates.
- **Expected learned AUROC** tests whether the learned reliability score anticipates predictable hard transitions.
- **Expected residual AUROC** is an oracle-style diagnostic: it checks whether hard transitions indeed cause larger realized prediction errors.
- **Observed learned AUROC** tests whether the pre-observation reliability score can anticipate injected post-observation surprises.
- **Observed residual AUROC** tests whether realized prediction error detects observed surprises after they happen.
- **Utility** is defined as

\[
U = -\text{mean error} - \lambda \cdot \text{mean compute}.
\]

The selector is successful if it improves the utility-vs-compute frontier relative to cheap-only and all-expensive static policies.

## Current interpretation

The synthetic result validates the intended mechanism: predictable difficult transitions can be anticipated by the reliability head, injected surprises are better detected by post-observation residuals, and adaptive refinement can outperform both cheap-only and all-expensive baselines in utility.

This result should not be overclaimed. It is a sanity benchmark. The next milestone is to replace synthetic latents with latents extracted from real image or video sequences using a frozen visual encoder.
