# Project overview

## Research object

A **Selective Predictive State Model** maps an observation history to a latent state, predicts future latent states, estimates reliability or surprise, and decides which optional computations to execute under deployment constraints.

## Core hypothesis

Prediction, reliability, and compute selection should be learned and evaluated jointly in latent space. A deployment-constrained agent should not execute every optional module by default; it should spend computation as a function of expected utility and self-estimated uncertainty.

## Minimal experimental claim

A compact predictive model with reliability-aware adaptive execution should dominate static policies on a utility-vs-compute frontier.

## What counts as evidence

- Retrieval improves or remains competitive at fixed model size.
- Reliability score detects corrupt, mismatched, or implausible transitions.
- Adaptive execution saves compute at comparable utility or improves utility at fixed compute.
- Latency and parameter count are measured, not only claimed.

## Non-goals for v0

- No frontier-scale pretraining.
- No large dataset construction before the synthetic smoke tests pass.
- No notebooks as core training logic.
