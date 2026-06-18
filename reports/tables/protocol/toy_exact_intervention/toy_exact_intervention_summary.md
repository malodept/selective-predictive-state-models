# Toy exact-intervention benchmark summary

## Goal

The toy exact-intervention benchmark was introduced after the TartanAir silver experiments failed to produce a robust action-grounded candidate-matching signal.

The purpose is not realism. The purpose is to verify that the evaluation protocol can detect action grounding when the data truly contains exact intervention branches.

## Dataset

Each group contains the same synthetic state branched under five actions:

```text
stay, right, left, down, up

The dataset contains:

quantity	value
groups	1000
candidates per group	5
samples	5000
chance top-1	0.200000
latent shape	16 x 8
action dimension	2
Oracle validation

The delta-transfer oracle confirms that the candidate-matching metric behaves correctly.

model	alpha	top-1	mean rank	positive margin frac
identity	0.00	0.200000	3.000000	0.200000
delta oracle original	1.00	1.000000	1.000000	1.000000
delta oracle deranged	1.00	0.000000	3.512400	0.000000
random shuffle	1.00	0.206000	3.004800	0.206000

This verifies that the metric recognizes exact action-future correspondence and fails when the correspondence is deliberately broken.

Learned action-conditioned model

A small residual action MLP was trained on 800 groups, validated on 100 groups, and tested on 100 groups.

split	mode	top-1	mean rank	positive margin frac
train	original	1.000000	1.000000	1.000000
val	original:	---:		
train	original	1.000000	1.000000	1.000000
val	original	1.000000	1.000000	1.000000
test	original	1.000000	1.000000	1.000000
test	zero	0.200000	3.000000	0.200000
test	within-group shuffle	0.148000	3.054000	0.148000
test	within-group reverse	0.200000	2.798000	0.200000
Interpretation

This validates the exact-intervention evaluation pipeline.

The project now has a clean contrast:

TartanAir observational silver:
    oracle barely above chance
    learned models at chance

Toy exact intervention:
    oracle perfect
    learned action model perfect

Therefore, the previous TartanAir failure should not be interpreted as a metric bug. It is evidence that observational pseudo-counterfactual groups were not sufficiently identifiable.

Decision

The next step is to move from toy exact interventions to a minimal physical simulator with exact state reset.

Recommended next branch:

v4.3-pybullet-exact-intervention

Before moving to CALVIN or Habitat, validate the same protocol in a simple PyBullet environment.
