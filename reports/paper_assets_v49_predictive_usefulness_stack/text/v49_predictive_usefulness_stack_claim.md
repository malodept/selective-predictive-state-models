# v49 predictive usefulness stack

## Main idea

v47 and v48 can be unified into a single diagnostic stack for latent world models. A prediction is useful only if it is counterfactually discriminative, diagnostically interpretable, compute-worthwhile, physically calibrated, and actionable.

## Scientific claim

Predictive usefulness is not scalar accuracy. SPSM factorizes it into five levels:
1. counterfactual discrimination;
2. failure-axis diagnosis;
3. interventional value-of-computation;
4. predicted-latent calibration;
5. downstream actionability.

## Why this matters

Standard world-model evaluation can hide several failure modes: a model can rank futures correctly but fail physical calibration, a larger model can be worse than a smaller one on some queries, a regime can have oracle compute value that current routers cannot recover, and raw decision quality can disagree with latency-normalized decision utility.

## Next scientific step

The next major extension should not be another formatting pass. It should test whether this stack transfers to a second exact-intervention environment, ideally contact-rich object manipulation, where failure axes include object pose, contact mode, occlusion, and dynamics parameters.
