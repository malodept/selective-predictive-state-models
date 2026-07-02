# SPSM v49 Predictive Usefulness Stack

v49 unifies v47 and v48 into one diagnostic object. The point is not to add another metric, but to define the layers required for a predicted latent future to be useful.

## Five-level stack

| level | diagnostic question | hidden failure | evidence |
|---|---|---|---|
| 1. Counterfactual discrimination | Can the model select the correct future under exact interventions? | Average trajectory prediction can hide shortcut use and action-state entanglement. | Full state-action predictor reaches ID moving-only top-1 0.999, while action-only and state-only ablations fail on complementary hard-negative axes. |
| 2. Failure-axis diagnosis | When prediction fails, which axis fails? | A single OOD score does not reveal whether the issue is action grounding, state geometry, or another factor. | Controlled 3-block OOD shifts primarily degrade state discrimination; H=72 full model has state discrimination 0.847 while action discrimination remains 0.983. |
| 3. Compute-regime map | Is extra predictive compute useful, harmful, unnecessary, or insufficient? | Bigger-model fallback assumes monotonic improvement, but larger capacities can introduce different failures. | H=48 is the cleanest routable regime: useful=0.096, anti-rescue=0.051, stable router lambdas=4. H=72 has larger value but harder routing: useful=0.126, anti-rescue=0.089, oracle gap=0.120. |
| 4. Predicted-latent calibration | Are predicted futures directly interpretable by downstream probes? | Latent ranking can be correct even if predicted latents are not in the same operational coordinate system as encoded futures. | True-future probes fail on predicted latents with xy error 2.006-2.859, while model-specific probes recover 0.026-0.031 (77.7x-94.3x error reduction ratio). |
| 5. Decision actionability | Do calibrated predicted latents support action choice? | A prediction can be accurate in latent space but not useful for downstream decisions or latency-normalized utility. | Model-specific calibration improves one-step action accuracy by 0.383-0.500 and reduces regret by 0.207-0.259. On obstacle-sensitive challenge goals, calibrated predictors reduce regret by 0.065-0.073, stable in 3 blocks. |

## Compact compute-regime summary

| shift | useful | anti-rescue | oracle gap | stable router λ | recommendation |
|---|---:|---:|---:|---:|---|
| 3-block, H=36 | 0.071 | 0.067 | 0.070 | 1 | mostly fixed-capacity regime; limited routing value |
| 3-block, H=48 | 0.096 | 0.051 | 0.080 | 4 | deploy margin-gated routing; useful signal is routable |
| 3-block, H=72 | 0.126 | 0.089 | 0.120 | 2 | high oracle value; current routing only partially recovers it |

## Compact calibration/actionability summary

| model | pred + true probe | pred + specific probe | decision regret reduction | challenge regret reduction |
|---|---:|---:|---:|---:|
| Tiny | 2.859 | 0.031 | 0.259 | 0.073 |
| Medium | 2.553 | 0.027 | 0.255 | 0.065 |
| Full | 2.006 | 0.026 | 0.207 | 0.073 |

## Interpretation

- v47 says compute usefulness is query-regime structured: extra compute can be unnecessary, useful, insufficient, or harmful.
- v48 says latent usefulness is calibration-structured: predicted latents can contain physical information without being directly actionable.
- Together they support the broader thesis: useful world-model prediction requires reliability, diagnosis, compute allocation, calibration, and decision actionability.
- This stack should become the scientific compass for the next environment extension.
