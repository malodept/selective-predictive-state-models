# Exact-intervention feasibility audit

## Motivation

TartanAir observational silver groups were not sufficiently identifiable for robust action-grounded future ranking. The next protocol must use exact interventions: multiple futures generated from the same simulator state under different actions.

## Required capability

The benchmark requires:

1. load an environment;
2. reset or restore an exact simulator state;
3. execute several alternative actions from the same state;
4. render observations after each action/horizon;
5. encode observations with the frozen representation;
6. evaluate candidate matching under true interventions.

## Local environment audit

To be filled from NEOHPC import checks.

| package | status | note |
| --- | --- | --- |
| calvin_env | TODO | candidate P1 |
| habitat / habitat_sim | TODO | candidate P2 |
| pybullet | TODO | useful for CALVIN / simple control |
| gym / gymnasium | TODO | common API |
| robomimic | TODO | possible fallback dataset |

## Decision criteria

Use an environment only if it supports exact or reconstructible state reset. Offline demonstrations alone are not enough.

## Next milestone

Build a tiny exact-intervention dataset:

| quantity | target |
| --- | ---: |
| anchor states | 100 |
| actions per anchor | 5 |
| horizons | 1, 4, 8 |
| observations | RGB |
| metadata | simulator state, action, horizon |

Before training, evaluate:

- identity baseline;
- random action baseline;
- delta-transfer oracle;
- frozen V-JEPA feature ranking.

If the oracle is not clearly above chance, the benchmark is invalid.
