# Exact-intervention protocol plan

The TartanAir silver experiments showed that observational pseudo-counterfactual groups are not sufficiently identifiable. The next protocol must use exact simulator interventions.

## Required primitive

The benchmark requires the ability to:

1. initialize a simulator state `s_t`;
2. save or reconstruct the exact same state;
3. execute action `a_i` for horizon `h`;
4. observe future representation `z_{t+h}^{(i)}`;
5. reset to the same `s_t`;
6. repeat for multiple actions.

## Candidate environments

| environment | expected role | priority |
| --- | --- | --- |
| CALVIN | controlled manipulation, exact state reset likely feasible | P1 |
| Habitat | navigation/rearrangement, exact simulator states possible | P2 |
| CausalWorld | simpler causal-control sanity benchmark | P3 |
| TartanAir | observational only unless simulator reset is available | stop for now |

## First milestone

Before training, produce a small exact-intervention dataset:

```text
N = 100 anchor states
K = 5 actions per state
horizon h ∈ {1, 4, 8}
RGB observations + simulator state metadata + action labels
First diagnostic

Evaluate an oracle and simple baselines:

identity baseline;
random action baseline;
delta-transfer oracle;
V-JEPA frozen feature ranking.

If the oracle is not clearly above chance, the benchmark is not valid.
