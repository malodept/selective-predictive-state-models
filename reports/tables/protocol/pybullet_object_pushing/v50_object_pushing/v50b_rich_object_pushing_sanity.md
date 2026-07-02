# SPSM v50B rich object-pushing sanity

v50B enriches v50A with a rectangular object, object yaw, tangent offsets, variable gaps, mass/friction variation, and non-trivial contact modes.

## Repeatability

- repeat tests: `160`
- max object xy repeat diff: `0`
- mean object xy repeat diff: `0`
- max object rotation repeat diff: `0`
- contact repeat match rate: `1.000`

## Action summary

| action | contact rate | mean disp | median disp | p90 disp | mean rot | mean contact steps | mean force |
|---|---:|---:|---:|---:|---:|---:|---:|
| `backward` | 0.205 | 0.024 | 0.000 | 0.123 | 0.017 | 14.06 | 2.855 |
| `forward` | 0.205 | 0.026 | 0.000 | 0.140 | 0.019 | 14.70 | 2.944 |
| `left` | 0.203 | 0.025 | 0.000 | 0.138 | 0.019 | 14.71 | 2.786 |
| `right` | 0.211 | 0.025 | 0.000 | 0.138 | 0.019 | 14.24 | 3.049 |
| `stay` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

## Group diversity

- states: `512`
- nondegenerate groups: `0.766`
- contact-mode nontrivial groups, 0 < contact actions < 4: `0.824`
- mean moving contact action count: `0.824`
- contact count std: `0.381`
- mean future xy spread: `0.040`
- mean max future xy spread: `0.100`
- mean strong displacement actions: `0.668`
- mean rotation actions: `0.340`

## Interpretation

- v50B should replace v50A if repeatability remains exact and contact-mode diversity is non-trivial.
- The target is not maximum realism yet; it is to create a second exact-intervention regime where failures depend on contact geometry and object pose.
- If contact-mode diversity is high enough, v50C should generate the actual RGB intervention dataset.
- If contact-mode diversity is still too simple, increase tangent-offset range, add diagonal actions, or add multiple object shapes.
