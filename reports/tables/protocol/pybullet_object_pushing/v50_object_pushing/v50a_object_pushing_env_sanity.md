# SPSM v50A object-pushing exact-intervention sanity

This is the first SPSM-v2 prototype. It tests whether a contact-rich object-pushing environment can support exact-intervention counterfactual groups.

## Setup

- environment: PyBullet DIRECT
- scene: plane, dynamic cube, dynamic spherical pusher
- actions: `stay`, `right`, `left`, `forward`, `backward`
- state sampling: cube xy/yaw plus pusher initialized on a cardinal side
- protocol: reset exact same state, apply each action, compare future object xy

## Repeatability

- repeat tests: `160`
- max object xy repeat diff: `0`
- mean object xy repeat diff: `0`
- max pusher xy repeat diff: `0`
- contact repeat match rate: `1.000`

## Action summary

| action | contact rate | mean object disp | median object disp | p90 object disp | pusher disp |
|---|---:|---:|---:|---:|---:|
| `backward` | 0.250 | 0.048 | 0.000 | 0.195 | 0.254 |
| `forward` | 0.250 | 0.049 | 0.000 | 0.198 | 0.254 |
| `left` | 0.250 | 0.049 | 0.000 | 0.200 | 0.254 |
| `right` | 0.250 | 0.049 | 0.000 | 0.199 | 0.254 |
| `stay` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Counterfactual group diversity

- states: `256`
- nondegenerate groups, max pairwise future xy spread > 0.02: `1.000`
- mean future xy spread: `0.078`
- median future xy spread: `0.078`
- mean max future xy spread: `0.194`
- mean contact action count: `1.000`
- mean moving contact action count: `1.000`
- mean max object displacement: `0.194`

## Interpretation

- If repeat diffs are near zero, exact reset/action replay is deterministic.
- If nondegenerate group fraction is high, the environment can generate meaningful counterfactual hard negatives.
- If contact rates vary across actions and states, the environment creates contact-mode structure beyond obstacle navigation.
- v50B should turn this into a dataset generator with RGB futures, state variables, candidate sets, and DINOv2 features.
- The target failure axes for SPSM-v2 are object pose, contact mode, pusher-object relation, and later hidden dynamics such as friction or mass.
