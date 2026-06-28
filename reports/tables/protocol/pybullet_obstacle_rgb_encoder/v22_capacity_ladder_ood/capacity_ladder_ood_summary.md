# SPSM v22 OOD capacity ladder

This evaluates the same mixed-hard moving-only OOD protocol across four predictor capacities: tiny, small, medium, and full.
ID performance is saturated for all capacities, so this table focuses on OOD reliability.

## Top-1 by capacity and OOD variant

| variant | tiny | small | medium | full | best | range |
|---|---:|---:|---:|---:|---|---:|
| 2-block, H=72 | 0.953125 | 0.929087 | 0.955529 | 0.974760 | `Full` | 0.045673 |
| 2-block, V=1.8 | 0.994924 | 0.987310 | 0.979695 | 0.994924 | `Tiny` | 0.015229 |
| 3-block, H=36 | 0.894866 | 0.897311 | 0.863081 | 0.828851 | `Small` | 0.068460 |
| 3-block, H=48 | 0.868159 | 0.853234 | 0.883085 | 0.840796 | `Medium` | 0.042289 |
| 3-block, H=72 | 0.815981 | 0.731235 | 0.842615 | 0.818402 | `Medium` | 0.111380 |

## State/action decomposition

| variant | model | same-action/diff-state win | same-state/diff-action win |
|---|---|---:|---:|
| 2-block, H=72 | Tiny | 0.968750 | 0.986779 |
| 2-block, H=72 | Small | 0.953125 | 0.985577 |
| 2-block, H=72 | Medium | 0.974760 | 0.986779 |
| 2-block, H=72 | Full | 0.986779 | 0.986779 |
| 2-block, V=1.8 | Tiny | 1.000000 | 0.997462 |
| 2-block, V=1.8 | Small | 0.992386 | 0.998731 |
| 2-block, V=1.8 | Medium | 0.988579 | 0.997462 |
| 2-block, V=1.8 | Full | 0.998731 | 0.998731 |
| 3-block, H=36 | Tiny | 0.933985 | 0.991443 |
| 3-block, H=36 | Small | 0.936430 | 0.995110 |
| 3-block, H=36 | Medium | 0.902200 | 0.984108 |
| 3-block, H=36 | Full | 0.863081 | 0.979218 |
| 3-block, H=48 | Tiny | 0.909204 | 0.992537 |
| 3-block, H=48 | Small | 0.896766 | 0.997512 |
| 3-block, H=48 | Medium | 0.916667 | 0.995025 |
| 3-block, H=48 | Full | 0.879353 | 0.992537 |
| 3-block, H=72 | Tiny | 0.868039 | 0.992736 |
| 3-block, H=72 | Small | 0.786925 | 0.991525 |
| 3-block, H=72 | Medium | 0.887409 | 0.979419 |
| 3-block, H=72 | Full | 0.857143 | 0.976998 |

## Checkpoint size

| model | checkpoint MB |
|---|---:|
| Tiny | 0.39 |
| Small | 1.15 |
| Medium | 3.99 |
| Full | 21.48 |

## Interpretation guide

- If larger models are not consistently better, this supports the non-uniform cheap/full ordering observed in VoC.
- If `medium` often matches or beats `full`, the expensive endpoint is not the right upper model.
- If the spread grows on block3 shifts, capacity matters specifically under harder OOD.
- The next step is a cost-normalized capacity VoC table, not another router family.
