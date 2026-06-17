# V-JEPA 2.1 regularization sweep

This sweep compares intra-epoch model selection under different regularization settings.

| config | best OOD error step | epoch | best OOD error | OOD gain at best error | OOD cosine | best gain step | best gain epoch | max OOD gain | seen error at best OOD | best seen error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 900 | 0.9688 | 0.032728 | 0.005527 | 0.330267 | 900 | 0.9688 | 0.005527 | 0.024103 | 0.023963 |
| drop015_wd001 | 900 | 0.9688 | 0.032871 | 0.005384 | 0.324822 | 900 | 0.9688 | 0.005384 | 0.024167 | 0.024058 |
| drop025_wd001 | 900 | 0.9688 | 0.032872 | 0.005382 | 0.321655 | 900 | 0.9688 | 0.005382 | 0.024300 | 0.024172 |

## Best configuration by OOD global error

- Config: `base`
- Best OOD error: `0.032728`
- OOD gain: `0.005527`
- OOD cosine: `0.330267`
- Step: `900`
- Epoch fraction: `0.9688`
