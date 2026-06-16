# Action sensitivity diagnostic

Checkpoint: `vjepa2_1_mse`.

For each current frame, this measures how much predictions vary across different candidate actions compared with how much the true futures vary.

| checkpoint | inference action | prediction pairwise spread | true future pairwise spread | spread ratio |
| --- | --- | ---: | ---: | ---: |
| vjepa2_1_mse | original | 0.004249 | 0.019682 | 0.240714 |
| vjepa2_1_mse | zero | 0.000000 | 0.019682 | 0.000000 |
