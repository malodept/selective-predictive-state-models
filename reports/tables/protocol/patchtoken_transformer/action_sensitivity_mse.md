# Action sensitivity diagnostic

Checkpoint: `mse`.

For each current frame, this measures how much predictions vary across different candidate actions compared with how much the true futures vary.

| checkpoint | inference action | prediction pairwise spread | true future pairwise spread | spread ratio |
| --- | --- | ---: | ---: | ---: |
| mse | original | 0.002420 | 1.404202 | 0.003038 |
| mse | zero | 0.000000 | 1.404202 | 0.000000 |
