# Action sensitivity diagnostic

Checkpoint: `candidate`.

For each current frame, this measures how much predictions vary across different candidate actions compared with how much the true futures vary.

| checkpoint | inference action | prediction pairwise spread | true future pairwise spread | spread ratio |
| --- | --- | ---: | ---: | ---: |
| candidate | original | 0.027714 | 1.404202 | 0.035856 |
| candidate | zero | 0.000000 | 1.404202 | 0.000000 |
