# Action candidate matching diagnostic

For each current frame, the model predicts multiple candidate futures using different actions from the same state. The prediction should be closest to the matching future if the action is used.

Groups: `8000`. Candidates per group: `5`. Alpha: `0.580000`.

| inference action | top-1 candidate accuracy | mean diag MSE | mean best offdiag MSE | margin best offdiag - diag | positive margin frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.200325 | 0.027274 | 0.013778 | -0.013495 | 0.200325 |
| zero | 0.200000 | 0.028514 | 0.010856 | -0.017658 | 0.200000 |
| within_group_shuffle | 0.200000 | 0.028272 | 0.013542 | -0.014729 | 0.200000 |
| within_group_reverse | 0.199700 | 0.029253 | 0.013418 | -0.015835 | 0.199700 |
