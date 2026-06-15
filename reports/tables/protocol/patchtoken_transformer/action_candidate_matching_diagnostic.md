# Action candidate matching diagnostic

For each current frame, the model predicts multiple candidate futures using different actions from the same state. The prediction should be closest to the matching future if the action is used.

Groups: `8000`. Candidates per group: `5`. Alpha: `0.625000`.

| inference action | top-1 candidate accuracy | mean diag MSE | mean best offdiag MSE | margin best offdiag - diag | positive margin frac |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 0.200075 | 1.065376 | 0.399334 | -0.666042 | 0.199875 |
| zero | 0.200000 | 1.065149 | 0.395445 | -0.669704 | 0.199800 |
| within_group_shuffle | 0.200075 | 1.065876 | 0.399194 | -0.666683 | 0.199875 |
| within_group_reverse | 0.199900 | 1.066324 | 0.399089 | -0.667235 | 0.199700 |
