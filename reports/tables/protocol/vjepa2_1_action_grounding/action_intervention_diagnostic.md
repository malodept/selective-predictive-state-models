# Action intervention diagnostic

A patch-token Transformer trained with the original 7D action is evaluated under action interventions at inference time.

Original validation alpha: `0.580000`.

| inference action | alpha original | alpha recalibrated | fixed-alpha gain | fixed-alpha error | fixed-alpha cosine | recalibrated gain | recalibrated error | recalibrated cosine | positive cosine frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.580000 | 0.580000 | 0.003826 | 0.027311 | 0.310250 | 0.003826 | 0.027311 | 0.310250 | 0.998725 |
| zero | 0.580000 | 1.000000 | 0.002575 | 0.028562 | 0.306749 | 0.003239 | 0.027898 | 0.306749 | 0.999549 |
| shuffled | 0.580000 | 0.470000 | 0.002251 | 0.028887 | 0.296646 | 0.002635 | 0.028503 | 0.296646 | 0.995998 |
