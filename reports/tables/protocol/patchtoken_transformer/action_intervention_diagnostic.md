# Action intervention diagnostic

A patch-token Transformer trained with the original 7D action is evaluated under action interventions at inference time.

Original validation alpha: `0.625000`.

| inference action | alpha original | alpha recalibrated | fixed-alpha gain | fixed-alpha error | fixed-alpha cosine | recalibrated gain | recalibrated error | recalibrated cosine | positive cosine frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 0.625000 | 0.625000 | 0.046457 | 1.062886 | 0.183513 | 0.046457 | 1.062886 | 0.183513 | 0.997254 |
| zero | 0.625000 | 0.670000 | 0.046684 | 1.062659 | 0.183435 | 0.045406 | 1.063937 | 0.183435 | 0.997313 |
| shuffled | 0.625000 | 0.615000 | 0.045113 | 1.064230 | 0.182283 | 0.045423 | 1.063919 | 0.182283 | 0.996999 |
