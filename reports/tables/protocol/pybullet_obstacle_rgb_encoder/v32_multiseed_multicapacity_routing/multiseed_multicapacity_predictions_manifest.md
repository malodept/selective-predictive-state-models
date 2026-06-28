# SPSM v32A multi-seed multi-capacity predictions

This exports per-instance predictions for three seed-aligned capacity ladders.
Each ladder contains Tiny, Small, Medium, and Full models trained with the same seed.

- rows: `24408`
- variants: `5`
- ladder seeds: `[0, 1, 2]`
- model labels: `['Full', 'Medium', 'Small', 'Tiny']`

## Counts

```
ladder_seed  variant            model_label
0            block2_h72_seed11  Full           416
                                Medium         416
                                Small          416
                                Tiny           416
             block2_v18_seed12  Full           394
                                Medium         394
                                Small          394
                                Tiny           394
             block3_h36_seed13  Full           409
                                Medium         409
                                Small          409
                                Tiny           409
             block3_h48_seed10  Full           402
                                Medium         402
                                Small          402
                                Tiny           402
             block3_h72_seed14  Full           413
                                Medium         413
                                Small          413
                                Tiny           413
1            block2_h72_seed11  Full           416
                                Medium         416
                                Small          416
                                Tiny           416
             block2_v18_seed12  Full           394
                                Medium         394
                                Small          394
                                Tiny           394
             block3_h36_seed13  Full           409
                                Medium         409
                                Small          409
                                Tiny           409
             block3_h48_seed10  Full           402
                                Medium         402
                                Small          402
                                Tiny           402
             block3_h72_seed14  Full           413
                                Medium         413
                                Small          413
                                Tiny           413
2            block2_h72_seed11  Full           416
                                Medium         416
                                Small          416
                                Tiny           416
             block2_v18_seed12  Full           394
                                Medium         394
                                Small          394
                                Tiny           394
             block3_h36_seed13  Full           409
                                Medium         409
                                Small          409
                                Tiny           409
             block3_h48_seed10  Full           402
                                Medium         402
                                Small          402
                                Tiny           402
             block3_h72_seed14  Full           413
                                Medium         413
                                Small          413
                                Tiny           413
dtype: int64
```

## Next step

Use this file to compute seed-aligned multi-capacity oracle headroom and learned routing.
The correct aggregation is not to let the oracle choose among all 12 models at once, but to evaluate each seed-aligned ladder and then average across seeds.
