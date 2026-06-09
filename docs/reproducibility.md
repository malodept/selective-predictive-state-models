# Reproducibility

Rules:

1. Every experiment must have a YAML config.
2. Every training run writes `metrics.json`.
3. Every selector experiment writes a utility-vs-compute CSV.
4. Core metrics must live in `src/spsm/eval`, not notebooks.
5. Every model change must pass `pytest`.

The week-1 command is:

```bash
bash scripts/run_week1_mre.sh
```
