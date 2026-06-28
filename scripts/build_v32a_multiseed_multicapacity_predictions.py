from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
import torch

sys.path.insert(0, "scripts")

from build_v26_multicapacity_oracle_headroom import (
    predict_variant_model,
    load_costs,
    VARIANTS,
    VARIANT_LABELS,
)

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v32_multiseed_multicapacity_routing")
CACHE = OUT_DIR / "prediction_parts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "multiseed_multicapacity_predictions.csv"
OUT_MD = OUT_DIR / "multiseed_multicapacity_predictions_manifest.md"

SEEDS = [0, 1, 2]

def models_for_seed(seed: int):
    return [
        (f"tiny_full_seed{seed}", "Tiny"),
        (f"small_full_seed{seed}", "Small"),
        (f"medium_full_seed{seed}", "Medium"),
        (f"full_seed{seed}", "Full"),
    ]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    costs = load_costs()

    parts = []

    for seed in SEEDS:
        for variant in VARIANTS:
            for model_name, label in models_for_seed(seed):
                out_part = CACHE / f"pred_seed{seed}_{variant}_{model_name}.csv"

                print("=" * 70, flush=True)
                print(f"ladder_seed={seed} variant={variant} model={model_name} label={label}", flush=True)
                print("=" * 70, flush=True)

                if out_part.exists():
                    print(f"[SKIP] {out_part}", flush=True)
                    part = pd.read_csv(out_part)
                else:
                    part = predict_variant_model(
                        variant=variant,
                        model_name=model_name,
                        label=label,
                        device=device,
                        cost=costs[label],
                        batch_groups=128,
                    )
                    part.insert(0, "ladder_seed", seed)
                    part.to_csv(out_part, index=False)

                parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v32A multi-seed multi-capacity predictions\n")
    lines.append("This exports per-instance predictions for three seed-aligned capacity ladders.")
    lines.append("Each ladder contains Tiny, Small, Medium, and Full models trained with the same seed.")
    lines.append("")
    lines.append(f"- rows: `{len(df)}`")
    lines.append(f"- variants: `{df['variant'].nunique()}`")
    lines.append(f"- ladder seeds: `{sorted(df['ladder_seed'].unique().tolist())}`")
    lines.append(f"- model labels: `{sorted(df['model_label'].unique().tolist())}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("```")
    lines.append(str(df.groupby(['ladder_seed', 'variant', 'model_label']).size()))
    lines.append("```")
    lines.append("")
    lines.append("## Next step")
    lines.append("")
    lines.append("Use this file to compute seed-aligned multi-capacity oracle headroom and learned routing.")
    lines.append("The correct aggregation is not to let the oracle choose among all 12 models at once, but to evaluate each seed-aligned ladder and then average across seeds.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())
    print("DONE v32A multiseed multicapacity predictions", flush=True)

if __name__ == "__main__":
    main()
