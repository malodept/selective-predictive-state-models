from pathlib import Path
import json
import pandas as pd

dirs = [
    Path("outputs/envsplit_dinov2_directional_residual_seed0_lcos005_lnorm0001"),
    Path("outputs/envsplit_dinov2_directional_residual_seed1_lcos005_lnorm0001"),
    Path("outputs/envsplit_dinov2_directional_residual_seed2_lcos005_lnorm0001"),
]

rows = []

for d in dirs:
    m = json.loads((d / "metrics.json").read_text())
    seed = m["seed"]
    for split in ["val", "test"]:
        r = m[split]
        rows.append({
            "seed": seed,
            "split": split,
            "identity_error": r["identity_error"],
            "raw_error": r["raw_error"],
            "global_alpha": r["global_alpha"],
            "global_error": r["global_error"],
            "global_improvement_vs_identity": r["global_improvement_vs_identity"],
            "cosine_mean": r["cosine_mean"],
            "cosine_positive_frac": r["cosine_positive_frac"],
            "true_delta_norm_median": r["true_delta_norm_median"],
            "pred_delta_norm_median": r["pred_delta_norm_median"],
        })

df = pd.DataFrame(rows)

out_dir = Path("reports/tables/protocol/directional_residual")
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "directional_residual_seed_results.csv", index=False)

def pm(x):
    return f"{x.mean():.6f} ± {x.std(ddof=1):.6f}"

lines = [
    "# Directional residual predictor summary",
    "",
    "Loss: `MSE + 0.05 * cosine_loss + 0.001 * lognorm_loss`.",
    "",
    "| split | seeds | identity error | raw error | global alpha | global error | improvement vs identity | cosine mean | cosine positive frac | true Δ norm med | pred Δ norm med |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for split, g in df.groupby("split", sort=False):
    lines.append(
        f"| {split} | {len(g)} | "
        f"{pm(g['identity_error'])} | {pm(g['raw_error'])} | "
        f"{pm(g['global_alpha'])} | {pm(g['global_error'])} | "
        f"{pm(g['global_improvement_vs_identity'])} | "
        f"{pm(g['cosine_mean'])} | {pm(g['cosine_positive_frac'])} | "
        f"{pm(g['true_delta_norm_median'])} | {pm(g['pred_delta_norm_median'])} |"
    )

(out_dir / "directional_residual_summary.md").write_text("\n".join(lines) + "\n")

print(out_dir / "directional_residual_summary.md")
print((out_dir / "directional_residual_summary.md").read_text())
