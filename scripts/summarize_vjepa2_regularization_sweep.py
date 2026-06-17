from pathlib import Path
import json

base = Path("outputs/vjepa2_1_regularization_sweep")
out_dir = Path("reports/tables/protocol/vjepa2_1_regularization_sweep")
out_dir.mkdir(parents=True, exist_ok=True)

rows = []

for run in sorted(base.iterdir()):
    p = run / "intraepoch_rows.json"
    if not p.exists():
        continue

    data = json.loads(p.read_text())

    best_ood_error = min(data, key=lambda r: r["ood_global_error"])
    best_ood_gain = max(data, key=lambda r: r["ood_gain"])
    best_seen_error = min(data, key=lambda r: r["seen_global_error"])

    rows.append({
        "config": run.name,
        "best_ood_error_step": best_ood_error["step"],
        "best_ood_error_epoch": best_ood_error["epoch_frac"],
        "best_ood_error": best_ood_error["ood_global_error"],
        "best_ood_gain": best_ood_error["ood_gain"],
        "best_ood_cosine": best_ood_error["ood_cosine"],
        "best_gain_step": best_ood_gain["step"],
        "best_gain_epoch": best_ood_gain["epoch_frac"],
        "max_ood_gain": best_ood_gain["ood_gain"],
        "seen_error_at_best_ood": best_ood_error["seen_global_error"],
        "best_seen_error": best_seen_error["seen_global_error"],
    })

lines = [
    "# V-JEPA 2.1 regularization sweep",
    "",
    "This sweep compares intra-epoch model selection under different regularization settings.",
    "",
    "| config | best OOD error step | epoch | best OOD error | OOD gain at best error | OOD cosine | best gain step | best gain epoch | max OOD gain | seen error at best OOD | best seen error |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]

for r in rows:
    lines.append(
        f"| {r['config']} | {r['best_ood_error_step']} | {r['best_ood_error_epoch']:.4f} | "
        f"{r['best_ood_error']:.6f} | {r['best_ood_gain']:.6f} | {r['best_ood_cosine']:.6f} | "
        f"{r['best_gain_step']} | {r['best_gain_epoch']:.4f} | {r['max_ood_gain']:.6f} | "
        f"{r['seen_error_at_best_ood']:.6f} | {r['best_seen_error']:.6f} |"
    )

if rows:
    best = min(rows, key=lambda r: r["best_ood_error"])
    lines += [
        "",
        "## Best configuration by OOD global error",
        "",
        f"- Config: `{best['config']}`",
        f"- Best OOD error: `{best['best_ood_error']:.6f}`",
        f"- OOD gain: `{best['best_ood_gain']:.6f}`",
        f"- OOD cosine: `{best['best_ood_cosine']:.6f}`",
        f"- Step: `{best['best_ood_error_step']}`",
        f"- Epoch fraction: `{best['best_ood_error_epoch']:.4f}`",
    ]

out = out_dir / "vjepa2_1_regularization_sweep_summary.md"
out.write_text("\n".join(lines) + "\n")

print(out)
print(out.read_text())
