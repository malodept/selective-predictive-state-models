from __future__ import annotations

import json
from pathlib import Path
import statistics as stats

RUNS = [
    ("cheap5/exp80", "cheap5_exp80"),
    ("cheap10/exp80", "cheap10_exp80"),
    ("cheap20/exp80", "cheap20_exp80"),
    ("cheap10/exp40", "cheap10_exp40"),
    ("cheap10/exp120", "cheap10_exp120"),
]
SEEDS = [0, 1, 2]

OUT_MD = Path("reports/tables/bestval_gap_ablation_multiseed.md")
OUT_MD.parent.mkdir(parents=True, exist_ok=True)


def mean_std(xs):
    return stats.mean(xs), stats.stdev(xs) if len(xs) > 1 else 0.0


rows = []

for label, slug in RUNS:
    metrics = []
    for seed in SEEDS:
        path = Path(f"outputs/bestval_dinov2_{slug}_seed{seed}/metrics.json")
        if not path.exists():
            raise FileNotFoundError(path)
        metrics.append(json.loads(path.read_text()))

    cheap_errors = [m["diagnostics"]["cheap_error"] for m in metrics]
    expensive_errors = [m["diagnostics"]["expensive_error"] for m in metrics]
    gaps = [m["diagnostics"]["gap"] for m in metrics]
    best_errors = [m["best_error"] for m in metrics]
    best_computes = [m["best_compute"] for m in metrics]
    best_selected = [m["best_selected"] for m in metrics]
    utilities = [m["best_utility"] for m in metrics]
    policies = [m["best_policy"] for m in metrics]
    exp_epochs = [m["training"]["expensive_best_epoch"] for m in metrics]

    row = {
        "run": label,
        "cheap_error": mean_std(cheap_errors),
        "expensive_error": mean_std(expensive_errors),
        "gap": mean_std(gaps),
        "best_error": mean_std(best_errors),
        "best_compute": mean_std(best_computes),
        "selected": mean_std(best_selected),
        "utility": mean_std(utilities),
        "policies": ", ".join(policies),
        "exp_epochs": ", ".join(str(x) for x in exp_epochs),
    }
    rows.append(row)


def fmt(ms):
    m, s = ms
    return f"{m:.4f} ± {s:.4f}"


lines = []
lines.append("# Best-validation cheap/expensive gap ablation\n\n")
lines.append("| run | cheap error | expensive error | gap | best error | best compute | selected | utility | policies | expensive best epochs |\n")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |\n")

for r in rows:
    lines.append(
        f"| {r['run']} | "
        f"{fmt(r['cheap_error'])} | "
        f"{fmt(r['expensive_error'])} | "
        f"{fmt(r['gap'])} | "
        f"{fmt(r['best_error'])} | "
        f"{fmt(r['best_compute'])} | "
        f"{fmt(r['selected'])} | "
        f"{fmt(r['utility'])} | "
        f"{r['policies']} | "
        f"{r['exp_epochs']} |\n"
    )

OUT_MD.write_text("".join(lines))
print(OUT_MD)
print(OUT_MD.read_text())
