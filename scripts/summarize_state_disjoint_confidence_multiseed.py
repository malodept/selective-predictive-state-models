from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def mean_std(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return f"{values.mean():.6f} ± 0.000000"
    return f"{values.mean():.6f} ± {values.std(ddof=1):.6f}"


def main():
    base = Path(
        "reports/tables/protocol/pybullet_obstacle_rgb_encoder/"
        "scale_5k/mixed_hard_state_disjoint/confidence"
    )
    seeds = [0, 1, 2]

    metrics = {
        "mean_confidence": [],
        "biased_top1": [],
        "strict_top1": [],
        "tieaware_top1": [],
        "correct_tied_frac": [],
        "ece_tieaware": [],
        "ece_strict": [],
    }

    coverage_rows = {}
    stay_conf = []
    moving_conf = []

    for seed in seeds:
        path = base / f"full_seed{seed}.json"
        d = json.loads(path.read_text())

        for key in metrics:
            metrics[key].append(d[key])

        for row in d["selective_rows"]:
            cov = f"{row['coverage']:.2f}"
            if cov not in coverage_rows:
                coverage_rows[cov] = {
                    "mean_confidence": [],
                    "strict_top1": [],
                    "tieaware_top1": [],
                    "stay_fraction": [],
                }

            coverage_rows[cov]["mean_confidence"].append(row["mean_confidence"])
            coverage_rows[cov]["strict_top1"].append(row["strict_top1"])
            coverage_rows[cov]["tieaware_top1"].append(row["tieaware_top1"])
            coverage_rows[cov]["stay_fraction"].append(row["stay_fraction"])

        for row in d["action_rows"]:
            if row["action"] == "stay":
                stay_conf.append(row["mean_confidence"])
            else:
                moving_conf.append(row["mean_confidence"])

    lines = [
        "# State-disjoint confidence multiseed summary",
        "",
        "Values are mean ± sample standard deviation over seeds 0, 1, 2.",
        "",
        "## Global reliability metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mean confidence | {mean_std(metrics['mean_confidence'])} |",
        f"| biased top-1 | {mean_std(metrics['biased_top1'])} |",
        f"| strict top-1 | {mean_std(metrics['strict_top1'])} |",
        f"| tie-aware top-1 | {mean_std(metrics['tieaware_top1'])} |",
        f"| correct tied with another candidate | {mean_std(metrics['correct_tied_frac'])} |",
        f"| ECE vs tie-aware target | {mean_std(metrics['ece_tieaware'])} |",
        f"| ECE vs strict target | {mean_std(metrics['ece_strict'])} |",
        "",
        "## Selective prediction curve",
        "",
        "| coverage | confidence | strict top-1 | tie-aware top-1 | stay fraction |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]

    for cov in sorted(coverage_rows.keys(), reverse=True):
        row = coverage_rows[cov]
        lines.append(
            f"| {cov} | "
            f"{mean_std(row['mean_confidence'])} | "
            f"{mean_std(row['strict_top1'])} | "
            f"{mean_std(row['tieaware_top1'])} | "
            f"{mean_std(row['stay_fraction'])} |"
        )

    lines += [
        "",
        "## Action confidence separation",
        "",
        "| action subset | mean confidence |",
        "| --- | ---: |",
        f"| stay | {mean_std(stay_conf)} |",
        f"| moving actions | {mean_std(moving_conf)} |",
        "",
        "## Interpretation",
        "",
        "The full Transformer assigns low confidence to the non-identifiable `stay` cases and high confidence to identifiable moving actions.",
        "At 80% coverage, the retained set contains no `stay` anchors and reaches perfect strict and tie-aware accuracy across all three seeds.",
        "This reconnects the exact-intervention benchmark to the original SPSM reliability goal: confidence is informative about when a latent prediction is identifiable and trustworthy.",
    ]

    out = base / "state_disjoint_confidence_multiseed_summary.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
