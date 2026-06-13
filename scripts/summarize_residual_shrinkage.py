from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, default=Path("reports/tables/protocol/residual_shrinkage_summary.csv"))
    p.add_argument("--out-md", type=Path, default=Path("reports/tables/protocol/residual_shrinkage_summary.md"))
    return p.parse_args()


def mean_pm(x):
    return f"{x.mean():.6f} ± {x.std(ddof=1):.6f}" if len(x) > 1 else f"{x.mean():.6f} ± 0.000000"


def main():
    args = parse_args()

    frames = []
    for p in args.inputs:
        seed = int(p.parent.name.replace("residual_shrinkage_seed", ""))
        df = pd.read_csv(p)
        df["seed"] = seed
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    group_cols = ["split", "model", "rule"]
    rows = []

    for keys, g in df.groupby(group_cols, sort=False):
        split, model, rule = keys
        rows.append(
            {
                "split": split,
                "model": model,
                "rule": rule,
                "n_seeds": len(g),
                "alpha": mean_pm(g["alpha"]),
                "error": mean_pm(g["error"]),
                "identity_error": mean_pm(g["identity_error"]),
                "improvement_vs_identity": mean_pm(g["improvement_vs_identity"]),
                "ratio_vs_identity": mean_pm(g["ratio_vs_identity"]),
            }
        )

    out = pd.DataFrame(rows)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    lines = [
        "# Residual shrinkage alpha summary over seeds",
        "",
        "Prediction family: `z_pred(alpha) = z_current + alpha * delta_hat`.",
        "",
        "| split | model | rule | seeds | alpha | error | identity error | improvement vs identity | ratio vs identity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in out.iterrows():
        lines.append(
            f"| {r['split']} | {r['model']} | {r['rule']} | {r['n_seeds']} | "
            f"{r['alpha']} | {r['error']} | {r['identity_error']} | "
            f"{r['improvement_vs_identity']} | {r['ratio_vs_identity']} |"
        )

    args.out_md.write_text("\n".join(lines) + "\n")

    print(args.out_md)
    print(args.out_md.read_text())


if __name__ == "__main__":
    main()
