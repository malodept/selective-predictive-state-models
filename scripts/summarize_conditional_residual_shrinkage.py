from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-md", type=Path, required=True)
    return p.parse_args()


def seed_from_path(p: Path) -> int:
    m = re.search(r"seed(\d+)", str(p))
    if not m:
        raise ValueError(f"Could not infer seed from {p}")
    return int(m.group(1))


def pm(s):
    return f"{s.mean():.6f} ± {s.std(ddof=1):.6f}"


def main():
    args = parse_args()

    frames = []
    for p in args.inputs:
        df = pd.read_csv(p)
        df["seed"] = seed_from_path(p)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    rows = []
    for (split, method), g in df.groupby(["split", "method"], sort=False):
        rows.append({
            "split": split,
            "method": method,
            "seeds": len(g),
            "error": pm(g["error"]),
            "identity_error": pm(g["identity_error"]),
            "improvement_vs_identity": pm(g["improvement_vs_identity"]),
            "ratio_vs_identity": pm(g["ratio_vs_identity"]),
            "alpha_mean": pm(g["alpha_mean"]),
            "alpha_std": pm(g["alpha_std"]),
        })

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    lines = [
        "# Conditional residual shrinkage summary over seeds",
        "",
        "| split | method | seeds | error | identity error | improvement vs identity | ratio vs identity | alpha mean | alpha std |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for _, r in out.iterrows():
        lines.append(
            f"| {r['split']} | {r['method']} | {r['seeds']} | "
            f"{r['error']} | {r['identity_error']} | {r['improvement_vs_identity']} | "
            f"{r['ratio_vs_identity']} | {r['alpha_mean']} | {r['alpha_std']} |"
        )

    args.out_md.write_text("\n".join(lines) + "\n")
    print(args.out_md)
    print(args.out_md.read_text())


if __name__ == "__main__":
    main()
