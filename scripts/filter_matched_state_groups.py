from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, required=True)
    p.add_argument("--keep-fracs", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    p.add_argument("--score", choices=["mean_latent", "max_latent"], default="mean_latent")
    return p.parse_args()


def subset_groups(d, idx):
    out = {}
    n = d["candidate_indices"].shape[0]

    for k in d.files:
        v = d[k]
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v

    out["filtered_from"] = str(args.input)
    out["filtered_groups"] = len(idx)
    return out


def qstats(x):
    q = np.quantile(x, [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    return {
        "min": float(q[0]),
        "p05": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p95": float(q[5]),
        "max": float(q[6]),
        "mean": float(np.mean(x)),
    }


def main(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input, allow_pickle=True)
    latent = d["latent_distances"][:, 1:]
    action = d["action_distances"][:, 1:]
    future = d["future_distances"][:, 1:]

    if args.score == "mean_latent":
        score = latent.mean(axis=1)
    else:
        score = latent.max(axis=1)

    order = np.argsort(score)
    n = len(order)

    summary_lines = [
        "# Filtered matched-state group audit",
        "",
        f"- input: `{args.input}`",
        f"- score: `{args.score}`",
        f"- total groups: `{n}`",
        "",
        "| keep frac | groups | score max | latent mean | action mean | future mean | same traj offdiag | same env offdiag |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    json_summary = {}

    for frac in args.keep_fracs:
        k = max(1, int(round(frac * n)))
        idx = order[:k]

        tag = f"top{int(round(frac * 100)):02d}_{args.score}"
        out_path = args.out_dir / f"{args.input.stem}_{tag}.npz"

        out = subset_groups(d, idx)
        out["filter_score"] = score[idx]
        out["filter_keep_frac"] = frac
        out["filter_score_name"] = args.score

        np.savez_compressed(out_path, **out)

        same_traj = d["same_trajectory"][idx, 1:].mean()
        same_env = d["same_environment"][idx, 1:].mean()

        stats = {
            "groups": int(k),
            "keep_frac": float(frac),
            "score": qstats(score[idx]),
            "latent": qstats(latent[idx].reshape(-1)),
            "action": qstats(action[idx].reshape(-1)),
            "future": qstats(future[idx].reshape(-1)),
            "same_trajectory_offdiag": float(same_traj),
            "same_environment_offdiag": float(same_env),
            "out_path": str(out_path),
        }

        json_summary[tag] = stats

        summary_lines.append(
            f"| {frac:.2f} | {k} | {stats['score']['max']:.6f} | "
            f"{stats['latent']['mean']:.6f} | {stats['action']['mean']:.6f} | "
            f"{stats['future']['mean']:.6f} | {same_traj:.6f} | {same_env:.6f} |"
        )

    report_json = args.report_dir / f"{args.input.stem}_filtered_{args.score}.json"
    report_md = args.report_dir / f"{args.input.stem}_filtered_{args.score}.md"

    report_json.write_text(json.dumps(json_summary, indent=2) + "\n")
    report_md.write_text("\n".join(summary_lines) + "\n")

    print(report_md)
    print(report_md.read_text())


if __name__ == "__main__":
    args = parse_args()
    main(args)
