from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trainval_gain_router import eval_arrays, tune_threshold, utility
from scripts.trainval_weighted_gain_router import train_router


def seed_from_path(p: Path) -> int:
    m = re.search(r"seed(\d+)", str(p))
    return int(m.group(1)) if m else 0


def fmt(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.std())


def summarize(rows):
    lambdas = sorted(set(r["lambda"] for r in rows))
    methods = ["cheap-only", "all-expensive", "gain classifier", "gain classifier + fallback", "oracle upper bound"]

    out = []
    for lam in lambdas:
        for method in methods:
            rs = [r for r in rows if r["lambda"] == lam and r["method"] == method]
            if not rs:
                continue
            item = {"lambda": lam, "method": method}
            for key in ["error", "compute", "selected", "utility", "delta_utility"]:
                m, s = mean_std([r[key] for r in rs])
                item[key + "_mean"] = m
                item[key + "_std"] = s
            out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-glob",
        default="outputs/bestval_dinov2_cheap10_exp80_seed*/bestval_checkpoint.pt",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument("--compute-gap", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    ckpts = sorted(Path(".").glob(args.run_glob))
    if not ckpts:
        raise FileNotFoundError(args.run_glob)

    rows = []

    print("Using checkpoints:")
    for ckpt in ckpts:
        run_dir = ckpt.parent
        seed = seed_from_path(run_dir)
        print(f" - seed={seed}: {run_dir}")

        train = eval_arrays(run_dir, "train", args.batch_size, args.device)
        val = eval_arrays(run_dir, "val", args.batch_size, args.device)

        train_gain = train["cheap_error"] - train["expensive_error"]
        val_gain = val["cheap_error"] - val["expensive_error"]

        for lam in args.lambdas:
            marginal_threshold = lam * args.compute_gap

            torch.manual_seed(seed)
            np.random.seed(seed)

            cheap_selected = np.zeros(len(val["cheap_error"]), dtype=bool)
            exp_selected = np.ones(len(val["cheap_error"]), dtype=bool)
            oracle_selected = val_gain > marginal_threshold

            cheap_err, cheap_comp, cheap_sel, cheap_util = utility(
                val["cheap_error"], val["expensive_error"],
                cheap_selected, lam, args.compute_gap,
            )

            fixed_methods = {
                "cheap-only": cheap_selected,
                "all-expensive": exp_selected,
                "oracle upper bound": oracle_selected,
            }

            for method, selected in fixed_methods.items():
                err, comp, sel, util = utility(
                    val["cheap_error"], val["expensive_error"],
                    selected, lam, args.compute_gap,
                )
                rows.append(
                    {
                        "seed": seed,
                        "lambda": lam,
                        "method": method,
                        "error": err,
                        "compute": comp,
                        "selected": sel,
                        "utility": util,
                        "delta_utility": util - cheap_util,
                    }
                )

            # Deployment-specific value classifier:
            # label = 1 iff expensive gain exceeds marginal compute cost.
            train_margin = train_gain - marginal_threshold

            train_scores, val_scores = train_router(
                train["router_x"],
                train_margin,
                val["router_x"],
                mode="unweighted classifier",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )

            best = tune_threshold(
                train_scores,
                train["cheap_error"],
                train["expensive_error"],
                lam,
                args.compute_gap,
            )

            selected = val_scores >= best["threshold"]

            err, comp, sel, util = utility(
                val["cheap_error"], val["expensive_error"],
                selected, lam, args.compute_gap,
            )

            rows.append(
                {
                    "seed": seed,
                    "lambda": lam,
                    "method": "gain classifier",
                    "error": err,
                    "compute": comp,
                    "selected": sel,
                    "utility": util,
                    "delta_utility": util - cheap_util,
                }
            )

            # Deployment-safe fallback:
            # if the tuned router does not beat cheap-only on the training/calibration
            # split, deploy cheap-only instead of forcing a harmful adaptive policy.
            train_selected = train_scores >= best["threshold"]
            _, _, _, train_router_util = utility(
                train["cheap_error"],
                train["expensive_error"],
                train_selected,
                lam,
                args.compute_gap,
            )
            _, _, _, train_cheap_util = utility(
                train["cheap_error"],
                train["expensive_error"],
                np.zeros(len(train["cheap_error"]), dtype=bool),
                lam,
                args.compute_gap,
            )

            if train_router_util > train_cheap_util:
                fallback_selected = selected
            else:
                fallback_selected = np.zeros(len(val["cheap_error"]), dtype=bool)

            ferr, fcomp, fsel, futil = utility(
                val["cheap_error"],
                val["expensive_error"],
                fallback_selected,
                lam,
                args.compute_gap,
            )

            rows.append(
                {
                    "seed": seed,
                    "lambda": lam,
                    "method": "gain classifier + fallback",
                    "error": ferr,
                    "compute": fcomp,
                    "selected": fsel,
                    "utility": futil,
                    "delta_utility": futil - cheap_util,
                }
            )

    summary = summarize(rows)

    out_table = Path("reports/tables/trainval_gain_classifier_lambda_sweep.md")
    out_table.parent.mkdir(parents=True, exist_ok=True)

    method_order = ["cheap-only", "all-expensive", "gain classifier", "gain classifier + fallback", "oracle upper bound"]

    lines = []
    lines.append("# Train-to-validation gain classifier lambda sweep\n")
    lines.append(f"Compute gap: `{args.compute_gap:.4f}`.\n")
    lines.append("| lambda | method | error | compute | selected | utility | delta utility vs cheap-only |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")

    for lam in sorted(set(r["lambda"] for r in summary)):
        for method in method_order:
            rs = [r for r in summary if r["lambda"] == lam and r["method"] == method]
            if not rs:
                continue
            r = rs[0]
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"{lam:.3f}",
                        method,
                        fmt(r["error_mean"], r["error_std"]),
                        fmt(r["compute_mean"], r["compute_std"]),
                        fmt(r["selected_mean"], r["selected_std"]),
                        fmt(r["utility_mean"], r["utility_std"]),
                        fmt(r["delta_utility_mean"], r["delta_utility_std"]),
                    ]
                )
                + " |"
            )

    out_table.write_text("\n".join(lines) + "\n")
    print(out_table)

    fig_dir = Path("reports/figures/gain_router")
    fig_dir.mkdir(parents=True, exist_ok=True)

    lambdas = np.array(sorted(set(r["lambda"] for r in summary)), dtype=float)

    style_names = {
        "cheap-only": "cheap-only",
        "all-expensive": "all-expensive",
        "gain classifier": "gain classifier",
        "gain classifier + fallback": "gain classifier + fallback",
        "oracle upper bound": "oracle",
    }

    # Figure 1: utility across lambda
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)

    for method in method_order:
        vals = []
        stds = []
        for lam in lambdas:
            r = next(x for x in summary if x["lambda"] == float(lam) and x["method"] == method)
            vals.append(r["utility_mean"])
            stds.append(r["utility_std"])
        vals = np.array(vals)
        stds = np.array(stds)
        ax.plot(lambdas, vals, marker="o", label=style_names[method])
        ax.fill_between(lambdas, vals - stds, vals + stds, alpha=0.12)

    ax.set_title("Gain classifier utility across deployment compute penalties")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("utility = -error - λ compute")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_classifier_lambda_utility.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2: utility improvement over cheap-only
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)

    for method in ["all-expensive", "gain classifier", "gain classifier + fallback", "oracle upper bound"]:
        vals = []
        stds = []
        for lam in lambdas:
            r = next(x for x in summary if x["lambda"] == float(lam) and x["method"] == method)
            vals.append(r["delta_utility_mean"])
            stds.append(r["delta_utility_std"])
        vals = np.array(vals)
        stds = np.array(stds)
        ax.plot(lambdas, vals, marker="o", label=style_names[method])
        ax.fill_between(lambdas, vals - stds, vals + stds, alpha=0.12)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Gain classifier deployment improvement over cheap-only")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("utility improvement over cheap-only")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_classifier_lambda_delta_utility.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 3: selected fraction and compute
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)

    gain_selected = []
    gain_selected_std = []
    gain_compute = []
    gain_compute_std = []

    for lam in lambdas:
        r = next(x for x in summary if x["lambda"] == float(lam) and x["method"] == "gain classifier")
        gain_selected.append(r["selected_mean"])
        gain_selected_std.append(r["selected_std"])
        gain_compute.append(r["compute_mean"])
        gain_compute_std.append(r["compute_std"])

    gain_selected = np.array(gain_selected)
    gain_selected_std = np.array(gain_selected_std)
    gain_compute = np.array(gain_compute)
    gain_compute_std = np.array(gain_compute_std)

    ax.plot(lambdas, gain_selected, marker="o", label="selected fraction")
    ax.fill_between(lambdas, gain_selected - gain_selected_std, gain_selected + gain_selected_std, alpha=0.12)
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("selected fraction")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(lambdas, gain_compute, marker="s", linestyle="--", label="mean compute")
    ax2.fill_between(lambdas, gain_compute - gain_compute_std, gain_compute + gain_compute_std, alpha=0.08)
    ax2.set_ylabel("mean compute cost")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title("Gain classifier uses less compute as λ increases")
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_classifier_lambda_compute.png", bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "trainval_gain_classifier_lambda_utility.png")
    print(fig_dir / "trainval_gain_classifier_lambda_delta_utility.png")
    print(fig_dir / "trainval_gain_classifier_lambda_compute.png")


if __name__ == "__main__":
    main()
