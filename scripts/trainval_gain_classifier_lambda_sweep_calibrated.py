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

def utility_per_sample(cheap_error, expensive_error, selected, lam, compute_gap):
    selected = np.asarray(selected, dtype=bool)
    err = np.where(selected, expensive_error, cheap_error)
    comp = 1.0 + compute_gap * selected.astype(np.float32)
    return -err - lam * comp


def lower_confidence_gain(router_u, cheap_u, z):
    delta = np.asarray(router_u - cheap_u, dtype=np.float64)
    mean = float(delta.mean())
    if len(delta) <= 1:
        return mean, mean
    se = float(delta.std(ddof=1) / np.sqrt(len(delta)))
    return mean, mean - z * se


def summarize(rows):
    lambdas = sorted(set(r["lambda"] for r in rows))
    methods = [
        "cheap-only",
        "all-expensive",
        "calibrated gain classifier",
        "calibrated gain classifier + fallback",
        "oracle upper bound",
    ]

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
    parser.add_argument("--run-glob", default="outputs/bestval_dinov2_cheap10_exp80_seed*/bestval_checkpoint.pt")
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30],
    )
    parser.add_argument("--compute-gap", type=float, default=3.0)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--fallback-z", type=float, default=1.645)
    parser.add_argument("--fallback-margin", type=float, default=0.005)
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

        full_train = eval_arrays(run_dir, "train", args.batch_size, args.device)
        val = eval_arrays(run_dir, "val", args.batch_size, args.device)

        full_gain = full_train["cheap_error"] - full_train["expensive_error"]
        val_gain = val["cheap_error"] - val["expensive_error"]

        rng = np.random.default_rng(seed)
        n = len(full_gain)
        perm = rng.permutation(n)
        n_calib = int(round(args.calib_fraction * n))
        calib_idx = perm[:n_calib]
        router_idx = perm[n_calib:]

        router_x = full_train["router_x"][router_idx]
        calib_x = full_train["router_x"][calib_idx]

        router_cheap = full_train["cheap_error"][router_idx]
        router_exp = full_train["expensive_error"][router_idx]

        calib_cheap = full_train["cheap_error"][calib_idx]
        calib_exp = full_train["expensive_error"][calib_idx]

        router_gain = router_cheap - router_exp

        for lam in args.lambdas:
            marginal_threshold = lam * args.compute_gap

            torch.manual_seed(seed)
            np.random.seed(seed)

            cheap_selected = np.zeros(len(val["cheap_error"]), dtype=bool)
            exp_selected = np.ones(len(val["cheap_error"]), dtype=bool)
            oracle_selected = val_gain > marginal_threshold

            cheap_err, cheap_comp, cheap_sel, cheap_util = utility(
                val["cheap_error"],
                val["expensive_error"],
                cheap_selected,
                lam,
                args.compute_gap,
            )

            fixed_methods = {
                "cheap-only": cheap_selected,
                "all-expensive": exp_selected,
                "oracle upper bound": oracle_selected,
            }

            for method, selected in fixed_methods.items():
                err, comp, sel, util = utility(
                    val["cheap_error"],
                    val["expensive_error"],
                    selected,
                    lam,
                    args.compute_gap,
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

            router_margin = router_gain - marginal_threshold

            eval_x = np.concatenate([calib_x, val["router_x"]], axis=0)

            _, eval_scores = train_router(
                router_x,
                router_margin,
                eval_x,
                mode="unweighted classifier",
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )

            calib_scores = eval_scores[: len(calib_x)]
            val_scores = eval_scores[len(calib_x) :]

            best = tune_threshold(
                calib_scores,
                calib_cheap,
                calib_exp,
                lam,
                args.compute_gap,
            )

            selected = val_scores >= best["threshold"]

            err, comp, sel, util = utility(
                val["cheap_error"],
                val["expensive_error"],
                selected,
                lam,
                args.compute_gap,
            )

            rows.append(
                {
                    "seed": seed,
                    "lambda": lam,
                    "method": "calibrated gain classifier",
                    "error": err,
                    "compute": comp,
                    "selected": sel,
                    "utility": util,
                    "delta_utility": util - cheap_util,
                }
            )

            calib_selected = calib_scores >= best["threshold"]

            calib_router_u = utility_per_sample(
                calib_cheap,
                calib_exp,
                calib_selected,
                lam,
                args.compute_gap,
            )
            calib_cheap_u = utility_per_sample(
                calib_cheap,
                calib_exp,
                np.zeros(len(calib_cheap), dtype=bool),
                lam,
                args.compute_gap,
            )

            calib_delta_mean, calib_delta_lcb = lower_confidence_gain(
                calib_router_u,
                calib_cheap_u,
                args.fallback_z,
            )

            # Conservative deployment rule:
            # deploy the adaptive router only when calibration supports a positive
            # utility gain with a safety margin. Otherwise deploy cheap-only.
            if calib_delta_lcb > args.fallback_margin:
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
                    "method": "calibrated gain classifier + fallback",
                    "error": ferr,
                    "compute": fcomp,
                    "selected": fsel,
                    "utility": futil,
                    "delta_utility": futil - cheap_util,
                }
            )

    summary = summarize(rows)

    method_order = [
        "cheap-only",
        "all-expensive",
        "calibrated gain classifier",
        "calibrated gain classifier + fallback",
        "oracle upper bound",
    ]

    out_table = Path("reports/tables/trainval_gain_classifier_lambda_sweep_calibrated.md")
    out_table.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Calibrated train-to-validation gain classifier lambda sweep\n")
    lines.append(
        f"Compute gap: `{args.compute_gap:.4f}`. Calibration fraction: `{args.calib_fraction:.2f}`. "
        f"Fallback rule: deploy router only if calibration lower-confidence gain exceeds `{args.fallback_margin:.4f}`.\n"
    )
    lines.append("| lambda | method | error | compute | selected | utility | delta utility vs cheap-only |")
    lines.append("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")

    for lam in sorted(set(r["lambda"] for r in summary)):
        for method in method_order:
            r = next(x for x in summary if x["lambda"] == lam and x["method"] == method)
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

    short = {
        "cheap-only": "cheap-only",
        "all-expensive": "all-expensive",
        "calibrated gain classifier": "calibrated gain classifier",
        "calibrated gain classifier + fallback": "calibrated gain classifier + fallback",
        "oracle upper bound": "oracle",
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=220)

    for method in method_order:
        vals, stds = [], []
        for lam in lambdas:
            r = next(x for x in summary if x["lambda"] == float(lam) and x["method"] == method)
            vals.append(r["utility_mean"])
            stds.append(r["utility_std"])
        vals = np.array(vals)
        stds = np.array(stds)
        ax.plot(lambdas, vals, marker="o", label=short[method])
        ax.fill_between(lambdas, vals - stds, vals + stds, alpha=0.10)

    ax.set_title("Calibrated gain classifier utility across deployment compute penalties")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("utility = -error - λ compute")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_classifier_lambda_utility_calibrated.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=220)

    for method in [
        "all-expensive",
        "calibrated gain classifier",
        "calibrated gain classifier + fallback",
        "oracle upper bound",
    ]:
        vals, stds = [], []
        for lam in lambdas:
            r = next(x for x in summary if x["lambda"] == float(lam) and x["method"] == method)
            vals.append(r["delta_utility_mean"])
            stds.append(r["delta_utility_std"])
        vals = np.array(vals)
        stds = np.array(stds)
        ax.plot(lambdas, vals, marker="o", label=short[method])
        ax.fill_between(lambdas, vals - stds, vals + stds, alpha=0.10)

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title("Calibrated gain classifier deployment improvement over cheap-only")
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("utility improvement over cheap-only")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_classifier_lambda_delta_utility_calibrated.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=220)

    selected_mean, selected_std = [], []
    compute_mean, compute_std = [], []

    for lam in lambdas:
        r = next(
            x
            for x in summary
            if x["lambda"] == float(lam) and x["method"] == "calibrated gain classifier + fallback"
        )
        selected_mean.append(r["selected_mean"])
        selected_std.append(r["selected_std"])
        compute_mean.append(r["compute_mean"])
        compute_std.append(r["compute_std"])

    selected_mean = np.array(selected_mean)
    selected_std = np.array(selected_std)
    compute_mean = np.array(compute_mean)
    compute_std = np.array(compute_std)

    ax.plot(lambdas, selected_mean, marker="o", label="selected fraction")
    ax.fill_between(lambdas, selected_mean - selected_std, selected_mean + selected_std, alpha=0.12)
    ax.set_xlabel("compute penalty λ")
    ax.set_ylabel("selected fraction")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(lambdas, compute_mean, marker="s", linestyle="--", label="mean compute")
    ax2.fill_between(lambdas, compute_mean - compute_std, compute_mean + compute_std, alpha=0.08)
    ax2.set_ylabel("mean compute cost")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax.set_title("Calibrated gain classifier reduces compute as λ increases")
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_classifier_lambda_compute_calibrated.png", bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "trainval_gain_classifier_lambda_utility_calibrated.png")
    print(fig_dir / "trainval_gain_classifier_lambda_delta_utility_calibrated.png")
    print(fig_dir / "trainval_gain_classifier_lambda_compute_calibrated.png")


if __name__ == "__main__":
    main()
