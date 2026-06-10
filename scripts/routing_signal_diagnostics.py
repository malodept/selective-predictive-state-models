from __future__ import annotations

from pathlib import Path
import argparse
import math

import numpy as np
import matplotlib.pyplot as plt


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUROC, without scipy/sklearn dependency."""
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = ranks[labels].sum()
    return float((sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / (x.std() + 1e-12)


def topk_delta_utility(signal: np.ndarray, true_gain: np.ndarray, threshold: float, fraction: float) -> tuple[float, float, float]:
    """Route top fraction according to signal. Return selected, compute, utility gain over cheap-only."""
    n = len(signal)
    k = int(round(fraction * n))
    if k <= 0:
        return 0.0, 1.0, 0.0

    order = np.argsort(-signal)
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True

    selected_fraction = float(selected.mean())
    compute = 1.0 + 3.0 * selected_fraction
    delta_utility = float(((true_gain - threshold) * selected).mean())
    return selected_fraction, compute, delta_utility


def binned_curve(signal: np.ndarray, values: np.ndarray, bins: int = 10):
    q = np.quantile(signal, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    xs, ys, ns = [], [], []
    for lo, hi in zip(q[:-1], q[1:]):
        mask = (signal >= lo) & (signal <= hi)
        if mask.sum() == 0:
            continue
        xs.append(float(signal[mask].mean()))
        ys.append(float(values[mask].mean()))
        ns.append(int(mask.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def fmt(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="outputs/bestval_dinov2_cheap10_exp80_seed*/eval_arrays.npz")
    parser.add_argument("--lambda-compute", type=float, default=0.04)
    parser.add_argument("--compute-gap", type=float, default=3.0)
    parser.add_argument("--out-table", default="reports/tables/routing_signal_diagnostics.md")
    parser.add_argument("--out-fig", default="reports/figures/gain_router/routing_signal_diagnostics.png")
    args = parser.parse_args()

    paths = sorted(Path(".").glob(args.run_glob))
    if not paths:
        raise FileNotFoundError(f"No eval arrays found with glob: {args.run_glob}")

    threshold = args.lambda_compute * args.compute_gap

    rows = []
    all_true_gain = []
    all_reliability = []
    all_action = []
    all_hybrid = []

    print("Using eval arrays:")
    for p in paths:
        print(" -", p)

        d = np.load(p)
        cheap_errors = d["cheap_errors"].astype(np.float64)
        expensive_errors = d["expensive_errors"].astype(np.float64)
        reliability = d["reliability_scores"].astype(np.float64)
        action_norm = d["action_norm"].astype(np.float64)

        true_gain = cheap_errors - expensive_errors
        profitable = true_gain > threshold

        hybrid = 0.5 * zscore(reliability) + 0.5 * zscore(action_norm)

        profitable_fraction = float(profitable.mean())
        oracle_delta_utility = float(np.maximum(true_gain - threshold, 0.0).mean())
        all_exp_delta_utility = float((true_gain - threshold).mean())

        rel_selected, rel_compute, rel_delta_u = topk_delta_utility(
            reliability, true_gain, threshold, profitable_fraction
        )
        act_selected, act_compute, act_delta_u = topk_delta_utility(
            action_norm, true_gain, threshold, profitable_fraction
        )
        hyb_selected, hyb_compute, hyb_delta_u = topk_delta_utility(
            hybrid, true_gain, threshold, profitable_fraction
        )

        row = {
            "seed": p.parent.name.split("seed")[-1],
            "n": len(true_gain),
            "mean_true_gain": float(true_gain.mean()),
            "profitable_fraction": profitable_fraction,
            "threshold": threshold,
            "oracle_delta_utility": oracle_delta_utility,
            "all_exp_delta_utility": all_exp_delta_utility,
            "reliability_auc": rank_auc(reliability, profitable),
            "action_auc": rank_auc(action_norm, profitable),
            "hybrid_auc": rank_auc(hybrid, profitable),
            "rel_topk_delta_u": rel_delta_u,
            "action_topk_delta_u": act_delta_u,
            "hybrid_topk_delta_u": hyb_delta_u,
            "topk_fraction": profitable_fraction,
        }
        rows.append(row)

        all_true_gain.append(true_gain)
        all_reliability.append(reliability)
        all_action.append(action_norm)
        all_hybrid.append(hybrid)

    keys = [
        "mean_true_gain",
        "profitable_fraction",
        "oracle_delta_utility",
        "all_exp_delta_utility",
        "reliability_auc",
        "action_auc",
        "hybrid_auc",
        "rel_topk_delta_u",
        "action_topk_delta_u",
        "hybrid_topk_delta_u",
    ]

    mean_row = {"seed": "mean", "n": sum(r["n"] for r in rows), "threshold": threshold, "topk_fraction": float(np.mean([r["topk_fraction"] for r in rows]))}
    std_row = {"seed": "std", "n": "", "threshold": threshold, "topk_fraction": float(np.std([r["topk_fraction"] for r in rows]))}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        mean_row[k] = float(vals.mean())
        std_row[k] = float(vals.std())

    table_path = Path(args.out_table)
    table_path.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "seed",
        "n",
        "mean true gain",
        "profitable fraction",
        "oracle ΔU",
        "all-exp ΔU",
        "reliability AUROC",
        "action AUROC",
        "hybrid AUROC",
        "reliability top-k ΔU",
        "action top-k ΔU",
        "hybrid top-k ΔU",
    ]

    def row_to_md(r):
        return [
            str(r["seed"]),
            str(r["n"]),
            fmt(r["mean_true_gain"]),
            fmt(r["profitable_fraction"]),
            fmt(r["oracle_delta_utility"]),
            fmt(r["all_exp_delta_utility"]),
            fmt(r["reliability_auc"]),
            fmt(r["action_auc"]),
            fmt(r["hybrid_auc"]),
            fmt(r["rel_topk_delta_u"]),
            fmt(r["action_topk_delta_u"]),
            fmt(r["hybrid_topk_delta_u"]),
        ]

    lines = []
    lines.append("# Routing signal diagnostics, DINOv2 best-validation\n")
    lines.append(f"Marginal expensive-compute threshold: `{threshold:.4f}`.\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows + [mean_row, std_row]:
        lines.append("| " + " | ".join(row_to_md(r)) + " |")
    table_path.write_text("\n".join(lines) + "\n")
    print(table_path)

    true_gain = np.concatenate(all_true_gain)
    reliability = np.concatenate(all_reliability)
    action_norm = np.concatenate(all_action)
    hybrid = np.concatenate(all_hybrid)
    profitable = true_gain > threshold

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.hist(true_gain, bins=80, alpha=0.8)
    ax.axvline(0.0, linestyle="--", linewidth=1, label="zero gain")
    ax.axvline(threshold, linestyle="-", linewidth=2, label=f"routing threshold={threshold:.2f}")
    ax.set_title("True expensive-predictor gain distribution")
    ax.set_xlabel("cheap error - expensive error")
    ax.set_ylabel("count")
    ax.legend()

    ax = axes[0, 1]
    xs, ys, ns = binned_curve(reliability, true_gain)
    ax.plot(xs, ys, marker="o")
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_title("Reliability score vs true gain")
    ax.set_xlabel("predicted unreliability / reliability score")
    ax.set_ylabel("mean true gain per bin")

    ax = axes[1, 0]
    xs, ys, ns = binned_curve(action_norm, true_gain)
    ax.plot(xs, ys, marker="o")
    ax.axhline(threshold, linestyle="--", linewidth=1)
    ax.set_title("Action norm vs true gain")
    ax.set_xlabel("action norm")
    ax.set_ylabel("mean true gain per bin")

    ax = axes[1, 1]
    methods = ["reliability", "action norm", "hybrid"]
    aucs = [
        rank_auc(reliability, profitable),
        rank_auc(action_norm, profitable),
        rank_auc(hybrid, profitable),
    ]
    ax.bar(methods, aucs)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_ylim(0.45, max(0.75, max(aucs) + 0.05))
    ax.set_title("Detection of profitable expensive calls")
    ax.set_ylabel("AUROC")

    fig.suptitle("Routing signal diagnostics: why the oracle is still far away", fontsize=16)
    fig.tight_layout()

    fig_path = Path(args.out_fig)
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(fig_path)


if __name__ == "__main__":
    main()
