from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metrics(run_dir: Path) -> dict:
    with (run_dir / "metrics.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_selector_rows(run_dir: Path) -> list[dict[str, object]]:
    path = run_dir / "selector_utility.csv"
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        item: dict[str, object] = {"policy": row.get("policy", "")}
        for key in ["threshold", "mean_error", "mean_compute", "utility", "selected_fraction"]:
            value = row.get(key, "")
            if value in {"", "None", "none", "null"}:
                item[key] = None
            else:
                item[key] = float(value)
        parsed.append(item)
    return parsed


def policy_label(row: dict[str, object]) -> str:
    policy = str(row.get("policy") or "")
    if policy in {"cheap-only", "all-expensive"}:
        return policy
    threshold = row.get("threshold")
    if threshold is None:
        return policy or "policy"
    return f"τ={float(threshold):.2f}"


def plot_losses(metrics: dict, fig_dir: Path, title_prefix: str) -> None:
    history = metrics["history"]
    epochs = [h["epoch"] for h in history]
    plt.figure()
    plt.plot(epochs, [h["loss"] for h in history], label="total")
    plt.plot(epochs, [h["prediction_loss"] for h in history], label="prediction")
    plt.plot(epochs, [h["reliability_loss"] for h in history], label="reliability")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix}: training losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "training_losses.png", dpi=220)
    plt.close()


def plot_error_vs_compute(rows: list[dict[str, object]], fig_dir: Path) -> None:
    plt.figure()
    for row in rows:
        x = float(row["mean_compute"])
        y = float(row["mean_error"])
        plt.scatter([x], [y])
        plt.annotate(policy_label(row), (x, y), fontsize=8)
    plt.xlabel("Mean compute cost")
    plt.ylabel("Mean prediction error")
    plt.title("Prediction error vs compute")
    plt.tight_layout()
    plt.savefig(fig_dir / "error_vs_compute.png", dpi=220)
    plt.close()


def plot_utility_vs_compute(rows: list[dict[str, object]], fig_dir: Path) -> None:
    plt.figure()
    for row in rows:
        x = float(row["mean_compute"])
        y = float(row["utility"])
        plt.scatter([x], [y])
        plt.annotate(policy_label(row), (x, y), fontsize=8)
    plt.xlabel("Mean compute cost")
    plt.ylabel("Utility = -error - λ compute")
    plt.title("Utility vs compute")
    plt.tight_layout()
    plt.savefig(fig_dir / "utility_vs_compute.png", dpi=220)
    plt.close()


def plot_aurocs(metrics: dict, fig_dir: Path) -> None:
    surprise = metrics["surprise"]
    names = [
        "Expected\nlearned",
        "Expected\nresidual",
        "Observed\nlearned",
        "Observed\nresidual",
    ]
    values = [
        surprise["expected_learned_auroc"],
        surprise["expected_residual_auroc"],
        surprise["observed_learned_auroc"],
        surprise["observed_residual_auroc"],
    ]
    plt.figure()
    plt.bar(names, values)
    plt.ylim(0.0, 1.0)
    plt.ylabel("AUROC")
    plt.title("Reliability and surprise diagnostics")
    plt.tight_layout()
    plt.savefig(fig_dir / "reliability_surprise_aurocs.png", dpi=220)
    plt.close()


def plot_reliability_vs_error(run_dir: Path, fig_dir: Path) -> None:
    arrays_path = run_dir / "eval_arrays.npz"
    if not arrays_path.exists():
        print(f"Skipping reliability-vs-error plot: missing {arrays_path}")
        return
    arrays = np.load(arrays_path)
    probs = arrays["probs"].reshape(-1)
    residuals_clean = arrays["residuals_clean"].reshape(-1)
    observed_labels = arrays["observed_labels"].reshape(-1)

    keep = observed_labels < 0.5
    probs = probs[keep]
    residuals_clean = residuals_clean[keep]

    bins = np.linspace(0.0, 1.0, 8)
    centers = 0.5 * (bins[:-1] + bins[1:])
    means = []
    counts = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        counts.append(int(mask.sum()))
        means.append(float(residuals_clean[mask].mean()) if mask.any() else np.nan)

    plt.figure()
    plt.plot(centers, means, marker="o")
    for x, y, n in zip(centers, means, counts):
        if not np.isnan(y):
            plt.annotate(f"n={n}", (x, y), fontsize=8)
    plt.xlabel("Predicted unreliability score")
    plt.ylabel("Mean latent prediction error after transition")
    plt.title("Predicted unreliability tracks realized error")
    plt.tight_layout()
    plt.savefig(fig_dir / "reliability_vs_realized_error.png", dpi=220)
    plt.close()


def write_readme(metrics: dict, rows: list[dict[str, object]], fig_dir: Path) -> None:
    best = max(rows, key=lambda r: float(r["utility"]))
    surprise = metrics["surprise"]
    text = f"""# Experiment figures

This folder contains diagnostic figures for a selective predictive-state run.

Key metrics:

- Retrieval R@1: {metrics['retrieval']['R@1']:.3f}
- Retrieval R@5: {metrics['retrieval']['R@5']:.3f}
- Expected learned AUROC: {surprise['expected_learned_auroc']:.3f}
- Observed residual AUROC: {surprise['observed_residual_auroc']:.3f}
- Best policy: {policy_label(best)}
- Best utility: {float(best['utility']):.3f}
- Best mean compute: {float(best['mean_compute']):.3f}
- Best mean error: {float(best['mean_error']):.3f}

Interpretation: the learned reliability score anticipates predictable hard transitions,
while residual error detects post-observation surprises. The adaptive policy uses the
pre-observation reliability score to decide when optional refinement computation is worth paying for.
"""
    (fig_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="outputs/synthetic_selective_compute")
    parser.add_argument("--figure-dir", default="reports/figures/synthetic_selective_compute")
    parser.add_argument("--title-prefix", default="Selective compute")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(run_dir)
    rows = load_selector_rows(run_dir)
    plot_losses(metrics, fig_dir, args.title_prefix)
    plot_error_vs_compute(rows, fig_dir)
    plot_utility_vs_compute(rows, fig_dir)
    plot_aurocs(metrics, fig_dir)
    plot_reliability_vs_error(run_dir, fig_dir)
    write_readme(metrics, rows, fig_dir)
    print(f"Wrote figures to {fig_dir}")


if __name__ == "__main__":
    main()
