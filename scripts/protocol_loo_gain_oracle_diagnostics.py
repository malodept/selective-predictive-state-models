from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAMBDA_COMPUTE = 0.04
CHEAP_COMPUTE = 1.0
EXPENSIVE_COMPUTE = 4.0
COMPUTE_GAP = EXPENSIVE_COMPUTE - CHEAP_COMPUTE
MARGINAL_THRESHOLD = LAMBDA_COMPUTE * COMPUTE_GAP


def utility(cheap_error: np.ndarray, expensive_error: np.ndarray, selected: np.ndarray) -> tuple[float, float, float, float]:
    selected = selected.astype(bool)
    err = float(np.where(selected, expensive_error, cheap_error).mean())
    compute = float(CHEAP_COMPUTE + COMPUTE_GAP * selected.mean())
    selected_frac = float(selected.mean())
    util = float(-err - LAMBDA_COMPUTE * compute)
    return err, compute, selected_frac, util


def parse_run_name(path: Path) -> tuple[str, str]:
    m = re.search(r"protocol_loo_(P\d+)_(.*?)_seed0", path.parent.name)
    if not m:
        return "unknown", path.parent.name
    return m.group(1), m.group(2)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main() -> None:
    runs = sorted(Path("outputs").glob("protocol_loo_P*_cheap*_exp20_seed0/eval_arrays.npz"))
    if not runs:
        raise FileNotFoundError("No LOO eval_arrays.npz files found. Run export_bestval_eval_arrays.py first.")

    rows = []

    for p in runs:
        heldout, run = parse_run_name(p)
        d = np.load(p, allow_pickle=True)

        cheap = d["cheap_errors"].astype(np.float64)
        exp = d["expensive_errors"].astype(np.float64)
        gain = cheap - exp

        cheap_selected = np.zeros_like(gain, dtype=bool)
        all_selected = np.ones_like(gain, dtype=bool)
        oracle_selected = gain > MARGINAL_THRESHOLD

        cheap_err, cheap_comp, cheap_sel, cheap_util = utility(cheap, exp, cheap_selected)
        all_err, all_comp, all_sel, all_util = utility(cheap, exp, all_selected)
        oracle_err, oracle_comp, oracle_sel, oracle_util = utility(cheap, exp, oracle_selected)

        metrics_path = p.parent / "metrics.json"
        best_policy = ""
        best_error = np.nan
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text())
            best_policy = str(m.get("best_policy", ""))
            best_error = float(m.get("best_error", np.nan))

        rows.append(
            {
                "heldout": heldout,
                "run": run,
                "n": len(gain),
                "mean_gain": float(gain.mean()),
                "median_gain": float(np.median(gain)),
                "positive_fraction": float((gain > 0).mean()),
                "profitable_fraction": float((gain > MARGINAL_THRESHOLD).mean()),
                "cheap_error": cheap_err,
                "all_exp_error": all_err,
                "oracle_error": oracle_err,
                "oracle_compute": oracle_comp,
                "oracle_selected": oracle_sel,
                "cheap_utility": cheap_util,
                "all_exp_delta_u": all_util - cheap_util,
                "oracle_delta_u": oracle_util - cheap_util,
                "best_policy": best_policy,
                "best_error": best_error,
            }
        )

    out = Path("reports/tables/protocol/loo_oracle_gain_diagnostics_seed0.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Leave-one-trajectory-out oracle gain diagnostics, DINOv2, seed 0",
        "",
        f"Lambda compute: `{LAMBDA_COMPUTE:.4f}`. Marginal expensive threshold: `{MARGINAL_THRESHOLD:.4f}`.",
        "",
        "| heldout | run | n | mean gain | positive frac | profitable frac | cheap error | oracle error | oracle compute | oracle selected | all-exp ΔU | oracle ΔU | best policy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for r in rows:
        lines.append(
            f"| {r['heldout']} | {r['run']} | {r['n']} | {fmt(r['mean_gain'])} | "
            f"{fmt(r['positive_fraction'])} | {fmt(r['profitable_fraction'])} | "
            f"{fmt(r['cheap_error'])} | {fmt(r['oracle_error'])} | {fmt(r['oracle_compute'])} | "
            f"{fmt(r['oracle_selected'])} | {fmt(r['all_exp_delta_u'])} | {fmt(r['oracle_delta_u'])} | "
            f"{r['best_policy']} |"
        )

    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())

    fig_dir = Path("reports/figures/protocol")
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['heldout']}\n{r['run'].replace('_', '/')}" for r in rows]
    x = np.arange(len(rows))

    oracle_delta = np.array([r["oracle_delta_u"] for r in rows])
    profitable = np.array([r["profitable_fraction"] for r in rows])
    mean_gain = np.array([r["mean_gain"] for r in rows])

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, oracle_delta)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("oracle utility improvement over cheap-only")
    ax.set_title("LOO oracle value of selective expensive computation")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_oracle_delta_utility_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, profitable)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("fraction with gain > lambda * compute gap")
    ax.set_title("LOO fraction of profitable expensive predictions")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_profitable_fraction_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x, mean_gain)
    ax.axhline(MARGINAL_THRESHOLD, linestyle="--", linewidth=1, label="marginal compute threshold")
    ax.axhline(0.0, linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("mean cheap - expensive error")
    ax.set_title("LOO mean expensive gain is usually below deployment threshold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_mean_gain_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "loo_oracle_delta_utility_seed0.png")
    print(fig_dir / "loo_profitable_fraction_seed0.png")
    print(fig_dir / "loo_mean_gain_seed0.png")


if __name__ == "__main__":
    main()
