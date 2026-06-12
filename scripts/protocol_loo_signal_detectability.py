from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LAMBDA_COMPUTE = 0.04
CHEAP_COMPUTE = 1.0
EXPENSIVE_COMPUTE = 4.0
COMPUTE_GAP = EXPENSIVE_COMPUTE - CHEAP_COMPUTE
THRESHOLD = LAMBDA_COMPUTE * COMPUTE_GAP


def parse_run_name(path: Path) -> tuple[str, str]:
    m = re.search(r"protocol_loo_(P\d+)_(.*?)_seed0", path.parent.name)
    if not m:
        return "unknown", path.parent.name
    return m.group(1), m.group(2)


def rank_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)

    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)

    rank_sum_pos = ranks[labels].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def utility_delta(cheap: np.ndarray, exp: np.ndarray, selected: np.ndarray) -> float:
    selected = selected.astype(bool)
    cheap_utility = -float(cheap.mean()) - LAMBDA_COMPUTE * CHEAP_COMPUTE

    routed_error = float(np.where(selected, exp, cheap).mean())
    routed_compute = float(CHEAP_COMPUTE + COMPUTE_GAP * selected.mean())
    routed_utility = -routed_error - LAMBDA_COMPUTE * routed_compute

    return float(routed_utility - cheap_utility)


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.quantile(x, [0.02, 0.98])
    x = np.clip(x, lo, hi)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def topk_delta(cheap: np.ndarray, exp: np.ndarray, scores: np.ndarray, k_frac: float) -> float:
    n = len(scores)
    k = int(round(k_frac * n))
    if k <= 0:
        selected = np.zeros(n, dtype=bool)
    else:
        idx = np.argsort(-scores)[:k]
        selected = np.zeros(n, dtype=bool)
        selected[idx] = True
    return utility_delta(cheap, exp, selected)


def main() -> None:
    paths = sorted(Path("outputs").glob("protocol_loo_P*_cheap*_exp20_seed0/eval_arrays.npz"))
    if not paths:
        raise FileNotFoundError("No LOO eval arrays found.")

    rows = []

    for p in paths:
        heldout, run = parse_run_name(p)
        d = np.load(p, allow_pickle=True)

        cheap = d["cheap_errors"].astype(np.float64)
        exp = d["expensive_errors"].astype(np.float64)
        gain = cheap - exp
        profitable = gain > THRESHOLD

        reliability = d["reliability_scores"].astype(np.float64)
        action_norm = d["action_norm"].astype(np.float64)
        hybrid = 0.5 * normalize(reliability) + 0.5 * normalize(action_norm)

        oracle_delta = utility_delta(cheap, exp, profitable)
        k_frac = float(profitable.mean())

        signals = {
            "reliability": reliability,
            "action_norm": action_norm,
            "hybrid": hybrid,
        }

        for signal_name, score in signals.items():
            auc_raw = rank_auc(score, profitable)
            auc_inv = rank_auc(-score, profitable)
            if np.isnan(auc_raw):
                auc = float("nan")
                direction = "+"
                delta = float("nan")
            elif auc_inv > auc_raw:
                auc = auc_inv
                direction = "-"
                delta = topk_delta(cheap, exp, -score, k_frac)
            else:
                auc = auc_raw
                direction = "+"
                delta = topk_delta(cheap, exp, score, k_frac)

            rows.append({
                "heldout": heldout,
                "run": run,
                "signal": signal_name,
                "direction": direction,
                "profitable_frac": k_frac,
                "auroc": auc,
                "topk_delta_u": delta,
                "oracle_delta_u": oracle_delta,
                "capture_ratio": delta / oracle_delta if oracle_delta > 1e-12 else float("nan"),
            })

    out = Path("reports/tables/protocol/loo_signal_detectability_seed0.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Leave-one-trajectory-out profitable-gain signal detectability, DINOv2, seed 0",
        "",
        f"Profitable label: `cheap_error - expensive_error > {THRESHOLD:.4f}`.",
        "",
        "| heldout | run | signal | direction | profitable frac | AUROC | top-k ΔU | oracle ΔU | capture ratio |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['heldout']} | {r['run']} | {r['signal']} | {r['direction']} | "
            f"{r['profitable_frac']:.4f} | {r['auroc']:.4f} | {r['topk_delta_u']:.4f} | "
            f"{r['oracle_delta_u']:.4f} | {r['capture_ratio']:.4f} |"
        )

    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())

    fig_dir = Path("reports/figures/protocol")
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{r['heldout']}\n{r['run'].replace('_', '/')}\n{r['signal']}" for r in rows]
    x = np.arange(len(rows))
    aucs = np.array([r["auroc"] for r in rows])
    deltas = np.array([r["topk_delta_u"] for r in rows])

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.bar(x, aucs)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.set_ylabel("AUROC for profitable expensive call")
    ax.set_title("Can simple routing signals identify profitable expensive predictions?")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_signal_auroc_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5.5))
    ax.bar(x, deltas)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.set_ylabel("top-k utility improvement over cheap-only")
    ax.set_title("Utility captured by simple top-k routing signals")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_signal_topk_delta_utility_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "loo_signal_auroc_seed0.png")
    print(fig_dir / "loo_signal_topk_delta_utility_seed0.png")


if __name__ == "__main__":
    main()
