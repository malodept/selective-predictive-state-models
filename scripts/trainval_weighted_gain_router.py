from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.trainval_gain_router import eval_arrays, utility, tune_threshold, zscore_from_train, RouterMLP


def seed_from_path(p: Path) -> int:
    m = re.search(r"seed(\d+)", str(p))
    return int(m.group(1)) if m else 0


def train_router(
    x_train: np.ndarray,
    gain_margin_train: np.ndarray,
    x_val: np.ndarray,
    mode: str,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    x_train_z, x_val_z = zscore_from_train(x_train, x_val)

    xtr = torch.tensor(x_train_z, dtype=torch.float32, device=device)
    xva = torch.tensor(x_val_z, dtype=torch.float32, device=device)

    margin = torch.tensor(gain_margin_train, dtype=torch.float32, device=device)
    labels = (margin > 0).float()

    model = RouterMLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = float(labels.mean().item())
    pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-6)], device=device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    n = xtr.shape[0]

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)

        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(xtr[idx]).reshape(-1)

            if mode == "unweighted classifier":
                loss = bce(logits, labels[idx]).mean()

            elif mode == "weighted classifier":
                w = margin[idx].abs() + 1e-3
                w = w / w.mean().clamp_min(1e-6)
                loss = (bce(logits, labels[idx]) * w).mean()

            elif mode == "sqrt-weighted classifier":
                w = torch.sqrt(margin[idx].abs() + 1e-3)
                w = w / w.mean().clamp_min(1e-6)
                loss = (bce(logits, labels[idx]) * w).mean()

            elif mode == "hardband weighted classifier":
                w = margin[idx].abs()
                keep = w > 0.03
                if keep.sum() < 8:
                    continue
                w = w[keep] / w[keep].mean().clamp_min(1e-6)
                loss = (bce(logits[keep], labels[idx][keep]) * w).mean()

            elif mode == "soft utility router":
                p = torch.sigmoid(logits)
                # Directly maximize expected marginal utility p * (gain - marginal compute cost).
                # Negative margin should push p down, positive margin should push p up.
                loss = -(p * margin[idx]).mean()

            else:
                raise ValueError(mode)

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(xtr).detach().cpu().numpy().reshape(-1)
        val_logits = model(xva).detach().cpu().numpy().reshape(-1)

    return train_logits, val_logits


def summarize(rows):
    methods = sorted(set(r["method"] for r in rows))
    out = []

    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        d = {"method": method}
        for key in ["error", "compute", "selected", "utility", "delta_utility"]:
            vals = np.array([r[key] for r in rs], dtype=np.float64)
            d[key + "_mean"] = float(vals.mean())
            d[key + "_std"] = float(vals.std())
        out.append(d)

    return out


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="outputs/bestval_dinov2_cheap10_exp80_seed*/bestval_checkpoint.pt")
    parser.add_argument("--lambda-compute", type=float, default=0.04)
    parser.add_argument("--compute-gap", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    ckpts = sorted(Path(".").glob(args.run_glob))
    if not ckpts:
        raise FileNotFoundError(args.run_glob)

    marginal_compute_threshold = args.lambda_compute * args.compute_gap

    learned_modes = [
        "unweighted classifier",
        "sqrt-weighted classifier",
        "weighted classifier",
        "hardband weighted classifier",
        "soft utility router",
    ]

    rows = []

    print("Using checkpoints:")
    for ckpt in ckpts:
        run_dir = ckpt.parent
        seed = seed_from_path(run_dir)
        print(f" - seed={seed}: {run_dir}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        train = eval_arrays(run_dir, "train", args.batch_size, args.device)
        val = eval_arrays(run_dir, "val", args.batch_size, args.device)

        train_gain = train["cheap_error"] - train["expensive_error"]
        val_gain = val["cheap_error"] - val["expensive_error"]

        train_margin = train_gain - marginal_compute_threshold

        cheap_base = utility(
            val["cheap_error"],
            val["expensive_error"],
            np.zeros(len(val["cheap_error"]), dtype=bool),
            args.lambda_compute,
            args.compute_gap,
        )[3]

        fixed_methods = {
            "cheap-only": np.zeros(len(val["cheap_error"]), dtype=bool),
            "all-expensive": np.ones(len(val["cheap_error"]), dtype=bool),
            "oracle upper bound": val_gain > marginal_compute_threshold,
        }

        for method, selected in fixed_methods.items():
            err, comp, sel, util = utility(
                val["cheap_error"],
                val["expensive_error"],
                selected,
                args.lambda_compute,
                args.compute_gap,
            )
            rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "error": err,
                    "compute": comp,
                    "selected": sel,
                    "utility": util,
                    "delta_utility": util - cheap_base,
                }
            )

        for mode in learned_modes:
            train_scores, val_scores = train_router(
                train["router_x"],
                train_margin,
                val["router_x"],
                mode=mode,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
            )

            best = tune_threshold(
                train_scores,
                train["cheap_error"],
                train["expensive_error"],
                args.lambda_compute,
                args.compute_gap,
            )

            selected = val_scores >= best["threshold"]

            err, comp, sel, util = utility(
                val["cheap_error"],
                val["expensive_error"],
                selected,
                args.lambda_compute,
                args.compute_gap,
            )

            rows.append(
                {
                    "seed": seed,
                    "method": mode,
                    "error": err,
                    "compute": comp,
                    "selected": sel,
                    "utility": util,
                    "delta_utility": util - cheap_base,
                }
            )

    order = [
        "cheap-only",
        "all-expensive",
        "unweighted classifier",
        "sqrt-weighted classifier",
        "weighted classifier",
        "hardband weighted classifier",
        "soft utility router",
        "oracle upper bound",
    ]

    summary = summarize(rows)
    summary = sorted(summary, key=lambda r: order.index(r["method"]))

    out_table = Path("reports/tables/trainval_weighted_gain_router_summary.md")
    out_table.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Train-to-validation weighted gain-router experiment\n")
    lines.append(
        f"Lambda compute: `{args.lambda_compute:.4f}`. "
        f"Marginal expensive threshold: `{marginal_compute_threshold:.4f}`.\n"
    )
    lines.append("| method | error | compute | selected | utility | delta utility vs cheap-only |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")

    for r in summary:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["method"],
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

    labels = [r["method"] for r in summary]
    short = {
        "cheap-only": "cheap-only",
        "all-expensive": "all-exp.",
        "unweighted classifier": "unweighted",
        "sqrt-weighted classifier": "sqrt-weighted",
        "weighted classifier": "weighted",
        "hardband weighted classifier": "hardband",
        "soft utility router": "soft utility",
        "oracle upper bound": "oracle",
    }

    gains = np.array([r["delta_utility_mean"] for r in summary])
    gain_stds = np.array([r["delta_utility_std"] for r in summary])
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=220)
    ax.barh(y, gains, xerr=gain_stds, capsize=4)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels([short[l] for l in labels])
    ax.invert_yaxis()
    ax.set_title("Weighted gain routers at lambda=0.04")
    ax.set_xlabel("mean utility improvement over cheap-only")
    ax.grid(True, axis="x", alpha=0.25)

    for yi, g in zip(y, gains):
        if abs(g) < 0.0005:
            ax.text(0.002, yi, "+0.000", va="center", ha="left", fontsize=9)
        elif g > 0:
            ax.text(g + 0.003, yi, f"{g:+.3f}", va="center", ha="left", fontsize=9)
        else:
            ax.text(g - 0.003, yi, f"{g:+.3f}", va="center", ha="right", fontsize=9)

    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_weighted_gain_router_delta_utility.png", bbox_inches="tight")
    plt.close(fig)

    x = np.array([r["compute_mean"] for r in summary])
    e = np.array([r["error_mean"] for r in summary])
    xerr = np.array([r["compute_std"] for r in summary])
    eerr = np.array([r["error_std"] for r in summary])

    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=220)
    ax.errorbar(x, e, xerr=xerr, yerr=eerr, fmt="o", capsize=4)

    offsets = {
        "cheap-only": (8, 6),
        "all-expensive": (-58, -8),
        "unweighted classifier": (8, 6),
        "sqrt-weighted classifier": (8, -10),
        "weighted classifier": (8, 8),
        "hardband weighted classifier": (8, 6),
        "soft utility router": (8, 6),
        "oracle upper bound": (8, 4),
    }

    for xi, ei, lab in zip(x, e, labels):
        ax.annotate(
            short[lab],
            (xi, ei),
            xytext=offsets.get(lab, (8, 6)),
            textcoords="offset points",
            fontsize=8.5,
        )

    ax.set_title("Weighted gain routers: error--compute trade-off")
    ax.set_xlabel("mean compute cost")
    ax.set_ylabel("mean prediction error")
    ax.text(0.02, 0.035, "Lower-left is better.", transform=ax.transAxes, fontsize=9, alpha=0.75)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0.85, 4.15)
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_weighted_gain_router_error_compute.png", bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "trainval_weighted_gain_router_delta_utility.png")
    print(fig_dir / "trainval_weighted_gain_router_error_compute.png")


if __name__ == "__main__":
    main()
