from __future__ import annotations

from pathlib import Path
import argparse
import re

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def seed_from_path(p: Path) -> int:
    m = re.search(r"seed(\d+)", str(p))
    return int(m.group(1)) if m else 0


def utility(cheap_err, expensive_err, selected, lam, compute_gap):
    selected = selected.astype(bool)
    err = np.where(selected, expensive_err, cheap_err).mean()
    compute = 1.0 + compute_gap * selected.mean()
    util = -err - lam * compute
    return float(err), float(compute), float(selected.mean()), float(util)


def tune_threshold(scores, cheap_err, expensive_err, lam, compute_gap):
    qs = np.linspace(0.0, 1.0, 41)
    candidates = np.unique(np.quantile(scores, qs))
    candidates = np.concatenate([[np.inf], candidates, [-np.inf]])

    best = None
    for thr in candidates:
        selected = scores >= thr
        err, comp, sel, util = utility(cheap_err, expensive_err, selected, lam, compute_gap)
        if best is None or util > best["utility"]:
            best = {
                "threshold": float(thr),
                "error": err,
                "compute": comp,
                "selected": sel,
                "utility": util,
            }
    return best


def standardize(train_x, test_x):
    mu = train_x.mean(axis=0, keepdims=True)
    sig = train_x.std(axis=0, keepdims=True) + 1e-6
    return (train_x - mu) / sig, (test_x - mu) / sig


class MLP(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_binary_router(x_train, y_train, x_test, epochs, batch_size, lr, device):
    x_train, x_test = standardize(x_train, x_test)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)
    xte = torch.tensor(x_test, dtype=torch.float32, device=device)

    model = MLP(xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = float(y_train.mean())
    pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-6)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_scores = torch.sigmoid(model(xtr)).cpu().numpy().reshape(-1)
        test_scores = torch.sigmoid(model(xte)).cpu().numpy().reshape(-1)

    return train_scores, test_scores


def train_gain_regressor(x_train, y_train, x_test, epochs, batch_size, lr, device):
    x_train, x_test = standardize(x_train, x_test)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)
    xte = torch.tensor(x_test, dtype=torch.float32, device=device)

    model = MLP(xtr.shape[1], 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = len(xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        train_scores = model(xtr).cpu().numpy().reshape(-1)
        test_scores = model(xte).cpu().numpy().reshape(-1)

    return train_scores, test_scores


def summarize(rows):
    methods = sorted(set(r["method"] for r in rows))
    out = []
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        d = {"method": m}
        for key in ["error", "compute", "selected", "utility", "delta_utility"]:
            vals = np.array([r[key] for r in rs], dtype=float)
            d[key + "_mean"] = float(vals.mean())
            d[key + "_std"] = float(vals.std())
        out.append(d)
    return out


def fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="outputs/bestval_dinov2_cheap10_exp80_seed*/eval_arrays.npz")
    parser.add_argument("--lambda-compute", type=float, default=0.04)
    parser.add_argument("--compute-gap", type=float, default=3.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    paths = sorted(Path(".").glob(args.run_glob))
    if not paths:
        raise FileNotFoundError(args.run_glob)

    threshold = args.lambda_compute * args.compute_gap
    rows = []

    print("Using eval arrays:")
    for p in paths:
        print(" -", p)
        seed = seed_from_path(p)
        rng = np.random.default_rng(1000 + seed)

        d = np.load(p)
        cheap_err = d["cheap_errors"].astype(np.float64)
        expensive_err = d["expensive_errors"].astype(np.float64)
        reliability = d["reliability_scores"].astype(np.float64)
        action_norm = d["action_norm"].astype(np.float64)
        val_x = d["val_x"].astype(np.float32)
        cheap_pred = d["cheap_pred"].astype(np.float32)

        true_gain = cheap_err - expensive_err
        profitable = (true_gain > threshold).astype(np.float32)
        net_gain = true_gain - threshold

        n = len(true_gain)
        idx = rng.permutation(n)
        n_train = int(0.7 * n)
        tr = idx[:n_train]
        te = idx[n_train:]

        # Features available before running the expensive predictor:
        # current latent + action, cheap prediction, cheap reliability score, action norm.
        x = np.concatenate(
            [
                val_x,
                cheap_pred,
                reliability[:, None].astype(np.float32),
                action_norm[:, None].astype(np.float32),
            ],
            axis=1,
        )

        hybrid = 0.5 * ((reliability - reliability.mean()) / (reliability.std() + 1e-12))
        hybrid += 0.5 * ((action_norm - action_norm.mean()) / (action_norm.std() + 1e-12))

        train_scores_cls, test_scores_cls = train_binary_router(
            x[tr], profitable[tr], x[te], args.epochs, args.batch_size, args.lr, args.device
        )

        train_scores_reg, test_scores_reg = train_gain_regressor(
            x[tr], net_gain[tr], x[te], args.epochs, args.batch_size, args.lr, args.device
        )

        methods = {
            "cheap-only": None,
            "all-expensive": None,
            "oracle upper bound": None,
            "reliability threshold": (reliability[tr], reliability[te]),
            "action norm threshold": (action_norm[tr], action_norm[te]),
            "hybrid threshold": (hybrid[tr], hybrid[te]),
            "direct gain classifier": (train_scores_cls, test_scores_cls),
            "direct gain regressor": (train_scores_reg, test_scores_reg),
        }

        cheap_base = utility(
            cheap_err[te], expensive_err[te], np.zeros(len(te), dtype=bool), args.lambda_compute, args.compute_gap
        )[3]

        for method, scores in methods.items():
            if method == "cheap-only":
                selected = np.zeros(len(te), dtype=bool)
            elif method == "all-expensive":
                selected = np.ones(len(te), dtype=bool)
            elif method == "oracle upper bound":
                selected = true_gain[te] > threshold
            else:
                train_score, test_score = scores
                best = tune_threshold(
                    train_score,
                    cheap_err[tr],
                    expensive_err[tr],
                    args.lambda_compute,
                    args.compute_gap,
                )
                selected = test_score >= best["threshold"]

            err, comp, sel, util = utility(
                cheap_err[te],
                expensive_err[te],
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

    summary = summarize(rows)

    method_order = [
        "cheap-only",
        "all-expensive",
        "reliability threshold",
        "action norm threshold",
        "hybrid threshold",
        "direct gain classifier",
        "direct gain regressor",
        "oracle upper bound",
    ]
    summary = sorted(summary, key=lambda r: method_order.index(r["method"]))

    out_table = Path("reports/tables/gain_router_holdout_diagnostic.md")
    out_table.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Holdout gain-router diagnostic, DINOv2 best-validation\n")
    lines.append(f"Lambda compute: `{args.lambda_compute:.4f}`. Marginal expensive threshold: `{threshold:.4f}`.\n")
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
    x = np.array([r["compute_mean"] for r in summary])
    y = np.array([r["error_mean"] for r in summary])
    xerr = np.array([r["compute_std"] for r in summary])
    yerr = np.array([r["error_std"] for r in summary])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4)
    for xi, yi, lab in zip(x, y, labels):
        short = lab.replace(" threshold", "").replace(" upper bound", "")
        ax.annotate(short, (xi, yi), xytext=(7, 5), textcoords="offset points", fontsize=9)
    ax.set_title("Holdout gain-router diagnostic: error--compute trade-off")
    ax.set_xlabel("mean compute")
    ax.set_ylabel("mean prediction error")
    ax.grid(True, alpha=0.25)
    ax.text(0.02, 0.03, "Lower-left is better.", transform=ax.transAxes, alpha=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "gain_router_holdout_error_compute.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    gains = np.array([r["delta_utility_mean"] for r in summary])
    gain_stds = np.array([r["delta_utility_std"] for r in summary])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    yy = np.arange(len(labels))
    ax.barh(yy, gains)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([l.replace(" threshold", "").replace(" upper bound", "") for l in labels])
    ax.set_xlabel("mean utility improvement over cheap-only")
    ax.set_title("Holdout gain-router diagnostic at lambda=0.04")
    ax.grid(True, axis="x", alpha=0.25)
    for yi, g in zip(yy, gains):
        ax.text(g + (0.003 if g >= 0 else -0.003), yi, f"{g:+.3f}", va="center", ha="left" if g >= 0 else "right")
    fig.tight_layout()
    fig.savefig(fig_dir / "gain_router_holdout_delta_utility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "gain_router_holdout_error_compute.png")
    print(fig_dir / "gain_router_holdout_delta_utility.png")


if __name__ == "__main__":
    main()
