from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_real_selective_refinement_bestval import (
    MLP,
    load_feature_npz,
    per_sample_mse,
    predict,
    reliability_scores,
)


def seed_from_path(p: Path) -> int:
    m = re.search(r"seed(\d+)", str(p))
    return int(m.group(1)) if m else 0


def as_xy_action(path: str):
    loaded = load_feature_npz(path)

    if isinstance(loaded, tuple):
        if len(loaded) >= 2:
            x, y = loaded[0], loaded[1]
            x = np.asarray(x, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)

            # In the TartanAir feature files, x is [z_current, action].
            # Example for DINOv2: x_dim=391, latent_dim=384, action_dim=7.
            action_dim = x.shape[1] - y.shape[1]
            if action_dim <= 0:
                raise ValueError(
                    f"Cannot infer action from x/y shapes for {path}: "
                    f"x={x.shape}, y={y.shape}"
                )
            action = x[:, -action_dim:].astype(np.float32)
            return x, y, action

    # Robust fallback for standard npz layouts.
    d = np.load(path)
    keys = set(d.files)

    if {"x", "y", "action"}.issubset(keys):
        return (
            d["x"].astype(np.float32),
            d["y"].astype(np.float32),
            d["action"].astype(np.float32),
        )

    if {"z_current", "z_future", "action"}.issubset(keys):
        x = np.concatenate([d["z_current"], d["action"]], axis=1).astype(np.float32)
        return x, d["z_future"].astype(np.float32), d["action"].astype(np.float32)

    if {"x", "y"}.issubset(keys):
        x = d["x"].astype(np.float32)
        y = d["y"].astype(np.float32)
        action_dim = x.shape[1] - y.shape[1]
        if action_dim <= 0:
            raise ValueError(
                f"Cannot infer action from x/y shapes for {path}: "
                f"x={x.shape}, y={y.shape}"
            )
        action = x[:, -action_dim:].astype(np.float32)
        return x, y, action

    raise ValueError(f"Unsupported feature npz layout for {path}. Keys: {sorted(keys)}")


def build_models(run_dir: Path, device: str):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["model"]

    input_dim = int(cfg["input_dim"])
    latent_dim = int(cfg["latent_dim"])
    dropout = float(cfg.get("dropout", 0.0))

    cheap = MLP(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dim=int(cfg["cheap_hidden"]),
        layers=int(cfg["cheap_layers"]),
        dropout=dropout,
    ).to(device)

    expensive = MLP(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dim=int(cfg["expensive_hidden"]),
        layers=int(cfg["expensive_layers"]),
        dropout=dropout,
    ).to(device)

    reliability = MLP(
        input_dim=input_dim,
        output_dim=1,
        hidden_dim=int(cfg["reliability_hidden"]),
        layers=int(cfg.get("reliability_layers", 2)),
        dropout=dropout,
    ).to(device)

    ckpt = torch.load(run_dir / "bestval_checkpoint.pt", map_location=device)
    cheap.load_state_dict(ckpt["cheap"])
    expensive.load_state_dict(ckpt["expensive"])
    reliability.load_state_dict(ckpt["reliability"])

    cheap.eval()
    expensive.eval()
    reliability.eval()

    return metrics, cheap, expensive, reliability


def eval_arrays(run_dir: Path, split: str, batch_size: int, device: str):
    metrics, cheap, expensive, reliability = build_models(run_dir, device)

    path = metrics[split]
    x, y, action = as_xy_action(path)

    cheap_pred = predict(cheap, x, batch_size=batch_size, device=device)
    expensive_pred = predict(expensive, x, batch_size=batch_size, device=device)
    rel_score = reliability_scores(reliability, x, batch_size=batch_size, device=device)

    cheap_err = per_sample_mse(cheap_pred, y)
    expensive_err = per_sample_mse(expensive_pred, y)
    action_norm = np.linalg.norm(action, axis=1)

    router_x = np.concatenate(
        [
            x,
            cheap_pred.astype(np.float32),
            rel_score[:, None].astype(np.float32),
            action_norm[:, None].astype(np.float32),
        ],
        axis=1,
    )

    return {
        "x": x,
        "y": y,
        "router_x": router_x.astype(np.float32),
        "cheap_error": cheap_err.astype(np.float64),
        "expensive_error": expensive_err.astype(np.float64),
        "reliability": rel_score.astype(np.float64),
        "action_norm": action_norm.astype(np.float64),
    }


def utility(cheap_err, expensive_err, selected, lam: float, compute_gap: float):
    selected = selected.astype(bool)
    err = np.where(selected, expensive_err, cheap_err).mean()
    compute = 1.0 + compute_gap * selected.mean()
    util = -err - lam * compute
    return float(err), float(compute), float(selected.mean()), float(util)


def tune_threshold(scores, cheap_err, expensive_err, lam: float, compute_gap: float):
    qs = np.linspace(0.0, 1.0, 101)
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


def zscore_from_train(train, val):
    mu = train.mean(axis=0, keepdims=True)
    sig = train.std(axis=0, keepdims=True) + 1e-6
    return (train - mu) / sig, (val - mu) / sig


class RouterMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_classifier(x_train, y_train, x_val, epochs, batch_size, lr, device):
    x_train, x_val = zscore_from_train(x_train, x_val)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)

    model = RouterMLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = float(y_train.mean())
    pos_weight = torch.tensor([(1.0 - pos) / max(pos, 1e-6)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = len(xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_scores = torch.sigmoid(model(xtr)).cpu().numpy().reshape(-1)
        val_scores = torch.sigmoid(model(xva)).cpu().numpy().reshape(-1)
    return train_scores, val_scores


def train_regressor(x_train, y_train, x_val, epochs, batch_size, lr, device):
    x_train, x_val = zscore_from_train(x_train, x_val)

    xtr = torch.tensor(x_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train[:, None], dtype=torch.float32, device=device)
    xva = torch.tensor(x_val, dtype=torch.float32, device=device)

    model = RouterMLP(xtr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    n = len(xtr)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = loss_fn(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        train_scores = model(xtr).cpu().numpy().reshape(-1)
        val_scores = model(xva).cpu().numpy().reshape(-1)
    return train_scores, val_scores


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

    threshold = args.lambda_compute * args.compute_gap
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

        y_cls = (train_gain > threshold).astype(np.float32)
        y_reg = (train_gain - threshold).astype(np.float32)

        rel_train, rel_val = train["reliability"], val["reliability"]
        act_train, act_val = train["action_norm"], val["action_norm"]

        rel_train_z, rel_val_z = zscore_from_train(rel_train[:, None], rel_val[:, None])
        act_train_z, act_val_z = zscore_from_train(act_train[:, None], act_val[:, None])
        hyb_train = 0.5 * rel_train_z.reshape(-1) + 0.5 * act_train_z.reshape(-1)
        hyb_val = 0.5 * rel_val_z.reshape(-1) + 0.5 * act_val_z.reshape(-1)

        cls_train_scores, cls_val_scores = train_classifier(
            train["router_x"], y_cls, val["router_x"], args.epochs, args.batch_size, args.lr, args.device
        )
        reg_train_scores, reg_val_scores = train_regressor(
            train["router_x"], y_reg, val["router_x"], args.epochs, args.batch_size, args.lr, args.device
        )

        score_methods = {
            "reliability threshold": (rel_train, rel_val),
            "action norm threshold": (act_train, act_val),
            "hybrid threshold": (hyb_train, hyb_val),
            "trainval gain classifier": (cls_train_scores, cls_val_scores),
            "trainval gain regressor": (reg_train_scores, reg_val_scores),
        }

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
            "oracle upper bound": val_gain > threshold,
        }

        for method, selected in fixed_methods.items():
            err, comp, sel, util = utility(
                val["cheap_error"], val["expensive_error"], selected, args.lambda_compute, args.compute_gap
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

        for method, (train_score, val_score) in score_methods.items():
            best = tune_threshold(
                train_score,
                train["cheap_error"],
                train["expensive_error"],
                args.lambda_compute,
                args.compute_gap,
            )
            selected = val_score >= best["threshold"]
            err, comp, sel, util = utility(
                val["cheap_error"], val["expensive_error"], selected, args.lambda_compute, args.compute_gap
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

    order = [
        "cheap-only",
        "all-expensive",
        "reliability threshold",
        "action norm threshold",
        "hybrid threshold",
        "trainval gain classifier",
        "trainval gain regressor",
        "oracle upper bound",
    ]
    summary = summarize(rows)
    summary = sorted(summary, key=lambda r: order.index(r["method"]))

    out_table = Path("reports/tables/trainval_gain_router_summary.md")
    out_table.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Train-to-validation gain-router experiment, DINOv2 best-validation\n")
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
    short = [
        "cheap-only",
        "all-exp.",
        "reliability",
        "action norm",
        "hybrid",
        "gain clf.",
        "gain reg.",
        "oracle",
    ]

    x = np.array([r["compute_mean"] for r in summary])
    y = np.array([r["error_mean"] for r in summary])
    xerr = np.array([r["compute_std"] for r in summary])
    yerr = np.array([r["error_std"] for r in summary])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", capsize=4)
    offsets = {
        "cheap-only": (8, 6),
        "all-exp.": (-45, -2),
        "reliability": (-65, -8),
        "action norm": (-45, 8),
        "hybrid": (8, 10),
        "gain clf.": (8, 4),
        "gain reg.": (8, 8),
        "oracle": (8, 4),
    }
    for xi, yi, lab in zip(x, y, short):
        ax.annotate(
            lab,
            (xi, yi),
            xytext=offsets.get(lab, (7, 5)),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_title("Train-to-validation gain router: error--compute trade-off")
    ax.set_xlabel("mean compute")
    ax.set_ylabel("mean prediction error")
    ax.grid(True, alpha=0.25)
    ax.text(0.02, 0.03, "Lower-left is better.", transform=ax.transAxes, alpha=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_router_error_compute.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    gains = np.array([r["delta_utility_mean"] for r in summary])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    yy = np.arange(len(short))
    ax.barh(yy, gains)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(short)
    ax.set_xlabel("mean utility improvement over cheap-only")
    ax.set_title("Train-to-validation gain router at lambda=0.04")
    ax.grid(True, axis="x", alpha=0.25)
    for yi, g in zip(yy, gains):
        if abs(g) < 0.0005:
            ax.text(0.0025, yi, f"{g:+.3f}", va="center", ha="left", fontsize=9)
        elif g > 0:
            ax.text(g + 0.003, yi, f"{g:+.3f}", va="center", ha="left", fontsize=9)
        else:
            ax.text(g - 0.003, yi, f"{g:+.3f}", va="center", ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "trainval_gain_router_delta_utility.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "trainval_gain_router_error_compute.png")
    print(fig_dir / "trainval_gain_router_delta_utility.png")


if __name__ == "__main__":
    main()
