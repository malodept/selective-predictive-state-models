from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_real_selective_refinement_bestval import (
    MLP,
    per_sample_mse,
    predict,
    reliability_scores,
)


LAMBDA_COMPUTE = 0.04
CHEAP_COMPUTE = 1.0
EXPENSIVE_COMPUTE = 4.0
COMPUTE_GAP = EXPENSIVE_COMPUTE - CHEAP_COMPUTE
THRESHOLD = LAMBDA_COMPUTE * COMPUTE_GAP


def parse_run_name(run_dir: Path) -> tuple[str, str]:
    m = re.search(r"protocol_loo_(P\d+)_(.*?)_seed0", run_dir.name)
    if not m:
        return "unknown", run_dir.name
    return m.group(1), m.group(2)


def as_xy_action(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path, allow_pickle=True)

    if {"z_current", "z_future", "action"}.issubset(d.files):
        z = np.asarray(d["z_current"], dtype=np.float32)
        y = np.asarray(d["z_future"], dtype=np.float32)
        action = np.asarray(d["action"], dtype=np.float32)
        x = np.concatenate([z, action], axis=1)
        return x, y, action

    if {"x", "y", "action"}.issubset(d.files):
        x = np.asarray(d["x"], dtype=np.float32)
        y = np.asarray(d["y"], dtype=np.float32)
        action = np.asarray(d["action"], dtype=np.float32)
        return x, y, action

    raise KeyError(f"Unsupported feature file keys in {path}: {d.files}")


def last_bias_dim(state: dict[str, torch.Tensor]) -> int:
    keys = sorted([k for k in state if k.endswith(".bias")])
    return int(state[keys[-1]].shape[0])


def load_models(run_dir: Path, device: str):
    metrics = json.loads((run_dir / "metrics.json").read_text())
    cfg = metrics["model"]

    ckpt = torch.load(run_dir / "bestval_checkpoint.pt", map_location=device)

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
    cheap.load_state_dict(ckpt["cheap"])
    cheap.eval()

    expensive = MLP(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dim=int(cfg["expensive_hidden"]),
        layers=int(cfg["expensive_layers"]),
        dropout=dropout,
    ).to(device)
    expensive.load_state_dict(ckpt["expensive"])
    expensive.eval()

    rel_out = last_bias_dim(ckpt["reliability"])
    reliability = MLP(
        input_dim=input_dim,
        output_dim=rel_out,
        hidden_dim=int(cfg["reliability_hidden"]),
        layers=2,
        dropout=dropout,
    ).to(device)
    reliability.load_state_dict(ckpt["reliability"])
    reliability.eval()

    return metrics, cheap, expensive, reliability


def eval_on_feature_file(
    feature_path: Path,
    cheap,
    expensive,
    reliability,
    batch_size: int,
    device: str,
) -> dict[str, np.ndarray]:
    x, y, action = as_xy_action(feature_path)

    cheap_pred = predict(cheap, x, batch_size=batch_size, device=device)
    expensive_pred = predict(expensive, x, batch_size=batch_size, device=device)

    cheap_errors = per_sample_mse(cheap_pred, y)
    expensive_errors = per_sample_mse(expensive_pred, y)

    rel = reliability_scores(reliability, x, batch_size=batch_size, device=device)
    action_norm = np.linalg.norm(action, axis=1)

    router_x = np.concatenate(
        [
            x.astype(np.float32),
            cheap_pred.astype(np.float32),
            rel[:, None].astype(np.float32),
            action_norm[:, None].astype(np.float32),
        ],
        axis=1,
    )

    return {
        "x": x,
        "y": y,
        "cheap_pred": cheap_pred,
        "expensive_pred": expensive_pred,
        "cheap_errors": cheap_errors,
        "expensive_errors": expensive_errors,
        "gain": cheap_errors - expensive_errors,
        "profitable": (cheap_errors - expensive_errors) > THRESHOLD,
        "router_x": router_x,
        "reliability": rel,
        "action_norm": action_norm,
    }


def utility(cheap_errors: np.ndarray, expensive_errors: np.ndarray, selected: np.ndarray) -> tuple[float, float, float, float]:
    selected = selected.astype(bool)
    error = float(np.where(selected, expensive_errors, cheap_errors).mean())
    compute = float(CHEAP_COMPUTE + COMPUTE_GAP * selected.mean())
    selected_frac = float(selected.mean())
    util = float(-error - LAMBDA_COMPUTE * compute)
    return error, compute, selected_frac, util


def calibrate_threshold(
    scores: np.ndarray,
    cheap_errors: np.ndarray,
    expensive_errors: np.ndarray,
) -> tuple[float, float, float]:
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    candidates = np.concatenate(
        [
            [float(scores.max()) + 1e-6],
            candidates,
            [float(scores.min()) - 1e-6],
        ]
    )

    cheap_util = utility(cheap_errors, expensive_errors, np.zeros_like(scores, dtype=bool))[3]

    best_thr = float(scores.max()) + 1e-6
    best_delta = -1e9
    best_selected = 0.0

    for thr in candidates:
        selected = scores >= thr
        _, _, selected_frac, util = utility(cheap_errors, expensive_errors, selected)
        delta = util - cheap_util
        if delta > best_delta:
            best_delta = float(delta)
            best_thr = float(thr)
            best_selected = float(selected_frac)

    return best_thr, best_delta, best_selected


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-glob", default="outputs/protocol_loo_P*_cheap*_exp20_seed0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calib-fraction", type=float, default=0.25)
    parser.add_argument("--fallback-margin", type=float, default=0.0)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    run_dirs = sorted(Path(".").glob(args.run_glob))
    if not run_dirs:
        raise FileNotFoundError(f"No runs found for glob: {args.run_glob}")

    rows = []

    for run_dir in run_dirs:
        heldout, run_name = parse_run_name(run_dir)
        print(f"\n=== {heldout} {run_name} ===")

        metrics, cheap, expensive, reliability = load_models(run_dir, args.device)

        train_path = Path(metrics["train"])
        val_path = Path(metrics["val"])

        print("train:", train_path)
        print("val  :", val_path)

        train = eval_on_feature_file(train_path, cheap, expensive, reliability, args.batch_size, args.device)
        val = eval_on_feature_file(val_path, cheap, expensive, reliability, args.batch_size, args.device)

        y = train["profitable"].astype(int)

        idx = np.arange(len(y))
        stratify = y if len(np.unique(y)) == 2 else None

        fit_idx, calib_idx = train_test_split(
            idx,
            test_size=args.calib_fraction,
            random_state=args.seed,
            stratify=stratify,
        )

        clf = RandomForestClassifier(
            n_estimators=args.trees,
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=args.seed,
            n_jobs=-1,
        )

        clf.fit(train["router_x"][fit_idx], y[fit_idx])

        calib_scores = clf.predict_proba(train["router_x"][calib_idx])[:, 1]
        thr, calib_delta, calib_selected = calibrate_threshold(
            calib_scores,
            train["cheap_errors"][calib_idx],
            train["expensive_errors"][calib_idx],
        )

        val_scores = clf.predict_proba(val["router_x"])[:, 1]
        selected = val_scores >= thr

        if calib_delta <= args.fallback_margin:
            selected_fallback = np.zeros_like(selected, dtype=bool)
        else:
            selected_fallback = selected.copy()

        cheap_sel = np.zeros_like(selected, dtype=bool)
        all_sel = np.ones_like(selected, dtype=bool)
        oracle_sel = val["gain"] > THRESHOLD

        cheap_err, cheap_comp, cheap_frac, cheap_util = utility(val["cheap_errors"], val["expensive_errors"], cheap_sel)
        all_err, all_comp, all_frac, all_util = utility(val["cheap_errors"], val["expensive_errors"], all_sel)
        rf_err, rf_comp, rf_frac, rf_util = utility(val["cheap_errors"], val["expensive_errors"], selected)
        fb_err, fb_comp, fb_frac, fb_util = utility(val["cheap_errors"], val["expensive_errors"], selected_fallback)
        oracle_err, oracle_comp, oracle_frac, oracle_util = utility(val["cheap_errors"], val["expensive_errors"], oracle_sel)

        rows.extend(
            [
                {
                    "heldout": heldout,
                    "run": run_name,
                    "method": "cheap-only",
                    "error": cheap_err,
                    "compute": cheap_comp,
                    "selected": cheap_frac,
                    "utility": cheap_util,
                    "delta": 0.0,
                    "calib_delta": np.nan,
                    "threshold": np.nan,
                },
                {
                    "heldout": heldout,
                    "run": run_name,
                    "method": "all-expensive",
                    "error": all_err,
                    "compute": all_comp,
                    "selected": all_frac,
                    "utility": all_util,
                    "delta": all_util - cheap_util,
                    "calib_delta": np.nan,
                    "threshold": np.nan,
                },
                {
                    "heldout": heldout,
                    "run": run_name,
                    "method": "learned RF gain router",
                    "error": rf_err,
                    "compute": rf_comp,
                    "selected": rf_frac,
                    "utility": rf_util,
                    "delta": rf_util - cheap_util,
                    "calib_delta": calib_delta,
                    "threshold": thr,
                },
                {
                    "heldout": heldout,
                    "run": run_name,
                    "method": "learned RF gain router + fallback",
                    "error": fb_err,
                    "compute": fb_comp,
                    "selected": fb_frac,
                    "utility": fb_util,
                    "delta": fb_util - cheap_util,
                    "calib_delta": calib_delta,
                    "threshold": thr,
                },
                {
                    "heldout": heldout,
                    "run": run_name,
                    "method": "oracle upper bound",
                    "error": oracle_err,
                    "compute": oracle_comp,
                    "selected": oracle_frac,
                    "utility": oracle_util,
                    "delta": oracle_util - cheap_util,
                    "calib_delta": np.nan,
                    "threshold": np.nan,
                },
            ]
        )

    out = Path("reports/tables/protocol/loo_learned_gain_router_seed0.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Leave-one-trajectory-out learned gain router, DINOv2, seed 0",
        "",
        f"Lambda compute: `{LAMBDA_COMPUTE:.4f}`. Marginal expensive threshold: `{THRESHOLD:.4f}`.",
        f"Router: random forest classifier trained on non-held-out trajectory data, calibrated on `{args.calib_fraction:.2f}` of training samples.",
        "",
        "| heldout | run | method | error | compute | selected | utility | ΔU vs cheap | calib ΔU | threshold |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        calib_s = "" if np.isnan(r["calib_delta"]) else f"{r['calib_delta']:.4f}"
        thr_s = "" if np.isnan(r["threshold"]) else f"{r['threshold']:.4f}"
        lines.append(
            f"| {r['heldout']} | {r['run']} | {r['method']} | "
            f"{r['error']:.4f} | {r['compute']:.4f} | {r['selected']:.4f} | "
            f"{r['utility']:.4f} | {r['delta']:.4f} | {calib_s} | {thr_s} |"
        )

    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())

    fig_dir = Path("reports/figures/protocol")
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_rows = [r for r in rows if r["method"] in ["all-expensive", "learned RF gain router", "learned RF gain router + fallback", "oracle upper bound"]]
    labels = [f"{r['heldout']}\n{r['run'].replace('_', '/')}\n{r['method'].replace('learned RF gain router', 'RF')}" for r in plot_rows]
    x = np.arange(len(plot_rows))
    deltas = np.array([r["delta"] for r in plot_rows])

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x, deltas)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right")
    ax.set_ylabel("utility improvement over cheap-only")
    ax.set_title("LOO learned profitable-call router under trajectory shift")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_learned_gain_router_delta_utility_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    trade_rows = [r for r in rows if r["method"] in ["cheap-only", "all-expensive", "learned RF gain router + fallback", "oracle upper bound"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    for method in ["cheap-only", "all-expensive", "learned RF gain router + fallback", "oracle upper bound"]:
        rs = [r for r in trade_rows if r["method"] == method]
        ax.scatter(
            [r["compute"] for r in rs],
            [r["error"] for r in rs],
            label=method,
            s=50,
        )
    ax.set_xlabel("mean compute")
    ax.set_ylabel("mean prediction error")
    ax.set_title("LOO learned router: error--compute trade-off")
    ax.text(0.02, 0.04, "Lower-left is better.", transform=ax.transAxes, alpha=0.7)
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_dir / "loo_learned_gain_router_error_compute_seed0.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(fig_dir / "loo_learned_gain_router_delta_utility_seed0.png")
    print(fig_dir / "loo_learned_gain_router_error_compute_seed0.png")


if __name__ == "__main__":
    main()
