from __future__ import annotations

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "scripts")

from train_mixed_hard_delta_transformer import DeltaTransformer

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v26_multicapacity_oracle")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV = OUT_DIR / "multicapacity_predictions.csv"
SUMMARY_CSV = OUT_DIR / "multicapacity_oracle_summary.csv"
SUMMARY_MD = OUT_DIR / "multicapacity_oracle_summary.md"

LATENCY_CSV = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v24_measured_latency/capacity_latency_summary.csv")

CKPT_ROOT = Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer")

MODELS = [
    ("tiny_full_seed0", "Tiny"),
    ("small_full_seed0", "Small"),
    ("medium_full_seed0", "Medium"),
    ("full_seed0", "Full"),
]

VARIANTS = [
    "block2_h72_seed11",
    "block2_v18_seed12",
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

VARIANT_LABELS = {
    "block2_h72_seed11": "2-block, H=72",
    "block2_v18_seed12": "2-block, V=1.8",
    "block3_h36_seed13": "3-block, H=36",
    "block3_h48_seed10": "3-block, H=48",
    "block3_h72_seed14": "3-block, H=72",
}

DATA_PATHS = {
    v: Path(f"outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/{v}/pybullet_obstacle_rgb_v1_ood_{v}_dinov2_vits14_pool4.npz")
    for v in VARIANTS
}

GROUP_PATHS = {
    v: Path(f"outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/{v}/mixed_hard_state_disjoint_v0_groups.npz")
    for v in VARIANTS
}

LAMBDAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
TEMP = 0.02
EPS = 1e-8


def load_costs() -> dict[str, float]:
    lat = pd.read_csv(LATENCY_CSV)
    b128 = lat[lat["batch_size"] == 128].copy()
    return dict(zip(b128["model_label"], b128["relative_latency_b128"]))


def load_model(model_name: str, device: str):
    ckpt_path = CKPT_ROOT / model_name / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ca = ckpt["args"]

    model = DeltaTransformer(
        token_dim=int(ckpt["token_dim"]),
        action_dim=int(ckpt["action_dim"]),
        tokens=int(ckpt["tokens"]),
        model_dim=int(ca["model_dim"]),
        heads=int(ca["heads"]),
        layers=int(ca["layers"]),
        dropout=float(ca["dropout"]),
        mode=ca["mode"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = {
        "checkpoint": str(ckpt_path),
        "model_dim": int(ca["model_dim"]),
        "layers": int(ca["layers"]),
        "heads": int(ca["heads"]),
        "params": int(sum(p.numel() for p in model.parameters())),
    }
    return model, meta


def softmax_confidence(dist: np.ndarray, temp: float = TEMP) -> np.ndarray:
    logits = -dist / temp
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    p = exp / exp.sum(axis=1, keepdims=True)
    return p.max(axis=1)


@torch.no_grad()
def predict_variant_model(
    variant: str,
    model_name: str,
    label: str,
    device: str,
    cost: float,
    batch_groups: int = 128,
) -> pd.DataFrame:
    data_path = DATA_PATHS[variant]
    group_path = GROUP_PATHS[variant]

    d = np.load(data_path, allow_pickle=True)
    g = np.load(group_path, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)
    action_id = g["action_id"].astype(np.int64)
    action_names = d["action_names"].astype(str)

    if "split" in g.files:
        test_groups = np.where(g["split"].astype(str) == "test")[0]
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(anchor))
        n_train = int(0.8 * len(anchor))
        n_val = int(0.1 * len(anchor))
        test_groups = perm[n_train + n_val:]

    stay_ids = set(np.where(action_names == "stay")[0].tolist())
    moving_mask = np.array([action_id[i] not in stay_ids for i in test_groups], dtype=bool)
    test_groups = test_groups[moving_mask]

    model, meta = load_model(model_name, device)

    delta = y - z
    rows = []

    for start in range(0, len(test_groups), batch_groups):
        gids = test_groups[start:start + batch_groups]
        anch = anchor[gids]
        c = cand[gids]

        zt = torch.from_numpy(z[anch]).float().to(device)
        at = torch.from_numpy(a[anch]).float().to(device)

        pred_delta = model(zt, at).cpu().numpy()
        candidate_delta = delta[c]
        dist = ((pred_delta[:, None] - candidate_delta) ** 2).mean(axis=(2, 3))

        best = dist.argmin(axis=1)
        best_dist = dist.min(axis=1)
        second_dist = np.partition(dist, 1, axis=1)[:, 1]
        correct_dist = dist[:, 0]

        min_dist = dist.min(axis=1)
        tie_credit = []
        for i in range(dist.shape[0]):
            tied = np.where(np.abs(dist[i] - min_dist[i]) <= EPS)[0]
            if 0 in tied:
                tie_credit.append(1.0 / len(tied))
            else:
                tie_credit.append(0.0)

        strict_correct = correct_dist < np.min(dist[:, 1:], axis=1) - EPS
        conf = softmax_confidence(dist)

        for j, gid in enumerate(gids):
            aid = int(action_id[gid])
            rows.append({
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "group_id": int(gid),
                "model": model_name,
                "model_label": label,
                "relative_latency": float(cost),
                "action_id": aid,
                "action_name": str(action_names[aid]),
                "argmin_correct": float(best[j] == 0),
                "strict_correct": float(strict_correct[j]),
                "tie_credit": float(tie_credit[j]),
                "confidence": float(conf[j]),
                "best_distance": float(best_dist[j]),
                "second_best_distance": float(second_dist[j]),
                "pred_margin": float(second_dist[j] - best_dist[j]),
                "correct_distance": float(correct_dist[j]),
                "model_dim": meta["model_dim"],
                "layers": meta["layers"],
                "heads": meta["heads"],
                "params": meta["params"],
            })

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return pd.DataFrame(rows)


def oracle_for_variant(pred: pd.DataFrame, variant: str, lam: float) -> dict:
    sub = pred[pred["variant"] == variant].copy()

    piv_correct = sub.pivot_table(
        index="group_id",
        columns="model_label",
        values="tie_credit",
        aggfunc="first",
    )

    piv_cost = sub.drop_duplicates("model_label").set_index("model_label")["relative_latency"]

    model_order = ["Tiny", "Small", "Medium", "Full"]
    util = pd.DataFrame(index=piv_correct.index)

    for m in model_order:
        util[m] = piv_correct[m] - lam * float(piv_cost[m])

    fixed_rows = []
    for m in model_order:
        fixed_rows.append({
            "lambda": lam,
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "method": f"fixed_{m}",
            "utility": float(util[m].mean()),
            "top1": float(piv_correct[m].mean()),
            "relative_latency": float(piv_cost[m]),
            "select_Tiny": float(m == "Tiny"),
            "select_Small": float(m == "Small"),
            "select_Medium": float(m == "Medium"),
            "select_Full": float(m == "Full"),
            "n": int(len(util)),
        })

    costs = np.array([float(piv_cost[m]) for m in model_order])
    u = util[model_order].to_numpy()

    selected = []
    oracle_util = []
    for i in range(u.shape[0]):
        row = u[i]
        mx = row.max()
        tied = np.where(np.abs(row - mx) <= 1e-12)[0]
        chosen = tied[np.argmin(costs[tied])]
        selected.append(model_order[int(chosen)])
        oracle_util.append(float(row[int(chosen)]))

    selected = pd.Series(selected)
    oracle_row = {
        "lambda": lam,
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "method": "oracle_multicapacity",
        "utility": float(np.mean(oracle_util)),
        "top1": float(np.mean([
            piv_correct.loc[idx, selected.iloc[k]]
            for k, idx in enumerate(piv_correct.index)
        ])),
        "relative_latency": float(np.mean([piv_cost[m] for m in selected])),
        "select_Tiny": float((selected == "Tiny").mean()),
        "select_Small": float((selected == "Small").mean()),
        "select_Medium": float((selected == "Medium").mean()),
        "select_Full": float((selected == "Full").mean()),
        "n": int(len(util)),
    }

    return fixed_rows + [oracle_row]


def build_md(summary: pd.DataFrame) -> str:
    lines = []
    lines.append("# SPSM v26 multi-capacity oracle headroom\n")
    lines.append("This estimates the per-instance upper bound for routing among Tiny, Small, Medium, and Full.")
    lines.append("Utility uses measured batch-128 relative latency from v24: `tie_credit - lambda * relative_latency`.")
    lines.append("This is not a learned router yet; it asks whether multi-capacity routing is worth learning.\n")

    lines.append("## Oracle gain over best fixed capacity\n")
    lines.append("| lambda | variant | best fixed | best fixed util | oracle util | gain | oracle top-1 | oracle latency | selection Tiny/Small/Medium/Full |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---|")

    for lam in LAMBDAS:
        for variant in VARIANTS:
            sub = summary[(summary["lambda"] == lam) & (summary["variant"] == variant)]
            fixed = sub[sub["method"].str.startswith("fixed_")].copy()
            oracle = sub[sub["method"] == "oracle_multicapacity"].iloc[0]
            best = fixed.sort_values("utility", ascending=False).iloc[0]

            sel = (
                f"{oracle['select_Tiny']:.2f}/"
                f"{oracle['select_Small']:.2f}/"
                f"{oracle['select_Medium']:.2f}/"
                f"{oracle['select_Full']:.2f}"
            )

            lines.append(
                f"| {lam:.2f} | {VARIANT_LABELS[variant]} | `{best['method'].replace('fixed_', '')}` | "
                f"{best['utility']:.6f} | {oracle['utility']:.6f} | "
                f"{oracle['utility'] - best['utility']:.6f} | "
                f"{oracle['top1']:.6f} | {oracle['relative_latency']:.6f} | {sel} |"
            )

    lines.append("\n## Fixed and oracle utilities on hard OOD\n")
    lines.append("| lambda | variant | Tiny | Small | Medium | Full | oracle |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|")

    for lam in LAMBDAS:
        for variant in ["block3_h36_seed13", "block3_h48_seed10", "block3_h72_seed14"]:
            sub = summary[(summary["lambda"] == lam) & (summary["variant"] == variant)].set_index("method")
            lines.append(
                f"| {lam:.2f} | {VARIANT_LABELS[variant]} | "
                f"{sub.loc['fixed_Tiny', 'utility']:.6f} | "
                f"{sub.loc['fixed_Small', 'utility']:.6f} | "
                f"{sub.loc['fixed_Medium', 'utility']:.6f} | "
                f"{sub.loc['fixed_Full', 'utility']:.6f} | "
                f"{sub.loc['oracle_multicapacity', 'utility']:.6f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append("- If oracle gain over best fixed is large, there is real headroom for learned multi-capacity routing.")
    lines.append("- If oracle mostly selects one model, fixed capacity is sufficient for that shift.")
    lines.append("- If oracle selection is mixed, the shift contains instance-level heterogeneity that a router could exploit.")
    lines.append("- The next step should only train a router if this table shows meaningful oracle headroom.")

    return "\n".join(lines) + "\n"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    costs = load_costs()

    pred_parts = []
    for variant in VARIANTS:
        for model_name, label in MODELS:
            print(f"variant={variant} model={label}", flush=True)
            pred_parts.append(
                predict_variant_model(
                    variant=variant,
                    model_name=model_name,
                    label=label,
                    device=device,
                    cost=costs[label],
                )
            )

    pred = pd.concat(pred_parts, ignore_index=True)
    pred.to_csv(PRED_CSV, index=False)

    rows = []
    for lam in LAMBDAS:
        for variant in VARIANTS:
            rows.extend(oracle_for_variant(pred, variant, lam))

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_CSV, index=False)
    SUMMARY_MD.write_text(build_md(summary), encoding="utf-8")

    print()
    print(SUMMARY_MD)
    print(SUMMARY_CSV)
    print(PRED_CSV)
    print(SUMMARY_MD.read_text())


if __name__ == "__main__":
    main()
