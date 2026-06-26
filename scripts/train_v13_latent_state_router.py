from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from train_v8_geometry_aware_gain_router import (
    prepare,
    add_meta_features,
    context_cols,
    utility,
    best_conf_threshold,
    train_router,
    best_score_threshold,
    TRAIN_VARIANTS,
    TEST_VARIANTS,
    ALL_VARIANTS,
    LAMBDAS,
    META,
)

from train_v10_local_expected_gain_router import fit_knn_gain_router
from train_v11_rescue_harm_router import fit_local_rescue_harm_router

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v13_latent_state_router")
OUT_MD = OUT_DIR / "latent_state_router_loso_summary.md"
OUT_CSV = OUT_DIR / "latent_state_router_loso_summary.csv"


DATASET_PATHS = {
    "id_block2_h36_seed0": Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz"),
    "block2_h72_seed11": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14_pool4.npz"),
    "block2_v18_seed12": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14_pool4.npz"),
    "block3_h36_seed13": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14_pool4.npz"),
    "block3_h48_seed10": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14_pool4.npz"),
    "block3_h72_seed14": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14_pool4.npz"),
}

GROUP_PATHS = {
    "id_block2_h36_seed0": Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz"),
    "block2_h72_seed11": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/mixed_hard_state_disjoint_v0_groups.npz"),
    "block2_v18_seed12": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/mixed_hard_state_disjoint_v0_groups.npz"),
    "block3_h36_seed13": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/mixed_hard_state_disjoint_v0_groups.npz"),
    "block3_h48_seed10": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/mixed_hard_state_disjoint_v0_groups.npz"),
    "block3_h72_seed14": Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/mixed_hard_state_disjoint_v0_groups.npz"),
}


def action_cols(m: pd.DataFrame) -> list[str]:
    return sorted([c for c in m.columns if c.startswith("action_") and c.removeprefix("action_").isdigit()])


def token_features(z: np.ndarray, cls: np.ndarray | None = None) -> dict[str, np.ndarray]:
    # z: [N, 16, 384], pooled 4x4 patch tokens.
    z = z.astype(np.float32)
    n = z.shape[0]
    grid = z.reshape(n, 4, 4, z.shape[-1])

    token_norm = np.linalg.norm(z, axis=-1)
    token_mean = z.mean(axis=1)
    centered = z - token_mean[:, None, :]

    dx = grid[:, :, 1:, :] - grid[:, :, :-1, :]
    dy = grid[:, 1:, :, :] - grid[:, :-1, :, :]

    feats = {
        "latent_token_norm_mean": token_norm.mean(axis=1),
        "latent_token_norm_std": token_norm.std(axis=1),
        "latent_token_norm_max": token_norm.max(axis=1),
        "latent_token_norm_min": token_norm.min(axis=1),
        "latent_spatial_var": (centered * centered).mean(axis=(1, 2)),
        "latent_spatial_std": centered.std(axis=(1, 2)),
        "latent_dx_norm_mean": np.linalg.norm(dx, axis=-1).mean(axis=(1, 2)),
        "latent_dx_norm_std": np.linalg.norm(dx, axis=-1).std(axis=(1, 2)),
        "latent_dy_norm_mean": np.linalg.norm(dy, axis=-1).mean(axis=(1, 2)),
        "latent_dy_norm_std": np.linalg.norm(dy, axis=-1).std(axis=(1, 2)),
    }

    feats["latent_grad_norm_mean"] = 0.5 * (feats["latent_dx_norm_mean"] + feats["latent_dy_norm_mean"])
    feats["latent_grad_anisotropy"] = np.abs(feats["latent_dx_norm_mean"] - feats["latent_dy_norm_mean"])

    if cls is not None:
        cls = cls.astype(np.float32)
        feats["latent_cls_norm"] = np.linalg.norm(cls, axis=-1)

    return feats


def build_variant_latent_feature_frame(variant: str) -> pd.DataFrame:
    data_path = DATASET_PATHS[variant]
    group_path = GROUP_PATHS[variant]

    if not data_path.exists():
        raise FileNotFoundError(data_path)
    if not group_path.exists():
        raise FileNotFoundError(group_path)

    data = np.load(data_path, allow_pickle=True)
    groups = np.load(group_path, allow_pickle=True)

    z_current = data["z_current"]
    cls_current = data["cls_current"] if "cls_current" in data.files else None
    feats = token_features(z_current, cls_current)

    group_ids = np.arange(len(groups["anchor_indices"]), dtype=np.int64)
    anchor_idx = groups["anchor_indices"].astype(np.int64)

    rows = {
        "variant": np.array([variant] * len(group_ids)),
        "group_id": group_ids,
        "anchor_index": anchor_idx,
    }

    for k, arr in feats.items():
        rows[k] = arr[anchor_idx]

    # Current-state metadata only. Avoid future_xy because it is not available before rollout.
    if "state_xy" in data.files:
        state_xy = data["state_xy"].astype(np.float32)
        rows["state_x"] = state_xy[anchor_idx, 0]
        rows["state_y"] = state_xy[anchor_idx, 1]
        rows["state_radius"] = np.linalg.norm(state_xy[anchor_idx], axis=1)

    if "blocked_mask" in data.files:
        rows["blocked_mask_value"] = data["blocked_mask"][anchor_idx].astype(np.float32)

    return pd.DataFrame(rows)


def add_latent_state_features(m: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for variant in ALL_VARIANTS:
        f = build_variant_latent_feature_frame(variant)
        frames.append(f)

    lf = pd.concat(frames, ignore_index=True)

    out = m.merge(lf, on=["variant", "group_id"], how="left")
    latent_cols = [c for c in lf.columns if c not in {"variant", "group_id", "anchor_index"}]

    missing = out[latent_cols].isna().mean().sort_values(ascending=False)
    print("===== LATENT FEATURE JOIN MISSING RATE =====")
    print(missing.to_string())

    for c in latent_cols:
        out[c] = out[c].replace([np.inf, -np.inf], np.nan)
        out[c] = out[c].fillna(out.groupby("variant")[c].transform("mean")).fillna(0.0)

    return out


def feature_sets(m: pd.DataFrame) -> dict[str, list[str]]:
    uncertainty = ["cheap_confidence", "cheap_uncertainty"]
    margin = [
        "cheap_confidence",
        "cheap_uncertainty",
        "cheap_best_distance",
        "cheap_second_best_distance",
        "cheap_pred_margin",
    ]
    actions = action_cols(m)
    context = margin + actions + ["horizon_norm", "velocity_norm"]

    latent = sorted([
        c for c in m.columns
        if c.startswith("latent_") or c in ["state_x", "state_y", "state_radius", "blocked_mask_value"]
    ])

    out = {
        "uncertainty": uncertainty,
        "margin": margin,
        "context": context,
        "latent_only": latent,
        "context_latent": context + latent,
    }

    return {k: [c for c in cols if c in m.columns] for k, cols in out.items()}


def eval_route(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return float(acc), float(util), float(rate)


def evaluate_one(m: pd.DataFrame, heldout: str, lam: float):
    train_variants = [v for v in ALL_VARIANTS if v != heldout]
    train = m[m["variant"].isin(train_variants)].copy()
    test = m[m["variant"] == heldout].copy()

    rows = []

    cheap_util = float(test["cheap_correct"].mean())
    full_util = float(test["full_correct"].mean()) - lam

    oracle_route = (test["gain"].values - lam) > 0
    _, oracle_util, oracle_rate = eval_route(test, oracle_route, lam)

    _, conf_thr, _, _ = best_conf_threshold(train, lam)
    conf_route = test["cheap_confidence"].values < conf_thr
    _, conf_util, conf_rate = eval_route(test, conf_route, lam)

    ccols = context_cols(m)
    context_cls_fn = train_router(m, ccols, train_variants, lam, seed=0)
    train_context_scores = context_cls_fn(train)
    _, context_thr, _, _ = best_score_threshold(train_context_scores, train, lam)
    context_cls_route = context_cls_fn(test) >= context_thr
    _, context_cls_util, context_cls_rate = eval_route(test, context_cls_route, lam)

    base = {
        "lambda": lam,
        "heldout": heldout,
        "variant_label": META[heldout]["label"],
        "cheap": cheap_util,
        "full": full_util,
        "oracle": oracle_util,
        "oracle_route": oracle_rate,
        "conf": conf_util,
        "conf_route": conf_rate,
        "context_cls": context_cls_util,
        "context_cls_route": context_cls_rate,
    }

    for name, cols in feature_sets(m).items():
        knn_fn, knn_rule, knn_k, knn_thr = fit_knn_gain_router(m, cols, train_variants, lam)
        knn_scores = knn_fn(test)
        knn_route = knn_scores > lam if knn_rule == "gain_threshold" else knn_scores >= knn_thr
        _, knn_util, knn_rate = eval_route(test, knn_route, lam)

        row = dict(base)
        row.update({
            "router": "knn_gain",
            "feature_set": name,
            "util": knn_util,
            "route": knn_rate,
            "k": knn_k,
            "rule": knn_rule,
        })
        rows.append(row)

        rh_fn, rh_k, rh_thr, rh_cap = fit_local_rescue_harm_router(m, cols, train_variants, lam)
        rh_route, rescue, harm, gain = rh_fn(test)
        _, rh_util, rh_rate = eval_route(test, rh_route, lam)

        row = dict(base)
        row.update({
            "router": "rescue_harm",
            "feature_set": name,
            "util": rh_util,
            "route": rh_rate,
            "k": rh_k,
            "rule": f"thr={rh_thr:.4f},cap={rh_cap:.4f}",
        })
        rows.append(row)

    return rows


def to_md(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# SPSM v13 observable latent-state router — leave-one-hard-out\n")
    lines.append("This experiment adds features computed only from the current observed latent state `z_i = phi(x_i)`.")
    lines.append("Unlike candidate-mined geometry, these features are available before deciding whether to call the expensive model.\n")

    lines.append("## Best method per heldout and lambda\n")
    lines.append("| lambda | heldout | cheap | full | oracle | conf | context-cls | best learned | best feature set | best router |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---|---|")

    for (lam, heldout), sub in df.groupby(["lambda", "heldout"], sort=False):
        best = sub.sort_values("util", ascending=False).iloc[0]
        base = sub.iloc[0]
        lines.append(
            f"| {lam:.2f} | {base['variant_label']} | "
            f"{base['cheap']:.6f} | {base['full']:.6f} | {base['oracle']:.6f} | "
            f"{base['conf']:.6f} | {base['context_cls']:.6f} | "
            f"{best['util']:.6f} | `{best['feature_set']}` | `{best['router']}` |"
        )

    lines.append("\n## Full learned-router table\n")
    lines.append("| lambda | heldout | router | feature set | util | route | k | rule |")
    lines.append("|---:|---|---|---|---:|---:|---:|---|")

    for _, r in df.sort_values(["heldout", "lambda", "router", "feature_set"]).iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | `{r['router']}` | `{r['feature_set']}` | "
            f"{r['util']:.6f} | {r['route']:.6f} | {int(r['k'])} | `{r['rule']}` |"
        )

    lines.append("\n## Interpretation guide\n")
    lines.append("- If `latent_only` helps, current visual-state complexity contains value-of-computation signal.")
    lines.append("- If `context_latent` beats `context`, latent state complexity improves routing beyond uncertainty and horizon/velocity.")
    lines.append("- If it does not help, the current cheap-model risk features are already the dominant observable signal.")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = prepare()
    m = add_meta_features(m)
    m = add_latent_state_features(m)

    print("Rows:", len(m))
    print("Feature sets:")
    for k, cols in feature_sets(m).items():
        print(k, len(cols), cols)

    rows = []
    for heldout in TEST_VARIANTS:
        for lam in LAMBDAS:
            print(f"heldout={heldout} lambda={lam}")
            rows.extend(evaluate_one(m, heldout, lam))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(to_md(df), encoding="utf-8")

    print("\n===== WROTE =====")
    print(OUT_MD)
    print(OUT_CSV)
    print("\n===== SUMMARY =====")
    print(OUT_MD.read_text())


def main_wrapper():
    main()


if __name__ == "__main__":
    main_wrapper()
