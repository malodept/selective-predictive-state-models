from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from train_v8_geometry_aware_gain_router import (
    prepare,
    add_meta_features,
    add_optional_group_geometry,
    context_cols,
    geometry_cols,
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

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v12_router_feature_ablation")
OUT_MD = OUT_DIR / "router_feature_ablation_loso_summary.md"
OUT_CSV = OUT_DIR / "router_feature_ablation_loso_summary.csv"


def action_cols(m: pd.DataFrame) -> list[str]:
    return sorted([c for c in m.columns if c.startswith("action_") and c.removeprefix("action_").isdigit()])


def feature_sets(m: pd.DataFrame) -> dict[str, list[str]]:
    base_uncertainty = [
        "cheap_confidence",
        "cheap_uncertainty",
    ]

    margin = [
        "cheap_confidence",
        "cheap_uncertainty",
        "cheap_best_distance",
        "cheap_second_best_distance",
        "cheap_pred_margin",
    ]

    actions = action_cols(m)

    env_context = [
        "horizon_norm",
        "velocity_norm",
    ]

    mined_geometry = [
        "blocked_norm",
        "blocked_x_horizon",
        "blocked_x_velocity",
        "complexity_score",
        "npz_anchor_blocked",
        "npz_candidate_blocked_mean",
        "npz_candidate_blocked_std",
        "npz_blocked_mismatch_frac",
        "npz_state_dist_min",
        "npz_state_dist_mean",
        "npz_state_dist_max",
        "npz_state_dist_std",
        "npz_same_state_neg_frac",
        "npz_same_action_neg_frac",
    ]

    out = {
        "uncertainty": base_uncertainty,
        "margin": margin,
        "margin_action": margin + actions,
        "context": margin + actions + env_context,
        "mined_geometry_only": env_context + mined_geometry,
        "context_geometry": geometry_cols(m),
    }

    # Keep only columns that exist.
    return {k: [c for c in cols if c in m.columns] for k, cols in out.items()}


def eval_route(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return float(acc), float(util), float(rate)


def evaluate_one(m: pd.DataFrame, heldout: str, lam: float):
    train_variants = [v for v in ALL_VARIANTS if v != heldout]
    train = m[m["variant"].isin(train_variants)].copy()
    test = m[m["variant"] == heldout].copy()

    rows = []

    # Static baselines.
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

    fsets = feature_sets(m)

    for name, cols in fsets.items():
        # KNN expected gain.
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
            "cols": ",".join(cols),
        })
        rows.append(row)

        # Rescue-harm local router.
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
            "cols": ",".join(cols),
        })
        rows.append(row)

    return rows


def to_md(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# SPSM v12 router feature-source ablation — leave-one-hard-out\n")
    lines.append("This experiment asks where the value-of-computation signal comes from.")
    lines.append("It compares uncertainty-only, margin, action, context, mined-geometry-only, and context+geometry feature sets.")
    lines.append("Each feature set is evaluated with both KNN expected-gain routing and local rescue-harm routing.\n")

    lines.append("## Best method per heldout and lambda\n")
    lines.append("| lambda | heldout | cheap | full | oracle | conf | context-cls | best learned | best util | best feature set | best router |")
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
    lines.append("- If `uncertainty` or `margin` is best, the value-of-computation signal is mostly cheap-model risk.")
    lines.append("- If `context` beats `margin_action`, horizon/velocity add useful shift information.")
    lines.append("- If `mined_geometry_only` or `context_geometry` wins, the current geometry features are useful.")
    lines.append("- If rescue-harm wins over KNN gain, decomposing rescue and harm is useful.")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = prepare()
    m = add_meta_features(m)
    m = add_optional_group_geometry(m)

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


if __name__ == "__main__":
    main()
