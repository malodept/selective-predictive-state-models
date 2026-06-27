from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from train_v8_geometry_aware_gain_router import (
    add_meta_features,
    context_cols,
    utility,
    best_conf_threshold,
    train_router,
    best_score_threshold,
    TEST_VARIANTS,
    ALL_VARIANTS,
    LAMBDAS,
    META,
)

from train_v10_local_expected_gain_router import fit_knn_gain_router
from train_v11_rescue_harm_router import fit_local_rescue_harm_router
from train_v13_latent_state_router import add_latent_state_features, feature_sets

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v19_multiseed_locked_voc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER_SEED = OUT_DIR / "locked_voc_per_cheap_seed.csv"
OUT_AGG = OUT_DIR / "locked_voc_multiseed_summary.csv"
OUT_MD = OUT_DIR / "locked_voc_multiseed_summary.md"

CSV_BY_SEED = {
    0: Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/small_full_voc_examples.csv"),
    1: Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood/small_full_voc_examples_seed1.csv"),
    2: Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v18_multiseed_small_full_ood/small_full_voc_examples_seed2.csv"),
}


def prepare_from_csv(path: Path, cheap_seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)

    cheap = df[df["mode"] == "small_full"].copy()
    full = df[df["mode"] == "full"].copy()

    keys = ["variant", "group_id", "action_id", "action_name"]
    m = cheap.merge(
        full,
        on=keys,
        suffixes=("_cheap", "_full"),
        validate="one_to_one",
    )

    out = pd.DataFrame()
    out["cheap_seed"] = cheap_seed
    out["variant"] = m["variant"]
    out["group_id"] = m["group_id"]
    out["action_id"] = m["action_id"]
    out["action_name"] = m["action_name"]

    out["cheap_correct"] = m["correct_cheap"].astype(float)
    out["full_correct"] = m["correct_full"].astype(float)
    out["gain"] = out["full_correct"] - out["cheap_correct"]

    out["cheap_confidence"] = m["confidence_cheap"].astype(float)
    out["cheap_uncertainty"] = 1.0 - out["cheap_confidence"]
    out["cheap_best_distance"] = m["best_distance_cheap"].astype(float)
    out["cheap_second_best_distance"] = m["second_best_distance_cheap"].astype(float)
    out["cheap_pred_margin"] = m["pred_margin_cheap"].astype(float)
    out["cheap_correct_distance"] = m["correct_distance_cheap"].astype(float)

    out["full_confidence"] = m["confidence_full"].astype(float)
    out["full_best_distance"] = m["best_distance_full"].astype(float)
    out["full_second_best_distance"] = m["second_best_distance_full"].astype(float)
    out["full_pred_margin"] = m["pred_margin_full"].astype(float)
    out["full_correct_distance"] = m["correct_distance_full"].astype(float)

    # Moving-only action ids are usually 1..4. Keep explicit columns stable.
    for a in [1, 2, 3, 4]:
        out[f"action_{a}"] = (out["action_id"].astype(int) == a).astype(float)

    out = add_meta_features(out)
    out = add_latent_state_features(out)

    return out


def eval_route(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return float(acc), float(util), float(rate)


def evaluate_seed_case(m: pd.DataFrame, cheap_seed: int, heldout: str, lam: float):
    train_variants = [v for v in ALL_VARIANTS if v != heldout]
    train = m[m["variant"].isin(train_variants)].copy()
    test = m[m["variant"] == heldout].copy()

    rows = []

    cheap_route = np.zeros(len(test), dtype=bool)
    full_route = np.ones(len(test), dtype=bool)
    oracle_route = (test["gain"].values - lam) > 0

    methods = {}

    _, cheap_util, cheap_rate = eval_route(test, cheap_route, lam)
    methods["cheap"] = (cheap_util, cheap_rate)

    _, full_util, full_rate = eval_route(test, full_route, lam)
    methods["full"] = (full_util, full_rate)

    _, oracle_util, oracle_rate = eval_route(test, oracle_route, lam)
    methods["oracle"] = (oracle_util, oracle_rate)

    _, conf_thr, _, _ = best_conf_threshold(train, lam)
    conf_route = test["cheap_confidence"].to_numpy() < conf_thr
    _, conf_util, conf_rate = eval_route(test, conf_route, lam)
    methods["confidence"] = (conf_util, conf_rate)

    ccols = context_cols(m)
    context_cls_fn = train_router(m, ccols, train_variants, lam, seed=0)
    train_scores = context_cls_fn(train)
    _, context_thr, _, _ = best_score_threshold(train_scores, train, lam)
    context_cls_route = context_cls_fn(test) >= context_thr
    _, context_cls_util, context_cls_rate = eval_route(test, context_cls_route, lam)
    methods["context_cls"] = (context_cls_util, context_cls_rate)

    fsets = feature_sets(m)
    context_features = fsets["context"]
    context_latent_features = fsets["context_latent"]

    knn_context_fn, knn_context_rule, _, knn_context_thr = fit_knn_gain_router(
        m, context_features, train_variants, lam
    )
    s = knn_context_fn(test)
    route = s > lam if knn_context_rule == "gain_threshold" else s >= knn_context_thr
    _, util, rate = eval_route(test, route, lam)
    methods["knn_context"] = (util, rate)

    rh_context_fn, _, _, _ = fit_local_rescue_harm_router(
        m, context_features, train_variants, lam
    )
    route = rh_context_fn(test)[0]
    _, util, rate = eval_route(test, route, lam)
    methods["rh_context"] = (util, rate)

    knn_latent_fn, knn_latent_rule, _, knn_latent_thr = fit_knn_gain_router(
        m, context_latent_features, train_variants, lam
    )
    s = knn_latent_fn(test)
    route = s > lam if knn_latent_rule == "gain_threshold" else s >= knn_latent_thr
    _, util, rate = eval_route(test, route, lam)
    methods["knn_context_latent"] = (util, rate)

    rh_latent_fn, _, _, _ = fit_local_rescue_harm_router(
        m, context_latent_features, train_variants, lam
    )
    route = rh_latent_fn(test)[0]
    _, util, rate = eval_route(test, route, lam)
    methods["rh_context_latent"] = (util, rate)

    for method, (util, rate) in methods.items():
        rows.append({
            "cheap_seed": cheap_seed,
            "lambda": lam,
            "heldout": heldout,
            "variant_label": META[heldout]["label"],
            "method": method,
            "utility": util,
            "route_rate": rate,
            "n": len(test),
        })

    return rows


def build_markdown(per_seed: pd.DataFrame, agg: pd.DataFrame) -> str:
    lines = []
    lines.append("# SPSM v19 multi-cheap-seed locked VoC\n")
    lines.append("This repeats the locked v15 routing comparison across cheap model seeds 0, 1, and 2.")
    lines.append("The expensive model remains the same full seed0 predictor; the question is whether the routing conclusion is stable under cheap-model retraining.\n")

    h72 = agg[
        (agg["heldout"] == "block3_h72_seed14")
        & (agg["lambda"].isin([0.02, 0.05, 0.10, 0.20, 0.30]))
    ].copy()

    order = [
        "cheap",
        "full",
        "confidence",
        "context_cls",
        "knn_context",
        "rh_context",
        "knn_context_latent",
        "rh_context_latent",
        "oracle",
    ]

    lines.append("## Mean utility on 3-block, H=72 across cheap seeds\n")
    lines.append("| lambda | method | mean utility | std | mean route |")
    lines.append("|---:|---|---:|---:|---:|")

    for lam in [0.02, 0.05, 0.10, 0.20, 0.30]:
        sub = h72[h72["lambda"] == lam].set_index("method")
        for method in order:
            r = sub.loc[method]
            lines.append(
                f"| {lam:.2f} | `{method}` | {r['utility_mean']:.6f} | {r['utility_std']:.6f} | {r['route_rate_mean']:.6f} |"
            )

    lines.append("\n## Best fixed non-oracle method by cheap seed on 3-block, H=72\n")
    lines.append("| lambda | cheap seed | best method | utility |")
    lines.append("|---:|---:|---|---:|")

    non_oracle = [m for m in order if m != "oracle"]
    h72_seed = per_seed[
        (per_seed["heldout"] == "block3_h72_seed14")
        & (per_seed["lambda"].isin([0.02, 0.05, 0.10, 0.20, 0.30]))
        & (per_seed["method"].isin(non_oracle))
    ]

    for (lam, seed), sub in h72_seed.groupby(["lambda", "cheap_seed"], sort=True):
        best = sub.sort_values("utility", ascending=False).iloc[0]
        lines.append(f"| {lam:.2f} | {int(seed)} | `{best['method']}` | {best['utility']:.6f} |")

    lines.append("\n## Latent advantage over context-only on 3-block, H=72\n")
    lines.append("| lambda | Δ KNN latent-context mean ± std | Δ RH latent-context mean ± std |")
    lines.append("|---:|---:|---:|")

    rows = []
    h72_pivot = per_seed[per_seed["heldout"] == "block3_h72_seed14"].pivot_table(
        index=["cheap_seed", "lambda"],
        columns="method",
        values="utility",
    ).reset_index()

    for lam in [0.02, 0.05, 0.10, 0.20, 0.30]:
        sub = h72_pivot[h72_pivot["lambda"] == lam]
        d_knn = sub["knn_context_latent"] - sub["knn_context"]
        d_rh = sub["rh_context_latent"] - sub["rh_context"]
        lines.append(
            f"| {lam:.2f} | {d_knn.mean():.6f} ± {d_knn.std(ddof=1):.6f} | "
            f"{d_rh.mean():.6f} ± {d_rh.std(ddof=1):.6f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- If the latent advantage remains positive on average across cheap seeds, v15 is directionally stable.")
    lines.append("- If the best method changes a lot across cheap seeds, the next scientific claim should emphasize cheap-model realization sensitivity.")
    lines.append("- This result should be interpreted together with the v16 paired bootstrap over test queries.")

    return "\n".join(lines) + "\n"


def main():
    all_rows = []

    for seed, path in CSV_BY_SEED.items():
        print(f"===== prepare cheap_seed={seed} =====")
        m = prepare_from_csv(path, seed)
        print("rows", len(m), "variants", m["variant"].value_counts().to_dict())

        for heldout in TEST_VARIANTS:
            for lam in LAMBDAS:
                print(f"cheap_seed={seed} heldout={heldout} lambda={lam}")
                all_rows.extend(evaluate_seed_case(m, seed, heldout, lam))

    per_seed = pd.DataFrame(all_rows)
    per_seed.to_csv(OUT_PER_SEED, index=False)

    agg = (
        per_seed
        .groupby(["lambda", "heldout", "variant_label", "method"], as_index=False)
        .agg(
            utility_mean=("utility", "mean"),
            utility_std=("utility", "std"),
            route_rate_mean=("route_rate", "mean"),
            route_rate_std=("route_rate", "std"),
            n=("n", "first"),
        )
    )
    agg.to_csv(OUT_AGG, index=False)

    OUT_MD.write_text(build_markdown(per_seed, agg), encoding="utf-8")

    print("\n===== WROTE =====")
    print(OUT_PER_SEED)
    print(OUT_AGG)
    print(OUT_MD)
    print("\n===== SUMMARY =====")
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
