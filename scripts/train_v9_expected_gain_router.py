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
    train_router,
    best_score_threshold,
    utility,
    best_conf_threshold,
    TRAIN_VARIANTS,
    TEST_VARIANTS,
    ALL_VARIANTS,
    LAMBDAS,
    META,
)

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v9_expected_gain_router")
OUT_MD = OUT_DIR / "expected_gain_router_loso_summary.md"
OUT_CSV = OUT_DIR / "expected_gain_router_loso_summary.csv"


def clean_x(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return (
        frame[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float64)
    )


def standardize(train_x: np.ndarray, x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd, mu, sd


def fit_ridge_gain(
    m: pd.DataFrame,
    cols: list[str],
    train_variants: list[str],
    alpha: float = 1e-2,
):
    train = m[m["variant"].isin(train_variants)].copy()
    x_raw = clean_x(train, cols)
    y = train["gain"].to_numpy(np.float64)

    x_std, mu, sd = standardize(x_raw, x_raw)

    # Add intercept. Do not regularize intercept.
    x_aug = np.concatenate([np.ones((x_std.shape[0], 1)), x_std], axis=1)
    reg = alpha * np.eye(x_aug.shape[1])
    reg[0, 0] = 0.0

    w = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)

    def score(frame: pd.DataFrame) -> np.ndarray:
        xx = clean_x(frame, cols)
        xx = (xx - mu) / sd
        xx = np.concatenate([np.ones((xx.shape[0], 1)), xx], axis=1)
        return xx @ w

    return score


def fit_tuned_ridge_gain(
    m: pd.DataFrame,
    cols: list[str],
    train_variants: list[str],
    lam: float,
    alpha: float = 1e-2,
):
    score_fn = fit_ridge_gain(m, cols, train_variants, alpha=alpha)
    train = m[m["variant"].isin(train_variants)].copy()
    train_scores = score_fn(train)
    _, threshold, _, _ = best_score_threshold(train_scores, train, lam)

    return score_fn, threshold


def eval_route(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return float(acc), float(util), float(rate)


def evaluate_one_protocol(
    m: pd.DataFrame,
    train_variants: list[str],
    eval_variant: str,
    lam: float,
    seed: int = 0,
):
    ccols = context_cols(m)
    gcols = geometry_cols(m)

    train = m[m["variant"].isin(train_variants)].copy()
    v = m[m["variant"] == eval_variant].copy()

    cheap_acc = float(v["cheap_correct"].mean())
    full_acc = float(v["full_correct"].mean())
    cheap_util = cheap_acc
    full_util = full_acc - lam

    oracle_route = (v["gain"].values - lam) > 0
    _, oracle_util, oracle_rate = eval_route(v, oracle_route, lam)

    # Confidence baseline.
    _, conf_threshold, _, _ = best_conf_threshold(train, lam)
    conf_route = v["cheap_confidence"].values < conf_threshold
    _, conf_util, conf_rate = eval_route(v, conf_route, lam)

    # v5.8-style classifier, lambda-specific.
    context_cls_fn = train_router(m, ccols, train_variants, lam, seed=seed)
    train_context_scores = context_cls_fn(train)
    _, context_threshold, _, _ = best_score_threshold(train_context_scores, train, lam)
    context_route = context_cls_fn(v) >= context_threshold
    _, context_util, context_rate = eval_route(v, context_route, lam)

    # v8-style geometry classifier, lambda-specific.
    geometry_cls_fn = train_router(m, gcols, train_variants, lam, seed=seed)
    train_geometry_scores = geometry_cls_fn(train)
    _, geometry_threshold, _, _ = best_score_threshold(train_geometry_scores, train, lam)
    geometry_route = geometry_cls_fn(v) >= geometry_threshold
    _, geometry_cls_util, geometry_cls_rate = eval_route(v, geometry_route, lam)

    # v9 expected-gain ridge: same model for all lambdas, route if predicted gain > lambda.
    ridge_context_fn = fit_ridge_gain(m, ccols, train_variants, alpha=1e-2)
    ridge_context_score = ridge_context_fn(v)
    ridge_context_route = ridge_context_score > lam
    _, ridge_context_util, ridge_context_rate = eval_route(v, ridge_context_route, lam)

    ridge_geometry_fn = fit_ridge_gain(m, gcols, train_variants, alpha=1e-2)
    ridge_geometry_score = ridge_geometry_fn(v)
    ridge_geometry_route = ridge_geometry_score > lam
    _, ridge_geometry_util, ridge_geometry_rate = eval_route(v, ridge_geometry_route, lam)

    # Tuned ridge: still a gain score, but threshold selected on train utility.
    tuned_context_fn, tuned_context_threshold = fit_tuned_ridge_gain(m, ccols, train_variants, lam, alpha=1e-2)
    tuned_context_route = tuned_context_fn(v) >= tuned_context_threshold
    _, tuned_context_util, tuned_context_rate = eval_route(v, tuned_context_route, lam)

    tuned_geometry_fn, tuned_geometry_threshold = fit_tuned_ridge_gain(m, gcols, train_variants, lam, alpha=1e-2)
    tuned_geometry_route = tuned_geometry_fn(v) >= tuned_geometry_threshold
    _, tuned_geometry_util, tuned_geometry_rate = eval_route(v, tuned_geometry_route, lam)

    methods = {
        "cheap": cheap_util,
        "full": full_util,
        "conf": conf_util,
        "context_cls": context_util,
        "geometry_cls": geometry_cls_util,
        "ridge_context": ridge_context_util,
        "ridge_geometry": ridge_geometry_util,
        "tuned_context": tuned_context_util,
        "tuned_geometry": tuned_geometry_util,
    }
    best_method = max(methods, key=methods.get)

    return {
        "lambda": lam,
        "heldout": eval_variant,
        "variant_label": META[eval_variant]["label"],
        "cheap_util": cheap_util,
        "full_util": full_util,
        "oracle_util": oracle_util,
        "oracle_route": oracle_rate,
        "conf_util": conf_util,
        "conf_route": conf_rate,
        "context_cls_util": context_util,
        "context_cls_route": context_rate,
        "geometry_cls_util": geometry_cls_util,
        "geometry_cls_route": geometry_cls_rate,
        "ridge_context_util": ridge_context_util,
        "ridge_context_route": ridge_context_rate,
        "ridge_geometry_util": ridge_geometry_util,
        "ridge_geometry_route": ridge_geometry_rate,
        "tuned_context_util": tuned_context_util,
        "tuned_context_route": tuned_context_rate,
        "tuned_geometry_util": tuned_geometry_util,
        "tuned_geometry_route": tuned_geometry_rate,
        "best_non_oracle": best_method,
        "best_non_oracle_util": methods[best_method],
        "ridge_geom_minus_context_cls": ridge_geometry_util - context_util,
        "tuned_geom_minus_context_cls": tuned_geometry_util - context_util,
    }


def markdown(rows: list[dict]) -> str:
    lines = []
    lines.append("# SPSM v9 expected-gain router — leave-one-hard-out\n")
    lines.append("This evaluates routers that estimate expected value-of-computation gain directly.")
    lines.append("Instead of training a separate binary classifier for every lambda, ridge gain models predict `E[full_correct - small_correct | x]` and route when predicted gain exceeds the compute cost.\n")

    lines.append("## Main comparison\n")
    lines.append("| lambda | heldout | cheap | full | oracle | conf | context-cls | geometry-cls | ridge-context | ridge-geometry | tuned-context | tuned-geometry | best |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['cheap_util']:.6f} | {r['full_util']:.6f} | {r['oracle_util']:.6f} | "
            f"{r['conf_util']:.6f} | {r['context_cls_util']:.6f} | {r['geometry_cls_util']:.6f} | "
            f"{r['ridge_context_util']:.6f} | {r['ridge_geometry_util']:.6f} | "
            f"{r['tuned_context_util']:.6f} | {r['tuned_geometry_util']:.6f} | "
            f"`{r['best_non_oracle']}` |"
        )

    lines.append("\n## Route rates\n")
    lines.append("| lambda | heldout | oracle | conf | context-cls | geometry-cls | ridge-context | ridge-geometry | tuned-context | tuned-geometry |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['oracle_route']:.6f} | {r['conf_route']:.6f} | "
            f"{r['context_cls_route']:.6f} | {r['geometry_cls_route']:.6f} | "
            f"{r['ridge_context_route']:.6f} | {r['ridge_geometry_route']:.6f} | "
            f"{r['tuned_context_route']:.6f} | {r['tuned_geometry_route']:.6f} |"
        )

    lines.append("\n## Interpretation guide\n")
    lines.append("- If ridge-context beats context-cls, the expected-gain formulation is better than lambda-specific classification.")
    lines.append("- If ridge-geometry or tuned-geometry beats ridge-context, local geometry contains useful value-of-computation signal.")
    lines.append("- If all learned routers lose to confidence or cheap-only, the current features are not sufficient and we should design physical geometry features directly from the environment.")
    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = prepare()
    m = add_meta_features(m)
    m = add_optional_group_geometry(m)

    rows = []
    for heldout in TEST_VARIANTS:
        train_variants = [v for v in ALL_VARIANTS if v != heldout]
        for lam in LAMBDAS:
            print(f"heldout={heldout} lambda={lam}")
            rows.append(evaluate_one_protocol(m, train_variants, heldout, lam, seed=0))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(markdown(rows), encoding="utf-8")

    print("\n===== WROTE =====")
    print(OUT_MD)
    print(OUT_CSV)
    print("\n===== SUMMARY =====")
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
