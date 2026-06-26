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

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v10_local_expected_gain_router")
OUT_MD = OUT_DIR / "local_expected_gain_loso_summary.md"
OUT_CSV = OUT_DIR / "local_expected_gain_loso_summary.csv"

K_GRID = [15, 25, 50, 100, 200]


def clean_x(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return (
        frame[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float64)
    )


def standardize_fit(train_x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd


def knn_gain_score(
    train_x: np.ndarray,
    train_gain: np.ndarray,
    test_x: np.ndarray,
    k: int,
    chunk: int = 2048,
) -> np.ndarray:
    k = min(k, len(train_gain))
    out = np.empty((len(test_x),), dtype=np.float64)

    train_norm2 = np.sum(train_x * train_x, axis=1, keepdims=True).T

    for start in range(0, len(test_x), chunk):
        stop = min(start + chunk, len(test_x))
        x = test_x[start:stop]
        d2 = np.sum(x * x, axis=1, keepdims=True) + train_norm2 - 2.0 * (x @ train_x.T)

        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        out[start:stop] = train_gain[idx].mean(axis=1)

    return out


def best_threshold_from_scores(scores: np.ndarray, frame: pd.DataFrame, lam: float):
    thresholds = np.unique(np.concatenate([
        np.array([-1e9, 1e9]),
        np.quantile(scores, np.linspace(0, 1, 201)),
    ]))

    best = None
    for t in thresholds:
        route = scores >= t
        acc, util, rate = utility(frame, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def fit_knn_gain_router(
    m: pd.DataFrame,
    cols: list[str],
    train_variants: list[str],
    lam: float,
):
    train = m[m["variant"].isin(train_variants)].copy()

    x_train_raw = clean_x(train, cols)
    gain_train = train["gain"].to_numpy(np.float64)

    mu, sd = standardize_fit(x_train_raw)
    x_train = standardize_apply(x_train_raw, mu, sd)

    best = None
    for k in K_GRID:
        train_scores = knn_gain_score(x_train, gain_train, x_train, k=k)

        # Two decision rules:
        # 1. principled gain rule: score > lambda
        route_gain = train_scores > lam
        acc_g, util_g, rate_g = utility(train, route_gain, lam)
        cand_gain = (util_g, "gain_threshold", k, lam, acc_g, rate_g)

        # 2. tuned threshold on train utility, useful because KNN gain may be miscalibrated.
        util_t, thr_t, acc_t, rate_t = best_threshold_from_scores(train_scores, train, lam)
        cand_tuned = (util_t, "tuned_threshold", k, thr_t, acc_t, rate_t)

        for cand in [cand_gain, cand_tuned]:
            if best is None or cand[0] > best[0]:
                best = cand

    assert best is not None
    _, rule, best_k, threshold, _, _ = best

    def score(frame: pd.DataFrame):
        x = standardize_apply(clean_x(frame, cols), mu, sd)
        return knn_gain_score(x_train, gain_train, x, k=best_k)

    return score, rule, best_k, threshold


def eval_route(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return float(acc), float(util), float(rate)


def evaluate_protocol(
    m: pd.DataFrame,
    train_variants: list[str],
    heldout: str,
    lam: float,
):
    train = m[m["variant"].isin(train_variants)].copy()
    v = m[m["variant"] == heldout].copy()

    ccols = context_cols(m)

    # Keep geometry feature set, but also define a compact signal set from the audit.
    gcols = geometry_cols(m)
    signal_cols = [
        "cheap_confidence",
        "cheap_uncertainty",
        "cheap_pred_margin",
        "cheap_best_distance",
        "cheap_second_best_distance",
        "horizon_norm",
        "velocity_norm",
    ]
    signal_cols += sorted([c for c in m.columns if c.startswith("action_") and c.removeprefix("action_").isdigit()])

    # Baselines.
    cheap_acc = float(v["cheap_correct"].mean())
    full_acc = float(v["full_correct"].mean())
    cheap_util = cheap_acc
    full_util = full_acc - lam

    oracle_route = (v["gain"].values - lam) > 0
    _, oracle_util, oracle_rate = eval_route(v, oracle_route, lam)

    _, conf_threshold, _, _ = best_conf_threshold(train, lam)
    conf_route = v["cheap_confidence"].values < conf_threshold
    _, conf_util, conf_rate = eval_route(v, conf_route, lam)

    # Context classifier baseline from v5.8/v8.
    context_cls_fn = train_router(m, ccols, train_variants, lam, seed=0)
    train_context_scores = context_cls_fn(train)
    _, context_thr, _, _ = best_score_threshold(train_context_scores, train, lam)
    context_cls_route = context_cls_fn(v) >= context_thr
    _, context_cls_util, context_cls_rate = eval_route(v, context_cls_route, lam)

    # Local expected-gain routers.
    knn_signal_fn, signal_rule, signal_k, signal_thr = fit_knn_gain_router(m, signal_cols, train_variants, lam)
    signal_scores = knn_signal_fn(v)
    signal_route = signal_scores > lam if signal_rule == "gain_threshold" else signal_scores >= signal_thr
    _, knn_signal_util, knn_signal_rate = eval_route(v, signal_route, lam)

    knn_context_fn, context_rule, context_k, context_thr2 = fit_knn_gain_router(m, ccols, train_variants, lam)
    context_scores = knn_context_fn(v)
    context_route = context_scores > lam if context_rule == "gain_threshold" else context_scores >= context_thr2
    _, knn_context_util, knn_context_rate = eval_route(v, context_route, lam)

    knn_geometry_fn, geometry_rule, geometry_k, geometry_thr = fit_knn_gain_router(m, gcols, train_variants, lam)
    geometry_scores = knn_geometry_fn(v)
    geometry_route = geometry_scores > lam if geometry_rule == "gain_threshold" else geometry_scores >= geometry_thr
    _, knn_geometry_util, knn_geometry_rate = eval_route(v, geometry_route, lam)

    methods = {
        "cheap": cheap_util,
        "full": full_util,
        "conf": conf_util,
        "context_cls": context_cls_util,
        "knn_signal": knn_signal_util,
        "knn_context": knn_context_util,
        "knn_geometry": knn_geometry_util,
    }
    best_method = max(methods, key=methods.get)

    return {
        "lambda": lam,
        "heldout": heldout,
        "variant_label": META[heldout]["label"],
        "cheap_util": cheap_util,
        "full_util": full_util,
        "oracle_util": oracle_util,
        "oracle_route": oracle_rate,
        "conf_util": conf_util,
        "conf_route": conf_rate,
        "context_cls_util": context_cls_util,
        "context_cls_route": context_cls_rate,
        "knn_signal_util": knn_signal_util,
        "knn_signal_route": knn_signal_rate,
        "knn_signal_k": signal_k,
        "knn_signal_rule": signal_rule,
        "knn_context_util": knn_context_util,
        "knn_context_route": knn_context_rate,
        "knn_context_k": context_k,
        "knn_context_rule": context_rule,
        "knn_geometry_util": knn_geometry_util,
        "knn_geometry_route": knn_geometry_rate,
        "knn_geometry_k": geometry_k,
        "knn_geometry_rule": geometry_rule,
        "best_non_oracle": best_method,
        "best_non_oracle_util": methods[best_method],
        "knn_context_minus_context_cls": knn_context_util - context_cls_util,
        "knn_geometry_minus_context_cls": knn_geometry_util - context_cls_util,
    }


def to_md(rows: list[dict]) -> str:
    lines = []
    lines.append("# SPSM v10 local expected-gain router — leave-one-hard-out\n")
    lines.append("This evaluates non-parametric KNN expected-gain routers.")
    lines.append("The router estimates local expected gain from neighboring training examples, then routes when estimated gain exceeds cost or a tuned utility threshold.\n")

    lines.append("## Utility comparison\n")
    lines.append("| lambda | heldout | cheap | full | oracle | conf | context-cls | knn-signal | knn-context | knn-geometry | best |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['cheap_util']:.6f} | {r['full_util']:.6f} | {r['oracle_util']:.6f} | "
            f"{r['conf_util']:.6f} | {r['context_cls_util']:.6f} | "
            f"{r['knn_signal_util']:.6f} | {r['knn_context_util']:.6f} | {r['knn_geometry_util']:.6f} | "
            f"`{r['best_non_oracle']}` |"
        )

    lines.append("\n## Route rates and selected K\n")
    lines.append("| lambda | heldout | oracle route | conf route | context-cls route | knn-signal route / k | knn-context route / k | knn-geometry route / k |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['oracle_route']:.6f} | {r['conf_route']:.6f} | {r['context_cls_route']:.6f} | "
            f"{r['knn_signal_route']:.6f} / {r['knn_signal_k']} | "
            f"{r['knn_context_route']:.6f} / {r['knn_context_k']} | "
            f"{r['knn_geometry_route']:.6f} / {r['knn_geometry_k']} |"
        )

    lines.append("\n## Interpretation guide\n")
    lines.append("- If KNN beats ridge/classifier routers on 3-block H=72, local gain structure exists.")
    lines.append("- If KNN-geometry beats KNN-context, mined local geometry is useful.")
    lines.append("- If KNN-signal is as good as KNN-geometry, most useful information is already in cheap uncertainty/margin.")
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
            rows.append(evaluate_protocol(m, train_variants, heldout, lam))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(to_md(rows), encoding="utf-8")

    print("\n===== WROTE =====")
    print(OUT_MD)
    print(OUT_CSV)
    print("\n===== SUMMARY =====")
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
