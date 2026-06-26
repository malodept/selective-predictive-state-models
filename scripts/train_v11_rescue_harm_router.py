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

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v11_rescue_harm_router")
OUT_MD = OUT_DIR / "rescue_harm_loso_summary.md"
OUT_CSV = OUT_DIR / "rescue_harm_loso_summary.csv"

K_GRID = [15, 25, 50, 100, 200]
CAP_GRID = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.35, 1.00]


def clean_x(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return (
        frame[cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(np.float64)
    )


def standardize_fit(x: np.ndarray):
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return mu, sd


def standardize_apply(x: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (x - mu) / sd


def knn_probs(
    train_x: np.ndarray,
    train_rescue: np.ndarray,
    train_harm: np.ndarray,
    test_x: np.ndarray,
    k: int,
    leave_one_train: bool = False,
    chunk: int = 2048,
):
    k = min(k, len(train_rescue) - 1 if leave_one_train else len(train_rescue))
    k = max(k, 1)

    rescue_out = np.empty((len(test_x),), dtype=np.float64)
    harm_out = np.empty((len(test_x),), dtype=np.float64)

    train_norm2 = np.sum(train_x * train_x, axis=1, keepdims=True).T

    for start in range(0, len(test_x), chunk):
        stop = min(start + chunk, len(test_x))
        x = test_x[start:stop]
        d2 = np.sum(x * x, axis=1, keepdims=True) + train_norm2 - 2.0 * (x @ train_x.T)

        if leave_one_train:
            # Used only when test_x is train_x in the same order.
            for local_i, global_i in enumerate(range(start, stop)):
                if global_i < d2.shape[1]:
                    d2[local_i, global_i] = np.inf

        idx = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        rescue_out[start:stop] = train_rescue[idx].mean(axis=1)
        harm_out[start:stop] = train_harm[idx].mean(axis=1)

    return rescue_out, harm_out


def best_rescue_harm_policy(
    frame: pd.DataFrame,
    rescue: np.ndarray,
    harm: np.ndarray,
    lam: float,
):
    gain_score = rescue - harm

    thresholds = np.unique(np.concatenate([
        np.array([-1e9, lam, 1e9]),
        np.quantile(gain_score, np.linspace(0, 1, 151)),
    ]))

    caps = np.unique(np.concatenate([
        np.array(CAP_GRID),
        np.quantile(harm, np.linspace(0, 1, 21)),
    ]))

    best = None
    for t in thresholds:
        for cap in caps:
            route = (gain_score >= t) & (harm <= cap)
            acc, util, rate = utility(frame, route, lam)
            cand = (util, t, cap, acc, rate)
            if best is None or cand[0] > best[0]:
                best = cand

    return best


def fit_local_rescue_harm_router(
    m: pd.DataFrame,
    cols: list[str],
    train_variants: list[str],
    lam: float,
):
    train = m[m["variant"].isin(train_variants)].copy()

    rescue = ((train["cheap_correct"].values == 0) & (train["full_correct"].values == 1)).astype(np.float64)
    harm = ((train["cheap_correct"].values == 1) & (train["full_correct"].values == 0)).astype(np.float64)

    x_raw = clean_x(train, cols)
    mu, sd = standardize_fit(x_raw)
    x_train = standardize_apply(x_raw, mu, sd)

    best = None
    for k in K_GRID:
        r_train, h_train = knn_probs(x_train, rescue, harm, x_train, k=k, leave_one_train=True)
        util, thr, cap, acc, rate = best_rescue_harm_policy(train, r_train, h_train, lam)
        cand = (util, k, thr, cap, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand

    assert best is not None
    _, best_k, best_thr, best_cap, _, _ = best

    def predict(frame: pd.DataFrame):
        xx = standardize_apply(clean_x(frame, cols), mu, sd)
        r, h = knn_probs(x_train, rescue, harm, xx, k=best_k, leave_one_train=False)
        gain = r - h
        route = (gain >= best_thr) & (h <= best_cap)
        return route, r, h, gain

    return predict, best_k, best_thr, best_cap


def eval_route(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return float(acc), float(util), float(rate)


def evaluate_protocol(m: pd.DataFrame, train_variants: list[str], heldout: str, lam: float):
    train = m[m["variant"].isin(train_variants)].copy()
    v = m[m["variant"] == heldout].copy()

    ccols = context_cols(m)
    gcols = geometry_cols(m)

    cheap_acc = float(v["cheap_correct"].mean())
    full_acc = float(v["full_correct"].mean())
    cheap_util = cheap_acc
    full_util = full_acc - lam

    oracle_route = (v["gain"].values - lam) > 0
    _, oracle_util, oracle_rate = eval_route(v, oracle_route, lam)

    _, conf_threshold, _, _ = best_conf_threshold(train, lam)
    conf_route = v["cheap_confidence"].values < conf_threshold
    _, conf_util, conf_rate = eval_route(v, conf_route, lam)

    context_cls_fn = train_router(m, ccols, train_variants, lam, seed=0)
    train_context_scores = context_cls_fn(train)
    _, context_thr, _, _ = best_score_threshold(train_context_scores, train, lam)
    context_cls_route = context_cls_fn(v) >= context_thr
    _, context_cls_util, context_cls_rate = eval_route(v, context_cls_route, lam)

    rh_context_fn, k_c, thr_c, cap_c = fit_local_rescue_harm_router(m, ccols, train_variants, lam)
    rh_context_route, rh_context_rescue, rh_context_harm, rh_context_gain = rh_context_fn(v)
    _, rh_context_util, rh_context_rate = eval_route(v, rh_context_route, lam)

    rh_geometry_fn, k_g, thr_g, cap_g = fit_local_rescue_harm_router(m, gcols, train_variants, lam)
    rh_geometry_route, rh_geometry_rescue, rh_geometry_harm, rh_geometry_gain = rh_geometry_fn(v)
    _, rh_geometry_util, rh_geometry_rate = eval_route(v, rh_geometry_route, lam)

    methods = {
        "cheap": cheap_util,
        "full": full_util,
        "conf": conf_util,
        "context_cls": context_cls_util,
        "rh_context": rh_context_util,
        "rh_geometry": rh_geometry_util,
    }
    best_method = max(methods, key=methods.get)

    # Actual rescue/harm among routed examples, for interpretability.
    def actual_decomp(route):
        if route.sum() == 0:
            return 0.0, 0.0
        rescue_actual = ((v["cheap_correct"].values == 0) & (v["full_correct"].values == 1) & route).sum() / route.sum()
        harm_actual = ((v["cheap_correct"].values == 1) & (v["full_correct"].values == 0) & route).sum() / route.sum()
        return float(rescue_actual), float(harm_actual)

    rhc_rescue_actual, rhc_harm_actual = actual_decomp(rh_context_route)
    rhg_rescue_actual, rhg_harm_actual = actual_decomp(rh_geometry_route)

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
        "rh_context_util": rh_context_util,
        "rh_context_route": rh_context_rate,
        "rh_context_k": k_c,
        "rh_context_thr": thr_c,
        "rh_context_cap": cap_c,
        "rh_context_routed_rescue": rhc_rescue_actual,
        "rh_context_routed_harm": rhc_harm_actual,
        "rh_geometry_util": rh_geometry_util,
        "rh_geometry_route": rh_geometry_rate,
        "rh_geometry_k": k_g,
        "rh_geometry_thr": thr_g,
        "rh_geometry_cap": cap_g,
        "rh_geometry_routed_rescue": rhg_rescue_actual,
        "rh_geometry_routed_harm": rhg_harm_actual,
        "best_non_oracle": best_method,
        "best_non_oracle_util": methods[best_method],
    }


def to_md(rows: list[dict]) -> str:
    lines = []
    lines.append("# SPSM v11 rescue-harm local router — leave-one-hard-out\n")
    lines.append("This decomposes value-of-computation into local rescue and harm probabilities.")
    lines.append("The router estimates `P(full correct, small wrong | x)` and `P(full wrong, small correct | x)`, then routes when predicted rescue outweighs harm and compute cost.\n")

    lines.append("## Utility comparison\n")
    lines.append("| lambda | heldout | cheap | full | oracle | conf | context-cls | rescue-harm context | rescue-harm geometry | best |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['cheap_util']:.6f} | {r['full_util']:.6f} | {r['oracle_util']:.6f} | "
            f"{r['conf_util']:.6f} | {r['context_cls_util']:.6f} | "
            f"{r['rh_context_util']:.6f} | {r['rh_geometry_util']:.6f} | "
            f"`{r['best_non_oracle']}` |"
        )

    lines.append("\n## Route rates and routed decomposition\n")
    lines.append("| lambda | heldout | oracle route | context route | rh-context route/k | rh-context rescue/harm | rh-geometry route/k | rh-geometry rescue/harm |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['oracle_route']:.6f} | {r['context_cls_route']:.6f} | "
            f"{r['rh_context_route']:.6f}/{r['rh_context_k']} | "
            f"{r['rh_context_routed_rescue']:.3f}/{r['rh_context_routed_harm']:.3f} | "
            f"{r['rh_geometry_route']:.6f}/{r['rh_geometry_k']} | "
            f"{r['rh_geometry_routed_rescue']:.3f}/{r['rh_geometry_routed_harm']:.3f} |"
        )

    lines.append("\n## Interpretation guide\n")
    lines.append("- If rescue-harm beats local expected-gain, decomposing positive and negative transfer matters.")
    lines.append("- If routed harm is high, the large model is unsafe in that regime.")
    lines.append("- If geometry still loses to context, the current mined geometry features are not the right physical signal.")
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


def main_wrapper():
    main()


if __name__ == "__main__":
    main_wrapper()
