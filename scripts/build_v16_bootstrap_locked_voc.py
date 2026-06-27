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

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v16_bootstrap_locked_voc")
OUT_CSV = OUT_DIR / "bootstrap_locked_voc_summary.csv"
OUT_MD = OUT_DIR / "bootstrap_locked_voc_summary.md"

BOOT = 3000
SEED = 123


def route_utility_vector(frame: pd.DataFrame, route: np.ndarray, lam: float) -> np.ndarray:
    cheap = frame["cheap_correct"].to_numpy(np.float64)
    full = frame["full_correct"].to_numpy(np.float64)
    route = route.astype(bool)
    correct = np.where(route, full, cheap)
    return correct - lam * route.astype(np.float64)


def ci_mean(x: np.ndarray, rng: np.random.Generator, boot: int = BOOT):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    means = np.empty(boot, dtype=np.float64)
    for b in range(boot):
        idx = rng.integers(0, n, size=n)
        means[b] = x[idx].mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_delta_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, boot: int = BOOT):
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return ci_mean(d, rng, boot)


def evaluate_case(m: pd.DataFrame, heldout: str, lam: float, rng: np.random.Generator):
    train_variants = [v for v in ALL_VARIANTS if v != heldout]
    train = m[m["variant"].isin(train_variants)].copy()
    test = m[m["variant"] == heldout].copy()

    fsets = feature_sets(m)
    context_features = fsets["context"]
    context_latent_features = fsets["context_latent"]

    routes = {}

    routes["cheap"] = np.zeros(len(test), dtype=bool)
    routes["full"] = np.ones(len(test), dtype=bool)

    _, conf_thr, _, _ = best_conf_threshold(train, lam)
    routes["confidence"] = test["cheap_confidence"].to_numpy() < conf_thr

    ccols = context_cols(m)
    context_cls_fn = train_router(m, ccols, train_variants, lam, seed=0)
    train_scores = context_cls_fn(train)
    _, context_thr, _, _ = best_score_threshold(train_scores, train, lam)
    routes["context_cls"] = context_cls_fn(test) >= context_thr

    knn_context_fn, knn_context_rule, _, knn_context_thr = fit_knn_gain_router(
        m, context_features, train_variants, lam
    )
    s = knn_context_fn(test)
    routes["knn_context"] = s > lam if knn_context_rule == "gain_threshold" else s >= knn_context_thr

    rh_context_fn, _, _, _ = fit_local_rescue_harm_router(
        m, context_features, train_variants, lam
    )
    routes["rh_context"] = rh_context_fn(test)[0]

    knn_latent_fn, knn_latent_rule, _, knn_latent_thr = fit_knn_gain_router(
        m, context_latent_features, train_variants, lam
    )
    s = knn_latent_fn(test)
    routes["knn_context_latent"] = s > lam if knn_latent_rule == "gain_threshold" else s >= knn_latent_thr

    rh_latent_fn, _, _, _ = fit_local_rescue_harm_router(
        m, context_latent_features, train_variants, lam
    )
    routes["rh_context_latent"] = rh_latent_fn(test)[0]

    oracle_route = (
        test["full_correct"].to_numpy(np.float64)
        - test["cheap_correct"].to_numpy(np.float64)
    ) > lam
    routes["oracle"] = oracle_route

    util_vec = {
        name: route_utility_vector(test, route, lam)
        for name, route in routes.items()
    }

    rows = []

    for name, u in util_vec.items():
        mean, lo, hi = ci_mean(u, rng)
        rows.append({
            "lambda": lam,
            "heldout": heldout,
            "variant_label": META[heldout]["label"],
            "method": name,
            "utility": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "route_rate": float(routes[name].mean()),
            "n": len(test),
        })

    comparisons = [
        ("knn_context_latent", "knn_context"),
        ("rh_context_latent", "rh_context"),
        ("knn_context_latent", "confidence"),
        ("rh_context_latent", "confidence"),
        ("knn_context_latent", "context_cls"),
        ("rh_context_latent", "context_cls"),
        ("knn_context_latent", "cheap"),
        ("rh_context_latent", "cheap"),
    ]

    for a, b in comparisons:
        mean, lo, hi = paired_delta_ci(util_vec[a], util_vec[b], rng)
        rows.append({
            "lambda": lam,
            "heldout": heldout,
            "variant_label": META[heldout]["label"],
            "method": f"DELTA:{a}-{b}",
            "utility": mean,
            "ci95_low": lo,
            "ci95_high": hi,
            "route_rate": np.nan,
            "n": len(test),
        })

    return rows


def to_md(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# SPSM v16 bootstrap validation of locked VoC routing\n")
    lines.append("This validates v15 with paired bootstrap confidence intervals over test queries.")
    lines.append("Positive DELTA intervals whose lower bound is above zero indicate a stable improvement.\n")

    lines.append("## Main methods on 3-block, H=72\n")
    h72 = df[
        (df["heldout"] == "block3_h72_seed14")
        & (~df["method"].str.startswith("DELTA:"))
        & (df["lambda"].isin([0.02, 0.05, 0.10, 0.20, 0.30]))
    ].copy()

    lines.append("| lambda | method | utility | 95% CI | route rate |")
    lines.append("|---:|---|---:|---:|---:|")
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
    for lam in [0.02, 0.05, 0.10, 0.20, 0.30]:
        sub = h72[h72["lambda"] == lam].set_index("method")
        for method in order:
            r = sub.loc[method]
            lines.append(
                f"| {lam:.2f} | `{method}` | {r['utility']:.6f} | "
                f"[{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {r['route_rate']:.6f} |"
            )

    lines.append("\n## Paired latent improvement deltas on 3-block, H=72\n")
    deltas = df[
        (df["heldout"] == "block3_h72_seed14")
        & (df["method"].str.startswith("DELTA:"))
        & (df["lambda"].isin([0.05, 0.10, 0.20, 0.30]))
    ].copy()

    keep = [
        "DELTA:knn_context_latent-knn_context",
        "DELTA:rh_context_latent-rh_context",
        "DELTA:knn_context_latent-confidence",
        "DELTA:rh_context_latent-confidence",
        "DELTA:knn_context_latent-context_cls",
        "DELTA:rh_context_latent-context_cls",
    ]

    lines.append("| lambda | comparison | delta | 95% CI | stable positive? |")
    lines.append("|---:|---|---:|---:|---|")
    for lam in [0.05, 0.10, 0.20, 0.30]:
        sub = deltas[deltas["lambda"] == lam].set_index("method")
        for method in keep:
            r = sub.loc[method]
            stable = "yes" if r["ci95_low"] > 0 else "no"
            lines.append(
                f"| {lam:.2f} | `{method.replace('DELTA:', '')}` | "
                f"{r['utility']:.6f} | [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | {stable} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append("- If v15 survives this bootstrap, it becomes a defensible central result.")
    lines.append("- If intervals cross zero, we keep the result but describe it as promising rather than conclusive.")
    lines.append("- The next step after v16 is multi-seed small-full training, not more router variants.")

    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    m = prepare()
    m = add_meta_features(m)
    m = add_latent_state_features(m)

    rows = []
    for heldout in TEST_VARIANTS:
        for lam in LAMBDAS:
            print(f"heldout={heldout} lambda={lam}")
            rows.extend(evaluate_case(m, heldout, lam, rng))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    OUT_MD.write_text(to_md(df), encoding="utf-8")

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
