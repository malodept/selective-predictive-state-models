from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, "scripts")

from train_v5_7_small_full_gain_router import (
    prepare,
    feature_cols,
    train_model,
    best_score_threshold,
    best_conf_threshold,
    utility,
    TRAIN_VARIANTS,
    TEST_VARIANTS,
    LAMBDAS,
)

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_7_small_full_gain_router/gain_router_error_audit.md")
ALL_VARIANTS = TRAIN_VARIANTS + TEST_VARIANTS

def safe_div(a, b):
    return float(a / b) if b else 0.0

def audit_route(frame, route, lam):
    gain = frame["gain"].values
    route = route.astype(bool)

    good = gain > lam
    harmful = gain < 0
    wasteful = (gain >= 0) & (gain <= lam)

    good_routes = route & good
    harmful_routes = route & harmful
    wasteful_routes = route & wasteful
    missed_good = (~route) & good

    return {
        "route_rate": route.mean(),
        "good_available": good.mean(),
        "good_route_rate": good_routes.mean(),
        "harmful_route_rate": harmful_routes.mean(),
        "wasteful_route_rate": wasteful_routes.mean(),
        "missed_good_rate": missed_good.mean(),
        "route_precision": safe_div(good_routes.sum(), route.sum()),
        "route_recall": safe_div(good_routes.sum(), good.sum()),
    }

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    m = prepare()
    cols = feature_cols(m)
    train = m[m["variant"].isin(TRAIN_VARIANTS)].copy()

    lines = []
    lines.append("# SPSM v5.7 gain-router error audit\n")
    lines.append("This audit decomposes routing decisions for `small_full -> full` into useful, harmful, wasteful, and missed routes.\n")
    lines.append("- `good_available`: fraction of examples where routing to full gives net positive gain.")
    lines.append("- `harmful_route_rate`: routed examples where full is worse than small_full.")
    lines.append("- `wasteful_route_rate`: routed examples where full is not worse, but the gain does not pay for lambda.")
    lines.append("- `missed_good_rate`: examples where full would have positive net gain but the router did not route.\n")

    lines.append("| lambda | method | variant | split | utility | route rate | good available | route precision | route recall | harmful route | wasteful route | missed good |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for lam in LAMBDAS:
        score_fn = train_model(m, cols, lam, seed=0)
        m_lam = m.copy()
        m_lam["score"] = score_fn(m_lam)

        train_lam = m_lam[m_lam["variant"].isin(TRAIN_VARIANTS)].copy()
        train_scores = train_lam["score"].values

        _, learned_threshold, _, _ = best_score_threshold(train_scores, train_lam, lam)
        _, conf_threshold, _, _ = best_conf_threshold(train_lam, lam)

        for variant in ALL_VARIANTS:
            v = m_lam[m_lam["variant"] == variant].copy()
            split = "train/easy" if variant in TRAIN_VARIANTS else "hard_ood"

            methods = {
                "confidence": v["cheap_confidence"].values < conf_threshold,
                "learned": v["score"].values >= learned_threshold,
            }

            for name, route in methods.items():
                _, util, _ = utility(v, route, lam)
                a = audit_route(v, route, lam)
                lines.append(
                    f"| {lam:.2f} | {name} | {variant} | {split} | "
                    f"{util:.6f} | {a['route_rate']:.6f} | {a['good_available']:.6f} | "
                    f"{a['route_precision']:.6f} | {a['route_recall']:.6f} | "
                    f"{a['harmful_route_rate']:.6f} | {a['wasteful_route_rate']:.6f} | {a['missed_good_rate']:.6f} |"
                )

    lines.append("\n## Interpretation\n")
    lines.append(
        "The key diagnostic is whether the learned router reduces harmful routing on variants where small_full is stronger, "
        "while preserving recall on variants where full has positive net gain. "
        "If harmful routing remains high on block3_h36, the next router must include geometry/OOD features rather than only cheap-model confidence and margins."
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(OUT.read_text())

if __name__ == "__main__":
    main()
