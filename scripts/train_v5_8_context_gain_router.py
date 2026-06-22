from pathlib import Path
import sys

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

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_8_context_gain_router/context_gain_router_summary.md")

META = {
    "id_block2_h36_seed0": {"horizon": 36.0, "velocity": 1.2},
    "block2_h72_seed11": {"horizon": 72.0, "velocity": 1.2},
    "block2_v18_seed12": {"horizon": 36.0, "velocity": 1.8},
    "block3_h36_seed13": {"horizon": 36.0, "velocity": 1.2},
    "block3_h48_seed10": {"horizon": 48.0, "velocity": 1.2},
    "block3_h72_seed14": {"horizon": 72.0, "velocity": 1.2},
}

ALL_VARIANTS = TRAIN_VARIANTS + TEST_VARIANTS

def add_context_features(m):
    m = m.copy()
    m["horizon"] = m["variant"].map(lambda v: META[v]["horizon"])
    m["velocity"] = m["variant"].map(lambda v: META[v]["velocity"])

    # Normalized continuous context. Do not include blocked_count because it is constant in train
    # and would create uncontrolled extrapolation on block3.
    m["horizon_norm"] = (m["horizon"] - 36.0) / 36.0
    m["velocity_norm"] = (m["velocity"] - 1.2) / 0.6
    return m

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    m = add_context_features(prepare())
    cols = feature_cols(m) + ["horizon_norm", "velocity_norm"]

    train = m[m["variant"].isin(TRAIN_VARIANTS)].copy()

    lines = []
    lines.append("# SPSM v5.8 context-aware gain router\n")
    lines.append("Cheap model: `small_full_seed0`. Expensive model: `full_seed0`.")
    lines.append("This repeats v5.7 but adds observable context features: `horizon_norm` and `velocity_norm`.")
    lines.append("Blocked count is intentionally excluded because it is constant in the training variants.\n")
    lines.append("Features: `" + "`, `".join(cols) + "`.\n")

    lines.append("| lambda | variant | split | cheap util | full util | oracle util | oracle route | conf util | conf route | context-learned util | context route | context acc |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for lam in LAMBDAS:
        score_fn = train_model(m, cols, lam, seed=0)

        m_lam = m.copy()
        m_lam["score"] = score_fn(m_lam)
        train_lam = m_lam[m_lam["variant"].isin(TRAIN_VARIANTS)].copy()

        _, learned_threshold, _, _ = best_score_threshold(train_lam["score"].values, train_lam, lam)
        _, conf_threshold, _, _ = best_conf_threshold(train_lam, lam)

        for variant in ALL_VARIANTS:
            v = m_lam[m_lam["variant"] == variant].copy()
            split = "train/easy" if variant in TRAIN_VARIANTS else "hard_ood"

            cheap_acc = float(v["cheap_correct"].mean())
            full_acc = float(v["full_correct"].mean())
            cheap_util = cheap_acc
            full_util = full_acc - lam

            oracle_route = (v["gain"].values - lam) > 0
            _, oracle_util, oracle_rate = utility(v, oracle_route, lam)

            conf_route = v["cheap_confidence"].values < conf_threshold
            _, conf_util, conf_rate = utility(v, conf_route, lam)

            learned_route = v["score"].values >= learned_threshold
            learned_acc, learned_util, learned_rate = utility(v, learned_route, lam)

            lines.append(
                f"| {lam:.2f} | {variant} | {split} | "
                f"{cheap_util:.6f} | {full_util:.6f} | "
                f"{oracle_util:.6f} | {oracle_rate:.6f} | "
                f"{conf_util:.6f} | {conf_rate:.6f} | "
                f"{learned_util:.6f} | {learned_rate:.6f} | {learned_acc:.6f} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append(
        "This tests whether cheap-model uncertainty plus simple environment context can improve value-of-computation routing. "
        "A strong result would route more on horizon-72 cases where the full model helps, while avoiding harmful routing on block3_h36 where small_full is stronger."
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(OUT.read_text())

if __name__ == "__main__":
    main()
