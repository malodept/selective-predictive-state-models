from pathlib import Path
import numpy as np
import pandas as pd

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/small_full_voc_examples.csv")
OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/small_full_voc_summary.md")

LAMBDAS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
TRAIN_VARIANTS = ["id_block2_h36_seed0", "block2_h72_seed11", "block2_v18_seed12"]
TEST_VARIANTS = ["block3_h36_seed13", "block3_h48_seed10", "block3_h72_seed14"]
ALL_VARIANTS = TRAIN_VARIANTS + TEST_VARIANTS

df = pd.read_csv(IN)

key = ["variant", "group_id", "action_id", "action_name"]
cheap = df[df["mode"] == "small_full"].copy()
full = df[df["mode"] == "full"].copy()
m = cheap.merge(full, on=key, suffixes=("_cheap", "_full"))

m["cheap_correct"] = m["correct_cheap"].astype(float)
m["full_correct"] = m["correct_full"].astype(float)
m["gain"] = m["full_correct"] - m["cheap_correct"]

def utility(frame, route, lam):
    route = route.astype(float)
    acc = frame["cheap_correct"].values + route * (frame["full_correct"].values - frame["cheap_correct"].values)
    util = acc - lam * route
    return float(acc.mean()), float(util.mean()), float(route.mean())

def best_conf_threshold(train, lam):
    conf = train["confidence_cheap"].values
    thresholds = np.unique(np.concatenate([[0.0, 1.01], np.quantile(conf, np.linspace(0, 1, 101))]))
    best = None
    for t in thresholds:
        route = conf < t
        acc, util, rate = utility(train, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand
    return best

train = m[m["variant"].isin(TRAIN_VARIANTS)].copy()

lines = []
lines.append("# SPSM v5.6 small-full value-of-computation summary\n")
lines.append("Cheap model: `small_full_seed0` reduced full Transformer. Expensive model: `full_seed0` v5.2 Transformer.\n")
lines.append("| lambda | variant | split | cheap acc | full acc | cheap util | full util | oracle util | oracle route | conf-threshold | routed util | routed rate | routed acc |")
lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for lam in LAMBDAS:
    _, threshold, _, _ = best_conf_threshold(train, lam)

    for variant in ALL_VARIANTS:
        v = m[m["variant"] == variant].copy()
        split = "train/easy" if variant in TRAIN_VARIANTS else "hard_ood"

        cheap_acc = float(v["cheap_correct"].mean())
        full_acc = float(v["full_correct"].mean())
        cheap_util = cheap_acc
        full_util = full_acc - lam

        oracle_route = (v["full_correct"].values - v["cheap_correct"].values) > lam
        _, oracle_util, oracle_rate = utility(v, oracle_route, lam)

        route = v["confidence_cheap"].values < threshold
        routed_acc, routed_util, routed_rate = utility(v, route, lam)

        lines.append(
            f"| {lam:.2f} | {variant} | {split} | {cheap_acc:.6f} | {full_acc:.6f} | "
            f"{cheap_util:.6f} | {full_util:.6f} | {oracle_util:.6f} | {oracle_rate:.6f} | "
            f"{threshold:.6f} | {routed_util:.6f} | {routed_rate:.6f} | {routed_acc:.6f} |"
        )

lines.append("\n## Interpretation\n")
lines.append(
    "This table tests the meaningful cheap/expensive setup. A useful VoC setting requires a non-trivial oracle gap and a router that improves over always-cheap and always-full at positive compute cost. "
    "If the confidence router fails while oracle utility is high, the next step is a learned gain router using small-full uncertainty and action/state diagnostics."
)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
print(OUT.read_text())
