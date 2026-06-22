from pathlib import Path
import numpy as np
import pandas as pd

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_4_voc/voc_examples.csv")
OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_4_voc/voc_routing_summary.md")

LAMBDAS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
CHEAP_MODES = ["action_only", "state_only", "no_context"]
ID_VARIANT = "id_block2_h36_seed0"

df = pd.read_csv(IN)

def make_pair(cheap_mode: str) -> pd.DataFrame:
    full = df[df["mode"] == "full"].copy()
    cheap = df[df["mode"] == cheap_mode].copy()

    key = ["variant", "group_id", "action_id", "action_name"]
    m = cheap.merge(full, on=key, suffixes=("_cheap", "_full"))

    m["cheap_correct"] = m["correct_cheap"].astype(float)
    m["full_correct"] = m["correct_full"].astype(float)
    m["gain"] = m["full_correct"] - m["cheap_correct"]
    return m

def utility(m: pd.DataFrame, route: np.ndarray, lam: float):
    route = route.astype(float)
    acc = m["cheap_correct"].values + route * (m["full_correct"].values - m["cheap_correct"].values)
    util = acc - lam * route
    return float(acc.mean()), float(util.mean()), float(route.mean())

def best_threshold_on_id(m: pd.DataFrame, lam: float):
    idm = m[m["variant"] == ID_VARIANT].copy()
    conf = idm["confidence_cheap"].values

    thresholds = np.unique(np.concatenate([
        np.array([0.0, 1.01]),
        np.quantile(conf, np.linspace(0, 1, 101)),
    ]))

    best = None
    for t in thresholds:
        route = conf < t
        acc, util, rate = utility(idm, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand
    return best  # util, threshold, acc, rate

lines = []
lines.append("# SPSM v5.4 value-of-computation routing pilot\n")
lines.append("This pilot treats an incomplete-context model as the cheap predictor and the full state-action Transformer as the expensive predictor.")
lines.append("The route decision is learned as a simple ID-fitted confidence threshold on the cheap model, then evaluated on ID and OOD variants.\n")

for cheap_mode in CHEAP_MODES:
    m = make_pair(cheap_mode)

    lines.append(f"## Cheap model: `{cheap_mode}`\n")
    lines.append("| lambda | variant | cheap acc | full acc | oracle utility | oracle route | ID-threshold | routed acc | routed utility | routed rate |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for lam in LAMBDAS:
        best_util, threshold, id_acc, id_rate = best_threshold_on_id(m, lam)

        for variant in sorted(m["variant"].unique()):
            v = m[m["variant"] == variant].copy()

            cheap_acc = float(v["cheap_correct"].mean())
            full_acc = float(v["full_correct"].mean())

            oracle_route = (v["full_correct"].values - lam) > v["cheap_correct"].values
            oracle_acc, oracle_util, oracle_rate = utility(v, oracle_route, lam)

            route = v["confidence_cheap"].values < threshold
            routed_acc, routed_util, routed_rate = utility(v, route, lam)

            lines.append(
                f"| {lam:.2f} | {variant} | {cheap_acc:.6f} | {full_acc:.6f} | "
                f"{oracle_util:.6f} | {oracle_rate:.6f} | {threshold:.6f} | "
                f"{routed_acc:.6f} | {routed_util:.6f} | {routed_rate:.6f} |"
            )

    lines.append("")

lines.append("## Interpretation\n")
lines.append(
    "This is not yet the final value-of-computation model. It is the zero-new-training routing audit. "
    "The key quantities are the oracle utility and the gap between oracle routing and the ID-fitted confidence-threshold router. "
    "A large oracle gap means that a learned gain model is justified; a strong ID-threshold transfer would mean cheap confidence already carries much of the routing signal."
)

OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT)
print(OUT.read_text())
