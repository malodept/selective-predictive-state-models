from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_4_voc/voc_examples.csv")
OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_5_learned_voc")
OUT = OUT_DIR / "learned_voc_router_summary.md"

CHEAP_MODES = ["action_only", "state_only", "no_context"]
LAMBDAS = [0.10, 0.20, 0.30, 0.50]

TRAIN_VARIANTS = [
    "id_block2_h36_seed0",
    "block2_h72_seed11",
    "block2_v18_seed12",
]

TEST_VARIANTS = [
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

ALL_VARIANTS = TRAIN_VARIANTS + TEST_VARIANTS


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_pair(df: pd.DataFrame, cheap_mode: str) -> pd.DataFrame:
    key = ["variant", "group_id", "action_id", "action_name"]
    cheap = df[df["mode"] == cheap_mode].copy()
    full = df[df["mode"] == "full"].copy()
    m = cheap.merge(full, on=key, suffixes=("_cheap", "_full"))

    m["cheap_correct"] = m["correct_cheap"].astype(float)
    m["full_correct"] = m["correct_full"].astype(float)
    m["gain"] = m["full_correct"] - m["cheap_correct"]

    # Non-leaky routing features: observable from the cheap model output only.
    m["cheap_confidence"] = m["confidence_cheap"].astype(float)
    m["cheap_best_distance"] = m["best_distance_cheap"].astype(float)
    m["cheap_second_best_distance"] = m["second_best_distance_cheap"].astype(float)
    m["cheap_pred_margin"] = m["cheap_second_best_distance"] - m["cheap_best_distance"]

    # Avoid using correct_distance or margin_cheap because they depend on the true candidate.
    action_ids = sorted(m["action_id"].unique())
    for aid in action_ids:
        m[f"action_{aid}"] = (m["action_id"] == aid).astype(float)

    return m


def feature_columns(m: pd.DataFrame):
    cols = [
        "cheap_confidence",
        "cheap_best_distance",
        "cheap_second_best_distance",
        "cheap_pred_margin",
    ]
    # Only one-hot action columns, e.g. action_0, action_1.
    # Do NOT include action_name, which is a string column.
    cols += sorted([
        c for c in m.columns
        if c.startswith("action_") and c.removeprefix("action_").isdigit()
    ])
    return cols


def standardize(train_x: np.ndarray, x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma < 1e-8] = 1.0
    return (x - mu) / sigma, mu, sigma


def utility(m: pd.DataFrame, route: np.ndarray, lam: float):
    route = route.astype(float)
    acc = m["cheap_correct"].values + route * (m["full_correct"].values - m["cheap_correct"].values)
    util = acc - lam * route
    return float(acc.mean()), float(util.mean()), float(route.mean())


def best_threshold(scores: np.ndarray, m: pd.DataFrame, lam: float):
    thresholds = np.unique(np.concatenate([
        np.array([-1e-9, 1.0 + 1e-9]),
        np.quantile(scores, np.linspace(0, 1, 101)),
    ]))

    best = None
    for t in thresholds:
        route = scores >= t
        acc, util, rate = utility(m, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand
    return best  # utility, threshold, acc, route_rate


def confidence_threshold_baseline(m_train: pd.DataFrame, m_eval: pd.DataFrame, lam: float):
    # Route to full when cheap confidence is below threshold.
    conf = m_train["cheap_confidence"].values
    thresholds = np.unique(np.concatenate([
        np.array([0.0, 1.01]),
        np.quantile(conf, np.linspace(0, 1, 101)),
    ]))

    best = None
    for t in thresholds:
        route = conf < t
        acc, util, rate = utility(m_train, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand

    _, threshold, _, _ = best
    eval_route = m_eval["cheap_confidence"].values < threshold
    return threshold, *utility(m_eval, eval_route, lam)


def train_router(m: pd.DataFrame, cols: list[str], seed: int = 0):
    set_seed(seed)

    train = m[m["variant"].isin(TRAIN_VARIANTS)].copy()
    x_train_raw = train[cols].to_numpy(np.float32)
    y_train = ((train["full_correct"].values - train["cheap_correct"].values) > 0).astype(np.float32)

    x_train, mu, sigma = standardize(x_train_raw, x_train_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RouterMLP(x_train.shape[1]).to(device)

    x = torch.tensor(x_train, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.float32, device=device)

    pos = float(y.sum().item())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    for _ in range(400):
        model.train()
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    def score(frame: pd.DataFrame):
        xx = frame[cols].to_numpy(np.float32)
        xx = (xx - mu) / sigma
        with torch.no_grad():
            t = torch.tensor(xx, dtype=torch.float32, device=device)
            return torch.sigmoid(model(t)).cpu().numpy()

    return score


def fmt(x):
    if isinstance(x, str):
        return x
    return f"{x:.6f}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(IN)

    lines = []
    lines.append("# SPSM v5.5 learned value-of-computation router\n")
    lines.append("This experiment trains a lightweight router to predict when the expensive full state-action Transformer improves over a cheap incomplete-context predictor.")
    lines.append("Training variants: `" + "`, `".join(TRAIN_VARIANTS) + "`.")
    lines.append("Held-out hard OOD variants: `" + "`, `".join(TEST_VARIANTS) + "`.\n")

    for cheap_mode in CHEAP_MODES:
        m = make_pair(df, cheap_mode)
        cols = feature_columns(m)
        score_fn = train_router(m, cols, seed=0)

        m = m.copy()
        m["learned_gain_score"] = score_fn(m)

        train_m = m[m["variant"].isin(TRAIN_VARIANTS)].copy()

        lines.append(f"## Cheap model: `{cheap_mode}`\n")
        lines.append("| lambda | variant | split | cheap util | full util | oracle util | oracle route | conf-threshold util | conf route | learned util | learned route | learned acc |")
        lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for lam in LAMBDAS:
            train_scores = train_m["learned_gain_score"].values
            _, learned_threshold, _, _ = best_threshold(train_scores, train_m, lam)

            for variant in ALL_VARIANTS:
                v = m[m["variant"] == variant].copy()
                split = "train/easy" if variant in TRAIN_VARIANTS else "hard_ood"

                cheap_acc = float(v["cheap_correct"].mean())
                full_acc = float(v["full_correct"].mean())

                cheap_util = cheap_acc
                full_util = full_acc - lam

                oracle_route = (v["full_correct"].values - v["cheap_correct"].values) > lam
                _, oracle_util, oracle_rate = utility(v, oracle_route, lam)

                _conf_threshold, conf_acc, conf_util, conf_rate = confidence_threshold_baseline(train_m, v, lam)

                learned_route = v["learned_gain_score"].values >= learned_threshold
                learned_acc, learned_util, learned_rate = utility(v, learned_route, lam)

                lines.append(
                    f"| {lam:.2f} | {variant} | {split} | "
                    f"{cheap_util:.6f} | {full_util:.6f} | "
                    f"{oracle_util:.6f} | {oracle_rate:.6f} | "
                    f"{conf_util:.6f} | {conf_rate:.6f} | "
                    f"{learned_util:.6f} | {learned_rate:.6f} | {learned_acc:.6f} |"
                )

        lines.append("")
        lines.append("Feature columns: `" + "`, `".join(cols) + "`.\n")

    lines.append("## Interpretation\n")
    lines.append(
        "A useful learned router should approach oracle utility more closely than the confidence-threshold baseline, especially on hard OOD variants. "
        "If it improves over always-full at non-trivial cost while keeping high accuracy, this supports the SPSM value-of-computation claim: "
        "the system can learn when expensive predictive computation is worth paying for."
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(OUT.read_text())


if __name__ == "__main__":
    main()
