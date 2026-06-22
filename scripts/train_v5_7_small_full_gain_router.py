from __future__ import annotations

from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn

IN = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/small_full_voc_examples.csv")
OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_7_small_full_gain_router")
OUT = OUT_DIR / "small_full_gain_router_summary.md"

LAMBDAS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]

TRAIN_VARIANTS = ["id_block2_h36_seed0", "block2_h72_seed11", "block2_v18_seed12"]
TEST_VARIANTS = ["block3_h36_seed13", "block3_h48_seed10", "block3_h72_seed14"]
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


def prepare() -> pd.DataFrame:
    df = pd.read_csv(IN)
    key = ["variant", "group_id", "action_id", "action_name"]

    cheap = df[df["mode"] == "small_full"].copy()
    full = df[df["mode"] == "full"].copy()

    m = cheap.merge(full, on=key, suffixes=("_cheap", "_full"))
    m["cheap_correct"] = m["correct_cheap"].astype(float)
    m["full_correct"] = m["correct_full"].astype(float)
    m["gain"] = m["full_correct"] - m["cheap_correct"]

    m["cheap_confidence"] = m["confidence_cheap"].astype(float)
    m["cheap_uncertainty"] = 1.0 - m["cheap_confidence"]
    m["cheap_best_distance"] = m["best_distance_cheap"].astype(float)
    m["cheap_second_best_distance"] = m["second_best_distance_cheap"].astype(float)
    m["cheap_pred_margin"] = m["pred_margin_cheap"].astype(float)

    for aid in sorted(m["action_id"].unique()):
        m[f"action_{int(aid)}"] = (m["action_id"] == aid).astype(float)

    return m


def feature_cols(m: pd.DataFrame) -> list[str]:
    cols = [
        "cheap_confidence",
        "cheap_uncertainty",
        "cheap_best_distance",
        "cheap_second_best_distance",
        "cheap_pred_margin",
    ]
    cols += sorted([c for c in m.columns if c.startswith("action_") and c.removeprefix("action_").isdigit()])
    return cols


def standardize(train_x: np.ndarray, x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd, mu, sd


def utility(frame: pd.DataFrame, route: np.ndarray, lam: float):
    route = route.astype(float)
    acc = frame["cheap_correct"].values + route * (frame["full_correct"].values - frame["cheap_correct"].values)
    util = acc - lam * route
    return float(acc.mean()), float(util.mean()), float(route.mean())


def best_score_threshold(train_scores: np.ndarray, train: pd.DataFrame, lam: float):
    thresholds = np.unique(np.concatenate([
        np.array([-1e-9, 1 + 1e-9]),
        np.quantile(train_scores, np.linspace(0, 1, 201)),
    ]))

    best = None
    for t in thresholds:
        route = train_scores >= t
        acc, util, rate = utility(train, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def best_conf_threshold(train: pd.DataFrame, lam: float):
    conf = train["cheap_confidence"].values
    thresholds = np.unique(np.concatenate([
        np.array([0.0, 1.01]),
        np.quantile(conf, np.linspace(0, 1, 201)),
    ]))

    best = None
    for t in thresholds:
        route = conf < t
        acc, util, rate = utility(train, route, lam)
        cand = (util, t, acc, rate)
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def train_model(m: pd.DataFrame, cols: list[str], lam: float, seed: int = 0):
    set_seed(seed)

    train = m[m["variant"].isin(TRAIN_VARIANTS)].copy()
    x_raw = train[cols].to_numpy(np.float32)
    y_np = ((train["gain"].values - lam) > 0).astype(np.float32)

    x_np, mu, sd = standardize(x_raw, x_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RouterMLP(x_np.shape[1]).to(device)

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    pos = float(y.sum().item())
    neg = float(len(y_np) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    for _ in range(500):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    def score(frame: pd.DataFrame):
        xx = frame[cols].to_numpy(np.float32)
        xx = (xx - mu) / sd
        with torch.no_grad():
            t = torch.tensor(xx, dtype=torch.float32, device=device)
            return torch.sigmoid(model(t)).cpu().numpy()

    return score


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    m = prepare()
    cols = feature_cols(m)

    lines = []
    lines.append("# SPSM v5.7 learned small-full gain router\n")
    lines.append("Cheap model: `small_full_seed0`. Expensive model: `full_seed0`.")
    lines.append("The learned router is trained on ID/easy variants and evaluated on held-out hard OOD block3 variants.")
    lines.append("Targets are lambda-specific: route if `full_correct - small_correct > lambda`.\n")
    lines.append("Features: `" + "`, `".join(cols) + "`.\n")

    lines.append("| lambda | variant | split | cheap util | full util | oracle util | oracle route | conf util | conf route | learned util | learned route | learned acc |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    train = m[m["variant"].isin(TRAIN_VARIANTS)].copy()

    for lam in LAMBDAS:
        score_fn = train_model(m, cols, lam, seed=0)

        m_lam = m.copy()
        m_lam["score"] = score_fn(m_lam)
        train_scores = m_lam[m_lam["variant"].isin(TRAIN_VARIANTS)]["score"].values

        _, learned_threshold, _, _ = best_score_threshold(train_scores, train, lam)
        _, conf_threshold, _, _ = best_conf_threshold(train, lam)

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
        "This is the first non-trivial SPSM value-of-computation setup: the cheap model is already competent, and the expensive model is not uniformly better. "
        "A successful router should improve over always-cheap and always-full at positive compute cost, especially on block3_h72 where full helps, while avoiding unnecessary or harmful routing on block3_h36 where small_full is stronger."
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(OUT.read_text())


if __name__ == "__main__":
    main()
