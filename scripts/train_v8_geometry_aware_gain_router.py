from __future__ import annotations

from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, "scripts")

from train_v5_7_small_full_gain_router import (
    prepare,
    feature_cols,
    utility,
    best_conf_threshold,
    TRAIN_VARIANTS,
    TEST_VARIANTS,
    LAMBDAS,
)

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v8_geometry_aware_gain_router")
OUT_MAIN = OUT_DIR / "geometry_aware_gain_router_summary.md"
OUT_MAIN_CSV = OUT_DIR / "geometry_aware_gain_router_summary.csv"
OUT_LOSO = OUT_DIR / "geometry_aware_loso_hard_summary.md"
OUT_LOSO_CSV = OUT_DIR / "geometry_aware_loso_hard_summary.csv"

ALL_VARIANTS = TRAIN_VARIANTS + TEST_VARIANTS

META = {
    "id_block2_h36_seed0": {"horizon": 36.0, "velocity": 1.2, "blocked": 2.0, "label": "ID 2-block, H=36"},
    "block2_h72_seed11": {"horizon": 72.0, "velocity": 1.2, "blocked": 2.0, "label": "2-block, H=72"},
    "block2_v18_seed12": {"horizon": 36.0, "velocity": 1.8, "blocked": 2.0, "label": "2-block, V=1.8"},
    "block3_h36_seed13": {"horizon": 36.0, "velocity": 1.2, "blocked": 3.0, "label": "3-block, H=36"},
    "block3_h48_seed10": {"horizon": 48.0, "velocity": 1.2, "blocked": 3.0, "label": "3-block, H=48"},
    "block3_h72_seed14": {"horizon": 72.0, "velocity": 1.2, "blocked": 3.0, "label": "3-block, H=72"},
}


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RouterMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.ReLU(),
            nn.LayerNorm(48),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.LayerNorm(24),
            nn.Linear(24, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def standardize(train_x: np.ndarray, x: np.ndarray):
    mu = train_x.mean(axis=0, keepdims=True)
    sd = train_x.std(axis=0, keepdims=True)
    sd[sd < 1e-8] = 1.0
    return (x - mu) / sd, mu, sd


def best_score_threshold(train_scores: np.ndarray, train: pd.DataFrame, lam: float):
    thresholds = np.unique(np.concatenate([
        np.array([-1e-9, 1.0 + 1e-9]),
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


def train_router(
    m: pd.DataFrame,
    cols: list[str],
    train_variants: list[str],
    lam: float,
    seed: int = 0,
    epochs: int = 600,
):
    set_seed(seed)

    train = m[m["variant"].isin(train_variants)].copy()
    x_raw = train[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    y_np = ((train["gain"].values - lam) > 0).astype(np.float32)

    pos = float(y_np.sum())
    neg = float(len(y_np) - pos)

    # Degenerate case: if there is no positive or no negative route target,
    # return a constant score. This keeps high-lambda regimes robust.
    if pos < 1.0 or neg < 1.0:
        const = pos / max(pos + neg, 1.0)
        def score_const(frame: pd.DataFrame):
            return np.full(len(frame), const, dtype=np.float32)
        return score_const

    x_np, mu, sd = standardize(x_raw, x_raw)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = RouterMLP(x_np.shape[1]).to(device)

    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)

    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)

    model.train()
    for _ in range(epochs):
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()

    def score(frame: pd.DataFrame):
        xx = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        xx = (xx - mu) / sd
        with torch.no_grad():
            t = torch.tensor(xx, dtype=torch.float32, device=device)
            return torch.sigmoid(model(t)).cpu().numpy()

    return score


def add_meta_features(m: pd.DataFrame) -> pd.DataFrame:
    m = m.copy()

    m["horizon"] = m["variant"].map(lambda v: META[v]["horizon"])
    m["velocity"] = m["variant"].map(lambda v: META[v]["velocity"])
    m["blocked_count"] = m["variant"].map(lambda v: META[v]["blocked"])
    m["variant_label"] = m["variant"].map(lambda v: META[v]["label"])

    # Normalized environment context.
    m["horizon_norm"] = (m["horizon"] - 36.0) / 36.0
    m["velocity_norm"] = (m["velocity"] - 1.2) / 0.6
    m["blocked_norm"] = m["blocked_count"] - 2.0

    # Geometry / OOD complexity interactions.
    m["blocked_x_horizon"] = m["blocked_norm"] * m["horizon_norm"]
    m["blocked_x_velocity"] = m["blocked_norm"] * m["velocity_norm"]
    m["complexity_score"] = m["blocked_norm"] * (1.0 + m["horizon_norm"]) + 0.5 * m["velocity_norm"]

    m["uncertainty_x_blocked"] = m["cheap_uncertainty"] * m["blocked_norm"]
    m["uncertainty_x_horizon"] = m["cheap_uncertainty"] * m["horizon_norm"]
    m["uncertainty_x_velocity"] = m["cheap_uncertainty"] * m["velocity_norm"]

    m["margin_x_blocked"] = m["cheap_pred_margin"] * m["blocked_norm"]
    m["best_dist_x_blocked"] = m["cheap_best_distance"] * m["blocked_norm"]
    m["second_dist_x_blocked"] = m["cheap_second_best_distance"] * m["blocked_norm"]

    # Action-specific geometry interactions: useful because some actions may be more sensitive
    # to constrained state geometry than others.
    for c in sorted([c for c in m.columns if c.startswith("action_") and c.removeprefix("action_").isdigit()]):
        m[f"{c}_x_blocked"] = m[c] * m["blocked_norm"]
        m[f"{c}_x_horizon"] = m[c] * m["horizon_norm"]

    return m


def candidate_group_npz_paths(variant: str) -> list[Path]:
    paths: list[Path] = []

    # ID / in-distribution groups.
    if variant == "id_block2_h36_seed0":
        paths.append(Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz"))

    # v5.3 OOD groups are stored under a variant directory, while the group
    # filename itself does not contain the variant name. This explicit path is
    # therefore essential; globbing only on filename misses it.
    paths.append(Path(f"outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/{variant}/mixed_hard_state_disjoint_v0_groups.npz"))

    # Additional robust fallbacks.
    roots = [
        Path("outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood"),
        Path("outputs/counterfactual/pybullet_obstacles_rgb_scale"),
        Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder"),
    ]

    for root in roots:
        if root.exists():
            # Variant may be in parent directory, not in filename.
            variant_dir = root / variant
            if variant_dir.exists():
                paths.extend(sorted(variant_dir.glob("*groups*.npz")))
                paths.extend(sorted(variant_dir.glob("*mixed*hard*.npz")))

            # Filename-based fallback.
            paths.extend(sorted(root.glob(f"**/*{variant}*groups*.npz")))
            paths.extend(sorted(root.glob(f"**/*{variant}*mixed*hard*.npz")))

    # Deduplicate while preserving order.
    out = []
    seen = set()
    for path in paths:
        rp = str(path)
        if rp not in seen and path.exists():
            out.append(path)
            seen.add(rp)
    return out


def add_optional_group_geometry(m: pd.DataFrame) -> pd.DataFrame:
    """
    If group-level NPZ files are available, join per-instance geometry features.
    The script is intentionally robust: if the files are not found or group_id does
    not index rows directly, it simply keeps the variant-level geometry features.
    """
    m = m.copy()
    optional_cols = [
        "npz_anchor_blocked",
        "npz_candidate_blocked_mean",
        "npz_candidate_blocked_std",
        "npz_blocked_mismatch_frac",
        "npz_state_dist_min",
        "npz_state_dist_mean",
        "npz_state_dist_max",
        "npz_state_dist_std",
        "npz_same_state_neg_frac",
        "npz_same_action_neg_frac",
    ]
    for c in optional_cols:
        m[c] = np.nan

    found = {}

    for variant in ALL_VARIANTS:
        sub = m[m["variant"] == variant]
        if sub.empty or "group_id" not in sub.columns:
            continue

        chosen = None
        for p in candidate_group_npz_paths(variant):
            try:
                data = np.load(p, allow_pickle=True)
                needed = {"anchor_state_xy", "candidate_state_xy"}
                if needed.issubset(set(data.files)):
                    chosen = (p, data)
                    break
            except Exception:
                continue

        if chosen is None:
            found[variant] = "missing"
            continue

        p, data = chosen
        idx = sub["group_id"].astype(int).to_numpy()
        n = len(data["anchor_state_xy"])
        if len(idx) == 0 or idx.min() < 0 or idx.max() >= n:
            found[variant] = f"found {p}, but group_id not direct index"
            continue

        anchor_xy = data["anchor_state_xy"][idx].astype(np.float32)
        cand_xy = data["candidate_state_xy"][idx].astype(np.float32)
        d = np.linalg.norm(cand_xy - anchor_xy[:, None, :], axis=-1)

        m.loc[sub.index, "npz_state_dist_min"] = d.min(axis=1)
        m.loc[sub.index, "npz_state_dist_mean"] = d.mean(axis=1)
        m.loc[sub.index, "npz_state_dist_max"] = d.max(axis=1)
        m.loc[sub.index, "npz_state_dist_std"] = d.std(axis=1)

        if "anchor_blocked" in data.files and "candidate_blocked" in data.files:
            ab = data["anchor_blocked"][idx].astype(np.float32)
            cb = data["candidate_blocked"][idx].astype(np.float32)
            m.loc[sub.index, "npz_anchor_blocked"] = ab
            m.loc[sub.index, "npz_candidate_blocked_mean"] = cb.mean(axis=1)
            m.loc[sub.index, "npz_candidate_blocked_std"] = cb.std(axis=1)
            m.loc[sub.index, "npz_blocked_mismatch_frac"] = (cb != ab[:, None]).mean(axis=1)

        if "negative_type" in data.files:
            nt = data["negative_type"][idx].astype(str)
            same_state = (np.char.find(nt, "same_state") >= 0).mean(axis=1)
            same_action = (np.char.find(nt, "same_action") >= 0).mean(axis=1)
            m.loc[sub.index, "npz_same_state_neg_frac"] = same_state
            m.loc[sub.index, "npz_same_action_neg_frac"] = same_action

        found[variant] = str(p)

    # Fill missing optional features conservatively.
    for c in optional_cols:
        m[c] = m[c].fillna(m.groupby("variant")[c].transform("mean")).fillna(0.0)

    print("===== OPTIONAL GROUP GEOMETRY JOIN =====")
    for k, v in found.items():
        print(f"{k}: {v}")

    return m


def context_cols(m: pd.DataFrame) -> list[str]:
    return feature_cols(m) + ["horizon_norm", "velocity_norm"]


def geometry_cols(m: pd.DataFrame) -> list[str]:
    cols = context_cols(m)
    extra = [
        "blocked_norm",
        "blocked_x_horizon",
        "blocked_x_velocity",
        "complexity_score",
        "uncertainty_x_blocked",
        "uncertainty_x_horizon",
        "uncertainty_x_velocity",
        "margin_x_blocked",
        "best_dist_x_blocked",
        "second_dist_x_blocked",
        "npz_anchor_blocked",
        "npz_candidate_blocked_mean",
        "npz_candidate_blocked_std",
        "npz_blocked_mismatch_frac",
        "npz_state_dist_min",
        "npz_state_dist_mean",
        "npz_state_dist_max",
        "npz_state_dist_std",
        "npz_same_state_neg_frac",
        "npz_same_action_neg_frac",
    ]
    extra += sorted([
        c for c in m.columns
        if (c.startswith("action_") and ("_x_blocked" in c or "_x_horizon" in c))
    ])

    for c in extra:
        if c in m.columns:
            cols.append(c)

    return cols


def eval_one(frame: pd.DataFrame, route: np.ndarray, lam: float):
    acc, util, rate = utility(frame, route, lam)
    return acc, util, rate


def evaluate_scheme(
    m: pd.DataFrame,
    train_variants: list[str],
    eval_variants: list[str],
    lam: float,
    seed: int = 0,
):
    ccols = context_cols(m)
    gcols = geometry_cols(m)

    context_score_fn = train_router(m, ccols, train_variants, lam, seed=seed)
    geometry_score_fn = train_router(m, gcols, train_variants, lam, seed=seed)

    m_lam = m.copy()
    m_lam["context_score"] = context_score_fn(m_lam)
    m_lam["geometry_score"] = geometry_score_fn(m_lam)

    train = m_lam[m_lam["variant"].isin(train_variants)].copy()

    _, conf_threshold, _, _ = best_conf_threshold(train, lam)
    _, context_threshold, _, _ = best_score_threshold(train["context_score"].values, train, lam)
    _, geometry_threshold, _, _ = best_score_threshold(train["geometry_score"].values, train, lam)

    rows = []

    for variant in eval_variants:
        v = m_lam[m_lam["variant"] == variant].copy()

        cheap_acc = float(v["cheap_correct"].mean())
        full_acc = float(v["full_correct"].mean())
        cheap_util = cheap_acc
        full_util = full_acc - lam

        oracle_route = (v["gain"].values - lam) > 0
        _, oracle_util, oracle_rate = eval_one(v, oracle_route, lam)

        conf_route = v["cheap_confidence"].values < conf_threshold
        conf_acc, conf_util, conf_rate = eval_one(v, conf_route, lam)

        context_route = v["context_score"].values >= context_threshold
        context_acc, context_util, context_rate = eval_one(v, context_route, lam)

        geometry_route = v["geometry_score"].values >= geometry_threshold
        geometry_acc, geometry_util, geometry_rate = eval_one(v, geometry_route, lam)

        rows.append({
            "lambda": lam,
            "variant": variant,
            "variant_label": META[variant]["label"],
            "cheap_util": cheap_util,
            "full_util": full_util,
            "oracle_util": oracle_util,
            "oracle_route": oracle_rate,
            "conf_util": conf_util,
            "conf_route": conf_rate,
            "context_util": context_util,
            "context_route": context_rate,
            "geometry_util": geometry_util,
            "geometry_route": geometry_rate,
            "geometry_acc": geometry_acc,
            "geometry_minus_context": geometry_util - context_util,
            "geometry_minus_cheap": geometry_util - cheap_util,
            "geometry_minus_full": geometry_util - full_util,
        })

    return rows


def rows_to_markdown(rows: list[dict], title: str, description: list[str]) -> str:
    lines = []
    lines.append(f"# {title}\n")
    lines.extend(description)
    lines.append("")
    lines.append("| lambda | variant | cheap util | full util | oracle util | oracle route | conf util | conf route | context util | context route | geometry util | geometry route | geom - context |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['cheap_util']:.6f} | {r['full_util']:.6f} | "
            f"{r['oracle_util']:.6f} | {r['oracle_route']:.6f} | "
            f"{r['conf_util']:.6f} | {r['conf_route']:.6f} | "
            f"{r['context_util']:.6f} | {r['context_route']:.6f} | "
            f"{r['geometry_util']:.6f} | {r['geometry_route']:.6f} | "
            f"{r['geometry_minus_context']:.6f} |"
        )

    return "\n".join(lines) + "\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = prepare()
    m = add_meta_features(m)
    m = add_optional_group_geometry(m)

    print("Rows:", len(m))
    print("Context features:", context_cols(m))
    print("Geometry features:", geometry_cols(m))

    # Protocol A: comparable to v5.8, trained only on ID/easy.
    main_rows = []
    for lam in LAMBDAS:
        main_rows.extend(
            evaluate_scheme(
                m=m,
                train_variants=TRAIN_VARIANTS,
                eval_variants=ALL_VARIANTS,
                lam=lam,
                seed=0,
            )
        )

    pd.DataFrame(main_rows).to_csv(OUT_MAIN_CSV, index=False)

    main_md = rows_to_markdown(
        main_rows,
        "SPSM v8 geometry-aware gain router — easy-train comparison",
        [
            "Cheap model: `small_full_seed0`. Expensive model: `full_seed0`.",
            "This repeats v5.8 with additional geometry/OOD-complexity features.",
            "The router is trained on ID/easy variants and evaluated on both easy and hard OOD variants.",
            "",
            "Important caveat: `blocked_count` is out-of-support when training only on ID/easy variants, so this table is mainly a comparability check against v5.8.",
        ],
    )
    OUT_MAIN.write_text(main_md, encoding="utf-8")

    # Protocol B: scientific protocol, leave one hard OOD shift out.
    loso_rows = []
    for heldout in TEST_VARIANTS:
        train_variants = [v for v in ALL_VARIANTS if v != heldout]
        for lam in LAMBDAS:
            rows = evaluate_scheme(
                m=m,
                train_variants=train_variants,
                eval_variants=[heldout],
                lam=lam,
                seed=0,
            )
            for r in rows:
                r["heldout"] = heldout
                r["heldout_label"] = META[heldout]["label"]
                r["train_variants"] = ",".join(train_variants)
            loso_rows.extend(rows)

    pd.DataFrame(loso_rows).to_csv(OUT_LOSO_CSV, index=False)

    loso_md = rows_to_markdown(
        loso_rows,
        "SPSM v8 geometry-aware gain router — leave-one-hard-out",
        [
            "This is the main v8 scientific protocol.",
            "For each hard OOD variant, the router is trained on all other variants and evaluated on the held-out hard shift.",
            "The goal is to test whether geometry/OOD-complexity features improve value-of-computation generalization beyond confidence and simple context.",
        ],
    )
    OUT_LOSO.write_text(loso_md, encoding="utf-8")

    print("\n===== WROTE =====")
    print(OUT_MAIN)
    print(OUT_MAIN_CSV)
    print(OUT_LOSO)
    print(OUT_LOSO_CSV)

    print("\n===== LOSO SUMMARY =====")
    print(OUT_LOSO.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
