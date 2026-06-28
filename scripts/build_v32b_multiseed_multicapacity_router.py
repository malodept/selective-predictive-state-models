from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v32_multiseed_multicapacity_routing")
PRED = OUT_DIR / "multiseed_multicapacity_predictions.csv"

OUT_PER = OUT_DIR / "multiseed_router_per_seed.csv"
OUT_AGG = OUT_DIR / "multiseed_router_aggregate.csv"
OUT_MD = OUT_DIR / "multiseed_router_summary.md"

LAMBDAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
MODELS = ["Tiny", "Small", "Medium", "Full"]
SEEDS = [0, 1, 2]

TEST_VARIANTS = [
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

ALL_VARIANTS = [
    "block2_h72_seed11",
    "block2_v18_seed12",
    "block3_h36_seed13",
    "block3_h48_seed10",
    "block3_h72_seed14",
]

META = {
    "block2_h72_seed11": {"horizon": 72.0, "velocity": 1.2, "blocked": 2.0, "label": "2-block, H=72"},
    "block2_v18_seed12": {"horizon": 36.0, "velocity": 1.8, "blocked": 2.0, "label": "2-block, V=1.8"},
    "block3_h36_seed13": {"horizon": 36.0, "velocity": 1.2, "blocked": 3.0, "label": "3-block, H=36"},
    "block3_h48_seed10": {"horizon": 48.0, "velocity": 1.2, "blocked": 3.0, "label": "3-block, H=48"},
    "block3_h72_seed14": {"horizon": 72.0, "velocity": 1.2, "blocked": 3.0, "label": "3-block, H=72"},
}


def prepare() -> pd.DataFrame:
    raw = pd.read_csv(PRED)

    tiny = raw[raw["model_label"] == "Tiny"].copy()
    tiny = tiny.rename(columns={
        "tie_credit": "tiny_tie_credit",
        "strict_correct": "tiny_strict_correct",
        "argmin_correct": "tiny_argmin_correct",
        "confidence": "tiny_confidence",
        "best_distance": "tiny_best_distance",
        "second_best_distance": "tiny_second_best_distance",
        "pred_margin": "tiny_pred_margin",
        "correct_distance": "tiny_correct_distance",
    })

    base_cols = [
        "ladder_seed",
        "variant",
        "variant_label",
        "group_id",
        "action_id",
        "action_name",
        "tiny_tie_credit",
        "tiny_strict_correct",
        "tiny_argmin_correct",
        "tiny_confidence",
        "tiny_best_distance",
        "tiny_second_best_distance",
        "tiny_pred_margin",
        "tiny_correct_distance",
    ]
    base = tiny[base_cols].copy()

    for v, meta in META.items():
        m = base["variant"] == v
        base.loc[m, "horizon"] = meta["horizon"]
        base.loc[m, "velocity"] = meta["velocity"]
        base.loc[m, "blocked"] = meta["blocked"]

    base["horizon_norm"] = base["horizon"] / 72.0
    base["velocity_norm"] = base["velocity"] / 1.8
    base["blocked_norm"] = base["blocked"] / 3.0
    base["tiny_uncertainty"] = 1.0 - base["tiny_confidence"]
    base["tiny_margin_norm"] = base["tiny_pred_margin"] / (base["tiny_second_best_distance"].abs() + 1e-8)

    for a in [1, 2, 3, 4]:
        base[f"action_{a}"] = (base["action_id"].astype(int) == a).astype(float)

    corr = raw.pivot_table(
        index=["ladder_seed", "variant", "group_id"],
        columns="model_label",
        values="tie_credit",
        aggfunc="first",
    ).reset_index()

    cost = raw.drop_duplicates("model_label").set_index("model_label")["relative_latency"].to_dict()

    df = base.merge(corr, on=["ladder_seed", "variant", "group_id"], how="left", validate="one_to_one")

    for m in MODELS:
        df[f"{m}_cost"] = float(cost[m])

    return df


def feature_cols() -> list[str]:
    return [
        "tiny_confidence",
        "tiny_uncertainty",
        "tiny_best_distance",
        "tiny_second_best_distance",
        "tiny_pred_margin",
        "tiny_margin_norm",
        "tiny_correct_distance",
        "action_1",
        "action_2",
        "action_3",
        "action_4",
        "horizon_norm",
        "velocity_norm",
        "blocked_norm",
    ]


def utilities(df: pd.DataFrame, lam: float) -> pd.DataFrame:
    u = pd.DataFrame(index=df.index)
    for m in MODELS:
        u[m] = df[m] - lam * df[f"{m}_cost"]
    return u


def eval_choice(df: pd.DataFrame, choice: np.ndarray, lam: float) -> dict:
    u = utilities(df, lam)
    vals = np.array([u.iloc[i][choice[i]] for i in range(len(choice))], dtype=np.float64)
    top = np.array([df.iloc[i][choice[i]] for i in range(len(choice))], dtype=np.float64)
    costs = np.array([df.iloc[i][f"{choice[i]}_cost"] for i in range(len(choice))], dtype=np.float64)

    return {
        "utility": float(vals.mean()),
        "top1": float(top.mean()),
        "relative_latency": float(costs.mean()),
        "select_Tiny": float((choice == "Tiny").mean()),
        "select_Small": float((choice == "Small").mean()),
        "select_Medium": float((choice == "Medium").mean()),
        "select_Full": float((choice == "Full").mean()),
        "n": int(len(choice)),
    }


def oracle_choice(df: pd.DataFrame, lam: float) -> np.ndarray:
    u = utilities(df, lam)
    costs = {m: float(df[f"{m}_cost"].iloc[0]) for m in MODELS}
    out = []

    for _, row in u.iterrows():
        mx = row[MODELS].max()
        tied = [m for m in MODELS if abs(row[m] - mx) <= 1e-12]
        out.append(min(tied, key=lambda m: costs[m]))

    return np.asarray(out, dtype=object)


def best_fixed(df: pd.DataFrame, lam: float):
    rows = []
    for m in MODELS:
        choice = np.array([m] * len(df), dtype=object)
        r = eval_choice(df, choice, lam)
        r["method"] = f"fixed_{m}"
        rows.append(r)
    return rows, max(rows, key=lambda r: r["utility"])


def train_expected_utility_router(train: pd.DataFrame, lam: float, kind: str):
    X = train[feature_cols()].to_numpy(np.float64)
    u = utilities(train, lam)

    regs = {}
    for m in MODELS:
        y = u[m].to_numpy(np.float64)
        if kind == "ridge":
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif kind == "rf":
            reg = RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                random_state=0,
                n_jobs=-1,
            )
        else:
            raise ValueError(kind)

        reg.fit(X, y)
        regs[m] = reg

    def predict(test: pd.DataFrame):
        Xt = test[feature_cols()].to_numpy(np.float64)
        pred = np.stack([regs[m].predict(Xt) for m in MODELS], axis=1)
        return np.asarray([MODELS[i] for i in pred.argmax(axis=1)], dtype=object)

    return predict


def train_oracle_label_router(train: pd.DataFrame, lam: float, kind: str):
    X = train[feature_cols()].to_numpy(np.float64)
    y = oracle_choice(train, lam)

    if kind == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)

    clf.fit(X, y)

    def predict(test: pd.DataFrame):
        Xt = test[feature_cols()].to_numpy(np.float64)
        return clf.predict(Xt).astype(object)

    return predict


def main():
    df = prepare()
    rows = []

    learned_methods = {
        "ridge_expected_utility": lambda tr, lam: train_expected_utility_router(tr, lam, "ridge"),
        "rf_expected_utility": lambda tr, lam: train_expected_utility_router(tr, lam, "rf"),
        "logreg_oracle_label": lambda tr, lam: train_oracle_label_router(tr, lam, "logreg"),
        "rf_oracle_label": lambda tr, lam: train_oracle_label_router(tr, lam, "rf"),
    }

    for heldout in TEST_VARIANTS:
        train_variants = [v for v in ALL_VARIANTS if v != heldout]
        train = df[df["variant"].isin(train_variants)].copy()

        for lam in LAMBDAS:
            routers = {name: factory(train, lam) for name, factory in learned_methods.items()}

            for seed in SEEDS:
                test = df[(df["variant"] == heldout) & (df["ladder_seed"] == seed)].copy()

                fixed_rows, best = best_fixed(test, lam)
                for r in fixed_rows:
                    rows.append({
                        "lambda": lam,
                        "heldout": heldout,
                        "variant_label": META[heldout]["label"],
                        "ladder_seed": seed,
                        **r,
                    })

                oracle = eval_choice(test, oracle_choice(test, lam), lam)
                oracle["method"] = "oracle_multicapacity"
                rows.append({
                    "lambda": lam,
                    "heldout": heldout,
                    "variant_label": META[heldout]["label"],
                    "ladder_seed": seed,
                    **oracle,
                })

                for name, fn in routers.items():
                    choice = fn(test)
                    r = eval_choice(test, choice, lam)
                    r["method"] = name
                    rows.append({
                        "lambda": lam,
                        "heldout": heldout,
                        "variant_label": META[heldout]["label"],
                        "ladder_seed": seed,
                        **r,
                    })

            print(f"heldout={heldout} lambda={lam}", flush=True)

    per = pd.DataFrame(rows)
    per.to_csv(OUT_PER, index=False)

    agg = (
        per
        .groupby(["lambda", "heldout", "variant_label", "method"], as_index=False)
        .agg(
            utility_mean=("utility", "mean"),
            utility_std=("utility", "std"),
            top1_mean=("top1", "mean"),
            top1_std=("top1", "std"),
            latency_mean=("relative_latency", "mean"),
            latency_std=("relative_latency", "std"),
            select_Tiny_mean=("select_Tiny", "mean"),
            select_Small_mean=("select_Small", "mean"),
            select_Medium_mean=("select_Medium", "mean"),
            select_Full_mean=("select_Full", "mean"),
            n=("n", "first"),
        )
    )
    agg.to_csv(OUT_AGG, index=False)

    learned_names = list(learned_methods.keys())

    lines = []
    lines.append("# SPSM v32B seed-aligned multi-capacity routing\n")
    lines.append("This evaluates multi-capacity routing across three seed-aligned ladders.")
    lines.append("Each ladder contains Tiny/Small/Medium/Full with the same training seed; results are averaged across ladder seeds.")
    lines.append("The router is trained on all non-heldout variants and uses only Tiny diagnostics, action, and context features.\n")

    lines.append("## Best learned router vs best fixed and oracle, averaged over ladder seeds\n")
    lines.append("| lambda | heldout | best fixed | fixed util | best learned | learned util | oracle util | learned-fixed | oracle-learned | selection Tiny/Small/Medium/Full |")
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---|")

    for lam in LAMBDAS:
        for heldout in TEST_VARIANTS:
            sub = agg[(agg["lambda"] == lam) & (agg["heldout"] == heldout)]
            fixed = sub[sub["method"].str.startswith("fixed_")].sort_values("utility_mean", ascending=False).iloc[0]
            learned = sub[sub["method"].isin(learned_names)].sort_values("utility_mean", ascending=False).iloc[0]
            oracle = sub[sub["method"] == "oracle_multicapacity"].iloc[0]

            sel = (
                f"{learned['select_Tiny_mean']:.2f}/"
                f"{learned['select_Small_mean']:.2f}/"
                f"{learned['select_Medium_mean']:.2f}/"
                f"{learned['select_Full_mean']:.2f}"
            )

            lines.append(
                f"| {lam:.2f} | {META[heldout]['label']} | `{fixed['method'].replace('fixed_', '')}` | "
                f"{fixed['utility_mean']:.6f} | `{learned['method']}` | {learned['utility_mean']:.6f} | "
                f"{oracle['utility_mean']:.6f} | {learned['utility_mean'] - fixed['utility_mean']:.6f} | "
                f"{oracle['utility_mean'] - learned['utility_mean']:.6f} | {sel} |"
            )

    lines.append("\n## Per-seed best learned gains\n")
    lines.append("| lambda | heldout | seed | best fixed | best learned | learned-fixed |")
    lines.append("|---:|---|---:|---|---|---:|")

    for lam in LAMBDAS:
        for heldout in TEST_VARIANTS:
            for seed in SEEDS:
                sub = per[(per["lambda"] == lam) & (per["heldout"] == heldout) & (per["ladder_seed"] == seed)]
                fixed = sub[sub["method"].str.startswith("fixed_")].sort_values("utility", ascending=False).iloc[0]
                learned = sub[sub["method"].isin(learned_names)].sort_values("utility", ascending=False).iloc[0]
                lines.append(
                    f"| {lam:.2f} | {META[heldout]['label']} | {seed} | "
                    f"`{fixed['method'].replace('fixed_', '')}` | `{learned['method']}` | "
                    f"{learned['utility'] - fixed['utility']:.6f} |"
                )

    lines.append("\n## Interpretation\n")
    lines.append("- This is the robust version of v27/v28: routing is evaluated over seed-aligned capacity ladders instead of a single seed0 ladder.")
    lines.append("- If learned-fixed remains positive on H=48, the routing result survives multi-seed capacity retraining.")
    lines.append("- If H=72 keeps a large oracle gap, it remains a routing-signal problem rather than a solved regime.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_PER)
    print(OUT_AGG)
    print(OUT_MD)
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
