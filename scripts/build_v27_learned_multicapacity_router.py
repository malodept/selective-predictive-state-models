from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v27_learned_multicapacity_router")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v26_multicapacity_oracle/multicapacity_predictions.csv")
OUT_CSV = OUT_DIR / "learned_multicapacity_router_summary.csv"
OUT_MD = OUT_DIR / "learned_multicapacity_router_summary.md"

LAMBDAS = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

MODELS = ["Tiny", "Small", "Medium", "Full"]

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

    keep = [
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
    base = tiny[keep].copy()

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

    # Wide tables of correctness and costs.
    corr = raw.pivot_table(
        index=["variant", "group_id"],
        columns="model_label",
        values="tie_credit",
        aggfunc="first",
    ).reset_index()

    cost = raw.drop_duplicates("model_label").set_index("model_label")["relative_latency"].to_dict()

    merged = base.merge(corr, on=["variant", "group_id"], how="left", validate="one_to_one")

    for m in MODELS:
        merged[f"{m}_cost"] = float(cost[m])

    return merged


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
    }


def best_fixed(df: pd.DataFrame, lam: float):
    rows = []
    for m in MODELS:
        choice = np.array([m] * len(df), dtype=object)
        r = eval_choice(df, choice, lam)
        r["method"] = f"fixed_{m}"
        rows.append(r)
    best = max(rows, key=lambda r: r["utility"])
    return rows, best


def oracle_choice(df: pd.DataFrame, lam: float) -> np.ndarray:
    u = utilities(df, lam)
    costs = {m: float(df[f"{m}_cost"].iloc[0]) for m in MODELS}
    out = []
    for _, row in u.iterrows():
        mx = row[MODELS].max()
        tied = [m for m in MODELS if abs(row[m] - mx) <= 1e-12]
        out.append(min(tied, key=lambda m: costs[m]))
    return np.asarray(out, dtype=object)


def train_expected_utility_router(train: pd.DataFrame, lam: float, model_kind: str):
    X = train[feature_cols()].to_numpy(np.float64)
    u = utilities(train, lam)

    regressors = {}
    for m in MODELS:
        y = u[m].to_numpy(np.float64)
        if model_kind == "ridge":
            reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        elif model_kind == "rf":
            reg = RandomForestRegressor(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                random_state=0,
                n_jobs=-1,
            )
        else:
            raise ValueError(model_kind)
        reg.fit(X, y)
        regressors[m] = reg

    def predict(test: pd.DataFrame):
        Xt = test[feature_cols()].to_numpy(np.float64)
        pred = np.stack([regressors[m].predict(Xt) for m in MODELS], axis=1)
        return np.asarray([MODELS[i] for i in pred.argmax(axis=1)], dtype=object)

    return predict


def train_oracle_label_router(train: pd.DataFrame, lam: float, model_kind: str):
    X = train[feature_cols()].to_numpy(np.float64)
    y = oracle_choice(train, lam)

    if model_kind == "logreg":
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
            ),
        )
    elif model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=0,
            n_jobs=-1,
        )
    else:
        raise ValueError(model_kind)

    clf.fit(X, y)

    def predict(test: pd.DataFrame):
        Xt = test[feature_cols()].to_numpy(np.float64)
        return clf.predict(Xt).astype(object)

    return predict


def main():
    df = prepare()
    rows = []

    for heldout in TEST_VARIANTS:
        train_variants = [v for v in ALL_VARIANTS if v != heldout]
        train = df[df["variant"].isin(train_variants)].copy()
        test = df[df["variant"] == heldout].copy()

        for lam in LAMBDAS:
            fixed_rows, best = best_fixed(test, lam)

            for r in fixed_rows:
                rows.append({
                    "lambda": lam,
                    "heldout": heldout,
                    "variant_label": META[heldout]["label"],
                    **r,
                })

            oracle = eval_choice(test, oracle_choice(test, lam), lam)
            oracle["method"] = "oracle_multicapacity"
            rows.append({
                "lambda": lam,
                "heldout": heldout,
                "variant_label": META[heldout]["label"],
                **oracle,
            })

            routers = {
                "ridge_expected_utility": train_expected_utility_router(train, lam, "ridge"),
                "rf_expected_utility": train_expected_utility_router(train, lam, "rf"),
                "logreg_oracle_label": train_oracle_label_router(train, lam, "logreg"),
                "rf_oracle_label": train_oracle_label_router(train, lam, "rf"),
            }

            for name, fn in routers.items():
                choice = fn(test)
                r = eval_choice(test, choice, lam)
                r["method"] = name
                rows.append({
                    "lambda": lam,
                    "heldout": heldout,
                    "variant_label": META[heldout]["label"],
                    **r,
                })

            print(f"heldout={heldout} lambda={lam} best_fixed={best['method']} {best['utility']:.4f}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v27 learned multi-capacity router\n")
    lines.append("The router chooses among Tiny, Small, Medium, and Full using only Tiny diagnostics, action, and context features.")
    lines.append("This is a leave-one-hard-OOD-out protocol over the three hard variants.")
    lines.append("Utility uses measured batch-128 latency from v24.\n")

    lines.append("## Learned router vs best fixed and oracle\n")
    lines.append("| lambda | heldout | best fixed | best fixed util | best learned | learned util | oracle util | learned gain over fixed | oracle gap | selection Tiny/Small/Medium/Full |")
    lines.append("|---:|---|---|---:|---|---:|---:|---:|---:|---|")

    learned_methods = [
        "ridge_expected_utility",
        "rf_expected_utility",
        "logreg_oracle_label",
        "rf_oracle_label",
    ]

    for lam in LAMBDAS:
        for heldout in TEST_VARIANTS:
            sub = out[(out["lambda"] == lam) & (out["heldout"] == heldout)]
            fixed = sub[sub["method"].str.startswith("fixed_")].sort_values("utility", ascending=False).iloc[0]
            learned = sub[sub["method"].isin(learned_methods)].sort_values("utility", ascending=False).iloc[0]
            oracle = sub[sub["method"] == "oracle_multicapacity"].iloc[0]
            sel = (
                f"{learned['select_Tiny']:.2f}/"
                f"{learned['select_Small']:.2f}/"
                f"{learned['select_Medium']:.2f}/"
                f"{learned['select_Full']:.2f}"
            )
            lines.append(
                f"| {lam:.2f} | {META[heldout]['label']} | `{fixed['method'].replace('fixed_', '')}` | "
                f"{fixed['utility']:.6f} | `{learned['method']}` | {learned['utility']:.6f} | "
                f"{oracle['utility']:.6f} | {learned['utility'] - fixed['utility']:.6f} | "
                f"{oracle['utility'] - learned['utility']:.6f} | {sel} |"
            )

    lines.append("\n## All learned methods on hard OOD\n")
    lines.append("| lambda | heldout | method | utility | top-1 | latency | select Tiny/Small/Medium/Full |")
    lines.append("|---:|---|---|---:|---:|---:|---|")

    show_methods = learned_methods + ["oracle_multicapacity"]
    for _, r in out[out["method"].isin(show_methods)].iterrows():
        sel = (
            f"{r['select_Tiny']:.2f}/"
            f"{r['select_Small']:.2f}/"
            f"{r['select_Medium']:.2f}/"
            f"{r['select_Full']:.2f}"
        )
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | `{r['method']}` | "
            f"{r['utility']:.6f} | {r['top1']:.6f} | {r['relative_latency']:.6f} | {sel} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- If learned routers improve over best fixed on held-out hard variants, multi-capacity routing is empirically useful.")
    lines.append("- If learned routers fail while oracle is high, the available Tiny/context features are insufficient and better routing signals are needed.")
    lines.append("- This is intentionally conservative: the router cannot inspect Medium/Full outputs before choosing.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
