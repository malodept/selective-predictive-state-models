from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from build_v32b_multiseed_multicapacity_router import (
    prepare,
    utilities,
    train_expected_utility_router,
    train_oracle_label_router,
    LAMBDAS,
    MODELS,
    SEEDS,
    TEST_VARIANTS,
    ALL_VARIANTS,
    META,
)

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v32_multiseed_multicapacity_routing")
OUT_CSV = OUT_DIR / "multiseed_router_bootstrap_summary.csv"
OUT_MD = OUT_DIR / "multiseed_router_bootstrap_summary.md"

BOOT = 3000
RNG_SEED = 123

LEARNED_METHODS = {
    "ridge_expected_utility": lambda tr, lam: train_expected_utility_router(tr, lam, "ridge"),
    "rf_expected_utility": lambda tr, lam: train_expected_utility_router(tr, lam, "rf"),
    "logreg_oracle_label": lambda tr, lam: train_oracle_label_router(tr, lam, "logreg"),
    "rf_oracle_label": lambda tr, lam: train_oracle_label_router(tr, lam, "rf"),
}


def utility_vector(df: pd.DataFrame, choice: np.ndarray, lam: float) -> np.ndarray:
    u = utilities(df, lam)
    return np.array([u.iloc[i][choice[i]] for i in range(len(choice))], dtype=np.float64)


def fixed_choice(df: pd.DataFrame, model: str) -> np.ndarray:
    return np.array([model] * len(df), dtype=object)


def best_fixed_choice(df: pd.DataFrame, lam: float):
    best_model = None
    best_mean = -1e9
    best_vec = None

    for m in MODELS:
        ch = fixed_choice(df, m)
        vec = utility_vector(df, ch, lam)
        mu = float(vec.mean())
        if mu > best_mean:
            best_mean = mu
            best_model = m
            best_vec = vec

    return best_model, best_vec


def oracle_choice(df: pd.DataFrame, lam: float) -> np.ndarray:
    u = utilities(df, lam)
    costs = {m: float(df[f"{m}_cost"].iloc[0]) for m in MODELS}

    out = []
    for _, row in u.iterrows():
        mx = row[MODELS].max()
        tied = [m for m in MODELS if abs(row[m] - mx) <= 1e-12]
        out.append(min(tied, key=lambda m: costs[m]))

    return np.asarray(out, dtype=object)


def hierarchical_bootstrap_mean(x: np.ndarray, seeds: np.ndarray, rng: np.random.Generator):
    x = np.asarray(x, dtype=np.float64)
    seeds = np.asarray(seeds)
    unique_seeds = np.array(sorted(np.unique(seeds).tolist()))

    boot = np.empty(BOOT, dtype=np.float64)

    seed_to_idx = {s: np.where(seeds == s)[0] for s in unique_seeds}

    for b in range(BOOT):
        sampled_seeds = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
        vals = []
        for s in sampled_seeds:
            idx = seed_to_idx[s]
            sampled_idx = rng.choice(idx, size=len(idx), replace=True)
            vals.append(x[sampled_idx])
        boot[b] = np.concatenate(vals).mean()

    return float(x.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def eval_choice_stats(test: pd.DataFrame, choice: np.ndarray, lam: float):
    vec = utility_vector(test, choice, lam)
    top = np.array([test.iloc[i][choice[i]] for i in range(len(choice))], dtype=np.float64)
    cost = np.array([test.iloc[i][f"{choice[i]}_cost"] for i in range(len(choice))], dtype=np.float64)

    return {
        "utility": float(vec.mean()),
        "top1": float(top.mean()),
        "latency": float(cost.mean()),
        "select_Tiny": float((choice == "Tiny").mean()),
        "select_Small": float((choice == "Small").mean()),
        "select_Medium": float((choice == "Medium").mean()),
        "select_Full": float((choice == "Full").mean()),
        "utility_vector": vec,
    }


def main():
    rng = np.random.default_rng(RNG_SEED)
    df = prepare()

    rows = []

    for heldout in TEST_VARIANTS:
        train_variants = [v for v in ALL_VARIANTS if v != heldout]
        train = df[df["variant"].isin(train_variants)].copy()
        test = df[df["variant"] == heldout].copy()

        seed_arr = test["ladder_seed"].to_numpy()

        for lam in LAMBDAS:
            print(f"heldout={heldout} lambda={lam}", flush=True)

            fixed_model, fixed_vec = best_fixed_choice(test, lam)

            oracle_ch = oracle_choice(test, lam)
            oracle_stats = eval_choice_stats(test, oracle_ch, lam)

            routers = {
                name: factory(train, lam)
                for name, factory in LEARNED_METHODS.items()
            }

            learned_stats = {}
            for name, fn in routers.items():
                choice = fn(test)
                stats = eval_choice_stats(test, choice, lam)
                learned_stats[name] = stats

                delta_vec = stats["utility_vector"] - fixed_vec
                dmean, dlo, dhi = hierarchical_bootstrap_mean(delta_vec, seed_arr, rng)

                rows.append({
                    "lambda": lam,
                    "heldout": heldout,
                    "variant_label": META[heldout]["label"],
                    "method": name,
                    "fixed_model": fixed_model,
                    "utility": stats["utility"],
                    "top1": stats["top1"],
                    "latency": stats["latency"],
                    "delta_vs_fixed": dmean,
                    "delta_ci95_low": dlo,
                    "delta_ci95_high": dhi,
                    "stable_positive_vs_fixed": bool(dlo > 0),
                    "oracle_utility": oracle_stats["utility"],
                    "oracle_gap": oracle_stats["utility"] - stats["utility"],
                    "select_Tiny": stats["select_Tiny"],
                    "select_Small": stats["select_Small"],
                    "select_Medium": stats["select_Medium"],
                    "select_Full": stats["select_Full"],
                    "n": int(len(test)),
                })

            # Also record oracle delta over fixed.
            odelta = oracle_stats["utility_vector"] - fixed_vec
            odmean, odlo, odhi = hierarchical_bootstrap_mean(odelta, seed_arr, rng)

            rows.append({
                "lambda": lam,
                "heldout": heldout,
                "variant_label": META[heldout]["label"],
                "method": "oracle_multicapacity",
                "fixed_model": fixed_model,
                "utility": oracle_stats["utility"],
                "top1": oracle_stats["top1"],
                "latency": oracle_stats["latency"],
                "delta_vs_fixed": odmean,
                "delta_ci95_low": odlo,
                "delta_ci95_high": odhi,
                "stable_positive_vs_fixed": bool(odlo > 0),
                "oracle_utility": oracle_stats["utility"],
                "oracle_gap": 0.0,
                "select_Tiny": oracle_stats["select_Tiny"],
                "select_Small": oracle_stats["select_Small"],
                "select_Medium": oracle_stats["select_Medium"],
                "select_Full": oracle_stats["select_Full"],
                "n": int(len(test)),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v32C hierarchical bootstrap for multi-seed multi-capacity routing\n")
    lines.append("This validates v32B with a seed-aware paired bootstrap.")
    lines.append("Bootstrap resampling samples ladder seeds and examples within seeds, then estimates learned-router utility minus the best fixed-capacity utility.\n")

    lines.append("## Best learned router per heldout/cost\n")
    lines.append("| lambda | heldout | best learned | fixed | Δ vs fixed | 95% CI | stable positive? | oracle gap | selection Tiny/Small/Medium/Full |")
    lines.append("|---:|---|---|---|---:|---:|---|---:|---|")

    learned = out[out["method"].isin(LEARNED_METHODS.keys())].copy()

    for lam in LAMBDAS:
        for heldout in TEST_VARIANTS:
            sub = learned[(learned["lambda"] == lam) & (learned["heldout"] == heldout)]
            best = sub.sort_values("utility", ascending=False).iloc[0]
            stable = "yes" if best["stable_positive_vs_fixed"] else "no"
            sel = (
                f"{best['select_Tiny']:.2f}/"
                f"{best['select_Small']:.2f}/"
                f"{best['select_Medium']:.2f}/"
                f"{best['select_Full']:.2f}"
            )
            lines.append(
                f"| {lam:.2f} | {META[heldout]['label']} | `{best['method']}` | `{best['fixed_model']}` | "
                f"{best['delta_vs_fixed']:.6f} | [{best['delta_ci95_low']:.6f}, {best['delta_ci95_high']:.6f}] | "
                f"{stable} | {best['oracle_gap']:.6f} | {sel} |"
            )

    lines.append("\n## Pre-specified ridge expected-utility router\n")
    lines.append("| lambda | heldout | Δ vs fixed | 95% CI | stable positive? | oracle gap |")
    lines.append("|---:|---|---:|---:|---|---:|")

    ridge = out[out["method"] == "ridge_expected_utility"].copy()
    for _, r in ridge.iterrows():
        stable = "yes" if bool(r["stable_positive_vs_fixed"]) else "no"
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | {r['delta_vs_fixed']:.6f} | "
            f"[{r['delta_ci95_low']:.6f}, {r['delta_ci95_high']:.6f}] | {stable} | {r['oracle_gap']:.6f} |"
        )

    lines.append("\n## Oracle headroom\n")
    lines.append("| lambda | heldout | oracle Δ vs fixed | 95% CI | oracle latency | selection Tiny/Small/Medium/Full |")
    lines.append("|---:|---|---:|---:|---:|---|")

    oracle = out[out["method"] == "oracle_multicapacity"].copy()
    for _, r in oracle.iterrows():
        sel = (
            f"{r['select_Tiny']:.2f}/"
            f"{r['select_Small']:.2f}/"
            f"{r['select_Medium']:.2f}/"
            f"{r['select_Full']:.2f}"
        )
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | {r['delta_vs_fixed']:.6f} | "
            f"[{r['delta_ci95_low']:.6f}, {r['delta_ci95_high']:.6f}] | "
            f"{r['latency']:.6f} | {sel} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- Stable positive H=48 gains would support a robust learned multi-capacity routing claim.")
    lines.append("- H=72 should be treated as a remaining routing-signal problem if oracle headroom is large but learned gains are unstable.")
    lines.append("- The ridge expected-utility router is the cleanest pre-specified learned baseline; best-of-method results are useful but should be reported more cautiously.")

    OUT_MD.write_text('\\n'.join(lines) + '\\n', encoding="utf-8")

    print(OUT_CSV)
    print(OUT_MD)
    print()
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
