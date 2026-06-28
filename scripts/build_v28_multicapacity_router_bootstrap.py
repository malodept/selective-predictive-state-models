from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from build_v27_learned_multicapacity_router import (
    prepare,
    feature_cols,
    utilities,
    eval_choice,
    oracle_choice,
    train_expected_utility_router,
    train_oracle_label_router,
    LAMBDAS,
    MODELS,
    TEST_VARIANTS,
    ALL_VARIANTS,
    META,
)

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v28_multicapacity_router_bootstrap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "multicapacity_router_bootstrap_summary.csv"
OUT_MD = OUT_DIR / "multicapacity_router_bootstrap_summary.md"

BOOT = 3000
SEED = 123

LEARNED_METHODS = {
    "ridge_expected_utility": lambda train, lam: train_expected_utility_router(train, lam, "ridge"),
    "rf_expected_utility": lambda train, lam: train_expected_utility_router(train, lam, "rf"),
    "logreg_oracle_label": lambda train, lam: train_oracle_label_router(train, lam, "logreg"),
    "rf_oracle_label": lambda train, lam: train_oracle_label_router(train, lam, "rf"),
}


def utility_vector(df: pd.DataFrame, choice: np.ndarray, lam: float) -> np.ndarray:
    u = utilities(df, lam)
    vals = np.array([u.iloc[i][choice[i]] for i in range(len(choice))], dtype=np.float64)
    return vals


def ci_mean(x: np.ndarray, rng: np.random.Generator):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    means = np.empty(BOOT, dtype=np.float64)
    for b in range(BOOT):
        idx = rng.integers(0, n, size=n)
        means[b] = x[idx].mean()
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_delta_ci(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    return ci_mean(np.asarray(a) - np.asarray(b), rng)


def best_fixed_choice(test: pd.DataFrame, lam: float):
    best_m = None
    best_u = -1e9
    best_vec = None
    for m in MODELS:
        choice = np.array([m] * len(test), dtype=object)
        vec = utility_vector(test, choice, lam)
        mu = float(vec.mean())
        if mu > best_u:
            best_u = mu
            best_m = m
            best_vec = vec
    return best_m, best_vec


def main():
    rng = np.random.default_rng(SEED)
    df = prepare()

    rows = []

    for heldout in TEST_VARIANTS:
        train_variants = [v for v in ALL_VARIANTS if v != heldout]
        train = df[df["variant"].isin(train_variants)].copy()
        test = df[df["variant"] == heldout].copy()

        for lam in LAMBDAS:
            best_fixed, fixed_vec = best_fixed_choice(test, lam)

            oracle_ch = oracle_choice(test, lam)
            oracle_vec = utility_vector(test, oracle_ch, lam)

            fixed_mean, fixed_lo, fixed_hi = ci_mean(fixed_vec, rng)
            oracle_mean, oracle_lo, oracle_hi = ci_mean(oracle_vec, rng)

            rows.append({
                "lambda": lam,
                "heldout": heldout,
                "variant_label": META[heldout]["label"],
                "method": f"best_fixed_{best_fixed}",
                "utility": fixed_mean,
                "ci95_low": fixed_lo,
                "ci95_high": fixed_hi,
                "delta_vs_fixed": 0.0,
                "delta_ci95_low": 0.0,
                "delta_ci95_high": 0.0,
                "stable_positive_vs_fixed": False,
                "oracle_gap": oracle_mean - fixed_mean,
                "n": len(test),
            })

            rows.append({
                "lambda": lam,
                "heldout": heldout,
                "variant_label": META[heldout]["label"],
                "method": "oracle_multicapacity",
                "utility": oracle_mean,
                "ci95_low": oracle_lo,
                "ci95_high": oracle_hi,
                "delta_vs_fixed": oracle_mean - fixed_mean,
                "delta_ci95_low": np.nan,
                "delta_ci95_high": np.nan,
                "stable_positive_vs_fixed": True,
                "oracle_gap": 0.0,
                "n": len(test),
            })

            for name, factory in LEARNED_METHODS.items():
                fn = factory(train, lam)
                choice = fn(test)
                vec = utility_vector(test, choice, lam)

                mean, lo, hi = ci_mean(vec, rng)
                dmean, dlo, dhi = paired_delta_ci(vec, fixed_vec, rng)

                rows.append({
                    "lambda": lam,
                    "heldout": heldout,
                    "variant_label": META[heldout]["label"],
                    "method": name,
                    "utility": mean,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "delta_vs_fixed": dmean,
                    "delta_ci95_low": dlo,
                    "delta_ci95_high": dhi,
                    "stable_positive_vs_fixed": bool(dlo > 0),
                    "oracle_gap": oracle_mean - mean,
                    "n": len(test),
                })

            print(f"heldout={heldout} lambda={lam}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v28 bootstrap validation of learned multi-capacity routing\n")
    lines.append("This validates v27 with paired bootstrap confidence intervals over held-out hard-OOD test examples.")
    lines.append("The key quantity is learned-router utility minus the best fixed-capacity utility for that held-out shift and cost.\n")

    lines.append("## Best learned method vs best fixed\n")
    lines.append("| lambda | heldout | best learned | utility | Δ vs fixed | 95% CI for Δ | stable positive? | oracle gap |")
    lines.append("|---:|---|---|---:|---:|---:|---|---:|")

    learned = out[out["method"].isin(LEARNED_METHODS.keys())].copy()

    for lam in LAMBDAS:
        for heldout in TEST_VARIANTS:
            sub = learned[(learned["lambda"] == lam) & (learned["heldout"] == heldout)]
            best = sub.sort_values("utility", ascending=False).iloc[0]
            stable = "yes" if best["stable_positive_vs_fixed"] else "no"
            lines.append(
                f"| {lam:.2f} | {META[heldout]['label']} | `{best['method']}` | "
                f"{best['utility']:.6f} | {best['delta_vs_fixed']:.6f} | "
                f"[{best['delta_ci95_low']:.6f}, {best['delta_ci95_high']:.6f}] | "
                f"{stable} | {best['oracle_gap']:.6f} |"
            )

    lines.append("\n## All methods with bootstrap intervals\n")
    lines.append("| lambda | heldout | method | utility | 95% CI | Δ vs fixed | Δ 95% CI | stable positive? |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---|")

    show = out[out["method"].isin(list(LEARNED_METHODS.keys()) + ["oracle_multicapacity"])].copy()
    for _, r in show.iterrows():
        stable = "yes" if bool(r["stable_positive_vs_fixed"]) else "no"
        dci = (
            "—"
            if pd.isna(r["delta_ci95_low"])
            else f"[{r['delta_ci95_low']:.6f}, {r['delta_ci95_high']:.6f}]"
        )
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | `{r['method']}` | "
            f"{r['utility']:.6f} | [{r['ci95_low']:.6f}, {r['ci95_high']:.6f}] | "
            f"{r['delta_vs_fixed']:.6f} | {dci} | {stable} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- Stable positive gains on H=48 would make learned multi-capacity routing a defensible paper result.")
    lines.append("- Small or unstable gains on H=72 should be described as remaining headroom rather than solved routing.")
    lines.append("- If H=36 remains negative, that is useful evidence that routing should be shift-aware and not blindly applied.")

    OUT_MD.write_text('\\n'.join(lines) + '\\n', encoding="utf-8")

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
