from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_PRED = Path(
    "reports/tables/protocol/pybullet_obstacle_rgb_encoder/"
    "v32_multiseed_multicapacity_routing/multiseed_multicapacity_predictions.csv"
)

ROUTER_CANDIDATES = [
    Path(
        "reports/tables/protocol/pybullet_obstacle_rgb_encoder/"
        "v39_margin_gated_router/v39_margin_gated_router_loso_results.csv"
    ),
    Path(
        "reports/tables/protocol/pybullet_obstacle_rgb_encoder/"
        "v39_margin_gated_router/v39_margin_gated_router_loso_summary.csv"
    ),
    Path(
        "reports/paper_assets_v39_margin_gated_router/"
        "tables/table_v39_margin_gated_router_loso.csv"
    ),
]

OUT = Path(
    "reports/tables/protocol/pybullet_obstacle_rgb_encoder/"
    "v47_compute_regime_map"
)
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PAPER = Path("reports/paper_assets_v47_compute_regime_map")
PTAB = PAPER / "tables"
PFIG = PAPER / "figures"
PTXT = PAPER / "text"
PTAB.mkdir(parents=True, exist_ok=True)
PFIG.mkdir(parents=True, exist_ok=True)
PTXT.mkdir(parents=True, exist_ok=True)

MODELS = ["Tiny", "Small", "Medium", "Full"]
LAT = {
    "Tiny": 0.186,
    "Small": 0.201,
    "Medium": 0.383,
    "Full": 1.000,
}
LAMBDAS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30]

VARIANT_LABELS = {
    "block3_h36_seed13": "3-block, H=36",
    "block3_h48_seed10": "3-block, H=48",
    "block3_h72_seed14": "3-block, H=72",
    "block2_h72_seed11": "2-block, H=72",
    "block2_v18_seed12": "2-block, V=1.8",
}

HARD_ORDER = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]


def model_correct_columns(df: pd.DataFrame, correctness_col: str = "tie_credit") -> pd.DataFrame:
    keys = ["ladder_seed", "variant", "group_id"]
    needed = keys + ["model_label", correctness_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prediction CSV: {missing}")

    x = df[needed].copy()
    x[correctness_col] = pd.to_numeric(x[correctness_col], errors="coerce").fillna(0.0)
    x["is_correct"] = x[correctness_col] > 0.5

    piv = (
        x.pivot_table(
            index=keys,
            columns="model_label",
            values="is_correct",
            aggfunc="first",
        )
        .reset_index()
    )

    for m in MODELS:
        if m not in piv.columns:
            piv[m] = False
        piv[m] = piv[m].astype(bool)

    piv["variant_label"] = piv["variant"].map(VARIANT_LABELS).fillna(piv["variant"])
    return piv


def classify_query(row: pd.Series) -> str:
    bits = "".join("1" if bool(row[m]) else "0" for m in MODELS)

    n = sum(bool(row[m]) for m in MODELS)
    if n == 4:
        return "all_correct"
    if n == 0:
        return "none_correct"

    tiny = bool(row["Tiny"])
    small = bool(row["Small"])
    medium = bool(row["Medium"])
    full = bool(row["Full"])

    if tiny and not small and not medium and not full:
        return "tiny_only"
    if full and not tiny and not small and not medium:
        return "full_only"
    if medium and not tiny and not small and not full:
        return "medium_only"
    if small and not tiny and not medium and not full:
        return "small_only"

    if (not tiny) and (small or medium or full):
        return "useful_compute"

    if tiny and (not full):
        return "anti_rescue"

    return "mixed"


def build_regime_rates(piv: pd.DataFrame) -> pd.DataFrame:
    z = piv.copy()
    z["pattern"] = z.apply(lambda r: "".join("1" if bool(r[m]) else "0" for m in MODELS), axis=1)
    z["regime_class"] = z.apply(classify_query, axis=1)

    rows = []
    for variant_label, sub in z.groupby("variant_label"):
        total = len(sub)
        row = {
            "variant_label": variant_label,
            "n_queries": total,
            "all_correct": float((sub["regime_class"] == "all_correct").mean()),
            "none_correct": float((sub["regime_class"] == "none_correct").mean()),
            "useful_compute": float((sub["regime_class"] == "useful_compute").mean()),
            "anti_rescue": float((sub["regime_class"] == "anti_rescue").mean()),
            "tiny_only": float((sub["regime_class"] == "tiny_only").mean()),
            "small_only": float((sub["regime_class"] == "small_only").mean()),
            "medium_only": float((sub["regime_class"] == "medium_only").mean()),
            "full_only": float((sub["regime_class"] == "full_only").mean()),
            "mixed": float((sub["regime_class"] == "mixed").mean()),
            "tiny_wrong_any_other_correct": float(((~sub["Tiny"]) & (sub[["Small", "Medium", "Full"]].any(axis=1))).mean()),
            "tiny_correct_full_wrong": float((sub["Tiny"] & (~sub["Full"])).mean()),
            "full_correct_tiny_wrong": float((sub["Full"] & (~sub["Tiny"])).mean()),
            "medium_correct_full_wrong": float((sub["Medium"] & (~sub["Full"])).mean()),
            "full_correct_medium_wrong": float((sub["Full"] & (~sub["Medium"])).mean()),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    if "variant_label" in out.columns:
        out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
        out = out.sort_values(["sort_key", "variant_label"]).drop(columns=["sort_key"])
    return out


def build_pattern_table(piv: pd.DataFrame) -> pd.DataFrame:
    z = piv.copy()
    z["pattern"] = z.apply(lambda r: "".join("1" if bool(r[m]) else "0" for m in MODELS), axis=1)
    rows = []
    for variant_label, sub in z.groupby("variant_label"):
        counts = sub["pattern"].value_counts()
        total = len(sub)
        for pattern, count in counts.items():
            rows.append({
                "variant_label": variant_label,
                "pattern_TinySmallMediumFull": pattern,
                "count": int(count),
                "frac": float(count / total),
            })
    out = pd.DataFrame(rows)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    out = out.sort_values(["sort_key", "variant_label", "frac"], ascending=[True, True, False]).drop(columns=["sort_key"])
    return out


def build_oracle_headroom(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["ladder_seed", "variant", "group_id"]

    needed = keys + ["model_label", "tie_credit"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for oracle analysis: {missing}")

    z = df[needed].copy()
    z["tie_credit"] = pd.to_numeric(z["tie_credit"], errors="coerce").fillna(0.0)
    z["variant_label"] = z["variant"].map(VARIANT_LABELS).fillna(z["variant"])
    z["relative_latency"] = z["model_label"].map(LAT)

    rows = []
    for variant_label, sub in z.groupby("variant_label"):
        for lam in LAMBDAS:
            t = sub.copy()
            t["utility"] = t["tie_credit"] - lam * t["relative_latency"]

            fixed = (
                t.groupby("model_label", as_index=False)
                .agg(mean_utility=("utility", "mean"), mean_top1=("tie_credit", "mean"))
            )
            fixed = fixed[fixed["model_label"].isin(MODELS)].copy()
            best_fixed = fixed.sort_values("mean_utility", ascending=False).iloc[0]

            q_oracle = (
                t.groupby(keys, as_index=False)
                .agg(oracle_utility=("utility", "max"))
            )
            oracle_mean = float(q_oracle["oracle_utility"].mean())

            rows.append({
                "variant_label": variant_label,
                "lambda": lam,
                "best_fixed_model": best_fixed["model_label"],
                "best_fixed_utility": float(best_fixed["mean_utility"]),
                "best_fixed_top1": float(best_fixed["mean_top1"]),
                "oracle_utility": oracle_mean,
                "oracle_headroom": oracle_mean - float(best_fixed["mean_utility"]),
            })

    out = pd.DataFrame(rows)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    out = out.sort_values(["sort_key", "variant_label", "lambda"]).drop(columns=["sort_key"])
    return out


def load_router_results() -> pd.DataFrame | None:
    for p in ROUTER_CANDIDATES:
        if p.exists():
            try:
                r = pd.read_csv(p)
                r["source_path"] = str(p)
                return r
            except Exception:
                continue
    return None


def build_recommendations(regime: pd.DataFrame, oracle: pd.DataFrame, router: pd.DataFrame | None) -> pd.DataFrame:
    rows = []

    # Mean oracle headroom over nonzero λ that matter for routing.
    oracle_focus = oracle[oracle["lambda"].isin([0.05, 0.10, 0.20, 0.30])].copy()
    oracle_mean = (
        oracle_focus.groupby("variant_label", as_index=False)
        .agg(mean_oracle_headroom=("oracle_headroom", "mean"))
    )

    merged = regime.merge(oracle_mean, on="variant_label", how="left")

    router_summary = None
    if router is not None:
        # Try to normalize common columns from v39 paper-assets CSV.
        r = router.copy()
        colmap = {}
        for c in r.columns:
            lc = c.lower()
            if lc in ["shift", "heldout", "held-out shift"]:
                colmap[c] = "variant_label"
            elif lc in ["stable?", "stable"]:
                colmap[c] = "stable"
            elif lc in ["delta", "∆", "delta_vs_train_fixed", "delta vs train fixed", "Δ"]:
                colmap[c] = "delta"
        r = r.rename(columns=colmap)

        if "variant_label" in r.columns:
            if "stable" in r.columns:
                st = r["stable"].astype(str).str.lower().isin(["yes", "true", "1", "\\textbf{yes}"])
                r["_stable_bool"] = st
            else:
                r["_stable_bool"] = False

            if "delta" in r.columns:
                r["_delta"] = pd.to_numeric(r["delta"], errors="coerce")
            elif "Delta" in r.columns:
                r["_delta"] = pd.to_numeric(r["Delta"], errors="coerce")
            else:
                r["_delta"] = np.nan

            router_summary = (
                r.groupby("variant_label", as_index=False)
                .agg(
                    stable_router_count=("_stable_bool", "sum"),
                    mean_router_delta=("_delta", "mean"),
                    max_router_delta=("_delta", "max"),
                )
            )

    if router_summary is not None:
        merged = merged.merge(router_summary, on="variant_label", how="left")
    else:
        merged["stable_router_count"] = np.nan
        merged["mean_router_delta"] = np.nan
        merged["max_router_delta"] = np.nan

    for _, r in merged.iterrows():
        useful = float(r["useful_compute"])
        anti = float(r["anti_rescue"])
        allc = float(r["all_correct"])
        none = float(r["none_correct"])
        oracle_head = float(r["mean_oracle_headroom"]) if pd.notna(r["mean_oracle_headroom"]) else np.nan
        stable_router_count = r.get("stable_router_count", np.nan)

        if oracle_head < 0.04 and useful < 0.10:
            recommendation = "fixed cheap/medium; limited routing headroom"
        elif useful > 0.10 and pd.notna(stable_router_count) and stable_router_count >= 3:
            recommendation = "margin-gated routing is deployable"
        elif useful > 0.10 and anti > 0.08:
            recommendation = "capacity selection needed; avoid monotone cascade"
        elif none > 0.08:
            recommendation = "improve representation/predictor; compute alone insufficient"
        else:
            recommendation = "use fixed best capacity; routing optional"

        rows.append({
            "variant_label": r["variant_label"],
            "all_correct": allc,
            "none_correct": none,
            "useful_compute": useful,
            "anti_rescue": anti,
            "mean_oracle_headroom": oracle_head,
            "stable_router_count": stable_router_count,
            "mean_router_delta": r.get("mean_router_delta", np.nan),
            "recommendation": recommendation,
        })

    out = pd.DataFrame(rows)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    out = out.sort_values(["sort_key", "variant_label"]).drop(columns=["sort_key"])
    return out


def plot_regime_bars(regime: pd.DataFrame, path: Path):
    cols = ["all_correct", "useful_compute", "anti_rescue", "none_correct", "mixed"]
    labels = {
        "all_correct": "All correct",
        "useful_compute": "Useful compute",
        "anti_rescue": "Anti-rescue",
        "none_correct": "None correct",
        "mixed": "Mixed/other",
    }

    hard = regime[regime["variant_label"].isin(HARD_ORDER)].copy()
    hard["sort_key"] = hard["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    hard = hard.sort_values("sort_key")

    x = np.arange(len(hard))
    bottom = np.zeros(len(hard))

    plt.figure(figsize=(7.2, 4.0))
    for col in cols:
        vals = hard[col].to_numpy()
        plt.bar(x, vals, bottom=bottom, label=labels[col])
        bottom += vals

    plt.xticks(x, hard["variant_label"], rotation=0)
    plt.ylabel("Fraction of queries")
    plt.ylim(0, 1.0)
    plt.title("Compute-regime map on hard OOD shifts")
    plt.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def write_tex_tables(regime: pd.DataFrame, rec: pd.DataFrame):
    hard_regime = regime[regime["variant_label"].isin(HARD_ORDER)].copy()
    hard_regime["sort_key"] = hard_regime["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    hard_regime = hard_regime.sort_values("sort_key").drop(columns=["sort_key"])

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\begin{tabular}{lrrrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Shift & All correct & None & Useful & Anti-rescue & Tiny$\rightarrow$Other & Tiny$\checkmark$/Full$\times$ \\")
    tex.append(r"\midrule")
    for _, r in hard_regime.iterrows():
        tex.append(
            f"{r['variant_label']} & "
            f"{r['all_correct']:.3f} & {r['none_correct']:.3f} & "
            f"{r['useful_compute']:.3f} & {r['anti_rescue']:.3f} & "
            f"{r['tiny_wrong_any_other_correct']:.3f} & {r['tiny_correct_full_wrong']:.3f} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\caption{Compute-regime map on hard OOD shifts. Useful compute denotes queries where Tiny is wrong and at least one larger capacity is correct. Anti-rescue denotes queries where Tiny is correct but Full is wrong, showing that additional compute can be harmful rather than merely unnecessary.}")
    tex.append(r"\label{tab:compute_regime_map}")
    tex.append(r"\end{table}")
    (PTAB / "table_v47_compute_regime_map.tex").write_text("\n".join(tex) + "\n")

    rec_hard = rec[rec["variant_label"].isin(HARD_ORDER)].copy()
    rec_hard["sort_key"] = rec_hard["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    rec_hard = rec_hard.sort_values("sort_key").drop(columns=["sort_key"])

    tex = []
    tex.append(r"\begin{table}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\resizebox{\linewidth}{!}{%")
    tex.append(r"\begin{tabular}{lrrrrl}")
    tex.append(r"\toprule")
    tex.append(r"Shift & Useful & Anti-rescue & Oracle gap & Stable router $\lambda$ & Recommendation \\")
    tex.append(r"\midrule")
    for _, r in rec_hard.iterrows():
        stable = ""
        if pd.notna(r["stable_router_count"]):
            stable = f"{int(r['stable_router_count'])}"
        else:
            stable = "--"
        rec_text = str(r["recommendation"]).replace("_", r"\_")
        tex.append(
            f"{r['variant_label']} & {r['useful_compute']:.3f} & {r['anti_rescue']:.3f} & "
            f"{r['mean_oracle_headroom']:.3f} & {stable} & {rec_text} \\\\"
        )
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}%")
    tex.append(r"}")
    tex.append(r"\caption{Compute-regime recommendations induced by SPSM. The table separates whether compute is useful, harmful, available only to an oracle, or recoverable by the current margin-gated router.}")
    tex.append(r"\label{tab:compute_regime_recommendations}")
    tex.append(r"\end{table}")
    (PTAB / "table_v47_compute_regime_recommendations.tex").write_text("\n".join(tex) + "\n")


def write_claim_text():
    txt = """
# v47 compute-regime-map claim

## Main idea

SPSM should not only report which capacity is best on average. It should identify the query-level compute regimes that make value-of-computation possible or impossible.

The compute-regime map decomposes each OOD shift into:
- all-correct queries, where extra compute is unnecessary;
- none-correct queries, where the current capacity ladder cannot solve the query;
- useful-compute queries, where Tiny fails but a larger capacity succeeds;
- anti-rescue queries, where Tiny succeeds but Full fails;
- mixed capacity-selection cases.

## Scientific claim

Additional predictive compute is not a monotone fallback. Its value depends on the query regime: compute can be unnecessary, useful, insufficient, or actively harmful. This turns value-of-computation from a scalar average into an interventional diagnostic map.

## How it strengthens the paper

The existing non-monotonic capacity result says that Full is not always best. The compute-regime map explains why: hard OOD shifts contain a mixture of rescuable, anti-rescuable, all-correct, and all-fail queries. H=48 is routable because useful-compute cases have an exploitable Tiny-margin signal. H=72 retains oracle headroom because useful compute exists, but anti-rescue and incomplete routing signals prevent fully closing the gap.

## Placement

This is a candidate main-paper conceptual figure/table. It can replace part of the current routing prose or become a short subsection after the margin-gated router.
"""
    (PTXT / "v47_compute_regime_map_claim.md").write_text(txt.strip() + "\n")


def write_markdown(regime, patterns, oracle, rec, router_path):
    lines = []
    lines.append("# SPSM v47 Compute Regime Map\n")
    lines.append("This analysis turns the multi-capacity predictions into query-level compute regimes. It asks not only which capacity is best on average, but whether additional predictive compute is useful, harmful, unnecessary, or insufficient for each OOD query.\n")

    lines.append("## Inputs\n")
    lines.append(f"- predictions: `{IN_PRED}`")
    if router_path:
        lines.append(f"- router results: `{router_path}`")
    else:
        lines.append("- router results: `not found`")
    lines.append("")

    lines.append("## Regime definitions\n")
    lines.append("- `all_correct`: Tiny, Small, Medium, and Full are all correct; extra compute is unnecessary.")
    lines.append("- `none_correct`: no capacity is correct; current compute ladder cannot solve the query.")
    lines.append("- `useful_compute`: Tiny is wrong but at least one larger capacity is correct.")
    lines.append("- `anti_rescue`: Tiny is correct but Full is wrong; a monotone fallback to Full would hurt.")
    lines.append("- `tiny_only`, `medium_only`, `full_only`: only that capacity is correct.")
    lines.append("- pattern order is `Tiny Small Medium Full`.\n")

    hard = regime[regime["variant_label"].isin(HARD_ORDER)].copy()
    lines.append("## Hard-OOD compute-regime rates\n")
    lines.append("| shift | n | all correct | none correct | useful compute | anti-rescue | Tiny wrong / other correct | Tiny correct / Full wrong | Full correct / Tiny wrong |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in hard.iterrows():
        lines.append(
            f"| {r['variant_label']} | {int(r['n_queries'])} | "
            f"{r['all_correct']:.3f} | {r['none_correct']:.3f} | "
            f"{r['useful_compute']:.3f} | {r['anti_rescue']:.3f} | "
            f"{r['tiny_wrong_any_other_correct']:.3f} | {r['tiny_correct_full_wrong']:.3f} | "
            f"{r['full_correct_tiny_wrong']:.3f} |"
        )

    lines.append("\n## Dominant correctness patterns\n")
    lines.append("| shift | pattern | count | frac |")
    lines.append("|---|---|---:|---:|")
    for _, r in patterns[patterns["variant_label"].isin(HARD_ORDER)].groupby("variant_label").head(6).iterrows():
        lines.append(
            f"| {r['variant_label']} | `{r['pattern_TinySmallMediumFull']}` | "
            f"{int(r['count'])} | {r['frac']:.3f} |"
        )

    lines.append("\n## Oracle headroom by λ\n")
    lines.append("| shift | λ | best fixed | fixed util | oracle util | oracle headroom |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for _, r in oracle[oracle["variant_label"].isin(HARD_ORDER)].iterrows():
        lines.append(
            f"| {r['variant_label']} | {r['lambda']:.2f} | `{r['best_fixed_model']}` | "
            f"{r['best_fixed_utility']:.3f} | {r['oracle_utility']:.3f} | {r['oracle_headroom']:.3f} |"
        )

    lines.append("\n## Compute recommendations\n")
    lines.append("| shift | useful | anti-rescue | mean oracle gap | stable router λ count | recommendation |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for _, r in rec[rec["variant_label"].isin(HARD_ORDER)].iterrows():
        stable = ""
        if pd.notna(r["stable_router_count"]):
            stable = str(int(r["stable_router_count"]))
        else:
            stable = "--"
        lines.append(
            f"| {r['variant_label']} | {r['useful_compute']:.3f} | {r['anti_rescue']:.3f} | "
            f"{r['mean_oracle_headroom']:.3f} | {stable} | {r['recommendation']} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- `useful_compute` is the deployable reason to leave Tiny.")
    lines.append("- `anti_rescue` shows why larger models are not a monotone fallback.")
    lines.append("- Oracle headroom measures available capacity-selection value; router gain measures how much of it is recoverable from observable signals.")
    lines.append("- The map converts value-of-computation from a scalar average into a regime-level diagnostic.")

    (OUT / "v47_compute_regime_map.md").write_text("\n".join(lines) + "\n")


def main():
    if not IN_PRED.exists():
        raise FileNotFoundError(IN_PRED)

    df = pd.read_csv(IN_PRED)
    df = df[df["model_label"].isin(MODELS)].copy()

    piv = model_correct_columns(df)
    regime = build_regime_rates(piv)
    patterns = build_pattern_table(piv)
    oracle = build_oracle_headroom(df)

    router = load_router_results()
    router_path = router["source_path"].iloc[0] if router is not None and "source_path" in router.columns else None
    rec = build_recommendations(regime, oracle, router)

    regime.to_csv(OUT / "v47_compute_regime_rates.csv", index=False)
    patterns.to_csv(OUT / "v47_correctness_patterns.csv", index=False)
    oracle.to_csv(OUT / "v47_oracle_headroom_by_lambda.csv", index=False)
    rec.to_csv(OUT / "v47_compute_recommendations.csv", index=False)

    # Paper assets copies.
    regime.to_csv(PTAB / "table_v47_compute_regime_map.csv", index=False)
    rec.to_csv(PTAB / "table_v47_compute_regime_recommendations.csv", index=False)

    plot_regime_bars(regime, FIG / "fig_v47_compute_regime_map_hard_ood.png")
    plot_regime_bars(regime, PFIG / "fig_v47_compute_regime_map_hard_ood.png")

    write_tex_tables(regime, rec)
    write_claim_text()
    write_markdown(regime, patterns, oracle, rec, router_path)

    manifest = ["# Paper assets v47 compute regime map\n", "## Tables\n"]
    for p in sorted(PTAB.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Figures\n")
    for p in sorted(PFIG.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Text\n")
    for p in sorted(PTXT.glob("*")):
        manifest.append(f"- `{p}`")
    (PAPER / "paper_assets_v47_manifest.md").write_text("\n".join(manifest) + "\n")

    print(OUT / "v47_compute_regime_map.md")
    print(PAPER / "paper_assets_v47_manifest.md")
    print((OUT / "v47_compute_regime_map.md").read_text())


if __name__ == "__main__":
    main()
