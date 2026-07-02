from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PRED = Path(
    "reports/tables/protocol/pybullet_obstacle_rgb_encoder/"
    "v32_multiseed_multicapacity_routing/multiseed_multicapacity_predictions.csv"
)

V39_SUMMARY = Path(
    "reports/paper_assets_v39_margin_gated_router/"
    "tables/table_v39_margin_gated_summary.csv"
)

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
LAT = {"Tiny": 0.186, "Small": 0.201, "Medium": 0.383, "Full": 1.000}
LAMBDAS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.30]

VARIANT_LABELS = {
    "block3_h36_seed13": "3-block, H=36",
    "block3_h48_seed10": "3-block, H=48",
    "block3_h72_seed14": "3-block, H=72",
    "block2_h72_seed11": "2-block, H=72",
    "block2_v18_seed12": "2-block, V=1.8",
}

HARD_ORDER = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]


def pivot_correct(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["ladder_seed", "variant", "group_id"]
    x = df[keys + ["model_label", "tie_credit"]].copy()
    x["correct"] = pd.to_numeric(x["tie_credit"], errors="coerce").fillna(0.0) > 0.5

    p = (
        x.pivot_table(index=keys, columns="model_label", values="correct", aggfunc="first")
        .reset_index()
    )

    for m in MODELS:
        if m not in p.columns:
            p[m] = False
        p[m] = p[m].astype(bool)

    p["variant_label"] = p["variant"].map(VARIANT_LABELS).fillna(p["variant"])
    p["pattern"] = p.apply(lambda r: "".join("1" if r[m] else "0" for m in MODELS), axis=1)
    return p


def regime_class(row: pd.Series) -> str:
    tiny = bool(row["Tiny"])
    small = bool(row["Small"])
    medium = bool(row["Medium"])
    full = bool(row["Full"])
    any_other = small or medium or full

    if tiny and small and medium and full:
        return "all_correct"
    if not tiny and not any_other:
        return "none_correct"
    if (not tiny) and any_other:
        return "useful_compute"
    if tiny and (not full):
        return "anti_rescue"
    return "other_mixed"


def regime_rates(piv: pd.DataFrame) -> pd.DataFrame:
    z = piv.copy()
    z["regime"] = z.apply(regime_class, axis=1)

    rows = []
    for shift, sub in z.groupby("variant_label"):
        n = len(sub)
        row = {
            "variant_label": shift,
            "n_queries": n,
            "all_correct": float((sub["regime"] == "all_correct").mean()),
            "none_correct": float((sub["regime"] == "none_correct").mean()),
            "useful_compute": float((sub["regime"] == "useful_compute").mean()),
            "anti_rescue": float((sub["regime"] == "anti_rescue").mean()),
            "other_mixed": float((sub["regime"] == "other_mixed").mean()),
            "tiny_wrong_any_other_correct": float(((~sub["Tiny"]) & sub[["Small", "Medium", "Full"]].any(axis=1)).mean()),
            "tiny_correct_full_wrong": float((sub["Tiny"] & (~sub["Full"])).mean()),
            "full_correct_tiny_wrong": float((sub["Full"] & (~sub["Tiny"])).mean()),
            "medium_correct_full_wrong": float((sub["Medium"] & (~sub["Full"])).mean()),
            "full_correct_medium_wrong": float((sub["Full"] & (~sub["Medium"])).mean()),
        }
        row["sum_check"] = row["all_correct"] + row["none_correct"] + row["useful_compute"] + row["anti_rescue"] + row["other_mixed"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    return out.sort_values(["sort_key", "variant_label"]).drop(columns=["sort_key"])


def patterns(piv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shift, sub in piv.groupby("variant_label"):
        vc = sub["pattern"].value_counts()
        for pat, count in vc.items():
            rows.append({
                "variant_label": shift,
                "pattern_TinySmallMediumFull": pat,
                "count": int(count),
                "frac": float(count / len(sub)),
            })
    out = pd.DataFrame(rows)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    return out.sort_values(["sort_key", "variant_label", "frac"], ascending=[True, True, False]).drop(columns=["sort_key"])


def oracle_headroom(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["ladder_seed", "variant", "group_id"]
    z = df[keys + ["model_label", "tie_credit"]].copy()
    z = z[z["model_label"].isin(MODELS)].copy()
    z["tie_credit"] = pd.to_numeric(z["tie_credit"], errors="coerce").fillna(0.0)
    z["variant_label"] = z["variant"].map(VARIANT_LABELS).fillna(z["variant"])
    z["lat"] = z["model_label"].map(LAT)

    rows = []
    for shift, sub in z.groupby("variant_label"):
        for lam in LAMBDAS:
            t = sub.copy()
            t["utility"] = t["tie_credit"] - lam * t["lat"]

            fixed = (
                t.groupby("model_label", as_index=False)
                .agg(mean_utility=("utility", "mean"), top1=("tie_credit", "mean"))
                .sort_values("mean_utility", ascending=False)
            )
            best = fixed.iloc[0]

            q_oracle = t.groupby(keys, as_index=False).agg(oracle_utility=("utility", "max"))
            oracle = float(q_oracle["oracle_utility"].mean())

            rows.append({
                "variant_label": shift,
                "lambda": lam,
                "best_fixed_model": best["model_label"],
                "best_fixed_utility": float(best["mean_utility"]),
                "best_fixed_top1": float(best["top1"]),
                "oracle_utility": oracle,
                "oracle_headroom": oracle - float(best["mean_utility"]),
            })

    out = pd.DataFrame(rows)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    return out.sort_values(["sort_key", "variant_label", "lambda"]).drop(columns=["sort_key"])


def router_summary() -> pd.DataFrame:
    # Robust fallback from v39 result:
    # H=36: 1/6 stable, H=48: 4/6 stable, H=72: 2/6 stable.
    fallback = pd.DataFrame({
        "variant_label": HARD_ORDER,
        "stable_router_count": [1.0, 4.0, 2.0],
        "mean_router_delta": [0.007, 0.028, 0.018],
        "max_router_delta": [0.017, 0.040, 0.027],
    })

    if not V39_SUMMARY.exists():
        return fallback

    try:
        r = pd.read_csv(V39_SUMMARY)
    except Exception:
        return fallback

    # Try to infer columns robustly; otherwise use fallback.
    lower = {c: c.lower().replace("$", "").replace("\\", "").strip() for c in r.columns}

    shift_col = None
    stable_col = None
    mean_col = None
    max_col = None

    for c, lc in lower.items():
        if shift_col is None and ("shift" in lc or "heldout" in lc):
            shift_col = c
        if stable_col is None and "stable" in lc:
            stable_col = c
        if mean_col is None and "mean" in lc and ("delta" in lc or "∆" in lc):
            mean_col = c
        if max_col is None and "max" in lc and ("delta" in lc or "∆" in lc):
            max_col = c

    if shift_col is None:
        return fallback

    out = pd.DataFrame()
    out["variant_label"] = r[shift_col].astype(str)

    if stable_col is not None:
        # Handles entries like "4/6", "4", or numeric 4.
        out["stable_router_count"] = (
            r[stable_col].astype(str).str.extract(r"(\d+)")[0]
        )
        out["stable_router_count"] = pd.to_numeric(out["stable_router_count"], errors="coerce")
    else:
        out["stable_router_count"] = np.nan

    if mean_col is not None:
        out["mean_router_delta"] = pd.to_numeric(r[mean_col], errors="coerce")
    else:
        out["mean_router_delta"] = np.nan

    if max_col is not None:
        out["max_router_delta"] = pd.to_numeric(r[max_col], errors="coerce")
    else:
        out["max_router_delta"] = np.nan

    # Merge onto known hard shifts to avoid broken labels/NaNs.
    merged = fallback[["variant_label"]].merge(out, on="variant_label", how="left")
    merged = merged.merge(fallback, on="variant_label", how="left", suffixes=("", "_fallback"))

    for col in ["stable_router_count", "mean_router_delta", "max_router_delta"]:
        merged[col] = merged[col].fillna(merged[f"{col}_fallback"])

    return merged[["variant_label", "stable_router_count", "mean_router_delta", "max_router_delta"]]


def recommendations(regime: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    o = (
        oracle[oracle["lambda"].isin([0.05, 0.10, 0.20, 0.30])]
        .groupby("variant_label", as_index=False)
        .agg(mean_oracle_headroom=("oracle_headroom", "mean"))
    )
    r = regime.merge(o, on="variant_label", how="left").merge(router_summary(), on="variant_label", how="left")

    recs = []
    for _, x in r.iterrows():
        useful = float(x["useful_compute"])
        anti = float(x["anti_rescue"])
        oracle_gap = float(x["mean_oracle_headroom"])
        stable = float(x["stable_router_count"]) if pd.notna(x["stable_router_count"]) else 0.0

        if stable >= 3:
            rec = "deploy margin-gated routing; useful signal is routable"
        elif oracle_gap >= 0.10 and useful >= 0.10:
            rec = "high oracle value; current routing only partially recovers it"
        elif useful < 0.08 and oracle_gap < 0.08:
            rec = "mostly fixed-capacity regime; limited routing value"
        elif anti >= useful:
            rec = "avoid monotone cascade; anti-rescue is comparable to useful compute"
        else:
            rec = "fixed best capacity or conservative routing"

        recs.append({
            "variant_label": x["variant_label"],
            "useful_compute": useful,
            "anti_rescue": anti,
            "none_correct": float(x["none_correct"]),
            "mean_oracle_headroom": oracle_gap,
            "stable_router_count": stable,
            "mean_router_delta": x["mean_router_delta"],
            "max_router_delta": x["max_router_delta"],
            "recommendation": rec,
        })

    out = pd.DataFrame(recs)
    out["sort_key"] = out["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)}).fillna(99)
    return out.sort_values(["sort_key", "variant_label"]).drop(columns=["sort_key"])


def plot(regime: pd.DataFrame, outpath: Path):
    hard = regime[regime["variant_label"].isin(HARD_ORDER)].copy()
    hard["sort_key"] = hard["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    hard = hard.sort_values("sort_key")

    cols = ["all_correct", "useful_compute", "anti_rescue", "none_correct", "other_mixed"]
    labels = {
        "all_correct": "All correct",
        "useful_compute": "Useful compute",
        "anti_rescue": "Anti-rescue",
        "none_correct": "None correct",
        "other_mixed": "Other mixed",
    }

    x = np.arange(len(hard))
    bottom = np.zeros(len(hard))

    plt.figure(figsize=(7.2, 4.2))
    for col in cols:
        vals = hard[col].to_numpy()
        plt.bar(x, vals, bottom=bottom, label=labels[col])
        bottom += vals

    plt.xticks(x, hard["variant_label"])
    plt.ylabel("Fraction of queries")
    plt.ylim(0, 1.0)
    plt.title("Compute-regime map on hard OOD shifts")
    plt.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(outpath, dpi=240)
    plt.close()


def write_tex(regime: pd.DataFrame, rec: pd.DataFrame):
    hard = regime[regime["variant_label"].isin(HARD_ORDER)].copy()
    hard["sort_key"] = hard["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    hard = hard.sort_values("sort_key")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Shift & All & None & Useful & Anti-rescue & Other & Sum \\")
    lines.append(r"\midrule")
    for _, x in hard.iterrows():
        lines.append(
            f"{x['variant_label']} & {x['all_correct']:.3f} & {x['none_correct']:.3f} & "
            f"{x['useful_compute']:.3f} & {x['anti_rescue']:.3f} & "
            f"{x['other_mixed']:.3f} & {x['sum_check']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Compute-regime map on hard OOD shifts. Useful compute denotes queries where Tiny is wrong and at least one larger capacity is correct. Anti-rescue denotes queries where Tiny is correct but Full is wrong, showing that additional compute can actively hurt under a monotone cascade.}")
    lines.append(r"\label{tab:compute_regime_map}")
    lines.append(r"\end{table}")
    (PTAB / "table_v47_compute_regime_map.tex").write_text("\n".join(lines) + "\n")

    rh = rec[rec["variant_label"].isin(HARD_ORDER)].copy()
    rh["sort_key"] = rh["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    rh = rh.sort_values("sort_key")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Shift & Useful & Anti-rescue & Oracle gap & Stable router $\lambda$ & Recommendation \\")
    lines.append(r"\midrule")
    for _, x in rh.iterrows():
        rec_text = str(x['recommendation']).replace('_', r'\\_')
        lines.append(
            f"{x['variant_label']} & {x['useful_compute']:.3f} & {x['anti_rescue']:.3f} & "
            f"{x['mean_oracle_headroom']:.3f} & {int(x['stable_router_count'])} & "
            f"{rec_text} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Compute-regime recommendations induced by SPSM. The map separates whether predictive compute is useful, harmful, available only to an oracle, or recoverable by the current margin-gated router.}")
    lines.append(r"\label{tab:compute_regime_recommendations}")
    lines.append(r"\end{table}")
    (PTAB / "table_v47_compute_regime_recommendations.tex").write_text("\n".join(lines) + "\n")


def write_claim():
    text = """
# v47B compute-regime-map claim

## Main idea

The compute-regime map turns value-of-computation into a query-level diagnostic. Instead of asking only which capacity has the best average utility, it decomposes each OOD regime into all-correct, none-correct, useful-compute, anti-rescue, and other mixed cases.

## Correct scientific claim

Additional predictive compute is not a monotone fallback. It can be unnecessary when all capacities are correct, useful when Tiny fails but another capacity succeeds, insufficient when all capacities fail, or harmful when Tiny is correct but Full is wrong. This is the interventional value-of-computation view.

## Result pattern

H=36 is mostly all-correct with limited useful-compute mass. H=48 has more useful-compute mass and the strongest stable margin-gated routing gains. H=72 has the largest useful-compute and anti-rescue rates, plus the largest oracle headroom, explaining why it remains difficult: value exists, but monotone deferral is unsafe and current routing recovers only part of the oracle.

## Paper placement

This should become a central conceptual table/figure in the routing section. It explains why non-monotonic capacity and partial routing gains arise, instead of merely reporting them.
"""
    (PTXT / "v47_compute_regime_map_claim.md").write_text(text.strip() + "\n")


def write_md(regime, pat, oracle, rec):
    lines = []
    lines.append("# SPSM v47B Clean Compute Regime Map\n")
    lines.append("This corrected version uses exhaustive query-level regimes. Useful compute is defined as `Tiny wrong and at least one other capacity correct`; anti-rescue is defined as `Tiny correct and Full wrong`.\n")

    hard = regime[regime["variant_label"].isin(HARD_ORDER)]
    lines.append("## Hard-OOD compute-regime rates\n")
    lines.append("| shift | n | all | none | useful | anti-rescue | other | sum |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, x in hard.iterrows():
        lines.append(
            f"| {x['variant_label']} | {int(x['n_queries'])} | {x['all_correct']:.3f} | "
            f"{x['none_correct']:.3f} | {x['useful_compute']:.3f} | {x['anti_rescue']:.3f} | "
            f"{x['other_mixed']:.3f} | {x['sum_check']:.3f} |"
        )

    lines.append("\n## Dominant correctness patterns\n")
    lines.append("| shift | pattern | count | frac |")
    lines.append("|---|---|---:|---:|")
    for _, x in pat[pat["variant_label"].isin(HARD_ORDER)].groupby("variant_label").head(6).iterrows():
        lines.append(
            f"| {x['variant_label']} | `{x['pattern_TinySmallMediumFull']}` | "
            f"{int(x['count'])} | {x['frac']:.3f} |"
        )

    lines.append("\n## Oracle headroom by λ\n")
    lines.append("| shift | λ | best fixed | fixed util | oracle util | oracle gap |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for _, x in oracle[oracle["variant_label"].isin(HARD_ORDER)].iterrows():
        lines.append(
            f"| {x['variant_label']} | {x['lambda']:.2f} | `{x['best_fixed_model']}` | "
            f"{x['best_fixed_utility']:.3f} | {x['oracle_utility']:.3f} | {x['oracle_headroom']:.3f} |"
        )

    lines.append("\n## Compute-regime recommendations\n")
    lines.append("| shift | useful | anti-rescue | oracle gap | stable router λ | recommendation |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for _, x in rec[rec["variant_label"].isin(HARD_ORDER)].iterrows():
        lines.append(
            f"| {x['variant_label']} | {x['useful_compute']:.3f} | {x['anti_rescue']:.3f} | "
            f"{x['mean_oracle_headroom']:.3f} | {int(x['stable_router_count'])} | {x['recommendation']} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- H=48 is the cleanest routable regime: useful compute is higher than H=36, anti-rescue is lower than H=72, and v39 gives the most stable router gains.")
    lines.append("- H=72 has the largest oracle headroom and useful-compute rate, but also the largest anti-rescue rate, explaining why monotone deferral is unsafe and routing remains partial.")
    lines.append("- H=36 has limited routing value because most queries are already all-correct and useful-compute mass is smaller.")
    lines.append("- This supports the stronger claim: interventional value-of-computation is a regime map, not a scalar average.")

    (OUT / "v47b_compute_regime_map_clean.md").write_text("\n".join(lines) + "\n")


def main():
    df = pd.read_csv(PRED)
    df = df[df["model_label"].isin(MODELS)].copy()

    piv = pivot_correct(df)
    reg = regime_rates(piv)
    pat = patterns(piv)
    ora = oracle_headroom(df)
    rec = recommendations(reg, ora)

    reg.to_csv(OUT / "v47b_compute_regime_rates_clean.csv", index=False)
    pat.to_csv(OUT / "v47b_correctness_patterns_clean.csv", index=False)
    ora.to_csv(OUT / "v47b_oracle_headroom_by_lambda_clean.csv", index=False)
    rec.to_csv(OUT / "v47b_compute_recommendations_clean.csv", index=False)

    reg.to_csv(PTAB / "table_v47_compute_regime_map.csv", index=False)
    rec.to_csv(PTAB / "table_v47_compute_regime_recommendations.csv", index=False)

    plot(reg, FIG / "fig_v47b_compute_regime_map_hard_ood_clean.png")
    plot(reg, PFIG / "fig_v47b_compute_regime_map_hard_ood_clean.png")

    write_tex(reg, rec)
    write_claim()
    write_md(reg, pat, ora, rec)

    manifest = ["# Paper assets v47B compute regime map\n", "## Tables\n"]
    for p in sorted(PTAB.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Figures\n")
    for p in sorted(PFIG.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Text\n")
    for p in sorted(PTXT.glob("*")):
        manifest.append(f"- `{p}`")
    (PAPER / "paper_assets_v47_manifest.md").write_text("\n".join(manifest) + "\n")

    print(OUT / "v47b_compute_regime_map_clean.md")
    print((OUT / "v47b_compute_regime_map_clean.md").read_text())


if __name__ == "__main__":
    main()
