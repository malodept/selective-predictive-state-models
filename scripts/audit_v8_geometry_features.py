from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")

from train_v8_geometry_aware_gain_router import (
    prepare,
    add_meta_features,
    add_optional_group_geometry,
    context_cols,
    geometry_cols,
    TRAIN_VARIANTS,
    TEST_VARIANTS,
    ALL_VARIANTS,
    LAMBDAS,
    META,
)

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v8_geometry_aware_gain_router")
OUT = OUT_DIR / "geometry_feature_audit.md"
OUT_CSV = OUT_DIR / "geometry_feature_signal_table.csv"


def safe_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 5:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def rank_auc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    ok = np.isfinite(scores)
    scores = scores[ok]
    labels = labels[ok]
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Average-rank AUC without scipy.
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)

    # Tie correction by averaging ranks for tied scores.
    unique, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for j, c in enumerate(counts):
            if c > 1:
                idx = inv == j
                ranks[idx] = ranks[idx].mean()

    rank_sum_pos = ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def summarize_feature_signal(m, cols, variant, lam):
    if variant == "ALL":
        f = m.copy()
    else:
        f = m[m["variant"] == variant].copy()

    y_gain = f["gain"].to_numpy(float)
    y_route = ((f["gain"].to_numpy(float) - lam) > 0).astype(int)

    rows = []
    for c in cols:
        x = f[c].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        rows.append({
            "variant": variant,
            "variant_label": "ALL" if variant == "ALL" else META[variant]["label"],
            "lambda": lam,
            "feature": c,
            "std": float(np.std(x)),
            "corr_gain": safe_corr(x, y_gain),
            "corr_route": safe_corr(x, y_route),
            "auc_route": rank_auc(x, y_route),
            "mean_route1": float(np.mean(x[y_route == 1])) if np.any(y_route == 1) else np.nan,
            "mean_route0": float(np.mean(x[y_route == 0])) if np.any(y_route == 0) else np.nan,
            "n": len(f),
            "route_rate": float(np.mean(y_route)),
        })
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    m = prepare()
    m = add_meta_features(m)
    m = add_optional_group_geometry(m)

    ccols = context_cols(m)
    gcols = geometry_cols(m)
    added = [c for c in gcols if c not in ccols]

    rows = []
    for lam in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
        for variant in ["ALL"] + ALL_VARIANTS:
            rows.extend(summarize_feature_signal(m, gcols, variant, lam))

    table = pd.DataFrame(rows)
    table.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v8 geometry feature audit\n")
    lines.append("This audit checks whether the added geometry/OOD features actually contain signal for the value-of-computation target.")
    lines.append("The target is `route = 1[full_correct - small_correct > lambda]`.\n")

    lines.append("## Dataset\n")
    lines.append(f"- rows after small/full merge: `{len(m)}`")
    lines.append(f"- context features: `{len(ccols)}`")
    lines.append(f"- added geometry features: `{len(added)}`")
    lines.append("")
    lines.append("Added geometry features:")
    for c in added:
        lines.append(f"- `{c}`")

    lines.append("\n## Feature standard deviations by variant\n")
    std_rows = []
    for variant in ALL_VARIANTS:
        f = m[m["variant"] == variant]
        for c in added:
            std_rows.append((META[variant]["label"], c, float(f[c].std())))
    std_df = pd.DataFrame(std_rows, columns=["variant", "feature", "std"])

    for variant in [META[v]["label"] for v in ALL_VARIANTS]:
        lines.append(f"\n### {variant}\n")
        sub = std_df[std_df["variant"] == variant].sort_values("std")
        lines.append("| feature | std |")
        lines.append("|---|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| `{r['feature']}` | {r['std']:.6g} |")

    lines.append("\n## Top feature signal at lambda = 0.10\n")
    lam = 0.10
    for variant in ["ALL"] + TEST_VARIANTS:
        sub = table[(table["lambda"] == lam) & (table["variant"] == variant)].copy()
        sub["auc_distance"] = (sub["auc_route"] - 0.5).abs()
        sub["corr_distance"] = sub["corr_route"].abs()
        sub = sub.sort_values(["auc_distance", "corr_distance"], ascending=False).head(15)

        label = "ALL" if variant == "ALL" else META[variant]["label"]
        lines.append(f"\n### {label}\n")
        lines.append("| feature | std | corr(gain) | corr(route) | AUC(route) | mean route=1 | mean route=0 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| `{r['feature']}` | {r['std']:.4g} | "
                f"{r['corr_gain']:.4f} | {r['corr_route']:.4f} | {r['auc_route']:.4f} | "
                f"{r['mean_route1']:.4g} | {r['mean_route0']:.4g} |"
            )

    lines.append("\n## Interpretation checklist\n")
    lines.append("- If most added geometry features have near-zero std within a variant, they cannot help per-instance routing.")
    lines.append("- If AUC(route) is close to 0.5, the feature is not predictive of oracle positive-gain routing.")
    lines.append("- If context features dominate added geometry features, the v8 router should not be expected to beat v5.8.")
    lines.append("- If some local geometry feature has strong AUC but the MLP fails, the next step is a simpler calibrated linear/tree router.")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)
    print(OUT.read_text())


if __name__ == "__main__":
    main()
