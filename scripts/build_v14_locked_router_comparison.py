from __future__ import annotations

from pathlib import Path
import pandas as pd

SRC = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v13_latent_state_router/latent_state_router_loso_summary.csv")

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v14_locked_router_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "locked_router_comparison.csv"
OUT_MD = OUT_DIR / "locked_router_comparison.md"


def get_util(df: pd.DataFrame, lam: float, heldout: str, router: str, feature_set: str) -> float:
    sub = df[
        (df["lambda"] == lam)
        & (df["heldout"] == heldout)
        & (df["router"] == router)
        & (df["feature_set"] == feature_set)
    ]
    if len(sub) != 1:
        raise ValueError((lam, heldout, router, feature_set, len(sub)))
    return float(sub.iloc[0]["util"])


def get_route(df: pd.DataFrame, lam: float, heldout: str, router: str, feature_set: str) -> float:
    sub = df[
        (df["lambda"] == lam)
        & (df["heldout"] == heldout)
        & (df["router"] == router)
        & (df["feature_set"] == feature_set)
    ]
    if len(sub) != 1:
        raise ValueError((lam, heldout, router, feature_set, len(sub)))
    return float(sub.iloc[0]["route"])


def main():
    df = pd.read_csv(SRC)

    rows = []
    keys = df[["lambda", "heldout", "variant_label"]].drop_duplicates()

    for _, k in keys.iterrows():
        lam = float(k["lambda"])
        heldout = str(k["heldout"])
        label = str(k["variant_label"])

        base = df[(df["lambda"] == lam) & (df["heldout"] == heldout)].iloc[0]

        row = {
            "lambda": lam,
            "heldout": heldout,
            "variant_label": label,

            "cheap": float(base["cheap"]),
            "full": float(base["full"]),
            "oracle": float(base["oracle"]),
            "conf": float(base["conf"]),
            "context_cls": float(base["context_cls"]),

            "knn_context": get_util(df, lam, heldout, "knn_gain", "context"),
            "rh_context": get_util(df, lam, heldout, "rescue_harm", "context"),

            "knn_context_latent": get_util(df, lam, heldout, "knn_gain", "context_latent"),
            "rh_context_latent": get_util(df, lam, heldout, "rescue_harm", "context_latent"),

            "route_knn_context": get_route(df, lam, heldout, "knn_gain", "context"),
            "route_rh_context": get_route(df, lam, heldout, "rescue_harm", "context"),
            "route_knn_context_latent": get_route(df, lam, heldout, "knn_gain", "context_latent"),
            "route_rh_context_latent": get_route(df, lam, heldout, "rescue_harm", "context_latent"),
        }

        fixed_methods = [
            "cheap",
            "full",
            "conf",
            "context_cls",
            "knn_context",
            "rh_context",
            "knn_context_latent",
            "rh_context_latent",
        ]
        best = max(fixed_methods, key=lambda c: row[c])
        row["best_fixed"] = best
        row["best_fixed_util"] = row[best]

        row["delta_knn_latent_vs_context"] = row["knn_context_latent"] - row["knn_context"]
        row["delta_rh_latent_vs_context"] = row["rh_context_latent"] - row["rh_context"]
        row["delta_best_latent_vs_best_context"] = max(row["knn_context_latent"], row["rh_context_latent"]) - max(row["knn_context"], row["rh_context"])

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v14 locked router comparison\n")
    lines.append("This table compares fixed routing families from v13 without selecting arbitrary feature sets per cell.")
    lines.append("The main question is whether adding observable latent-state features to context improves value-of-computation routing.\n")

    lines.append("## Utility comparison\n")
    lines.append("| lambda | heldout | cheap | full | oracle | conf | context-cls | KNN context | RH context | KNN context+latent | RH context+latent | best fixed |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for _, r in out.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['cheap']:.6f} | {r['full']:.6f} | {r['oracle']:.6f} | "
            f"{r['conf']:.6f} | {r['context_cls']:.6f} | "
            f"{r['knn_context']:.6f} | {r['rh_context']:.6f} | "
            f"{r['knn_context_latent']:.6f} | {r['rh_context_latent']:.6f} | "
            f"`{r['best_fixed']}` |"
        )

    lines.append("\n## Latent-state improvement over context-only routers\n")
    lines.append("| lambda | heldout | Δ KNN latent-context | Δ RH latent-context | Δ best latent-best context |")
    lines.append("|---:|---|---:|---:|---:|")

    for _, r in out.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['delta_knn_latent_vs_context']:.6f} | "
            f"{r['delta_rh_latent_vs_context']:.6f} | "
            f"{r['delta_best_latent_vs_best_context']:.6f} |"
        )

    lines.append("\n## Route-rate comparison\n")
    lines.append("| lambda | heldout | KNN context | RH context | KNN context+latent | RH context+latent |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    for _, r in out.iterrows():
        lines.append(
            f"| {r['lambda']:.2f} | {r['variant_label']} | "
            f"{r['route_knn_context']:.6f} | {r['route_rh_context']:.6f} | "
            f"{r['route_knn_context_latent']:.6f} | {r['route_rh_context_latent']:.6f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- Context-only routing is already strong on hard long-horizon shifts.")
    lines.append("- Adding observable latent-state features is most useful on `3-block, H=72`, especially at moderate compute costs.")
    lines.append("- Candidate-mined geometry is not the right signal; observable latent state complexity is more promising.")
    lines.append("- The paper should frame the strongest result as latent-state-aware value-of-computation, not merely geometry-aware routing.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
