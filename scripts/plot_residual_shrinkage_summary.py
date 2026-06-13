from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_pm(s: str) -> tuple[float, float]:
    a, b = s.split("±")
    return float(a.strip()), float(b.strip())


def main() -> None:
    summary_path = Path("reports/tables/protocol/residual_shrinkage_summary.csv")
    boot_path = Path("reports/tables/protocol/residual_shrinkage_trajectory_bootstrap/cheap_residual_trajectory_effects.csv")

    out_dir = Path("reports/figures/protocol/residual_shrinkage")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)

    keep = df[
        (df["split"] == "test")
        & (df["model"] == "cheap_residual")
        & (df["rule"].isin(["raw_alpha_1", "alpha_selected_on_val"]))
    ].copy()

    labels = {
        "raw_alpha_1": "raw residual α=1",
        "alpha_selected_on_val": "val-calibrated α",
    }

    keep["label"] = keep["rule"].map(labels)
    vals = [parse_pm(x) for x in keep["improvement_vs_identity"]]
    means = [v[0] for v in vals]
    stds = [v[1] for v in vals]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(keep["label"], means, yerr=stds, capsize=5)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_ylabel("improvement over identity")
    ax.set_title("Residual shrinkage restores OOD latent prediction value")
    ax.tick_params(axis="x", rotation=15)

    for i, v in enumerate(means):
        ax.text(i, v + (0.001 if v >= 0 else -0.003), f"{v:+.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(out_dir / "residual_shrinkage_test_improvement.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    traj = pd.read_csv(boot_path)
    by_traj = (
        traj.groupby("trajectory_id", as_index=False)
        .agg(improvement_vs_identity=("improvement_vs_identity", "mean"))
        .sort_values("improvement_vs_identity")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(by_traj["trajectory_id"], by_traj["improvement_vs_identity"])
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("improvement over identity")
    ax.set_title("Per-trajectory test improvement after validation-calibrated shrinkage")

    fig.tight_layout()
    fig.savefig(out_dir / "residual_shrinkage_per_trajectory.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(out_dir / "residual_shrinkage_test_improvement.png")
    print(out_dir / "residual_shrinkage_per_trajectory.png")


if __name__ == "__main__":
    main()
