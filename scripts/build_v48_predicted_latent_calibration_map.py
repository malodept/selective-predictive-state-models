from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v40_closed_loop_planning")

CAL_CANDIDATES = [
    ROOT / "v40i_model_specific_xy_calibration_summary.csv",
    ROOT / "v40h_model_specific_xy_calibration_summary.csv",
]
UNCAL_ONESTEP = ROOT / "v40f_online_one_step_alignment_summary.csv"
CAL_ONESTEP = ROOT / "v40h_online_one_step_model_specific_probe_summary.csv"
BLOCK_SWEEP = ROOT / "v40i_calibrated_onestep_block_sweep_summary.csv"
CHALLENGE_BOOT = ROOT / "v40k_challenge_goal_bootstrap.csv"
CHALLENGE_UTIL = ROOT / "v40k_challenge_goal_latency_utility.csv"

OUT = ROOT / "v48_predicted_latent_calibration_map"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PAPER = Path("reports/paper_assets_v48_predicted_latent_calibration")
PTAB = PAPER / "tables"
PFIG = PAPER / "figures"
PTXT = PAPER / "text"
PTAB.mkdir(parents=True, exist_ok=True)
PFIG.mkdir(parents=True, exist_ok=True)
PTXT.mkdir(parents=True, exist_ok=True)

MODELS = ["Tiny", "Medium", "Full"]


def find_file(candidates):
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("None found: " + ", ".join(str(p) for p in candidates))


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str:
    lower = {c.lower().replace(" ", "_").replace("-", "_"): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "_").replace("-", "_")
        if key in lower:
            return lower[key]
    # fuzzy
    for cand in candidates:
        ck = cand.lower().replace(" ", "_").replace("-", "_")
        for lk, orig in lower.items():
            if ck in lk:
                return orig
    raise KeyError(f"Could not find any of {candidates} in {list(df.columns)}")


def load_calibration() -> pd.DataFrame:
    p = find_file(CAL_CANDIDATES)
    df = norm_cols(pd.read_csv(p))

    model_col = pick_col(df, ["model"])
    probe_col = pick_col(df, ["probe_input", "probe input"])
    mean_col = pick_col(df, ["val_xy_error_mean", "mean xy error", "mean"])
    med_col = pick_col(df, ["val_xy_error_median", "median"])
    p90_col = pick_col(df, ["val_xy_error_p90", "p90"])

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "model": str(r[model_col]),
            "probe_input": str(r[probe_col]),
            "mean_xy_error": float(r[mean_col]),
            "median_xy_error": float(r[med_col]),
            "p90_xy_error": float(r[p90_col]),
            "source": str(p),
        })
    return pd.DataFrame(rows)


def build_calibration_map(cal: pd.DataFrame) -> pd.DataFrame:
    rows = []

    true_rows = cal[
        (cal["model"].astype(str) == "TrueFutureLatent")
        | (cal["probe_input"].astype(str).str.contains("true_z_future", regex=False))
    ]
    true_ref = float(true_rows["mean_xy_error"].iloc[0]) if len(true_rows) else np.nan

    for model in MODELS:
        sub = cal[cal["model"] == model].copy()

        true_probe = sub[sub["probe_input"].str.contains("true_future_probe", regex=False)]
        specific = sub[sub["probe_input"].str.contains("model_specific_probe", regex=False)]

        true_probe_err = float(true_probe["mean_xy_error"].iloc[0]) if len(true_probe) else np.nan
        specific_err = float(specific["mean_xy_error"].iloc[0]) if len(specific) else np.nan

        rows.append({
            "model": model,
            "true_future_latent_probe_error": true_ref,
            "predicted_latent_with_true_probe_error": true_probe_err,
            "predicted_latent_with_model_specific_probe_error": specific_err,
            "calibration_gap": true_probe_err - specific_err,
            "error_ratio_true_probe_over_specific": true_probe_err / specific_err if specific_err and specific_err > 0 else np.nan,
            "specific_probe_over_true_latent_probe": specific_err / true_ref if true_ref and true_ref > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def load_summary_table(path: Path, name: str) -> pd.DataFrame | None:
    if not path.exists():
        print(f"[WARN] missing {name}: {path}")
        return None
    df = norm_cols(pd.read_csv(path))
    df["source"] = str(path)
    return df


def build_decision_calibration_map() -> pd.DataFrame:
    unc = load_summary_table(UNCAL_ONESTEP, "uncalibrated one-step")
    cal = load_summary_table(CAL_ONESTEP, "calibrated one-step")
    if unc is None or cal is None:
        return pd.DataFrame()

    def standardize(df):
        policy_col = pick_col(df, ["policy", "method"])
        acc_col = pick_col(df, ["action acc", "action_acc", "decision_acc"])
        regret_col = pick_col(df, ["mean regret", "mean_regret"])
        out = df[[policy_col, acc_col, regret_col]].copy()
        out.columns = ["model", "action_acc", "mean_regret"]
        out["model"] = out["model"].astype(str).str.replace("`", "", regex=False)
        out = out[out["model"].isin(MODELS)].copy()
        out["action_acc"] = pd.to_numeric(out["action_acc"], errors="coerce")
        out["mean_regret"] = pd.to_numeric(out["mean_regret"], errors="coerce")
        return out

    u = standardize(unc).rename(columns={
        "action_acc": "uncalibrated_action_acc",
        "mean_regret": "uncalibrated_mean_regret",
    })
    c = standardize(cal).rename(columns={
        "action_acc": "calibrated_action_acc",
        "mean_regret": "calibrated_mean_regret",
    })

    m = u.merge(c, on="model", how="inner")
    m["action_acc_gain_from_calibration"] = m["calibrated_action_acc"] - m["uncalibrated_action_acc"]
    m["regret_reduction_from_calibration"] = m["uncalibrated_mean_regret"] - m["calibrated_mean_regret"]
    return m


def build_block_sweep_map() -> pd.DataFrame:
    df = load_summary_table(BLOCK_SWEEP, "calibrated block sweep")
    if df is None:
        return pd.DataFrame()

    blocked_col = pick_col(df, ["blocked", "num_blocked"])
    policy_col = pick_col(df, ["policy", "method"])
    acc_col = pick_col(df, ["action acc", "action_acc", "decision_acc"])
    regret_col = pick_col(df, ["mean regret", "mean_regret"])
    blocked_action_col = pick_col(df, ["blocked-action rate", "blocked_action_rate"])

    out = df[[blocked_col, policy_col, acc_col, regret_col, blocked_action_col]].copy()
    out.columns = ["blocked", "policy", "action_acc", "mean_regret", "blocked_action_rate"]
    out["policy"] = out["policy"].astype(str).str.replace("`", "", regex=False)
    out = out[out["policy"].isin(["greedy_vector"] + MODELS)].copy()

    for c in ["blocked", "action_acc", "mean_regret", "blocked_action_rate"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    rows = []
    for blocked, sub in out.groupby("blocked"):
        greedy = sub[sub["policy"] == "greedy_vector"]
        greedy_acc = float(greedy["action_acc"].iloc[0]) if len(greedy) else np.nan
        greedy_reg = float(greedy["mean_regret"].iloc[0]) if len(greedy) else np.nan

        for model in MODELS:
            r = sub[sub["policy"] == model]
            if len(r) == 0:
                continue
            rr = r.iloc[0]
            rows.append({
                "blocked": int(blocked),
                "model": model,
                "action_acc": float(rr["action_acc"]),
                "mean_regret": float(rr["mean_regret"]),
                "blocked_action_rate": float(rr["blocked_action_rate"]),
                "acc_gap_vs_greedy": float(rr["action_acc"]) - greedy_acc,
                "regret_reduction_vs_greedy": greedy_reg - float(rr["mean_regret"]),
            })

    return pd.DataFrame(rows)


def build_challenge_summary() -> pd.DataFrame:
    df = load_summary_table(CHALLENGE_BOOT, "challenge bootstrap")
    if df is None:
        return pd.DataFrame()

    needed = {
        "blocked": pick_col(df, ["num_blocked", "blocked"]),
        "policy": pick_col(df, ["policy", "model"]),
        "regret_reduction": pick_col(df, ["regret_reduction"]),
        "regret_reduction_ci_low": pick_col(df, ["regret_reduction_ci_low"]),
        "regret_reduction_ci_high": pick_col(df, ["regret_reduction_ci_high"]),
        "acc_gain": pick_col(df, ["acc_gain"]),
    }

    out = df[[needed[k] for k in needed]].copy()
    out.columns = list(needed.keys())
    out["policy"] = out["policy"].astype(str).str.replace("`", "", regex=False)
    out = out[out["policy"].isin(MODELS)].copy()

    for c in ["blocked", "regret_reduction", "regret_reduction_ci_low", "regret_reduction_ci_high", "acc_gain"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    summary = (
        out.groupby("policy", as_index=False)
        .agg(
            mean_regret_reduction=("regret_reduction", "mean"),
            min_regret_reduction=("regret_reduction", "min"),
            max_regret_reduction=("regret_reduction", "max"),
            mean_acc_gain=("acc_gain", "mean"),
            stable_blocks=("regret_reduction_ci_low", lambda x: int((x > 0).sum())),
            n_blocks=("blocked", "nunique"),
        )
        .rename(columns={"policy": "model"})
    )
    return summary


def plot_calibration(calmap: pd.DataFrame, path: Path):
    x = np.arange(len(calmap))
    width = 0.35

    plt.figure(figsize=(6.8, 4.0))
    plt.bar(x - width/2, calmap["predicted_latent_with_true_probe_error"], width, label="Predicted latent + true probe")
    plt.bar(x + width/2, calmap["predicted_latent_with_model_specific_probe_error"], width, label="Predicted latent + model-specific probe")
    plt.xticks(x, calmap["model"])
    plt.ylabel("Mean xy error")
    plt.title("Predicted-latent physical calibration gap")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_decision(decmap: pd.DataFrame, path: Path):
    if decmap.empty:
        return
    x = np.arange(len(decmap))
    width = 0.35

    plt.figure(figsize=(6.8, 4.0))
    plt.bar(x - width/2, decmap["uncalibrated_mean_regret"], width, label="Uncalibrated")
    plt.bar(x + width/2, decmap["calibrated_mean_regret"], width, label="Model-specific calibrated")
    plt.xticks(x, decmap["model"])
    plt.ylabel("Mean one-step decision regret")
    plt.title("Decision utility after predicted-latent calibration")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def write_tex(calmap, decmap, challenge):
    # Table 1: calibration gap.
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & True-latent probe & Pred. + true probe & Pred. + specific probe & Gap & Ratio \\")
    lines.append(r"\midrule")
    for _, r in calmap.iterrows():
        lines.append(
            f"{r['model']} & {r['true_future_latent_probe_error']:.3f} & "
            f"{r['predicted_latent_with_true_probe_error']:.3f} & "
            f"{r['predicted_latent_with_model_specific_probe_error']:.3f} & "
            f"{r['calibration_gap']:.3f} & "
            f"{r['error_ratio_true_probe_over_specific']:.1f}x \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Predicted-latent physical calibration map. Probes trained on true future latents fail on predicted latents, while model-specific probes recover accurate physical xy predictions. This separates latent discrimination from physical calibration.}")
    lines.append(r"\label{tab:predicted_latent_calibration_map}")
    lines.append(r"\end{table}")
    (PTAB / "table_v48_predicted_latent_calibration_map.tex").write_text("\n".join(lines) + "\n")

    # Table 2: decision calibration.
    if not decmap.empty:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Model & Acc. gain & Regret reduction & Cal. acc. & Cal. regret \\")
        lines.append(r"\midrule")
        for _, r in decmap.iterrows():
            lines.append(
                f"{r['model']} & {r['action_acc_gain_from_calibration']:.3f} & "
                f"{r['regret_reduction_from_calibration']:.3f} & "
                f"{r['calibrated_action_acc']:.3f} & {r['calibrated_mean_regret']:.3f} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{One-step decision calibration. Calibrating predicted latents with model-specific physical heads substantially improves action selection and regret relative to decoding predicted latents with a true-future probe.}")
        lines.append(r"\label{tab:decision_calibration_map}")
        lines.append(r"\end{table}")
        (PTAB / "table_v48_decision_calibration_map.tex").write_text("\n".join(lines) + "\n")

    # Table 3: challenge summary.
    if not challenge.empty:
        lines = []
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(r"\small")
        lines.append(r"\begin{tabular}{lrrrr}")
        lines.append(r"\toprule")
        lines.append(r"Model & Mean regret red. & Min regret red. & Mean acc. gain & Stable blocks \\")
        lines.append(r"\midrule")
        for _, r in challenge.iterrows():
            lines.append(
                f"{r['model']} & {r['mean_regret_reduction']:.3f} & "
                f"{r['min_regret_reduction']:.3f} & {r['mean_acc_gain']:.3f} & "
                f"{int(r['stable_blocks'])}/{int(r['n_blocks'])} \\\\"
            )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\caption{Obstacle-sensitive challenge-goal calibration summary. Calibrated predicted latents reduce regret over the nominal greedy-vector baseline across blocked regimes.}")
        lines.append(r"\label{tab:challenge_goal_calibration_summary}")
        lines.append(r"\end{table}")
        (PTAB / "table_v48_challenge_goal_calibration_summary.tex").write_text("\n".join(lines) + "\n")


def write_claim():
    text = """
# v48 predicted-latent calibration claim

## Main idea

v48 separates three notions that are often conflated in latent world models:
1. counterfactual latent discrimination;
2. physical calibration of predicted latents;
3. downstream decision actionability.

## Scientific claim

Predicted latent futures can contain physical information without being directly decodable by probes trained on true encoded futures. In SPSM, a true-future xy probe decodes true future latents accurately, but fails on predicted future latents. Model-specific calibration recovers accurate physical xy predictions and improves one-step action selection.

## Why this matters

This means that latent ranking correctness is not enough to guarantee actionability. A predicted latent can be useful for discriminating the correct counterfactual future while still living in a representation coordinate system that requires calibration before downstream planning.

## How it strengthens the paper

v47 shows that predictive compute has query-level regimes. v48 shows that predicted latent usefulness also has calibration regimes. Together, they support the broader thesis: predictive usefulness is multi-dimensional, not a scalar accuracy.
"""
    (PTXT / "v48_predicted_latent_calibration_claim.md").write_text(text.strip() + "\n")


def write_md(calmap, decmap, blockmap, challenge):
    lines = []
    lines.append("# SPSM v48 Predicted-Latent Calibration Map\n")
    lines.append("This analysis tests whether predicted future latents are physically calibrated and actionable, rather than only discriminative in latent ranking space.\n")

    lines.append("## Physical calibration map\n")
    lines.append("| model | true-latent probe | pred + true probe | pred + model-specific probe | calibration gap | error ratio |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in calmap.iterrows():
        lines.append(
            f"| {r['model']} | {r['true_future_latent_probe_error']:.3f} | "
            f"{r['predicted_latent_with_true_probe_error']:.3f} | "
            f"{r['predicted_latent_with_model_specific_probe_error']:.3f} | "
            f"{r['calibration_gap']:.3f} | {r['error_ratio_true_probe_over_specific']:.1f}x |"
        )

    if not decmap.empty:
        lines.append("\n## One-step decision calibration\n")
        lines.append("| model | uncal acc | cal acc | acc gain | uncal regret | cal regret | regret reduction |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for _, r in decmap.iterrows():
            lines.append(
                f"| {r['model']} | {r['uncalibrated_action_acc']:.3f} | {r['calibrated_action_acc']:.3f} | "
                f"{r['action_acc_gain_from_calibration']:.3f} | {r['uncalibrated_mean_regret']:.3f} | "
                f"{r['calibrated_mean_regret']:.3f} | {r['regret_reduction_from_calibration']:.3f} |"
            )

    if not blockmap.empty:
        lines.append("\n## Calibrated one-step block sweep vs greedy-vector\n")
        lines.append("| blocked | model | action acc | regret | acc gap vs greedy | regret reduction vs greedy |")
        lines.append("|---:|---|---:|---:|---:|---:|")
        for _, r in blockmap.sort_values(["blocked", "mean_regret"]).iterrows():
            lines.append(
                f"| {int(r['blocked'])} | {r['model']} | {r['action_acc']:.3f} | {r['mean_regret']:.3f} | "
                f"{r['acc_gap_vs_greedy']:.3f} | {r['regret_reduction_vs_greedy']:.3f} |"
            )

    if not challenge.empty:
        lines.append("\n## Obstacle-sensitive challenge-goal summary\n")
        lines.append("| model | mean regret reduction | min regret reduction | mean acc gain | stable blocks |")
        lines.append("|---|---:|---:|---:|---:|")
        for _, r in challenge.iterrows():
            lines.append(
                f"| {r['model']} | {r['mean_regret_reduction']:.3f} | {r['min_regret_reduction']:.3f} | "
                f"{r['mean_acc_gain']:.3f} | {int(r['stable_blocks'])}/{int(r['n_blocks'])} |"
            )

    lines.append("\n## Interpretation\n")
    lines.append("- Predicted latents are not directly physically calibrated: true-future probes fail badly on predicted futures.")
    lines.append("- Model-specific probes recover accurate xy predictions, so the physical information is present but represented differently.")
    lines.append("- Calibration improves one-step decisions and enables obstacle-sensitive challenge-goal gains.")
    lines.append("- Therefore latent ranking, physical calibration, and decision utility are distinct dimensions of predictive usefulness.")

    (OUT / "v48_predicted_latent_calibration_map.md").write_text("\n".join(lines) + "\n")


def main():
    cal = load_calibration()
    calmap = build_calibration_map(cal)
    decmap = build_decision_calibration_map()
    blockmap = build_block_sweep_map()
    challenge = build_challenge_summary()

    cal.to_csv(OUT / "v48_raw_calibration_rows.csv", index=False)
    calmap.to_csv(OUT / "v48_predicted_latent_calibration_map.csv", index=False)
    decmap.to_csv(OUT / "v48_decision_calibration_map.csv", index=False)
    blockmap.to_csv(OUT / "v48_calibrated_block_sweep_map.csv", index=False)
    challenge.to_csv(OUT / "v48_challenge_goal_calibration_summary.csv", index=False)

    calmap.to_csv(PTAB / "table_v48_predicted_latent_calibration_map.csv", index=False)
    decmap.to_csv(PTAB / "table_v48_decision_calibration_map.csv", index=False)
    challenge.to_csv(PTAB / "table_v48_challenge_goal_calibration_summary.csv", index=False)

    plot_calibration(calmap, FIG / "fig_v48_predicted_latent_calibration_gap.png")
    plot_calibration(calmap, PFIG / "fig_v48_predicted_latent_calibration_gap.png")
    plot_decision(decmap, FIG / "fig_v48_decision_calibration_regret.png")
    plot_decision(decmap, PFIG / "fig_v48_decision_calibration_regret.png")

    write_tex(calmap, decmap, challenge)
    write_claim()
    write_md(calmap, decmap, blockmap, challenge)

    manifest = ["# Paper assets v48 predicted latent calibration\n", "## Tables\n"]
    for p in sorted(PTAB.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Figures\n")
    for p in sorted(PFIG.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Text\n")
    for p in sorted(PTXT.glob("*")):
        manifest.append(f"- `{p}`")
    (PAPER / "paper_assets_v48_manifest.md").write_text("\n".join(manifest) + "\n")

    print(OUT / "v48_predicted_latent_calibration_map.md")
    print((OUT / "v48_predicted_latent_calibration_map.md").read_text())


if __name__ == "__main__":
    main()
