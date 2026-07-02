from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

V47 = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v47_compute_regime_map")
V48 = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v40_closed_loop_planning/v48_predicted_latent_calibration_map")

REGIME = V47 / "v47b_compute_regime_rates_clean.csv"
RECS = V47 / "v47b_compute_recommendations_clean.csv"
ORACLE = V47 / "v47b_oracle_headroom_by_lambda_clean.csv"

CAL = V48 / "v48_predicted_latent_calibration_map.csv"
DEC = V48 / "v48_decision_calibration_map.csv"
CHAL = V48 / "v48_challenge_goal_calibration_summary.csv"

OUT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v49_predictive_usefulness_stack")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PAPER = Path("reports/paper_assets_v49_predictive_usefulness_stack")
PTAB = PAPER / "tables"
PFIG = PAPER / "figures"
PTXT = PAPER / "text"
PTAB.mkdir(parents=True, exist_ok=True)
PFIG.mkdir(parents=True, exist_ok=True)
PTXT.mkdir(parents=True, exist_ok=True)

HARD_ORDER = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def esc(s: str) -> str:
    s = str(s)
    repl = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return s


def load_all():
    reg = pd.read_csv(require(REGIME))
    rec = pd.read_csv(require(RECS))
    oracle = pd.read_csv(require(ORACLE))
    cal = pd.read_csv(require(CAL))
    dec = pd.read_csv(require(DEC))
    chal = pd.read_csv(require(CHAL))
    return reg, rec, oracle, cal, dec, chal


def get_row(df: pd.DataFrame, shift: str) -> pd.Series:
    m = df[df["variant_label"] == shift]
    if len(m) == 0:
        raise KeyError(f"Missing shift: {shift}")
    return m.iloc[0]


def build_stack(reg, rec, oracle, cal, dec, chal) -> pd.DataFrame:
    h36 = get_row(rec, "3-block, H=36")
    h48 = get_row(rec, "3-block, H=48")
    h72 = get_row(rec, "3-block, H=72")

    cal_ratio_min = float(cal["error_ratio_true_probe_over_specific"].min())
    cal_ratio_max = float(cal["error_ratio_true_probe_over_specific"].max())
    cal_specific_min = float(cal["predicted_latent_with_model_specific_probe_error"].min())
    cal_specific_max = float(cal["predicted_latent_with_model_specific_probe_error"].max())
    cal_true_probe_min = float(cal["predicted_latent_with_true_probe_error"].min())
    cal_true_probe_max = float(cal["predicted_latent_with_true_probe_error"].max())

    dec_regret_min = float(dec["regret_reduction_from_calibration"].min())
    dec_regret_max = float(dec["regret_reduction_from_calibration"].max())
    dec_acc_min = float(dec["action_acc_gain_from_calibration"].min())
    dec_acc_max = float(dec["action_acc_gain_from_calibration"].max())

    chal_mean_min = float(chal["mean_regret_reduction"].min())
    chal_mean_max = float(chal["mean_regret_reduction"].max())
    stable_blocks = sorted(chal["stable_blocks"].astype(str).unique())
    stable_summary = stable_blocks[0] if len(stable_blocks) == 1 else ", ".join(stable_blocks)

    rows = [
        {
            "level": "1. Counterfactual discrimination",
            "question": "Can the model select the correct future under exact interventions?",
            "hidden_failure": "Average trajectory prediction can hide shortcut use and action-state entanglement.",
            "spsm_evidence": "Full state-action predictor reaches ID moving-only top-1 0.999, while action-only and state-only ablations fail on complementary hard-negative axes.",
            "paper_role": "Establishes that the protocol tests intervention-conditioned latent dynamics rather than observational correlation.",
        },
        {
            "level": "2. Failure-axis diagnosis",
            "question": "When prediction fails, which axis fails?",
            "hidden_failure": "A single OOD score does not reveal whether the issue is action grounding, state geometry, or another factor.",
            "spsm_evidence": "Controlled 3-block OOD shifts primarily degrade state discrimination; H=72 full model has state discrimination 0.847 while action discrimination remains 0.983.",
            "paper_role": "Turns OOD evaluation into an interpretable failure decomposition.",
        },
        {
            "level": "3. Compute-regime map",
            "question": "Is extra predictive compute useful, harmful, unnecessary, or insufficient?",
            "hidden_failure": "Bigger-model fallback assumes monotonic improvement, but larger capacities can introduce different failures.",
            "spsm_evidence": (
                f"H=48 is the cleanest routable regime: useful={h48['useful_compute']:.3f}, "
                f"anti-rescue={h48['anti_rescue']:.3f}, stable router lambdas={int(h48['stable_router_count'])}. "
                f"H=72 has larger value but harder routing: useful={h72['useful_compute']:.3f}, "
                f"anti-rescue={h72['anti_rescue']:.3f}, oracle gap={h72['mean_oracle_headroom']:.3f}."
            ),
            "paper_role": "Defines interventional value-of-computation as a query-regime map rather than a scalar average.",
        },
        {
            "level": "4. Predicted-latent calibration",
            "question": "Are predicted futures directly interpretable by downstream probes?",
            "hidden_failure": "Latent ranking can be correct even if predicted latents are not in the same operational coordinate system as encoded futures.",
            "spsm_evidence": (
                f"True-future probes fail on predicted latents with xy error {cal_true_probe_min:.3f}-{cal_true_probe_max:.3f}, "
                f"while model-specific probes recover {cal_specific_min:.3f}-{cal_specific_max:.3f} "
                f"({cal_ratio_min:.1f}x-{cal_ratio_max:.1f}x error reduction ratio)."
            ),
            "paper_role": "Separates discriminative latent prediction from physical calibration.",
        },
        {
            "level": "5. Decision actionability",
            "question": "Do calibrated predicted latents support action choice?",
            "hidden_failure": "A prediction can be accurate in latent space but not useful for downstream decisions or latency-normalized utility.",
            "spsm_evidence": (
                f"Model-specific calibration improves one-step action accuracy by {dec_acc_min:.3f}-{dec_acc_max:.3f} "
                f"and reduces regret by {dec_regret_min:.3f}-{dec_regret_max:.3f}. "
                f"On obstacle-sensitive challenge goals, calibrated predictors reduce regret by {chal_mean_min:.3f}-{chal_mean_max:.3f}, stable in {stable_summary} blocks."
            ),
            "paper_role": "Connects latent reliability to decision usefulness while preserving the diagnostic scope.",
        },
    ]

    return pd.DataFrame(rows)


def build_regime_summary(reg, rec) -> pd.DataFrame:
    cols = [
        "variant_label",
        "all_correct",
        "none_correct",
        "useful_compute",
        "anti_rescue",
        "other_mixed",
        "mean_oracle_headroom",
        "stable_router_count",
        "recommendation",
    ]
    x = reg.merge(
        rec[["variant_label", "mean_oracle_headroom", "stable_router_count", "recommendation"]],
        on="variant_label",
        how="left",
    )
    x = x[x["variant_label"].isin(HARD_ORDER)].copy()
    x["sort_key"] = x["variant_label"].map({v: i for i, v in enumerate(HARD_ORDER)})
    return x.sort_values("sort_key")[cols]


def build_calibration_summary(cal, dec, chal) -> pd.DataFrame:
    x = cal.merge(dec[[
        "model",
        "action_acc_gain_from_calibration",
        "regret_reduction_from_calibration",
        "calibrated_action_acc",
        "calibrated_mean_regret",
    ]], on="model", how="left")

    x = x.merge(chal[[
        "model",
        "mean_regret_reduction",
        "min_regret_reduction",
        "mean_acc_gain",
        "stable_blocks",
        "n_blocks",
    ]], on="model", how="left")

    return x


def write_tex_stack(stack: pd.DataFrame):
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(r"\begin{tabular}{p{0.17\textwidth}p{0.22\textwidth}p{0.25\textwidth}p{0.31\textwidth}}")
    lines.append(r"\toprule")
    lines.append(r"Level & Diagnostic question & Failure hidden by scalar accuracy & SPSM evidence \\")
    lines.append(r"\midrule")
    for _, r in stack.iterrows():
        lines.append(
            f"{esc(r['level'])} & {esc(r['question'])} & {esc(r['hidden_failure'])} & {esc(r['spsm_evidence'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Predictive usefulness stack. SPSM evaluates latent world models through multiple diagnostic levels rather than a single scalar accuracy: counterfactual discrimination, failure-axis diagnosis, interventional value-of-computation, predicted-latent calibration, and downstream actionability.}")
    lines.append(r"\label{tab:predictive_usefulness_stack}")
    lines.append(r"\end{table*}")
    (PTAB / "table_v49_predictive_usefulness_stack.tex").write_text("\n".join(lines) + "\n")


def write_tex_compact(regsum: pd.DataFrame, calsum: pd.DataFrame):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Shift & Useful & Anti-rescue & Oracle gap & Stable router $\lambda$ & Recommendation \\")
    lines.append(r"\midrule")
    for _, r in regsum.iterrows():
        rec = esc(r["recommendation"])
        lines.append(
            f"{esc(r['variant_label'])} & {r['useful_compute']:.3f} & {r['anti_rescue']:.3f} & "
            f"{r['mean_oracle_headroom']:.3f} & {int(r['stable_router_count'])} & {rec} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Interventional value-of-computation regimes. The same capacity ladder produces different compute recommendations across hard OOD shifts.}")
    lines.append(r"\label{tab:interventional_voc_regimes}")
    lines.append(r"\end{table}")
    (PTAB / "table_v49_interventional_voc_regimes.tex").write_text("\n".join(lines) + "\n")

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Pred.+true probe & Pred.+specific probe & Decision regret red. & Challenge red. \\")
    lines.append(r"\midrule")
    for _, r in calsum.iterrows():
        lines.append(
            f"{esc(r['model'])} & {r['predicted_latent_with_true_probe_error']:.3f} & "
            f"{r['predicted_latent_with_model_specific_probe_error']:.3f} & "
            f"{r['regret_reduction_from_calibration']:.3f} & "
            f"{r['mean_regret_reduction']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\caption{Predicted-latent calibration and actionability. Predicted latents are not directly decoded by true-future probes, but model-specific calibration recovers physical decision utility.}")
    lines.append(r"\label{tab:predicted_latent_actionability_compact}")
    lines.append(r"\end{table}")
    (PTAB / "table_v49_predicted_latent_actionability_compact.tex").write_text("\n".join(lines) + "\n")


def plot_stack(stack: pd.DataFrame, path: Path):
    labels = [r["level"].split(". ", 1)[1] for _, r in stack.iterrows()]
    y = np.arange(len(labels))[::-1]

    plt.figure(figsize=(9.5, 5.2))
    plt.scatter(np.ones(len(y)), y, s=260)
    for i, (_, r) in enumerate(stack.iterrows()):
        yy = y[i]
        plt.text(1.08, yy, r["level"], va="center", fontsize=10, fontweight="bold")
        plt.text(2.35, yy, r["question"], va="center", fontsize=8.5)
        plt.text(5.45, yy, r["paper_role"], va="center", fontsize=8.5)

    plt.text(1.08, len(y), "Diagnostic level", fontsize=10, fontweight="bold")
    plt.text(2.35, len(y), "Question", fontsize=10, fontweight="bold")
    plt.text(5.45, len(y), "Role in SPSM", fontsize=10, fontweight="bold")

    plt.xlim(0.8, 9.2)
    plt.ylim(-0.8, len(y) + 0.6)
    plt.yticks([])
    plt.xticks([])
    plt.title("Predictive usefulness is a diagnostic stack, not scalar accuracy")
    plt.box(False)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_two_maps(regsum: pd.DataFrame, calsum: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    x = np.arange(len(regsum))
    axes[0].bar(x - 0.2, regsum["useful_compute"], width=0.4, label="Useful compute")
    axes[0].bar(x + 0.2, regsum["anti_rescue"], width=0.4, label="Anti-rescue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(regsum["variant_label"], rotation=20, ha="right")
    axes[0].set_ylabel("Query fraction")
    axes[0].set_title("Compute-regime map")
    axes[0].legend(fontsize=8)

    y = np.arange(len(calsum))
    axes[1].bar(y - 0.2, calsum["predicted_latent_with_true_probe_error"], width=0.4, label="Pred. + true probe")
    axes[1].bar(y + 0.2, calsum["predicted_latent_with_model_specific_probe_error"], width=0.4, label="Pred. + specific probe")
    axes[1].set_xticks(y)
    axes[1].set_xticklabels(calsum["model"])
    axes[1].set_ylabel("Mean xy error")
    axes[1].set_title("Calibration-regime map")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def write_text(stack: pd.DataFrame):
    section = r"""
\paragraph{Predictive usefulness is a diagnostic stack.}
The results above suggest that world-model predictions should not be evaluated by a single scalar accuracy. In SPSM, predictive usefulness decomposes into five levels. First, exact-intervention hard negatives test whether a predictor is counterfactually discriminative. Second, same-state/different-action and same-action/different-state negatives localize failures to action or state axes. Third, the compute-regime map separates queries where extra predictive compute is unnecessary, useful, insufficient, or harmful. Fourth, the predicted-latent calibration map shows that latent predictions can contain physical information without being directly decodable by probes trained on true future latents. Fifth, downstream decision diagnostics test whether calibrated predicted latents support action choice. This stack is the main methodological point of SPSM: it turns latent world-model evaluation from average prediction quality into an interventional diagnostic of reliability, compute value, calibration, and actionability.
"""
    (PTXT / "section_v49_predictive_usefulness_stack.tex").write_text(section.strip() + "\n")

    claim = """
# v49 predictive usefulness stack

## Main idea

v47 and v48 can be unified into a single diagnostic stack for latent world models. A prediction is useful only if it is counterfactually discriminative, diagnostically interpretable, compute-worthwhile, physically calibrated, and actionable.

## Scientific claim

Predictive usefulness is not scalar accuracy. SPSM factorizes it into five levels:
1. counterfactual discrimination;
2. failure-axis diagnosis;
3. interventional value-of-computation;
4. predicted-latent calibration;
5. downstream actionability.

## Why this matters

Standard world-model evaluation can hide several failure modes: a model can rank futures correctly but fail physical calibration, a larger model can be worse than a smaller one on some queries, a regime can have oracle compute value that current routers cannot recover, and raw decision quality can disagree with latency-normalized decision utility.

## Next scientific step

The next major extension should not be another formatting pass. It should test whether this stack transfers to a second exact-intervention environment, ideally contact-rich object manipulation, where failure axes include object pose, contact mode, occlusion, and dynamics parameters.
"""
    (PTXT / "v49_predictive_usefulness_stack_claim.md").write_text(claim.strip() + "\n")


def write_md(stack, regsum, calsum):
    lines = []
    lines.append("# SPSM v49 Predictive Usefulness Stack\n")
    lines.append("v49 unifies v47 and v48 into one diagnostic object. The point is not to add another metric, but to define the layers required for a predicted latent future to be useful.\n")

    lines.append("## Five-level stack\n")
    lines.append("| level | diagnostic question | hidden failure | evidence |")
    lines.append("|---|---|---|---|")
    for _, r in stack.iterrows():
        lines.append(
            f"| {r['level']} | {r['question']} | {r['hidden_failure']} | {r['spsm_evidence']} |"
        )

    lines.append("\n## Compact compute-regime summary\n")
    lines.append("| shift | useful | anti-rescue | oracle gap | stable router λ | recommendation |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for _, r in regsum.iterrows():
        lines.append(
            f"| {r['variant_label']} | {r['useful_compute']:.3f} | {r['anti_rescue']:.3f} | "
            f"{r['mean_oracle_headroom']:.3f} | {int(r['stable_router_count'])} | {r['recommendation']} |"
        )

    lines.append("\n## Compact calibration/actionability summary\n")
    lines.append("| model | pred + true probe | pred + specific probe | decision regret reduction | challenge regret reduction |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in calsum.iterrows():
        lines.append(
            f"| {r['model']} | {r['predicted_latent_with_true_probe_error']:.3f} | "
            f"{r['predicted_latent_with_model_specific_probe_error']:.3f} | "
            f"{r['regret_reduction_from_calibration']:.3f} | {r['mean_regret_reduction']:.3f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- v47 says compute usefulness is query-regime structured: extra compute can be unnecessary, useful, insufficient, or harmful.")
    lines.append("- v48 says latent usefulness is calibration-structured: predicted latents can contain physical information without being directly actionable.")
    lines.append("- Together they support the broader thesis: useful world-model prediction requires reliability, diagnosis, compute allocation, calibration, and decision actionability.")
    lines.append("- This stack should become the scientific compass for the next environment extension.")

    (OUT / "v49_predictive_usefulness_stack.md").write_text("\n".join(lines) + "\n")


def main():
    reg, rec, oracle, cal, dec, chal = load_all()

    stack = build_stack(reg, rec, oracle, cal, dec, chal)
    regsum = build_regime_summary(reg, rec)
    calsum = build_calibration_summary(cal, dec, chal)

    stack.to_csv(OUT / "v49_predictive_usefulness_stack.csv", index=False)
    regsum.to_csv(OUT / "v49_compact_compute_regime_summary.csv", index=False)
    calsum.to_csv(OUT / "v49_compact_calibration_actionability_summary.csv", index=False)

    stack.to_csv(PTAB / "table_v49_predictive_usefulness_stack.csv", index=False)
    regsum.to_csv(PTAB / "table_v49_interventional_voc_regimes.csv", index=False)
    calsum.to_csv(PTAB / "table_v49_predicted_latent_actionability_compact.csv", index=False)

    write_tex_stack(stack)
    write_tex_compact(regsum, calsum)
    write_text(stack)
    write_md(stack, regsum, calsum)

    plot_stack(stack, FIG / "fig_v49_predictive_usefulness_stack.png")
    plot_stack(stack, PFIG / "fig_v49_predictive_usefulness_stack.png")
    plot_two_maps(regsum, calsum, FIG / "fig_v49_compute_and_calibration_maps.png")
    plot_two_maps(regsum, calsum, PFIG / "fig_v49_compute_and_calibration_maps.png")

    manifest = ["# Paper assets v49 predictive usefulness stack\n", "## Tables\n"]
    for p in sorted(PTAB.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Figures\n")
    for p in sorted(PFIG.glob("*")):
        manifest.append(f"- `{p}`")
    manifest.append("\n## Text\n")
    for p in sorted(PTXT.glob("*")):
        manifest.append(f"- `{p}`")
    (PAPER / "paper_assets_v49_manifest.md").write_text("\n".join(manifest) + "\n")

    print(OUT / "v49_predictive_usefulness_stack.md")
    print((OUT / "v49_predictive_usefulness_stack.md").read_text())


if __name__ == "__main__":
    main()
