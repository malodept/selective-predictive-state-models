from __future__ import annotations

from pathlib import Path
import pandas as pd

OUT = Path("reports/paper_assets_v30_cleanup")
TAB = OUT / "tables"
TXT = OUT / "text"
TAB.mkdir(parents=True, exist_ok=True)
TXT.mkdir(parents=True, exist_ok=True)

LAT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v24_measured_latency/capacity_latency_summary.csv")
lat = pd.read_csv(LAT)

b128 = lat[lat["batch_size"] == 128].copy()
b1 = lat[lat["batch_size"] == 1].copy()

arch = b128.merge(
    b1[["model_label", "latency_ms_median"]].rename(columns={"latency_ms_median": "latency_ms_b1"}),
    on="model_label",
    how="left",
)

arch = arch[[
    "model_label",
    "model_dim",
    "layers",
    "heads",
    "params",
    "checkpoint_mb",
    "latency_ms_b1",
    "latency_ms_median",
    "relative_latency_b128",
]].copy()

arch = arch.rename(columns={
    "model_label": "model",
    "latency_ms_median": "latency_ms_b128",
})

order = ["Tiny", "Small", "Medium", "Full"]
arch["order"] = arch["model"].map({m: i for i, m in enumerate(order)})
arch = arch.sort_values("order").drop(columns=["order"])

arch.to_csv(TAB / "table_v30_capacity_architecture_latency.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{lrrrrrrr}")
tex.append(r"\toprule")
tex.append(r"Model & Dim. & Layers & Heads & Params & MB & Lat. @1 & Rel. lat. @128 \\")
tex.append(r"\midrule")
for _, r in arch.iterrows():
    tex.append(
        f"{r['model']} & {int(r['model_dim'])} & {int(r['layers'])} & {int(r['heads'])} & "
        f"{int(r['params'])/1e6:.2f}M & {r['checkpoint_mb']:.2f} & "
        f"{r['latency_ms_b1']:.3f} ms & {r['relative_latency_b128']:.3f} \\\\"
    )
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Capacity ladder used for latency-normalized value-of-computation. Latency is measured as median forward-pass time on the same GPU setup; relative latency is normalized by the Full model at batch size 128.}")
tex.append(r"\label{tab:capacity_arch_latency}")
tex.append(r"\end{table}")
(TAB / "table_v30_capacity_architecture_latency.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

related = r"""
\section{Related Work}

\paragraph{Latent world models and predictive representations.}
World models learn compact predictive state representations that support forecasting, planning, or control. Classical formulations learn recurrent latent dynamics for model-based decision making, while recent joint-embedding predictive architectures emphasize prediction in representation space rather than pixel reconstruction. Our experiments follow this latent-prediction perspective: future observations are not generated as images, but compared through frozen DINOv2 patch-token displacements. This isolates the dynamics and reliability question from representation pretraining.

\paragraph{Intervention-based evaluation.}
A central difficulty in evaluating action-conditioned predictive models is that observational trajectories can entangle state, action, and environment correlations. SPSM instead uses exact simulator resets: several actions are applied from the same initial state, producing counterfactual futures with shared initial conditions. The hard-negative ranking protocol then separates same-state/different-action discrimination from same-action/different-state discrimination. This makes failures more diagnostic than aggregate rollout error alone.

\paragraph{Selective prediction and value-of-computation.}
Selective prediction studies when a model should abstain or defer under uncertainty, usually through risk-coverage tradeoffs. Adaptive computation and model cascades study when extra compute should be spent at inference time. SPSM adapts these ideas to action-conditioned latent world models: the router must decide whether additional predictive computation is worth its measured latency cost before the larger predictor is called.

\paragraph{Multi-capacity predictive routing.}
A common assumption in cascades is that larger models are more accurate but more expensive. Our results show that this assumption can fail under controlled OOD shifts: medium-capacity predictors can outperform the largest model, and the best capacity depends on the shift and cost. This motivates latency-normalized multi-capacity routing rather than a simple small-to-large cascade.
"""
(TXT / "related_work_v30.tex").write_text(related.strip() + "\n", encoding="utf-8")

discussion = r"""
\paragraph{What the current evidence supports.}
The strongest claim is not that routing always improves predictive inference, nor that larger models are safer. The evidence supports a more specific conclusion: under controlled interventions, predictive capacity is not uniformly ordered across OOD shifts, and this creates measurable value-of-computation structure. Learned multi-capacity routing gives stable gains on the H=48 hard OOD regime, while H=72 retains large oracle headroom.

\paragraph{What remains open.}
The current routing signals are intentionally conservative: they use Tiny-model diagnostics, action, and simple observable context. They do not fully explain the hardest long-horizon H=72 regime. This suggests that better environment-complexity features, dynamics-aware representations, or planning-level feedback may be needed before the hardest routing problem can be solved.
"""
(TXT / "discussion_insert_v30.tex").write_text(discussion.strip() + "\n", encoding="utf-8")

claim = """
# v30 recommended paper framing

Main claim:
SPSM is a controlled exact-intervention protocol for evaluating reliability and value-of-computation in action-conditioned latent world models.

Empirical claims supported by current results:
1. State-action latent dynamics are learned under state-disjoint exact interventions.
2. OOD degradation is driven more by constrained geometry than by horizon or velocity alone.
3. Cheap and expensive predictors are not uniformly ordered, making VoC non-trivial.
4. Capacity is non-monotonic under hard OOD: Medium can beat Full.
5. Measured latency changes the compute frontier relative to checkpoint-size proxies.
6. Learned multi-capacity routing gives stable gains on H=48, but not on H=36 or H=72.
7. H=72 remains an oracle-headroom problem, not a solved routing problem.

Claims to avoid:
- Do not claim general adaptive computation is solved.
- Do not claim larger models are generally worse.
- Do not claim latent features always help.
- Do not claim the router is robust across all OOD regimes.
"""
(TXT / "v30_claim_checklist.md").write_text(claim.strip() + "\n", encoding="utf-8")

manifest = []
manifest.append("# Paper assets v30 cleanup\n")
manifest.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Text inserts\n")
for p in sorted(TXT.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Purpose\n")
manifest.append("These assets clean up the paper without changing the experimental story: add an architecture/latency table, strengthen related work, and make the claim boundaries explicit.")
(OUT / "paper_assets_v30_manifest.md").write_text("\n".join(manifest) + "\n", encoding="utf-8")

print((OUT / "paper_assets_v30_manifest.md").read_text())
print()
print((TAB / "table_v30_capacity_architecture_latency.tex").read_text())
print()
print((TXT / "v30_claim_checklist.md").read_text())
