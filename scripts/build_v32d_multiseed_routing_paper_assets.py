from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("reports/paper_assets_v32_multiseed_routing")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

BOOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v32_multiseed_multicapacity_routing/multiseed_router_bootstrap_summary.csv")
df = pd.read_csv(BOOT)

HARD = ["3-block, H=36", "3-block, H=48", "3-block, H=72"]
LAM_SHOW = [0.00, 0.05, 0.10, 0.20, 0.30]

# ---------------------------------------------------------------------
# Table 1: pre-specified ridge expected-utility router.
# ---------------------------------------------------------------------
ridge = df[df["method"] == "ridge_expected_utility"].copy()
ridge_show = ridge[ridge["lambda"].isin(LAM_SHOW)].copy()
ridge_show.to_csv(TAB / "table_v32_ridge_expected_utility_bootstrap.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{llrrrl}")
tex.append(r"\toprule")
tex.append(r"$\lambda$ & Held-out shift & $\Delta$ vs fixed & 95\% CI & Oracle gap & Stable? \\")
tex.append(r"\midrule")

for _, r in ridge_show.sort_values(["lambda", "variant_label"]).iterrows():
    stable = r"\textbf{yes}" if bool(r["stable_positive_vs_fixed"]) else "no"
    tex.append(
        f"{r['lambda']:.2f} & {r['variant_label']} & "
        f"{r['delta_vs_fixed']:.3f} & "
        f"[{r['delta_ci95_low']:.3f}, {r['delta_ci95_high']:.3f}] & "
        f"{r['oracle_gap']:.3f} & {stable} \\\\"
    )

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Seed-aware bootstrap validation of the pre-specified ridge expected-utility router. The router chooses among Tiny, Small, Medium, and Full using only Tiny-model diagnostics, action, and observable context. Stable positive gains appear on the H=48 hard OOD shift, while H=72 retains large oracle headroom.}")
tex.append(r"\label{tab:multiseed_ridge_router_bootstrap}")
tex.append(r"\end{table}")
(TAB / "table_v32_ridge_expected_utility_bootstrap.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Table 2: oracle headroom at selected costs.
# ---------------------------------------------------------------------
oracle = df[df["method"] == "oracle_multicapacity"].copy()
oracle_show = oracle[oracle["lambda"].isin(LAM_SHOW)].copy()
oracle_show.to_csv(TAB / "table_v32_oracle_headroom.csv", index=False)

tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\small")
tex.append(r"\resizebox{\linewidth}{!}{%")
tex.append(r"\begin{tabular}{llrrr}")
tex.append(r"\toprule")
tex.append(r"$\lambda$ & Held-out shift & Oracle $\Delta$ vs fixed & 95\% CI & Oracle latency \\")
tex.append(r"\midrule")

for _, r in oracle_show.sort_values(["lambda", "variant_label"]).iterrows():
    tex.append(
        f"{r['lambda']:.2f} & {r['variant_label']} & "
        f"{r['delta_vs_fixed']:.3f} & "
        f"[{r['delta_ci95_low']:.3f}, {r['delta_ci95_high']:.3f}] & "
        f"{r['latency']:.3f} \\\\"
    )

tex.append(r"\bottomrule")
tex.append(r"\end{tabular}%")
tex.append(r"}")
tex.append(r"\caption{Multi-capacity oracle headroom under seed-aware bootstrap. The oracle selects among the four capacities per instance within each seed-aligned ladder. Large H=72 headroom indicates that the routing problem is not solved by current Tiny-only diagnostics.}")
tex.append(r"\label{tab:multiseed_oracle_headroom}")
tex.append(r"\end{table}")
(TAB / "table_v32_oracle_headroom.tex").write_text("\n".join(tex) + "\n")

# ---------------------------------------------------------------------
# Figure 1: ridge gain with CIs.
# ---------------------------------------------------------------------
plt.figure(figsize=(6.9, 4.25))
for variant in HARD:
    sub = ridge[ridge["variant_label"] == variant].sort_values("lambda")
    plt.plot(sub["lambda"], sub["delta_vs_fixed"], marker="o", linewidth=1.8, label=variant)
    plt.fill_between(
        sub["lambda"],
        sub["delta_ci95_low"],
        sub["delta_ci95_high"],
        alpha=0.14,
    )

plt.axhline(0.0, linewidth=1.0)
plt.xlabel("Compute cost $\\lambda$")
plt.ylabel("Utility gain over best fixed")
plt.title("Seed-aware learned multi-capacity routing")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v32_ridge_router_gain_seedaware_bootstrap.png", dpi=240)
plt.close()

# ---------------------------------------------------------------------
# Figure 2: oracle headroom.
# ---------------------------------------------------------------------
plt.figure(figsize=(6.9, 4.25))
for variant in HARD:
    sub = oracle[oracle["variant_label"] == variant].sort_values("lambda")
    plt.plot(sub["lambda"], sub["delta_vs_fixed"], marker="o", linewidth=1.8, label=variant)
    plt.fill_between(
        sub["lambda"],
        sub["delta_ci95_low"],
        sub["delta_ci95_high"],
        alpha=0.14,
    )

plt.axhline(0.0, linewidth=1.0)
plt.xlabel("Compute cost $\\lambda$")
plt.ylabel("Oracle gain over best fixed")
plt.title("Remaining multi-capacity oracle headroom")
plt.grid(True, alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG / "fig_v32_oracle_headroom_seedaware_bootstrap.png", dpi=240)
plt.close()

# ---------------------------------------------------------------------
# Claim note.
# ---------------------------------------------------------------------
claim = """
# Paper assets v32 multi-seed routing

## Main update

The learned routing claim is now seed-aware.
The pre-specified ridge expected-utility router is stably positive on H=48 for all tested compute costs.
It is not stably positive on H=36 or H=72.

## Recommended claim

A learned multi-capacity router can improve over the best fixed capacity on specific hard OOD regimes.
In the current benchmark, this effect is robust on the H=48 constrained-geometry shift.
The hardest H=72 shift retains large oracle headroom, indicating that better routing signals are needed.

## Key numbers

H=48 ridge expected-utility gains:
- λ=0.00: +0.0265, CI [0.0083, 0.0448].
- λ=0.05: +0.0350, CI [0.0173, 0.0527].
- λ=0.10: +0.0439, CI [0.0120, 0.0733].
- λ=0.20: +0.0460, CI [0.0152, 0.0749].
- λ=0.30: +0.0467, CI [0.0180, 0.0726].

H=72:
- learned gains are not stable;
- oracle headroom stays around +0.11 to +0.12;
- this should be framed as an unsolved routing-signal problem.

## Claim boundary

Report the pre-specified ridge router as the clean learned-routing result.
Best-of-method tables can be shown in appendix or diagnostics, but the main text should avoid cherry-picking routers.
"""
(OUT / "v32_multiseed_routing_claim.md").write_text(claim.strip() + "\n")

manifest = []
manifest.append("# Paper assets v32 multi-seed routing\n")
manifest.append("## Tables\n")
for p in sorted(TAB.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Figures\n")
for p in sorted(FIG.glob("*")):
    manifest.append(f"- `{p}`")
manifest.append("\n## Text\n")
manifest.append(f"- `{OUT / 'v32_multiseed_routing_claim.md'}`")

(OUT / "paper_assets_v32_manifest.md").write_text("\n".join(manifest) + "\n")

print((OUT / "paper_assets_v32_manifest.md").read_text())
print()
print((TAB / "table_v32_ridge_expected_utility_bootstrap.tex").read_text())
print()
print((TAB / "table_v32_oracle_headroom.tex").read_text())
print()
print((OUT / "v32_multiseed_routing_claim.md").read_text())
