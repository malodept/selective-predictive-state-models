from pathlib import Path
import matplotlib.pyplot as plt

out_table = Path("reports/tables/protocol/representation_loss_factorization")
out_fig = Path("reports/figures/protocol/representation_loss_factorization")
out_table.mkdir(parents=True, exist_ok=True)
out_fig.mkdir(parents=True, exist_ok=True)

rows = [
    {
        "representation": "CLS",
        "loss": "MSE residual",
        "identity": 1.606977,
        "gain": 0.007659,
        "gain_std": 0.000579,
        "relative": 100 * 0.007659 / 1.606977,
        "cosine": 0.057030,
        "positive": 0.757434,
    },
    {
        "representation": "CLS",
        "loss": "directional residual",
        "identity": 1.606977,
        "gain": 0.017654,
        "gain_std": 0.000224,
        "relative": 100 * 0.017654 / 1.606977,
        "cosine": 0.088062,
        "positive": 0.857781,
    },
    {
        "representation": "spatial",
        "loss": "MSE residual",
        "identity": 0.727133,
        "gain": 0.007356,
        "gain_std": 0.000311,
        "relative": 1.011591,
        "cosine": 0.087571,
        "positive": 0.886191,
    },
    {
        "representation": "spatial",
        "loss": "directional residual",
        "identity": 0.727133,
        "gain": 0.010625,
        "gain_std": 0.000179,
        "relative": 1.461195,
        "cosine": 0.105416,
        "positive": 0.928168,
    },
]

lines = [
    "# Representation/loss factorization summary",
    "",
    "This table isolates two factors: the latent representation (`CLS` vs `[CLS, patch_mean, patch_std]`) and the residual training loss (`MSE` vs directional).",
    "",
    "| representation | loss | identity error | calibrated gain | relative gain | cosine mean | positive cosine frac |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
]

for r in rows:
    lines.append(
        f"| {r['representation']} | {r['loss']} | "
        f"{r['identity']:.6f} | {r['gain']:.6f} ± {r['gain_std']:.6f} | "
        f"{r['relative']:.3f}% | {r['cosine']:.6f} | {r['positive']:.6f} |"
    )

(out_table / "representation_loss_factorization_summary.md").write_text("\n".join(lines) + "\n")

labels = [
    "CLS\nMSE",
    "CLS\ndirectional",
    "spatial\nMSE",
    "spatial\ndirectional",
]

gains = [r["gain"] for r in rows]
gain_err = [r["gain_std"] for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.bar(labels, gains, yerr=gain_err, capsize=5)
ax.axhline(0.0, linestyle="--", linewidth=1)
ax.set_ylabel("MSE improvement over identity")
ax.set_title("Representation and loss jointly control OOD latent prediction")
for i, v in enumerate(gains):
    ax.text(i, v + 0.0007, f"+{v:.4f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(out_fig / "representation_loss_gain_factorization.png", dpi=220, bbox_inches="tight")
plt.close(fig)

cosines = [r["cosine"] for r in rows]
positives = [r["positive"] for r in rows]

fig, ax = plt.subplots(figsize=(8, 4.8))
x = list(range(len(labels)))
ax.bar([i - 0.18 for i in x], cosines, width=0.36, label="mean cosine")
ax.bar([i + 0.18 for i in x], positives, width=0.36, label="positive cosine fraction")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("directional alignment metric")
ax.set_title("Spatial latents and directional loss improve residual alignment")
ax.legend()
fig.tight_layout()
fig.savefig(out_fig / "representation_loss_alignment_factorization.png", dpi=220, bbox_inches="tight")
plt.close(fig)

print(out_table / "representation_loss_factorization_summary.md")
print((out_table / "representation_loss_factorization_summary.md").read_text())
print(out_fig / "representation_loss_gain_factorization.png")
print(out_fig / "representation_loss_alignment_factorization.png")
