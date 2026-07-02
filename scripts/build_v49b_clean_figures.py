from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v49_predictive_usefulness_stack")
PAPER = Path("reports/paper_assets_v49_predictive_usefulness_stack")
FIG = ROOT / "figures"
PFIG = PAPER / "figures"
FIG.mkdir(parents=True, exist_ok=True)
PFIG.mkdir(parents=True, exist_ok=True)

STACK = ROOT / "v49_predictive_usefulness_stack.csv"
REG = ROOT / "v49_compact_compute_regime_summary.csv"
CAL = ROOT / "v49_compact_calibration_actionability_summary.csv"

stack = pd.read_csv(STACK)
reg = pd.read_csv(REG)
cal = pd.read_csv(CAL)

# ---------------------------------------------------------------------
# Clean conceptual stack figure.
# ---------------------------------------------------------------------
levels = [
    ("1", "Counterfactual\n discrimination", "Correct future under\n exact interventions"),
    ("2", "Failure-axis\n diagnosis", "Action vs state\n failure localization"),
    ("3", "Compute-regime\n map", "Useful / harmful /\n unnecessary compute"),
    ("4", "Predicted-latent\n calibration", "Discriminative does\n not imply calibrated"),
    ("5", "Decision\n actionability", "Calibrated latents\n support decisions"),
]

plt.figure(figsize=(10.5, 3.2))
ax = plt.gca()
ax.axis("off")

x_positions = np.linspace(0.08, 0.92, len(levels))
y = 0.55

for i, (num, title, subtitle) in enumerate(levels):
    x = x_positions[i]
    ax.text(
        x, y + 0.22, num,
        ha="center", va="center",
        fontsize=13, fontweight="bold",
        bbox=dict(boxstyle="circle,pad=0.35", linewidth=1.5, facecolor="white"),
        transform=ax.transAxes,
    )
    ax.text(
        x, y + 0.02, title,
        ha="center", va="center",
        fontsize=10, fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        x, y - 0.20, subtitle,
        ha="center", va="center",
        fontsize=8.5,
        transform=ax.transAxes,
    )
    if i < len(levels) - 1:
        ax.annotate(
            "",
            xy=(x_positions[i + 1] - 0.065, y + 0.22),
            xytext=(x + 0.065, y + 0.22),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.4),
        )

ax.text(
    0.5, 0.96,
    "Predictive usefulness is a diagnostic stack, not scalar accuracy",
    ha="center", va="top", fontsize=14, fontweight="bold",
    transform=ax.transAxes,
)

plt.tight_layout()
for out in [
    FIG / "fig_v49b_predictive_usefulness_stack_clean.png",
    PFIG / "fig_v49b_predictive_usefulness_stack_clean.png",
]:
    plt.savefig(out, dpi=240)
plt.close()

# ---------------------------------------------------------------------
# Clean two-map figure, with log-scale calibration so the orange bars are visible.
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

x = np.arange(len(reg))
w = 0.36
axes[0].bar(x - w/2, reg["useful_compute"], width=w, label="Useful compute")
axes[0].bar(x + w/2, reg["anti_rescue"], width=w, label="Anti-rescue")
axes[0].set_xticks(x)
axes[0].set_xticklabels(reg["variant_label"], rotation=20, ha="right")
axes[0].set_ylabel("Query fraction")
axes[0].set_title("Compute-regime map")
axes[0].legend(fontsize=8)
axes[0].grid(axis="y", alpha=0.25)

y = np.arange(len(cal))
axes[1].bar(y - w/2, cal["predicted_latent_with_true_probe_error"], width=w, label="Pred. + true probe")
axes[1].bar(y + w/2, cal["predicted_latent_with_model_specific_probe_error"], width=w, label="Pred. + specific probe")
axes[1].set_xticks(y)
axes[1].set_xticklabels(cal["model"])
axes[1].set_yscale("log")
axes[1].set_ylabel("Mean xy error, log scale")
axes[1].set_title("Calibration-regime map")
axes[1].legend(fontsize=8)
axes[1].grid(axis="y", alpha=0.25)

plt.tight_layout()
for out in [
    FIG / "fig_v49b_compute_and_calibration_maps_clean.png",
    PFIG / "fig_v49b_compute_and_calibration_maps_clean.png",
]:
    plt.savefig(out, dpi=240)
plt.close()

print(FIG / "fig_v49b_predictive_usefulness_stack_clean.png")
print(FIG / "fig_v49b_compute_and_calibration_maps_clean.png")
