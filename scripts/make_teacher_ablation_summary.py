from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


runs = [
    "ResNet18\nheuristic",
    "ResNet18\nerror-rel",
    "DINOv2\nheuristic",
    "DINOv2\nerror-rel",
]

metrics = {
    "R@10": [0.6105, 0.5450, 0.8394, 0.8334],
    "Expected learned AUROC": [0.9881, 0.7602, 0.9779, 0.7669],
    "Expected residual AUROC": [0.5790, 0.9797, 0.5506, 0.9780],
    "Observed residual AUROC": [0.9602, 0.9554, 0.9845, 0.9845],
}

x = np.arange(len(runs))
width = 0.18

fig, ax = plt.subplots(figsize=(12, 6))

for i, (name, values) in enumerate(metrics.items()):
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, values, width, label=name)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

ax.set_xticks(x)
ax.set_xticklabels(runs)
ax.set_ylim(0, 1.10)
ax.set_ylabel("Score")
ax.set_title("Teacher ablation: retrieval and reliability diagnostics")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

out = Path("reports/figures/teacher_ablation_summary.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=220)

print(f"Wrote {out}")
