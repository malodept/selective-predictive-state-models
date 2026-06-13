from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict

import numpy as np


BASE = Path("outputs/tartanair_dinov2_features_multi_env/env_split")
SPLITS = {
    "train": BASE / "features_train.npz",
    "val": BASE / "features_val.npz",
    "test": BASE / "features_test.npz",
}


def mse_per_sample(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b) ** 2).mean(axis=1)


def load(path: Path) -> dict[str, np.ndarray]:
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def describe_split(name: str, data: dict[str, np.ndarray], train_mean_y: np.ndarray) -> list[dict[str, object]]:
    z0 = data["z_current"].astype(np.float32)
    z1 = data["z_future"].astype(np.float32)
    action = data["action"].astype(np.float32)
    env = data["environment"].astype(str)
    traj = data["trajectory_id"].astype(str)

    identity_err = mse_per_sample(z0, z1)
    mean_pred = np.broadcast_to(train_mean_y[None, :], z1.shape)
    mean_err = mse_per_sample(mean_pred, z1)
    action_norm = np.linalg.norm(action, axis=1)
    z0_norm = np.linalg.norm(z0, axis=1)
    z1_norm = np.linalg.norm(z1, axis=1)

    rows = []

    def add_row(group: str, mask: np.ndarray) -> None:
        rows.append(
            {
                "split": name,
                "group": group,
                "samples": int(mask.sum()),
                "trajectories": len(set(traj[mask])),
                "identity_error": float(identity_err[mask].mean()),
                "train_mean_error": float(mean_err[mask].mean()),
                "action_norm": float(action_norm[mask].mean()),
                "z_current_norm": float(z0_norm[mask].mean()),
                "z_future_norm": float(z1_norm[mask].mean()),
            }
        )

    add_row("ALL", np.ones(len(z0), dtype=bool))

    for e in sorted(set(env)):
        add_row(e, env == e)

    return rows


def main() -> None:
    datasets = {k: load(p) for k, p in SPLITS.items()}

    train_mean_y = datasets["train"]["z_future"].astype(np.float32).mean(axis=0)

    rows = []
    for name, data in datasets.items():
        rows.extend(describe_split(name, data, train_mean_y))

    out = Path("reports/tables/protocol/envsplit_data_baseline_diagnostics.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Environment split data baseline diagnostics",
        "",
        "| split | group | samples | trajectories | identity error | train-mean error | action norm | z_current norm | z_future norm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['split']} | {r['group']} | {r['samples']} | {r['trajectories']} | "
            f"{r['identity_error']:.4f} | {r['train_mean_error']:.4f} | "
            f"{r['action_norm']:.4f} | {r['z_current_norm']:.4f} | {r['z_future_norm']:.4f} |"
        )

    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
