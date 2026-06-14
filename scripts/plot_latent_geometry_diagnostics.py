from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_conditional_residual_shrinkage import (
    load_npz,
    torch_stats_to_numpy,
    normalized_input,
    build_model,
    predict_delta,
)
from scripts.train_envsplit_residual_predictor import MLP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--mse-checkpoint", type=Path, required=True)
    p.add_argument("--directional-checkpoint", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def np_stats_from_directional_checkpoint(ckpt):
    stats = {}
    for k, v in ckpt["stats"].items():
        if hasattr(v, "detach"):
            stats[k] = v.detach().cpu().numpy().astype(np.float32)
        else:
            stats[k] = np.asarray(v, dtype=np.float32)
    return stats


@torch.no_grad()
def predict_delta_directional(ckpt_path, data, batch_size, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    stats = np_stats_from_directional_checkpoint(ckpt)

    x = normalized_input(data, stats)
    latent_dim = data["z_current"].shape[1]
    input_dim = x.shape[1]

    model = MLP(
        input_dim=input_dim,
        output_dim=latent_dim,
        hidden_dim=512,
        layers=3,
        dropout=0.05,
        zero_init_final=True,
    )
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    delta_std = torch.from_numpy(stats["delta_std"]).to(device)
    delta_mean = torch.from_numpy(stats["delta_mean"]).to(device)

    outs = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)

    for xb in loader:
        xb = xb.to(device)
        pred_norm = model(xb)
        pred_delta = pred_norm * delta_std[None, :] + delta_mean[None, :]
        outs.append(pred_delta.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outs, axis=0)


def cosine_and_angle(delta_hat, delta_true):
    dot = np.sum(delta_hat * delta_true, axis=1)
    nh = np.linalg.norm(delta_hat, axis=1)
    nt = np.linalg.norm(delta_true, axis=1)
    cos = dot / (nh * nt + 1e-12)
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos))
    return cos, angle


def pca_basis(x, n_components=2):
    x = x.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    xc = x - mean
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return mean.astype(np.float32), vt[:n_components].astype(np.float32)


def project(x, mean, basis):
    return (x - mean) @ basis.T


def save_angle_hist(angle_mse, angle_dir, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bins = np.linspace(0, 180, 60)
    ax.hist(angle_mse, bins=bins, alpha=0.55, density=True, label="MSE residual")
    ax.hist(angle_dir, bins=bins, alpha=0.55, density=True, label="Directional residual")
    ax.axvline(np.median(angle_mse), linestyle="--", linewidth=1)
    ax.axvline(np.median(angle_dir), linestyle="--", linewidth=1)
    ax.set_xlabel("angle between predicted and true latent displacement (degrees)")
    ax.set_ylabel("density")
    ax.set_title("Directional loss shifts residual angles toward the true latent motion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "angle_distribution_mse_vs_directional.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_cosine_box_by_gap(cos_mse, cos_dir, gaps, out_dir):
    for name, cos in [("mse", cos_mse), ("directional", cos_dir)]:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        data = [cos[gaps == g] for g in sorted(set(gaps.tolist()))]
        ax.boxplot(data, labels=[str(g) for g in sorted(set(gaps.tolist()))], showfliers=False)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_xlabel("prediction gap")
        ax.set_ylabel("cosine(predicted Δz, true Δz)")
        ax.set_title(f"Residual alignment by temporal gap: {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"cosine_by_gap_boxplot_{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def save_pca_scatter(coords, gaps, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=gaps, s=5, alpha=0.55)
    ax.set_xlabel("PC1 of true latent displacement")
    ax.set_ylabel("PC2 of true latent displacement")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("gap")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_arrow_overlay(true_2d, mse_2d, dir_2d, gaps, out_dir, n=250):
    rng = np.random.default_rng(0)
    idx = rng.choice(len(true_2d), size=min(n, len(true_2d)), replace=False)

    for name, pred_2d in [("mse", mse_2d), ("directional", dir_2d)]:
        fig, ax = plt.subplots(figsize=(7, 7))

        for i in idx:
            ax.arrow(
                0, 0,
                true_2d[i, 0], true_2d[i, 1],
                alpha=0.12,
                length_includes_head=True,
                head_width=0.03,
            )
            ax.arrow(
                0, 0,
                pred_2d[i, 0], pred_2d[i, 1],
                alpha=0.18,
                length_includes_head=True,
                head_width=0.03,
            )

        ax.axhline(0.0, linewidth=1)
        ax.axvline(0.0, linewidth=1)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"True vs predicted latent displacement directions: {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"latent_displacement_arrows_{name}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    data = load_npz(args.test)
    z0 = data["z_current"].astype(np.float32)
    z1 = data["z_future"].astype(np.float32)
    true_delta = z1 - z0
    gaps = data["gap"].astype(np.int32)

    print("Loading MSE residual checkpoint...")
    mse_ckpt = torch.load(args.mse_checkpoint, map_location=args.device)
    mse_stats = torch_stats_to_numpy(mse_ckpt["stats"])
    mse_model = build_model(mse_ckpt, "cheap_residual", args.device)
    delta_mse = predict_delta(
        mse_model,
        normalized_input(data, mse_stats),
        mse_stats,
        args.batch_size,
        args.device,
    )

    print("Loading directional residual checkpoint...")
    delta_dir = predict_delta_directional(
        args.directional_checkpoint,
        data,
        args.batch_size,
        args.device,
    )

    cos_mse, angle_mse = cosine_and_angle(delta_mse, true_delta)
    cos_dir, angle_dir = cosine_and_angle(delta_dir, true_delta)

    print("MSE mean cosine:", float(cos_mse.mean()))
    print("DIR mean cosine:", float(cos_dir.mean()))
    print("MSE positive frac:", float((cos_mse > 0).mean()))
    print("DIR positive frac:", float((cos_dir > 0).mean()))

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(true_delta), size=min(args.sample, len(true_delta)), replace=False)

    mean, basis = pca_basis(true_delta[idx], n_components=2)
    true_2d = project(true_delta[idx], mean, basis)
    mse_2d = project(delta_mse[idx], mean, basis)
    dir_2d = project(delta_dir[idx], mean, basis)
    gaps_s = gaps[idx]

    save_angle_hist(angle_mse, angle_dir, args.out_dir)
    save_cosine_box_by_gap(cos_mse, cos_dir, gaps, args.out_dir)

    save_pca_scatter(
        true_2d,
        gaps_s,
        "True latent displacement Δz projected by PCA",
        args.out_dir / "pca_true_delta_by_gap.png",
    )
    save_pca_scatter(
        mse_2d,
        gaps_s,
        "Predicted latent displacement Δẑ, MSE residual",
        args.out_dir / "pca_pred_delta_mse_by_gap.png",
    )
    save_pca_scatter(
        dir_2d,
        gaps_s,
        "Predicted latent displacement Δẑ, directional residual",
        args.out_dir / "pca_pred_delta_directional_by_gap.png",
    )

    save_arrow_overlay(true_2d, mse_2d, dir_2d, gaps_s, args.out_dir)

    summary = [
        "# Latent geometry visualization diagnostics",
        "",
        "| model | mean cosine | positive cosine fraction | median angle deg |",
        "| --- | ---: | ---: | ---: |",
        f"| MSE residual | {cos_mse.mean():.6f} | {(cos_mse > 0).mean():.6f} | {np.median(angle_mse):.6f} |",
        f"| Directional residual | {cos_dir.mean():.6f} | {(cos_dir > 0).mean():.6f} | {np.median(angle_dir):.6f} |",
        "",
    ]
    (args.out_dir / "latent_geometry_visual_summary.md").write_text("\n".join(summary))
    print(args.out_dir / "latent_geometry_visual_summary.md")
    print((args.out_dir / "latent_geometry_visual_summary.md").read_text())


if __name__ == "__main__":
    main()
