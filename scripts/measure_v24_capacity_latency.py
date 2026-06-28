from __future__ import annotations

from pathlib import Path
import time
import json
import numpy as np
import pandas as pd
import torch

from train_mixed_hard_delta_transformer import DeltaTransformer

OUT_DIR = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v24_measured_latency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "capacity_latency_summary.csv"
OUT_MD = OUT_DIR / "capacity_latency_summary.md"

DATA = Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz")

CKPT_ROOT = Path("outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer")

MODELS = [
    ("tiny_full_seed0", "Tiny"),
    ("small_full_seed0", "Small"),
    ("medium_full_seed0", "Medium"),
    ("full_seed0", "Full"),
]

BATCH_SIZES = [1, 8, 32, 128, 512]
WARMUP = 80
ITERS = 300


def load_model(path: Path, device: str):
    ckpt = torch.load(path, map_location="cpu")
    ca = ckpt["args"]

    model = DeltaTransformer(
        token_dim=int(ckpt["token_dim"]),
        action_dim=int(ckpt["action_dim"]),
        tokens=int(ckpt["tokens"]),
        model_dim=int(ca["model_dim"]),
        heads=int(ca["heads"]),
        layers=int(ca["layers"]),
        dropout=float(ca["dropout"]),
        mode=ca["mode"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    return model, ckpt, n_params


@torch.no_grad()
def measure(model, z_pool, a_pool, batch_size: int, device: str):
    n = len(z_pool)
    if batch_size > n:
        reps = int(np.ceil(batch_size / n))
        z_np = np.concatenate([z_pool] * reps, axis=0)[:batch_size]
        a_np = np.concatenate([a_pool] * reps, axis=0)[:batch_size]
    else:
        z_np = z_pool[:batch_size]
        a_np = a_pool[:batch_size]

    z = torch.from_numpy(z_np).float().to(device)
    a = torch.from_numpy(a_np).float().to(device)

    for _ in range(WARMUP):
        _ = model(z, a)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(ITERS):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model(z, a)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    arr = np.asarray(times, dtype=np.float64)
    return {
        "batch_size": batch_size,
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_std": float(arr.std(ddof=1)),
        "latency_ms_median": float(np.median(arr)),
        "latency_ms_p10": float(np.quantile(arr, 0.10)),
        "latency_ms_p90": float(np.quantile(arr, 0.90)),
        "per_item_ms_median": float(np.median(arr) / batch_size),
        "throughput_items_per_s_median": float(batch_size / (np.median(arr) / 1000.0)),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    d = np.load(DATA, allow_pickle=True)
    z_pool = d["z_current"].astype(np.float32)
    a_pool = d["action"].astype(np.float32)

    rows = []

    for model_name, label in MODELS:
        ckpt_path = CKPT_ROOT / model_name / "checkpoint.pt"
        model, ckpt, n_params = load_model(ckpt_path, device)
        ca = ckpt["args"]

        print(f"===== {label} =====", flush=True)
        print("checkpoint:", ckpt_path, flush=True)
        print("params:", n_params, flush=True)

        for b in BATCH_SIZES:
            m = measure(model, z_pool, a_pool, b, device)
            row = {
                "model": model_name,
                "model_label": label,
                "device": device,
                "checkpoint_mb": ckpt_path.stat().st_size / 1024 / 1024,
                "params": n_params,
                "model_dim": int(ca["model_dim"]),
                "layers": int(ca["layers"]),
                "heads": int(ca["heads"]),
                **m,
            }
            rows.append(row)
            print(row, flush=True)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    full = df[(df["model_label"] == "Full") & (df["batch_size"] == 128)]["latency_ms_median"].iloc[0]
    df["relative_latency_b128"] = df["latency_ms_median"] / full

    df.to_csv(OUT_CSV, index=False)

    lines = []
    lines.append("# SPSM v24 measured capacity latency\n")
    lines.append("This measures forward-pass latency for the capacity ladder using the actual DeltaTransformer checkpoints.")
    lines.append("The main cost proxy for v25 should use measured latency, not checkpoint size.\n")

    lines.append("## Median latency by batch size\n")
    lines.append("| model | params | checkpoint MB | batch | median ms | per-item ms | throughput items/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")

    for _, r in df.iterrows():
        lines.append(
            f"| {r['model_label']} | {int(r['params'])} | {r['checkpoint_mb']:.2f} | "
            f"{int(r['batch_size'])} | {r['latency_ms_median']:.4f} | "
            f"{r['per_item_ms_median']:.6f} | {r['throughput_items_per_s_median']:.1f} |"
        )

    lines.append("\n## Batch-128 relative latency\n")
    lines.append("| model | median ms @128 | relative latency vs Full |")
    lines.append("|---|---:|---:|")

    b128 = df[df["batch_size"] == 128].copy()
    for _, r in b128.iterrows():
        lines.append(
            f"| {r['model_label']} | {r['latency_ms_median']:.4f} | {r['relative_latency_b128']:.6f} |"
        )

    lines.append("\n## Interpretation\n")
    lines.append("- Use batch-128 latency as the first measured compute proxy because the evaluation scripts use grouped batches.")
    lines.append("- If measured latency differs strongly from checkpoint-size cost, v23 should be rebuilt as v25 using latency.")
    lines.append("- Per-item latency at batch 1 is also useful for real-time deployment analysis.")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print(OUT_MD)
    print(OUT_CSV)
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
