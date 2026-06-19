from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


DINO_REPO = "/shared/home/mdepastor/.cache/torch/hub/facebookresearch_dinov2_main"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--model", default="dinov2_vits14")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--keep-rgb", action="store_true")
    return p.parse_args()


def preprocess_uint8_rgb(batch: np.ndarray, image_size: int, device: str) -> torch.Tensor:
    """Convert uint8 RGB [B,H,W,3] to normalized DINOv2 tensor [B,3,S,S]."""
    x = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2) / 255.0

    x = F.interpolate(
        x,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_tokens(
    model: torch.nn.Module,
    rgb: np.ndarray,
    image_size: int,
    batch_size: int,
    device: str,
):
    model.eval()

    patch_chunks = []
    cls_chunks = []

    n = len(rgb)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = rgb[start:end]

        x = preprocess_uint8_rgb(batch, image_size=image_size, device=device)
        out = model.forward_features(x)

        patch = out["x_norm_patchtokens"].detach().cpu().to(torch.float16).numpy()
        cls = out["x_norm_clstoken"].detach().cpu().to(torch.float16).numpy()

        patch_chunks.append(patch)
        cls_chunks.append(cls)

        print(f"extract {end}/{n}", flush=True)

    return np.concatenate(patch_chunks, axis=0), np.concatenate(cls_chunks, axis=0)


def maybe_copy(src, key: str, dst: dict):
    if key in src.files:
        dst[key] = src[key]


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(args.input, allow_pickle=True)

    current_rgb = d["current_rgb"]
    future_rgb = d["future_rgb"]

    print("input:", args.input, flush=True)
    print("current_rgb:", current_rgb.shape, current_rgb.dtype, flush=True)
    print("future_rgb :", future_rgb.shape, future_rgb.dtype, flush=True)
    print("loading DINOv2:", args.model, flush=True)

    model = torch.hub.load(DINO_REPO, args.model, source="local", pretrained=True)
    model = model.to(args.device).eval()

    print("extract current", flush=True)
    z_current, cls_current = extract_tokens(
        model=model,
        rgb=current_rgb,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("extract future", flush=True)
    z_future, cls_future = extract_tokens(
        model=model,
        rgb=future_rgb,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    out = {
        "z_current": z_current,
        "z_future": z_future,
        "cls_current": cls_current,
        "cls_future": cls_future,
        "action": d["action"].astype(np.float32),
        "candidate_indices": d["candidate_indices"].astype(np.int64),
        "correct_candidate": d["correct_candidate"].astype(np.int64),
        "encoder": np.asarray(f"{args.model}_patchtokens_224"),
        "latent_shape": np.asarray(z_current.shape[1:]),
        "latent_tokens": np.asarray(z_current.shape[1]),
        "latent_dim": np.asarray(z_current.shape[2]),
        "action_dim": np.asarray(d["action"].shape[1]),
        "source_npz": np.asarray(str(args.input)),
        "image_size": np.asarray(args.image_size),
    }

    for key in [
        "group_id",
        "action_id",
        "action_names",
        "state_xy",
        "future_xy",
        "blocked_mask",
        "patch_grid",
        "horizon",
        "velocity",
        "num_blocked_directions",
        "dataset_version",
    ]:
        maybe_copy(d, key, out)

    if args.keep_rgb:
        out["current_rgb"] = current_rgb
        out["future_rgb"] = future_rgb

    np.savez_compressed(args.output, **out)

    report = {
        "input": str(args.input),
        "output": str(args.output),
        "model": args.model,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "device": args.device,
        "z_current_shape": list(z_current.shape),
        "z_future_shape": list(z_future.shape),
        "z_dtype": str(z_current.dtype),
        "cls_current_shape": list(cls_current.shape),
        "cls_future_shape": list(cls_future.shape),
        "action_shape": list(out["action"].shape),
        "candidate_indices_shape": list(out["candidate_indices"].shape),
        "output_size_gb": args.output.stat().st_size / 1e9,
    }

    args.report.write_text(json.dumps(report, indent=2) + "\n")

    md = args.report.with_suffix(".md")
    lines = [
        "# PyBullet obstacle DINOv2 feature extraction",
        "",
        f"- input: `{args.input}`",
        f"- output: `{args.output}`",
        f"- model: `{args.model}`",
        f"- image size: `{args.image_size}`",
        f"- z_current: `{tuple(z_current.shape)}` `{z_current.dtype}`",
        f"- z_future: `{tuple(z_future.shape)}` `{z_future.dtype}`",
        f"- cls_current: `{tuple(cls_current.shape)}` `{cls_current.dtype}`",
        f"- cls_future: `{tuple(cls_future.shape)}` `{cls_future.dtype}`",
        f"- action: `{tuple(out['action'].shape)}`",
        f"- candidate groups: `{tuple(out['candidate_indices'].shape)}`",
        f"- output size GB: `{report['output_size_gb']:.4f}`",
        "",
        "## Interpretation",
        "",
        "This replaces engineered PyBullet patch tokens with frozen DINOv2 patch-token representations extracted from RGB observations.",
        "The resulting file is compatible with the exact-intervention oracle and ablation scripts.",
    ]

    md.write_text("\n".join(lines) + "\n")

    print(md, flush=True)
    print(md.read_text(), flush=True)


if __name__ == "__main__":
    main()
