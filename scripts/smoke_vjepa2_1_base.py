from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VJEPA = ROOT / "external" / "vjepa2"

sys.path.insert(0, str(VJEPA))
sys.path.insert(0, str(VJEPA / "src"))

from src.hub.backbones import vjepa2_1_vit_base_384, _clean_backbone_key


def describe(x, prefix="out"):
    if torch.is_tensor(x):
        print(prefix, "tensor", tuple(x.shape), x.dtype, x.device)
    elif isinstance(x, (list, tuple)):
        print(prefix, type(x), "len", len(x))
        for i, y in enumerate(x):
            describe(y, f"{prefix}[{i}]")
    elif isinstance(x, dict):
        print(prefix, "dict", x.keys())
        for k, v in x.items():
            describe(v, f"{prefix}.{k}")
    else:
        print(prefix, type(x), x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch", type=int, default=1)
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    print("device:", args.device)
    print("checkpoint:", args.checkpoint)

    print("Building model...")
    encoder, predictor = vjepa2_1_vit_base_384(
        pretrained=False,
        num_frames=args.frames,
    )

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print("checkpoint keys:", ckpt.keys())

    key = "ema_encoder" if "ema_encoder" in ckpt else "target_encoder"
    print("using key:", key)

    sd = _clean_backbone_key(ckpt[key])
    msg = encoder.load_state_dict(sd, strict=False)
    print("load_state_dict:", msg)

    encoder = encoder.to(args.device).eval()

    # V-JEPA expects video tensor [B, C, T, H, W].
    x = torch.randn(args.batch, 3, args.frames, 384, 384, device=args.device)

    print("input:", tuple(x.shape), x.dtype)

    with torch.no_grad():
        if args.device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = encoder(x)
        else:
            out = encoder(x)

    describe(out, "encoder_out")


if __name__ == "__main__":
    main()
