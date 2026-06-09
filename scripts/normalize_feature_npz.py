from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eps", type=float, default=1e-8)
    return p.parse_args()


def l2_normalize(x: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norm, eps)).astype(np.float32)


def main() -> None:
    args = parse_args()

    raw = np.load(args.input, allow_pickle=True)
    out = {k: raw[k] for k in raw.files}

    for key in ["z_current", "z_future"]:
        if key not in out:
            raise KeyError(f"Missing key: {key}")
        out[key] = l2_normalize(out[key].astype(np.float32), args.eps)

    out["normalization"] = np.asarray("l2_latents")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)

    print(f"Wrote {args.output}")
    print("z_current norm mean:", np.linalg.norm(out["z_current"], axis=1).mean())
    print("z_future norm mean:", np.linalg.norm(out["z_future"], axis=1).mean())


if __name__ == "__main__":
    main()
