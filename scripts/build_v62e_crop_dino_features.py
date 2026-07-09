from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

def pick_image_keys(d):
    keys = list(d.keys())
    candidates = [k for k in keys if d[k].ndim == 4 and d[k].shape[-1] in (3, 4)]

    cur = [k for k in candidates if "current" in k.lower()]
    fut = [k for k in candidates if "future" in k.lower()]

    if cur and fut:
        return cur[0], fut[0]

    print("Available keys:")
    for k in keys:
        print(k, d[k].shape, d[k].dtype)
    raise SystemExit("Could not infer current/future image keys.")

def crop_center(img, cx, cy, crop_size, out_size):
    h, w = img.shape[:2]
    half = crop_size // 2

    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + crop_size
    y1 = y0 + crop_size

    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)

    if pad_l or pad_t or pad_r or pad_b:
        img = np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        x0 += pad_l
        x1 += pad_l
        y0 += pad_t
        y1 += pad_t

    crop = img[y0:y1, x0:x1, :3]
    crop = Image.fromarray(crop.astype(np.uint8)).resize((out_size, out_size), Image.BICUBIC)
    return np.asarray(crop, dtype=np.uint8)

def load_dino(device):
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model

def preprocess(batch):
    x = torch.as_tensor(batch, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
    return (x - mean) / std

@torch.no_grad()
def encode(model, images, batch_size, device, pool_grid):
    outs = []
    for s in range(0, len(images), batch_size):
        batch = preprocess(images[s:s + batch_size]).to(device)
        feats = model.forward_features(batch)

        if isinstance(feats, dict):
            tok = feats["x_norm_patchtokens"]
        else:
            tok = feats

        # tok: B x N x C, usually N=16*16 for 224 with ViT/14.
        b, n, c = tok.shape
        side = int(round(n ** 0.5))

        if side * side == n and pool_grid > 0 and side % pool_grid == 0:
            tok = tok.reshape(b, side, side, c)
            step = side // pool_grid
            tok = tok.reshape(b, pool_grid, step, pool_grid, step, c).mean(dim=(2, 4))
            tok = tok.reshape(b, pool_grid * pool_grid, c)
        else:
            tok = tok.mean(dim=1, keepdim=True)

        outs.append(tok.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(outs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-size", type=int, default=320)
    ap.add_argument("--resize", type=int, default=224)
    ap.add_argument("--pool-grid", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = dict(np.load(args.raw))
    cur_key, fut_key = pick_image_keys(d)
    print("current image key:", cur_key)
    print("future image key:", fut_key)

    cur_img = d[cur_key]
    fut_img = d[fut_key]
    state = d["state"].astype(np.float32)

    n = len(state)
    h, w = cur_img.shape[1:3]
    scale_x = w / 512.0
    scale_y = h / 512.0

    crops_cur = []
    crops_fut = []

    for i in range(n):
        # Block center in simulator coordinates is state[2:4].
        # Use the initial block center for both current and future crops,
        # so the future crop preserves the block displacement.
        cx = float(state[i, 2]) * scale_x
        cy = float(state[i, 3]) * scale_y

        crops_cur.append(crop_center(cur_img[i], cx, cy, args.crop_size, args.resize))
        crops_fut.append(crop_center(fut_img[i], cx, cy, args.crop_size, args.resize))

        if (i + 1) % 500 == 0:
            print(f"cropped {i+1}/{n}")

    crops_cur = np.asarray(crops_cur, dtype=np.uint8)
    crops_fut = np.asarray(crops_fut, dtype=np.uint8)

    model = load_dino(args.device)
    zc = encode(model, crops_cur, args.batch_size, args.device, args.pool_grid)
    zf = encode(model, crops_fut, args.batch_size, args.device, args.pool_grid)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "z_current": zc,
        "z_future": zf,
        "state": d["state"],
        "future_state": d["future_state"],
        "action": d["action"],
        "state_id": d["state_id"],
        "action_id": d["action_id"],
        "crop_size": np.asarray(args.crop_size),
        "resize": np.asarray(args.resize),
        "pool_grid": np.asarray(args.pool_grid),
    }

    if "coverage" in d:
        payload["coverage"] = d["coverage"]

    np.savez_compressed(out, **payload)

    print("wrote", out)
    print("z_current", zc.shape, zc.dtype)
    print("z_future", zf.shape, zf.dtype)

if __name__ == "__main__":
    main()
