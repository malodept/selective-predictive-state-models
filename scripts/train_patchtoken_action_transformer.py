from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--action-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-cos", type=float, default=0.01)
    p.add_argument("--lambda-norm", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def per_sample_mse(a, b):
    return np.mean((a - b) ** 2, axis=(1, 2))


def compute_stats(data, chunk=4096):
    z = data["z_current"]
    y = data["z_future"]
    action = data["action"].astype(np.float32)

    n, tokens, dim = z.shape

    z_sum = np.zeros((dim,), dtype=np.float64)
    z_sq = np.zeros((dim,), dtype=np.float64)
    d_sum = np.zeros((dim,), dtype=np.float64)
    d_sq = np.zeros((dim,), dtype=np.float64)
    count = 0

    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        zb = z[start:end].astype(np.float32)
        yb = y[start:end].astype(np.float32)
        db = yb - zb

        z_sum += zb.sum(axis=(0, 1))
        z_sq += (zb ** 2).sum(axis=(0, 1))
        d_sum += db.sum(axis=(0, 1))
        d_sq += (db ** 2).sum(axis=(0, 1))
        count += zb.shape[0] * zb.shape[1]

    z_mean = z_sum / count
    z_var = z_sq / count - z_mean ** 2
    z_std = np.sqrt(np.maximum(z_var, 1e-8))

    d_mean = d_sum / count
    d_var = d_sq / count - d_mean ** 2
    d_std = np.sqrt(np.maximum(d_var, 1e-8))

    return {
        "z_mean": z_mean.astype(np.float32),
        "z_std": z_std.astype(np.float32),
        "delta_mean": d_mean.astype(np.float32),
        "delta_std": d_std.astype(np.float32),
        "action_mean": action.mean(axis=0).astype(np.float32),
        "action_std": np.maximum(action.std(axis=0), 1e-6).astype(np.float32),
    }


class PatchTokenDataset(Dataset):
    def __init__(self, data):
        self.z = data["z_current"]
        self.y = data["z_future"]
        self.action = data["action"].astype(np.float32)

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.z[idx].astype(np.float32)),
            torch.from_numpy(self.y[idx].astype(np.float32)),
            torch.from_numpy(self.action[idx]),
        )


class TokenActionTransformer(nn.Module):
    def __init__(self, dim, action_dim, tokens, layers, heads, action_hidden, dropout):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, tokens, dim))

        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.GELU(),
            nn.Linear(action_hidden, dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, z, action):
        a = self.action_mlp(action).unsqueeze(1)
        h = z + self.pos + a
        h = self.encoder(h)
        return self.out(h)


def to_torch_stats(stats, device):
    return {k: torch.from_numpy(v).to(device) for k, v in stats.items()}


def normalize_batch(z, y, action, stats):
    z_norm = (z - stats["z_mean"][None, None, :]) / stats["z_std"][None, None, :]
    delta = y - z
    delta_norm = (delta - stats["delta_mean"][None, None, :]) / stats["delta_std"][None, None, :]
    action_norm = (action - stats["action_mean"][None, :]) / stats["action_std"][None, :]
    return z_norm, delta, delta_norm, action_norm


def cosine_loss(pred_delta, true_delta):
    p = pred_delta.flatten(1)
    t = true_delta.flatten(1)
    cos = torch.sum(p * t, dim=1) / (torch.norm(p, dim=1) * torch.norm(t, dim=1) + 1e-8)
    return 1.0 - cos.mean()


def lognorm_loss(pred_delta, true_delta):
    p = torch.norm(pred_delta.flatten(1), dim=1)
    t = torch.norm(true_delta.flatten(1), dim=1)
    return torch.mean((torch.log(p + 1e-6) - torch.log(t + 1e-6)) ** 2)


@torch.no_grad()
def predict_delta(model, data, stats_np, batch_size, device):
    ds = PatchTokenDataset(data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    stats = to_torch_stats(stats_np, device)
    model.eval()

    outs = []

    for z, y, action in loader:
        z = z.to(device)
        y = y.to(device)
        action = action.to(device)

        z_norm, _, _, action_norm = normalize_batch(z, y, action, stats)
        pred_norm = model(z_norm, action_norm)
        pred_delta = pred_norm * stats["delta_std"][None, None, :] + stats["delta_mean"][None, None, :]
        outs.append(pred_delta.cpu().numpy().astype(np.float32))

    return np.concatenate(outs, axis=0)


def evaluate_alpha(z, y, delta_hat):
    identity_error = float(np.mean(per_sample_mse(z, y)))
    raw_error = float(np.mean(per_sample_mse(z + delta_hat, y)))

    best_alpha = 0.0
    best_error = identity_error

    for alpha in np.linspace(0.0, 1.0, 201):
        err = float(np.mean(per_sample_mse(z + alpha * delta_hat, y)))
        if err < best_error:
            best_error = err
            best_alpha = float(alpha)

    return best_alpha, identity_error, raw_error, best_error


def eval_with_alpha(z, y, delta_hat, alpha):
    identity_error = float(np.mean(per_sample_mse(z, y)))
    raw_error = float(np.mean(per_sample_mse(z + delta_hat, y)))
    global_error = float(np.mean(per_sample_mse(z + alpha * delta_hat, y)))

    true_delta = y - z
    pred_flat = delta_hat.reshape(delta_hat.shape[0], -1)
    true_flat = true_delta.reshape(true_delta.shape[0], -1)

    cos = np.sum(pred_flat * true_flat, axis=1) / (
        np.linalg.norm(pred_flat, axis=1) * np.linalg.norm(true_flat, axis=1) + 1e-12
    )

    return {
        "identity_error": identity_error,
        "raw_error": raw_error,
        "global_alpha": float(alpha),
        "global_error": global_error,
        "global_improvement_vs_identity": identity_error - global_error,
        "cosine_mean": float(np.mean(cos)),
        "cosine_positive_frac": float(np.mean(cos > 0)),
        "true_delta_norm_median": float(np.median(np.linalg.norm(true_flat, axis=1))),
        "pred_delta_norm_median": float(np.median(np.linalg.norm(pred_flat, axis=1))),
    }


def train_one_epoch(model, loader, opt, stats, args):
    model.train()
    total = 0.0
    n = 0

    for z, y, action in loader:
        z = z.to(args.device)
        y = y.to(args.device)
        action = action.to(args.device)

        z_norm, true_delta, delta_norm, action_norm = normalize_batch(z, y, action, stats)

        pred_norm = model(z_norm, action_norm)
        pred_delta = pred_norm * stats["delta_std"][None, None, :] + stats["delta_mean"][None, None, :]

        mse = torch.mean((pred_norm - delta_norm) ** 2)
        loss = mse
        loss = loss + args.lambda_cos * cosine_loss(pred_delta, true_delta)
        loss = loss + args.lambda_norm * lognorm_loss(pred_delta, true_delta)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.detach().cpu()) * z.shape[0]
        n += z.shape[0]

    return total / n


@torch.no_grad()
def val_loss(model, loader, stats, args):
    model.eval()
    total = 0.0
    n = 0

    for z, y, action in loader:
        z = z.to(args.device)
        y = y.to(args.device)
        action = action.to(args.device)

        z_norm, _, delta_norm, action_norm = normalize_batch(z, y, action, stats)
        pred_norm = model(z_norm, action_norm)
        loss = torch.mean((pred_norm - delta_norm) ** 2)

        total += float(loss.detach().cpu()) * z.shape[0]
        n += z.shape[0]

    return total / n


def write_md(path, args, metrics):
    lines = [
        "# Patch-token Action Transformer diagnostic",
        "",
        f"Loss: `MSE + {args.lambda_cos} * cosine_loss + {args.lambda_norm} * lognorm_loss`.",
        "",
        "| split | identity error | raw error | global alpha | global error | global improvement vs identity | cosine mean | cosine positive frac | true Δ norm med | pred Δ norm med |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split in ["val", "test"]:
        r = metrics[split]
        lines.append(
            f"| {split} | {r['identity_error']:.6f} | {r['raw_error']:.6f} | "
            f"{r['global_alpha']:.6f} | {r['global_error']:.6f} | "
            f"{r['global_improvement_vs_identity']:.6f} | {r['cosine_mean']:.6f} | "
            f"{r['cosine_positive_frac']:.6f} | {r['true_delta_norm_median']:.6f} | "
            f"{r['pred_delta_norm_median']:.6f} |"
        )

    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    train_data = load_npz(args.train)
    val_data = load_npz(args.val)
    test_data = load_npz(args.test)

    print("Computing train statistics...")
    stats_np = compute_stats(train_data)
    stats = to_torch_stats(stats_np, args.device)

    _, tokens, dim = train_data["z_current"].shape
    action_dim = train_data["action"].shape[1]

    print(f"tokens={tokens}, dim={dim}, action_dim={action_dim}")

    model = TokenActionTransformer(
        dim=dim,
        action_dim=action_dim,
        tokens=tokens,
        layers=args.layers,
        heads=args.heads,
        action_hidden=args.action_hidden,
        dropout=args.dropout,
    ).to(args.device)

    train_loader = DataLoader(
        PatchTokenDataset(train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        PatchTokenDataset(val_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, stats, args)
        va = val_loss(model, val_loader, stats, args)

        print(f"epoch={epoch:03d} train={tr:.6f} val={va:.6f}", flush=True)

        if va < best_val:
            best_val = va
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if bad >= args.patience:
            print(f"early stop at epoch {epoch}, best epoch {best_epoch}", flush=True)
            break

    model.load_state_dict(best_state)

    torch.save(
        {
            "model": model.state_dict(),
            "stats": {k: torch.from_numpy(v) for k, v in stats_np.items()},
            "args": vars(args),
            "best_epoch": best_epoch,
            "best_val": best_val,
        },
        args.out_dir / "checkpoint.pt",
    )

    print("Predicting validation...")
    val_delta = predict_delta(model, val_data, stats_np, args.batch_size, args.device)
    val_z = val_data["z_current"].astype(np.float32)
    val_y = val_data["z_future"].astype(np.float32)

    alpha, _, _, _ = evaluate_alpha(val_z, val_y, val_delta)

    print("Predicting test...")
    test_delta = predict_delta(model, test_data, stats_np, args.batch_size, args.device)

    metrics = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val": best_val,
        "val": eval_with_alpha(val_z, val_y, val_delta, alpha),
        "test": eval_with_alpha(
            test_data["z_current"].astype(np.float32),
            test_data["z_future"].astype(np.float32),
            test_delta,
            alpha,
        ),
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_md(args.out_dir / "patchtoken_transformer_diagnostic.md", args, metrics)

    print(args.out_dir / "patchtoken_transformer_diagnostic.md")
    print((args.out_dir / "patchtoken_transformer_diagnostic.md").read_text())


if __name__ == "__main__":
    main()
