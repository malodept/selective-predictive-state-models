from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.train_patchtoken_action_transformer import (
    load_npz,
    compute_stats,
    PatchTokenDataset,
    to_torch_stats,
    train_one_epoch,
    val_loss,
    predict_delta,
    evaluate_alpha,
    eval_with_alpha,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--hidden", type=int, default=768)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--action-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--lambda-cos", type=float, default=0.0)
    p.add_argument("--lambda-norm", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class TokenWiseActionMLP(nn.Module):
    """
    Token-preserving baseline without token-token communication.

    Each token is predicted independently from:
      - its current DINOv2 patch-token state
      - its learned spatial position embedding
      - the global action embedding

    This tests whether the Transformer gain comes from preserving tokens alone
    or from attention-based interaction between tokens.
    """

    def __init__(self, dim, action_dim, tokens, hidden, layers, action_hidden, dropout):
        super().__init__()

        self.pos = nn.Parameter(torch.zeros(1, tokens, dim))

        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.GELU(),
            nn.Linear(action_hidden, dim),
        )

        blocks = []
        in_dim = 2 * dim

        for i in range(layers):
            blocks.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            blocks.append(nn.GELU())
            blocks.append(nn.Dropout(dropout))

        blocks.append(nn.Linear(hidden, dim))
        self.mlp = nn.Sequential(*blocks)

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, action):
        a = self.action_mlp(action).unsqueeze(1).expand(-1, z.shape[1], -1)
        h = torch.cat([z + self.pos, a], dim=-1)
        return self.mlp(h)


def write_md(path, args, metrics):
    lines = [
        "# Patch-token Action MLP diagnostic",
        "",
        "This baseline preserves the 16 DINOv2 patch tokens but removes token-token attention.",
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

    model = TokenWiseActionMLP(
        dim=dim,
        action_dim=action_dim,
        tokens=tokens,
        hidden=args.hidden,
        layers=args.layers,
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
    write_md(args.out_dir / "patchtoken_mlp_diagnostic.md", args, metrics)

    print(args.out_dir / "patchtoken_mlp_diagnostic.md")
    print((args.out_dir / "patchtoken_mlp_diagnostic.md").read_text())


if __name__ == "__main__":
    main()
