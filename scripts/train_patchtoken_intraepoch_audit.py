from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_patchtoken_action_transformer import (
    load_npz,
    compute_stats,
    to_torch_stats,
    normalize_batch,
    TokenActionTransformer,
    predict_delta,
    evaluate_alpha,
    eval_with_alpha,
)


class ArrayDataset(Dataset):
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--seen-val", type=Path, required=True)
    p.add_argument("--ood-val", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--action-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--validate-every", type=int, default=100)
    p.add_argument("--eval-max", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def subset_data(data, max_n: int, seed: int):
    if max_n <= 0 or data["z_current"].shape[0] <= max_n:
        return data

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(data["z_current"].shape[0], size=max_n, replace=False))

    out = {}
    n = data["z_current"].shape[0]
    for k, v in data.items():
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            out[k] = v[idx]
        else:
            out[k] = v
    return out


@torch.no_grad()
def normalized_eval_loss(model, data, stats, batch_size, device):
    model.eval()
    ds = ArrayDataset(data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    total = 0.0
    n = 0

    for z, y, action in loader:
        z = z.to(device)
        y = y.to(device)
        action = action.to(device)

        z_norm, _, delta_norm, action_norm = normalize_batch(z, y, action, stats)
        pred = model(z_norm, action_norm)
        loss = torch.mean((pred - delta_norm) ** 2)

        total += float(loss.cpu()) * z.shape[0]
        n += z.shape[0]

    return total / n


@torch.no_grad()
def predictive_metrics(model, data, stats_np, batch_size, device):
    delta = predict_delta(model, data, stats_np, batch_size, device)

    z = data["z_current"].astype(np.float32)
    y = data["z_future"].astype(np.float32)

    alpha, _, _, _ = evaluate_alpha(z, y, delta)
    r = eval_with_alpha(z, y, delta, alpha)

    return {
        "alpha": float(alpha),
        "identity_error": float(r["identity_error"]),
        "raw_error": float(r["raw_error"]),
        "global_error": float(r["global_error"]),
        "gain": float(r["global_improvement_vs_identity"]),
        "cosine": float(r["cosine_mean"]),
        "positive_frac": float(r["cosine_positive_frac"]),
    }


def save_ckpt(path, model, stats_np, args, step, score):
    torch.save(
        {
            "model": model.state_dict(),
            "stats": {k: torch.from_numpy(v) for k, v in stats_np.items()},
            "args": vars(args),
            "best_step": step,
            "best_val": score,
            "best_epoch": step,
        },
        path,
    )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train = load_npz(args.train)
    seen_val_full = load_npz(args.seen_val)
    ood_val_full = load_npz(args.ood_val)

    seen_val = subset_data(seen_val_full, args.eval_max, args.seed)
    ood_val = subset_data(ood_val_full, args.eval_max, args.seed + 1)

    stats_np = compute_stats(train)
    stats = to_torch_stats(stats_np, args.device)

    _, tokens, dim = train["z_current"].shape
    action_dim = train["action"].shape[1]

    model = TokenActionTransformer(
        dim=dim,
        action_dim=action_dim,
        tokens=tokens,
        layers=args.layers,
        heads=args.heads,
        action_hidden=args.action_hidden,
        dropout=args.dropout,
    ).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ds = ArrayDataset(train)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    steps_per_epoch = math.ceil(len(ds) / args.batch_size)
    max_steps = int(round(args.epochs * steps_per_epoch))

    rows = []
    best_seen = float("inf")
    best_ood = float("inf")

    def run_eval(step):
        nonlocal best_seen, best_ood

        seen_norm = normalized_eval_loss(model, seen_val, stats, args.batch_size, args.device)
        ood_norm = normalized_eval_loss(model, ood_val, stats, args.batch_size, args.device)

        seen_pred = predictive_metrics(model, seen_val, stats_np, args.batch_size, args.device)
        ood_pred = predictive_metrics(model, ood_val, stats_np, args.batch_size, args.device)

        row = {
            "step": step,
            "epoch_frac": step / steps_per_epoch,
            "seen_norm": seen_norm,
            "ood_norm": ood_norm,
            "seen_global_error": seen_pred["global_error"],
            "ood_global_error": ood_pred["global_error"],
            "seen_gain": seen_pred["gain"],
            "ood_gain": ood_pred["gain"],
            "seen_cosine": seen_pred["cosine"],
            "ood_cosine": ood_pred["cosine"],
            "seen_alpha": seen_pred["alpha"],
            "ood_alpha": ood_pred["alpha"],
        }

        rows.append(row)

        print(
            f"eval step={step} epoch={row['epoch_frac']:.3f} "
            f"seen_norm={seen_norm:.6f} ood_norm={ood_norm:.6f} "
            f"seen_err={seen_pred['global_error']:.6f} ood_err={ood_pred['global_error']:.6f} "
            f"ood_gain={ood_pred['gain']:.6f}",
            flush=True,
        )

        if seen_norm < best_seen:
            best_seen = seen_norm
            save_ckpt(args.out_dir / "checkpoint_best_seen.pt", model, stats_np, args, step, seen_norm)

        if ood_norm < best_ood:
            best_ood = ood_norm
            save_ckpt(args.out_dir / "checkpoint_best_ood.pt", model, stats_np, args, step, ood_norm)
            save_ckpt(args.out_dir / "checkpoint.pt", model, stats_np, args, step, ood_norm)

    print("train samples:", len(ds))
    print("steps_per_epoch:", steps_per_epoch)
    print("max_steps:", max_steps)
    print("seen_val eval samples:", seen_val["z_current"].shape[0])
    print("ood_val eval samples:", ood_val["z_current"].shape[0])

    run_eval(step=0)

    step = 0
    model.train()

    while step < max_steps:
        for z, y, action in loader:
            step += 1

            z = z.to(args.device)
            y = y.to(args.device)
            action = action.to(args.device)

            z_norm, _, delta_norm, action_norm = normalize_batch(z, y, action, stats)
            pred = model(z_norm, action_norm)
            loss = torch.mean((pred - delta_norm) ** 2)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step == 1 or step % args.validate_every == 0 or step == max_steps:
                print(f"train step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
                run_eval(step=step)
                model.train()

            if step >= max_steps:
                break

    save_ckpt(args.out_dir / "checkpoint_final.pt", model, stats_np, args, step, float(loss.detach().cpu()))

    # Write json/csv-like markdown
    (args.out_dir / "intraepoch_rows.json").write_text(json.dumps(rows, indent=2))

    lines = [
        "# Intra-epoch training audit",
        "",
        f"- train: `{args.train}`",
        f"- seen-val: `{args.seen_val}`",
        f"- ood-val: `{args.ood_val}`",
        f"- steps per epoch: `{steps_per_epoch}`",
        f"- max steps: `{max_steps}`",
        "",
        "| step | epoch frac | seen norm loss | ood norm loss | seen global error | ood global error | seen gain | ood gain | seen cosine | ood cosine | seen alpha | ood alpha |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in rows:
        lines.append(
            f"| {r['step']} | {r['epoch_frac']:.4f} | "
            f"{r['seen_norm']:.6f} | {r['ood_norm']:.6f} | "
            f"{r['seen_global_error']:.6f} | {r['ood_global_error']:.6f} | "
            f"{r['seen_gain']:.6f} | {r['ood_gain']:.6f} | "
            f"{r['seen_cosine']:.6f} | {r['ood_cosine']:.6f} | "
            f"{r['seen_alpha']:.6f} | {r['ood_alpha']:.6f} |"
        )

    out = args.out_dir / "intraepoch_training_audit.md"
    out.write_text("\n".join(lines) + "\n")
    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
