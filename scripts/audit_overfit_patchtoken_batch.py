from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_patchtoken_action_transformer import (
    load_npz,
    compute_stats,
    to_torch_stats,
    normalize_batch,
    TokenActionTransformer,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--action-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz(args.train)
    stats_np = compute_stats(data)
    stats = to_torch_stats(stats_np, args.device)

    n = data["z_current"].shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n, size=args.batch_size, replace=False)

    z = torch.from_numpy(data["z_current"][idx].astype(np.float32)).to(args.device)
    y = torch.from_numpy(data["z_future"][idx].astype(np.float32)).to(args.device)
    action = torch.from_numpy(data["action"][idx].astype(np.float32)).to(args.device)

    _, tokens, dim = z.shape
    action_dim = action.shape[1]

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

    z_norm, _, delta_norm, action_norm = normalize_batch(z, y, action, stats)

    with torch.no_grad():
        pred0 = model(z_norm, action_norm)
        initial_loss = torch.mean((pred0 - delta_norm) ** 2).item()

    rows = []
    print(f"initial_loss={initial_loss:.8f}", flush=True)

    for step in range(1, args.steps + 1):
        pred = model(z_norm, action_norm)
        loss = torch.mean((pred - delta_norm) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 25 == 0 or step == args.steps:
            v = float(loss.detach().cpu())
            rows.append((step, v))
            print(f"step={step:04d} loss={v:.8f}", flush=True)

    final_loss = rows[-1][1]
    ratio = final_loss / initial_loss

    lines = [
        "# Overfit-batch audit",
        "",
        f"- batch size: `{args.batch_size}`",
        f"- steps: `{args.steps}`",
        f"- initial loss: `{initial_loss:.8f}`",
        f"- final loss: `{final_loss:.8f}`",
        f"- final / initial: `{ratio:.8f}`",
        "",
        "| step | loss |",
        "| ---: | ---: |",
    ]

    for step, loss_value in rows:
        lines.append(f"| {step} | {loss_value:.8f} |")

    out = args.out_dir / "overfit_batch_audit.md"
    out.write_text("\n".join(lines) + "\n")

    print(out)
    print(out.read_text())


if __name__ == "__main__":
    main()
