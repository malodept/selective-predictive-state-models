from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--test", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--action-hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-groups", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--lambda-match", type=float, default=0.2)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--candidates", type=int, default=5)
    p.add_argument("--max-train-groups", type=int, default=0)
    p.add_argument("--max-eval-groups", type=int, default=8000)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def build_candidate_groups(data, candidates: int, max_groups: int, seed: int):
    buckets = defaultdict(list)
    tids = np.asarray(data["trajectory_id"]).astype(str)
    frame_i = data["frame_i"].astype(int)

    for idx, (tid, fi) in enumerate(zip(tids, frame_i)):
        buckets[(tid, int(fi))].append(idx)

    groups = []
    for _, idxs in buckets.items():
        idxs = sorted(idxs, key=lambda u: int(data["gap"][u]))
        if len(idxs) >= candidates:
            groups.append(idxs[:candidates])

    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    if max_groups > 0:
        groups = groups[:max_groups]

    return np.asarray(groups, dtype=np.int64)


class CandidateGroupDataset(Dataset):
    def __init__(self, data, groups):
        self.z = data["z_current"]
        self.y = data["z_future"]
        self.action = data["action"].astype(np.float32)
        self.groups = groups

    def __len__(self):
        return self.groups.shape[0]

    def __getitem__(self, idx):
        g = self.groups[idx]
        return (
            torch.from_numpy(self.z[g].astype(np.float32)),
            torch.from_numpy(self.y[g].astype(np.float32)),
            torch.from_numpy(self.action[g]),
        )


def candidate_loss(pred_delta_norm, true_delta_norm, batch_size, k, temperature):
    _, tokens, dim = pred_delta_norm.shape

    pred = pred_delta_norm.reshape(batch_size, k, tokens, dim)
    true = true_delta_norm.reshape(batch_size, k, tokens, dim)

    dist = torch.mean(
        (pred[:, :, None, :, :] - true[:, None, :, :, :]) ** 2,
        dim=(-1, -2),
    )  # [B, K, K]

    logits = -dist / temperature
    targets = torch.arange(k, device=logits.device).repeat(batch_size)

    row_loss = nn.functional.cross_entropy(logits.reshape(batch_size * k, k), targets)
    col_loss = nn.functional.cross_entropy(logits.transpose(1, 2).reshape(batch_size * k, k), targets)

    return 0.5 * (row_loss + col_loss), dist.detach()


def train_epoch(model, loader, opt, stats, args):
    model.train()
    total = 0.0
    total_mse = 0.0
    total_match = 0.0
    n = 0

    for z, y, action in loader:
        b, k, tokens, dim = z.shape

        z = z.reshape(b * k, tokens, dim).to(args.device)
        y = y.reshape(b * k, tokens, dim).to(args.device)
        action = action.reshape(b * k, action.shape[-1]).to(args.device)

        z_norm, _, delta_norm, action_norm = normalize_batch(z, y, action, stats)
        pred_norm = model(z_norm, action_norm)

        mse = torch.mean((pred_norm - delta_norm) ** 2)
        match, _ = candidate_loss(pred_norm, delta_norm, b, k, args.temperature)

        loss = mse + args.lambda_match * match

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += float(loss.detach().cpu()) * b
        total_mse += float(mse.detach().cpu()) * b
        total_match += float(match.detach().cpu()) * b
        n += b

    return total / n, total_mse / n, total_match / n


@torch.no_grad()
def eval_group_loss(model, loader, stats, args):
    model.eval()
    total = 0.0
    total_mse = 0.0
    total_match = 0.0
    n = 0

    for z, y, action in loader:
        b, k, tokens, dim = z.shape

        z = z.reshape(b * k, tokens, dim).to(args.device)
        y = y.reshape(b * k, tokens, dim).to(args.device)
        action = action.reshape(b * k, action.shape[-1]).to(args.device)

        z_norm, _, delta_norm, action_norm = normalize_batch(z, y, action, stats)
        pred_norm = model(z_norm, action_norm)

        mse = torch.mean((pred_norm - delta_norm) ** 2)
        match, _ = candidate_loss(pred_norm, delta_norm, b, k, args.temperature)

        loss = mse + args.lambda_match * match

        total += float(loss.detach().cpu()) * b
        total_mse += float(mse.detach().cpu()) * b
        total_match += float(match.detach().cpu()) * b
        n += b

    return total / n, total_mse / n, total_match / n


def make_subset(data, groups):
    flat = groups.reshape(-1)
    out = {}

    n = data["action"].shape[0]
    for k, v in data.items():
        if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == n:
            out[k] = v[flat]
        else:
            out[k] = v

    slices = []
    pos = 0
    for _ in range(groups.shape[0]):
        slices.append((pos, pos + groups.shape[1]))
        pos += groups.shape[1]

    return out, slices


def make_action_intervention(actions, slices, mode, seed):
    out = actions.copy()
    rng = np.random.default_rng(seed)

    if mode == "original":
        return out

    if mode == "zero":
        return np.zeros_like(out)

    if mode == "within_group_shuffle":
        for a, b in slices:
            perm = rng.permutation(b - a)
            out[a:b] = out[a:b][perm]
        return out

    if mode == "within_group_reverse":
        for a, b in slices:
            out[a:b] = out[a:b][::-1]
        return out

    raise ValueError(mode)


def mse_matrix(pred, true):
    diff = pred[:, None, :, :] - true[None, :, :, :]
    return np.mean(diff ** 2, axis=(2, 3))


def eval_candidate_matching(pred, true, slices):
    top1 = []
    margins = []
    diag_mse = []
    best_offdiag = []

    for a, b in slices:
        dist = mse_matrix(pred[a:b], true[a:b])
        k = b - a

        for r in range(k):
            order = np.argsort(dist[r])
            top1.append(float(order[0] == r))

            diag = float(dist[r, r])
            off = np.delete(dist[r], r)
            best_off = float(np.min(off))

            diag_mse.append(diag)
            best_offdiag.append(best_off)
            margins.append(best_off - diag)

    margins = np.asarray(margins)
    return {
        "top1_accuracy": float(np.mean(top1)),
        "mean_diag_mse": float(np.mean(diag_mse)),
        "mean_best_offdiag_mse": float(np.mean(best_offdiag)),
        "mean_margin": float(np.mean(margins)),
        "positive_margin_frac": float(np.mean(margins > 0)),
    }


def evaluate_full(model, stats_np, val_data, test_data, args):
    val_delta = predict_delta(model, val_data, stats_np, args.eval_batch_size, args.device)
    val_z = val_data["z_current"].astype(np.float32)
    val_y = val_data["z_future"].astype(np.float32)

    alpha, _, _, _ = evaluate_alpha(val_z, val_y, val_delta)

    test_delta = predict_delta(model, test_data, stats_np, args.eval_batch_size, args.device)
    test_z = test_data["z_current"].astype(np.float32)
    test_y = test_data["z_future"].astype(np.float32)

    return {
        "alpha": alpha,
        "val": eval_with_alpha(val_z, val_y, val_delta, alpha),
        "test": eval_with_alpha(test_z, test_y, test_delta, alpha),
    }


def evaluate_matching_modes(model, stats_np, test_data, groups, alpha, args):
    subset, slices = make_subset(test_data, groups)
    z0 = subset["z_current"].astype(np.float32)
    true = subset["z_future"].astype(np.float32)
    actions = subset["action"].astype(np.float32)

    rows = []

    for mode in ["original", "zero", "within_group_shuffle", "within_group_reverse"]:
        dmode = dict(subset)
        dmode["action"] = make_action_intervention(actions, slices, mode, args.seed)

        delta = predict_delta(model, dmode, stats_np, args.eval_batch_size, args.device)
        pred = z0 + alpha * delta

        r = eval_candidate_matching(pred, true, slices)
        r["mode"] = mode
        rows.append(r)

    return rows


def write_md(path, args, metrics, matching_rows):
    lines = [
        "# Action-grounded candidate objective diagnostic",
        "",
        "The model is trained with a diagonal candidate-matching objective over futures from the same current state.",
        "",
        f"Loss: `MSE + {args.lambda_match} * candidate_matching_CE`, temperature `{args.temperature}`.",
        "",
        "## Prediction metrics",
        "",
        "| split | identity error | raw error | global alpha | global error | gain vs identity | cosine mean | positive cosine frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for split in ["val", "test"]:
        r = metrics[split]
        lines.append(
            f"| {split} | {r['identity_error']:.6f} | {r['raw_error']:.6f} | "
            f"{r['global_alpha']:.6f} | {r['global_error']:.6f} | "
            f"{r['global_improvement_vs_identity']:.6f} | {r['cosine_mean']:.6f} | "
            f"{r['cosine_positive_frac']:.6f} |"
        )

    lines += [
        "",
        "## Candidate matching under action interventions",
        "",
        "| inference action | top-1 candidate accuracy | mean diag MSE | mean best offdiag MSE | margin best offdiag - diag | positive margin frac |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for r in matching_rows:
        lines.append(
            f"| {r['mode']} | {r['top1_accuracy']:.6f} | "
            f"{r['mean_diag_mse']:.6f} | {r['mean_best_offdiag_mse']:.6f} | "
            f"{r['mean_margin']:.6f} | {r['positive_margin_frac']:.6f} |"
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

    print("Building candidate groups...")
    train_groups = build_candidate_groups(train_data, args.candidates, args.max_train_groups, args.seed)
    val_groups = build_candidate_groups(val_data, args.candidates, args.max_eval_groups, args.seed)
    test_groups = build_candidate_groups(test_data, args.candidates, args.max_eval_groups, args.seed)

    print("train groups:", train_groups.shape)
    print("val groups:", val_groups.shape)
    print("test groups:", test_groups.shape)

    print("Computing train statistics...")
    stats_np = compute_stats(train_data)
    stats = to_torch_stats(stats_np, args.device)

    _, tokens, dim = train_data["z_current"].shape
    action_dim = train_data["action"].shape[1]

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
        CandidateGroupDataset(train_data, train_groups),
        batch_size=args.batch_groups,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        CandidateGroupDataset(val_data, val_groups),
        batch_size=args.batch_groups,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        tr, tr_mse, tr_match = train_epoch(model, train_loader, opt, stats, args)
        va, va_mse, va_match = eval_group_loss(model, val_loader, stats, args)

        print(
            f"epoch={epoch:03d} train={tr:.6f} mse={tr_mse:.6f} match={tr_match:.6f} "
            f"val={va:.6f} val_mse={va_mse:.6f} val_match={va_match:.6f}",
            flush=True,
        )

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

    print("Evaluating prediction metrics...")
    metrics = evaluate_full(model, stats_np, val_data, test_data, args)
    metrics["seed"] = args.seed
    metrics["best_epoch"] = best_epoch
    metrics["best_val"] = best_val

    print("Evaluating candidate matching...")
    matching_rows = evaluate_matching_modes(
        model=model,
        stats_np=stats_np,
        test_data=test_data,
        groups=test_groups,
        alpha=metrics["alpha"],
        args=args,
    )

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_md(args.out_dir / "candidate_objective_diagnostic.md", args, metrics, matching_rows)

    print(args.out_dir / "candidate_objective_diagnostic.md")
    print((args.out_dir / "candidate_objective_diagnostic.md").read_text())


if __name__ == "__main__":
    main()
