from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("outputs/counterfactual/pusht_exact_reset_public")
REPORT = Path("reports/tables/protocol/pusht_exact_reset_public/v62b_state")
REPORT.mkdir(parents=True, exist_ok=True)

def split_by_state(state_id, seed):
    rng = np.random.default_rng(seed)
    s = np.unique(state_id)
    rng.shuffle(s)
    n = len(s)
    train_s = set(s[: int(0.70 * n)])
    val_s = set(s[int(0.70 * n): int(0.85 * n)])
    test_s = set(s[int(0.85 * n):])
    idx = np.arange(len(state_id))
    train = idx[np.array([x in train_s for x in state_id])]
    val = idx[np.array([x in val_s for x in state_id])]
    test = idx[np.array([x in test_s for x in state_id])]
    return train, val, test

def build_candidates(state_id, action_id, rows):
    acts = sorted(np.unique(action_id).tolist())
    table = {(int(s), int(a)): int(i) for i, s, a in zip(np.arange(len(state_id)), state_id, action_id)}
    cand, correct = [], []
    for r in rows:
        sid = int(state_id[r])
        aid = int(action_id[r])
        c = [table[(sid, a)] for a in acts]
        cand.append(c)
        correct.append(acts.index(aid))
    return np.asarray(cand), np.asarray(correct)

def eval_rank(pred_delta, cur, fut, cand, correct, dims):
    delta = fut[:, dims] - cur[:, dims]
    cand_delta = delta[cand]
    pred = pred_delta[:, dims]
    dist = ((cand_delta - pred[:, None, :]) ** 2).mean(axis=2)
    return float((dist.argmin(axis=1) == correct).mean())

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256, depth=2):
        super().__init__()
        layers = []
        d = din
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, dout))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

def train(x, y, train_idx, val_idx, args, hidden=256, depth=2):
    device = args.device
    model = MLP(x.shape[1], y.shape[1], hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    best, best_val = None, 1e9
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(train_idx)
        model.train()
        for s in range(0, len(perm), args.batch_size):
            ids = torch.as_tensor(perm[s:s+args.batch_size], dtype=torch.long, device=device)
            loss = F.mse_loss(model(xt[ids]), yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vv = torch.as_tensor(val_idx, dtype=torch.long, device=device)
            val = float(F.mse_loss(model(xt[vv]), yt[vv]).detach().cpu())
        if val < best_val:
            best_val = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 25 == 0 or ep == args.epochs:
            print(f"epoch={ep:03d} val={val:.5f}")

    model.load_state_dict(best)
    return model

@torch.no_grad()
def pred(model, x, args):
    device = args.device
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    outs = []
    for s in range(0, len(x), args.batch_size):
        outs.append(model(xt[s:s+args.batch_size]).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-states", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--radius", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = OUT / f"v62b_state_cache_n{args.num_states}_h{args.horizon}_r{args.radius}_seed{args.seed}.npz"
    d = dict(np.load(cache))

    cur = d["state"].astype(np.float32)
    fut = d["future_state"].astype(np.float32)
    action = d["action"].astype(np.float32)
    state_id = d["state_id"]
    action_id = d["action_id"]

    train_idx, val_idx, test_idx = split_by_state(state_id, args.seed)
    cand, correct = build_candidates(state_id, action_id, test_idx)

    x = np.concatenate([cur, action], axis=1)
    x_mu = x[train_idx].mean(axis=0, keepdims=True)
    x_sd = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    x_n = (x - x_mu) / x_sd

    y = fut - cur
    y_mu = y[train_idx].mean(axis=0, keepdims=True)
    y_sd = y[train_idx].std(axis=0, keepdims=True) + 1e-6
    y_n = (y - y_mu) / y_sd

    model = train(x_n, y_n, train_idx, val_idx, args, hidden=256, depth=2)

    modes = {
        "clean_action": action[test_idx],
        "zero_action": np.zeros_like(action[test_idx]),
        "neg_action": -action[test_idx],
        "shuffled_action": action[np.roll(test_idx, 17)],
        "random_action": np.random.default_rng(args.seed + 9).normal(size=action[test_idx].shape).astype(np.float32),
    }

    dims = {
        "full_state": [0,1,2,3,4],
        "agent_only": [0,1],
        "block_only": [2,3,4],
        "block_xy": [2,3],
        "block_theta": [4],
    }

    rows = []
    for mode, act_test in modes.items():
        x_mode = np.concatenate([cur[test_idx], act_test], axis=1)
        x_mode = (x_mode - x_mu) / x_sd
        p_n = pred(model, x_mode, args)
        p = p_n * y_sd + y_mu

        for dname, didx in dims.items():
            rows.append({
                "system": mode,
                "dims": dname,
                "chance": 1.0 / cand.shape[1],
                "acc": eval_rank(p, cur, fut, cand, correct, didx),
            })

    # shuffled prediction control
    x_clean = (np.concatenate([cur[test_idx], action[test_idx]], axis=1) - x_mu) / x_sd
    p_n = pred(model, x_clean, args)
    p = p_n * y_sd + y_mu
    p = np.roll(p, 17, axis=0)
    for dname, didx in dims.items():
        rows.append({
            "system": "pred_shuffled",
            "dims": dname,
            "chance": 1.0 / cand.shape[1],
            "acc": eval_rank(p, cur, fut, cand, correct, didx),
        })

    df = pd.DataFrame(rows)
    tag = f"n{args.num_states}_h{args.horizon}_r{args.radius}_seed{args.seed}"
    csv = REPORT / f"v62b_state_rank_dims_{tag}.csv"
    md = REPORT / f"v62b_state_rank_dims_{tag}.md"
    df.to_csv(csv, index=False)

    piv = df.pivot(index="system", columns="dims", values="acc")
    order_system = ["clean_action", "zero_action", "neg_action", "shuffled_action", "random_action", "pred_shuffled"]
    order_dims = ["full_state", "agent_only", "block_only", "block_xy", "block_theta"]

    lines = [f"# v62B state-space rank by dimensions — {tag}\n"]
    lines.append("| system | chance | full_state | agent_only | block_only | block_xy | block_theta |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in order_system:
        lines.append(
            f"| `{s}` | {1.0/cand.shape[1]:.3f} | "
            + " | ".join(f"{piv.loc[s, d]:.3f}" for d in order_dims)
            + " |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    print("[v62B-dims] wrote", csv)

if __name__ == "__main__":
    main()
