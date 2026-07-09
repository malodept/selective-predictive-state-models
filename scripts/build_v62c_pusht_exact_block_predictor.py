from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = Path("outputs/counterfactual/pusht_exact_reset_public")
REPORT = Path("reports/tables/protocol/pusht_exact_reset_public/v62c_block")
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

def eval_rank(pred_delta_block, cur, fut, cand, correct):
    # block dims are x, y, theta = indices 2,3,4
    delta = fut[:, 2:5] - cur[:, 2:5]
    cand_delta = delta[cand]
    dist = ((cand_delta - pred_delta_block[:, None, :]) ** 2).mean(axis=2)
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

def train_model(x, y, train_idx, val_idx, args, hidden, depth, name):
    device = args.device
    model = MLP(x.shape[1], y.shape[1], hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    best, best_val = None, 1e9
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(train_idx)
        for s in range(0, len(perm), args.batch_size):
            ids = torch.as_tensor(perm[s:s + args.batch_size], dtype=torch.long, device=device)
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
            print(f"[v62C] {name} epoch={ep:03d} val={val:.5f}")

    model.load_state_dict(best)
    return model

@torch.no_grad()
def predict(model, x_rows, args):
    device = args.device
    xt = torch.as_tensor(x_rows, dtype=torch.float32, device=device)
    outs = []
    for s in range(0, len(x_rows), args.batch_size):
        outs.append(model(xt[s:s + args.batch_size]).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)

def run_seed(args):
    cache = OUT / f"v62b_state_cache_n{args.num_states}_h{args.horizon}_r{args.radius}_seed{args.seed}.npz"
    d = dict(np.load(cache))

    cur = d["state"].astype(np.float32)
    fut = d["future_state"].astype(np.float32)
    action = d["action"].astype(np.float32)
    state_id = d["state_id"]
    action_id = d["action_id"]

    train, val, test = split_by_state(state_id, args.seed)
    cand, correct = build_candidates(state_id, action_id, test)

    x = np.concatenate([cur, action], axis=1)
    x_mu = x[train].mean(axis=0, keepdims=True)
    x_sd = x[train].std(axis=0, keepdims=True) + 1e-6
    x_n = (x - x_mu) / x_sd

    y = fut[:, 2:5] - cur[:, 2:5]
    y_mu = y[train].mean(axis=0, keepdims=True)
    y_sd = y[train].std(axis=0, keepdims=True) + 1e-6
    y_n = (y - y_mu) / y_sd

    configs = {
        "Tiny": (64, 1),
        "Medium": (256, 2),
        "Full": (512, 3),
    }

    models = {}
    for name, (hidden, depth) in configs.items():
        models[name] = train_model(x_n, y_n, train, val, args, hidden, depth, name)

    modes = {
        "clean_action": action[test],
        "zero_action": np.zeros_like(action[test]),
        "neg_action": -action[test],
        "shuffled_action": action[np.roll(test, 17)],
        "random_action": np.random.default_rng(args.seed + 9).normal(size=action[test].shape).astype(np.float32),
    }

    rows = []
    for mode, act_test in modes.items():
        x_mode = np.concatenate([cur[test], act_test], axis=1)
        x_mode = (x_mode - x_mu) / x_sd
        row = {"system": mode, "chance": 1.0 / cand.shape[1]}
        for name, model in models.items():
            pred_n = predict(model, x_mode, args)
            pred = pred_n * y_sd + y_mu
            row[name] = eval_rank(pred, cur, fut, cand, correct)
        rows.append(row)

    row = {"system": "pred_shuffled", "chance": 1.0 / cand.shape[1]}
    x_clean = (np.concatenate([cur[test], action[test]], axis=1) - x_mu) / x_sd
    for name, model in models.items():
        pred_n = predict(model, x_clean, args)
        pred = pred_n * y_sd + y_mu
        pred = np.roll(pred, 17, axis=0)
        row[name] = eval_rank(pred, cur, fut, cand, correct)
    rows.append(row)

    rows.append({
        "system": "zero_delta",
        "chance": 1.0 / cand.shape[1],
        "Tiny": eval_rank(np.zeros((len(test), 3), dtype=np.float32), cur, fut, cand, correct),
        "Medium": eval_rank(np.zeros((len(test), 3), dtype=np.float32), cur, fut, cand, correct),
        "Full": eval_rank(np.zeros((len(test), 3), dtype=np.float32), cur, fut, cand, correct),
    })

    df = pd.DataFrame(rows)
    tag = f"n{args.num_states}_h{args.horizon}_r{args.radius}_seed{args.seed}"
    csv = REPORT / f"v62c_block_predictor_{tag}.csv"
    md = REPORT / f"v62c_block_predictor_{tag}.md"
    df.to_csv(csv, index=False)

    lines = [f"# v62C public PushT exact-reset block-only predictor — {tag}\n"]
    lines.append("| system | chance | Tiny | Medium | Full |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['system']}` | {r['chance']:.3f} | "
            f"{r['Tiny']:.3f} | {r['Medium']:.3f} | {r['Full']:.3f} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())

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
    run_seed(args)

if __name__ == "__main__":
    main()
