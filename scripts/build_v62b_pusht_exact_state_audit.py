from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

import gymnasium as gym
import gym_pusht

import torch
import torch.nn as nn
import torch.nn.functional as F


OUT = Path("outputs/counterfactual/pusht_exact_reset_public")
REPORT = Path("reports/tables/protocol/pusht_exact_reset_public/v62b_state")
OUT.mkdir(parents=True, exist_ok=True)
REPORT.mkdir(parents=True, exist_ok=True)

ENV_ID = "gym_pusht/PushT-v0"


def primitive_targets(agent_xy, block_xy, radius=80.0):
    offsets = np.array([
        [0.0, 0.0],
        [radius, 0.0],
        [-radius, 0.0],
        [0.0, radius],
        [0.0, -radius],
    ], dtype=np.float32)
    return np.clip(block_xy[None, :] + offsets, 5.0, 507.0).astype(np.float32)


def set_exact_state(env, state):
    env.unwrapped._set_state(np.asarray(state, dtype=np.float64))


def rollout(env, base_state, target, horizon):
    set_exact_state(env, base_state)
    for _ in range(horizon):
        obs, reward, terminated, truncated, info = env.step(target.astype(np.float32))
        if terminated or truncated:
            break
    return np.asarray(obs, dtype=np.float32), dict(info)


def build_cache(args):
    env = gym.make(ENV_ID, render_mode="rgb_array")

    states, futures, actions, state_ids, action_ids, coverages = [], [], [], [], [], []

    for i in range(args.num_states):
        obs, info = env.reset(seed=args.seed * 100000 + i)
        base = np.asarray(obs, dtype=np.float32)
        agent_xy = base[:2]
        block_xy = base[2:4]

        targets = primitive_targets(agent_xy, block_xy, radius=args.radius)

        for aid, target in enumerate(targets):
            fut, fut_info = rollout(env, base, target, args.horizon)

            states.append(base)
            futures.append(fut)
            actions.append((target - agent_xy) / 512.0)
            state_ids.append(i)
            action_ids.append(aid)
            coverages.append(float(fut_info.get("coverage", 0.0)))

        if (i + 1) % 200 == 0:
            print(f"[v62B] generated {i+1}/{args.num_states}")

    env.close()

    cache = OUT / f"v62b_state_cache_n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}.npz"
    np.savez_compressed(
        cache,
        state=np.asarray(states, dtype=np.float32),
        future_state=np.asarray(futures, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        state_id=np.asarray(state_ids, dtype=np.int64),
        action_id=np.asarray(action_ids, dtype=np.int64),
        coverage=np.asarray(coverages, dtype=np.float32),
    )
    print("[v62B] wrote", cache)
    return cache


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


def eval_rank(pred_delta, cur, fut, cand, correct):
    delta = fut - cur
    cand_delta = delta[cand]
    dist = ((cand_delta - pred_delta[:, None, :]) ** 2).mean(axis=2)
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


def train_model(x, y, train, val, args, hidden, depth, name):
    device = args.device
    model = MLP(x.shape[1], y.shape[1], hidden=hidden, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    best, best_val = None, 1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(train)
        losses = []

        for s in range(0, len(perm), args.batch_size):
            ids = torch.as_tensor(perm[s:s + args.batch_size], dtype=torch.long, device=device)
            pred = model(xt[ids])
            loss = F.mse_loss(pred, yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            vv = torch.as_tensor(val, dtype=torch.long, device=device)
            val_loss = float(F.mse_loss(model(xt[vv]), yt[vv]).detach().cpu())

        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 25 == 0 or ep == 1 or ep == args.epochs:
            print(f"[v62B] {name} epoch={ep:03d} train={np.mean(losses):.5f} val={val_loss:.5f}")

    model.load_state_dict(best)
    return model, best_val


@torch.no_grad()
def predict(model, x_rows, args):
    device = args.device
    x = torch.as_tensor(x_rows, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(x_rows), args.batch_size):
        out.append(model(x[s:s + args.batch_size]).detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-states", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--radius", type=float, default=80.0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cache = OUT / f"v62b_state_cache_n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}.npz"
    if not cache.exists() or not args.reuse_cache:
        cache = build_cache(args)

    d = dict(np.load(cache))
    cur = d["state"].astype(np.float32)
    fut = d["future_state"].astype(np.float32)
    action = d["action"].astype(np.float32)
    state_id = d["state_id"]
    action_id = d["action_id"]

    train, val, test = split_by_state(state_id, args.seed)
    cand, correct = build_candidates(state_id, action_id, test)
    chance = 1.0 / cand.shape[1]

    x = np.concatenate([cur, action], axis=1)
    x_mu = x[train].mean(axis=0, keepdims=True)
    x_sd = x[train].std(axis=0, keepdims=True) + 1e-6
    x_n = (x - x_mu) / x_sd

    y = fut - cur
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
        models[name], best_val = train_model(x_n, y_n, train, val, args, hidden, depth, name)

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

        row = {"system": mode, "chance": chance}
        for name, model in models.items():
            pred_n = predict(model, x_mode, args)
            pred = pred_n * y_sd + y_mu
            row[name] = eval_rank(pred, cur, fut, cand, correct)
        rows.append(row)

    row = {"system": "pred_shuffled", "chance": chance}
    x_clean = (np.concatenate([cur[test], action[test]], axis=1) - x_mu) / x_sd
    for name, model in models.items():
        pred_n = predict(model, x_clean, args)
        pred = pred_n * y_sd + y_mu
        pred = np.roll(pred, 17, axis=0)
        row[name] = eval_rank(pred, cur, fut, cand, correct)
    rows.append(row)

    zero = np.zeros_like(fut[test] - cur[test])
    rows.append({
        "system": "zero_delta",
        "chance": chance,
        "Tiny": eval_rank(zero, cur, fut, cand, correct),
        "Medium": eval_rank(zero, cur, fut, cand, correct),
        "Full": eval_rank(zero, cur, fut, cand, correct),
    })

    df = pd.DataFrame(rows)

    tag = f"n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}"
    csv = REPORT / f"v62b_state_exact_reset_{tag}.csv"
    md = REPORT / f"v62b_state_exact_reset_{tag}.md"
    df.to_csv(csv, index=False)

    lines = [f"# v62B public PushT exact-reset state-space audit — {tag}\n"]
    lines.append("| system | chance | Tiny | Medium | Full |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['system']}` | {r['chance']:.3f} | "
            f"{r['Tiny']:.3f} | {r['Medium']:.3f} | {r['Full']:.3f} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    print("[v62B] wrote", csv)


if __name__ == "__main__":
    main()
