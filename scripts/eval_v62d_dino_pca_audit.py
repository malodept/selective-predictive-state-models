from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPORT = Path("reports/tables/protocol/pusht_exact_reset_public/v62d_dino_pca")
REPORT.mkdir(parents=True, exist_ok=True)

def split_by_state(state_id, seed):
    rng = np.random.default_rng(seed)
    states = np.unique(state_id)
    rng.shuffle(states)
    n = len(states)
    train_s = set(states[:int(0.70*n)])
    val_s = set(states[int(0.70*n):int(0.85*n)])
    test_s = set(states[int(0.85*n):])
    idx = np.arange(len(state_id))
    train = idx[np.array([s in train_s for s in state_id])]
    val = idx[np.array([s in val_s for s in state_id])]
    test = idx[np.array([s in test_s for s in state_id])]
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

def fit_pca(x, train, k, whiten=True):
    mu = x[train].mean(axis=0, keepdims=True)
    xc = x[train] - mu
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    comp = vt[:k].astype(np.float32)
    scale = (s[:k] / np.sqrt(max(1, len(train)-1))).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    def transform(y):
        z = (y - mu) @ comp.T
        if whiten:
            z = z / scale[None, :]
        return z.astype(np.float32)
    return transform

def eval_rank(pred, all_delta, cand, correct):
    cand_delta = all_delta[cand]
    dist = ((cand_delta - pred[:, None, :]) ** 2).mean(axis=2)
    return float((dist.argmin(axis=1) == correct).mean())

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=512, depth=3):
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

def train_mlp(x, y, train, val, args):
    device = args.device
    model = MLP(x.shape[1], y.shape[1], hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)

    best = None
    best_val = 1e9

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = np.random.permutation(train)

        for s in range(0, len(perm), args.batch_size):
            ids = torch.as_tensor(perm[s:s+args.batch_size], dtype=torch.long, device=device)
            loss = F.mse_loss(model(xt[ids]), yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            vv = torch.as_tensor(val, dtype=torch.long, device=device)
            val_loss = float(F.mse_loss(model(xt[vv]), yt[vv]).detach().cpu())

        if val_loss < best_val:
            best_val = val_loss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 25 == 0 or ep == args.epochs:
            print(f"epoch={ep:03d} val={val_loss:.5f}")

    model.load_state_dict(best)
    return model

@torch.no_grad()
def predict(model, x, args):
    device = args.device
    xt = torch.as_tensor(x, dtype=torch.float32, device=device)
    out = []
    for s in range(0, len(x), args.batch_size):
        out.append(model(xt[s:s+args.batch_size]).detach().cpu().numpy())
    return np.concatenate(out, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input-mode", choices=["state", "dino"], default="state")
    ap.add_argument("--components", default="8,16,32,64")
    ap.add_argument("--input-components", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = dict(np.load(args.features))

    zc = d["z_current"].astype(np.float32).reshape(len(d["z_current"]), -1)
    zf = d["z_future"].astype(np.float32).reshape(len(d["z_future"]), -1)
    dz = zf - zc

    state = d["state"].astype(np.float32)
    action = d["action"].astype(np.float32)
    state_id = d["state_id"]
    action_id = d["action_id"]

    train, val, test = split_by_state(state_id, args.seed)
    cand, correct = build_candidates(state_id, action_id, test)
    chance = 1.0 / cand.shape[1]

    if args.input_mode == "state":
        x_all = np.concatenate([state, action], axis=1)
        def make_x(act):
            return np.concatenate([state[test], act], axis=1)
    else:
        zc_pca_fn = fit_pca(zc, train, args.input_components, whiten=True)
        zc_pca = zc_pca_fn(zc)
        x_all = np.concatenate([zc_pca, action], axis=1)
        def make_x(act):
            return np.concatenate([zc_pca[test], act], axis=1)

    x_mu = x_all[train].mean(axis=0, keepdims=True)
    x_sd = x_all[train].std(axis=0, keepdims=True) + 1e-6
    x_norm = (x_all - x_mu) / x_sd

    rows = []
    for k in [int(x) for x in args.components.split(",")]:
        print(f"\n### PCA dim {k}, input={args.input_mode}")

        dz_pca_fn = fit_pca(dz, train, k, whiten=True)
        dz_pca = dz_pca_fn(dz)

        oracle = eval_rank(dz_pca[test], dz_pca, cand, correct)
        zero = eval_rank(np.zeros_like(dz_pca[test]), dz_pca, cand, correct)

        model = train_mlp(x_norm, dz_pca, train, val, args)

        modes = {
            "clean_action": action[test],
            "zero_action": np.zeros_like(action[test]),
            "neg_action": -action[test],
            "shuffled_action": action[np.roll(test, 17)],
            "random_action": np.random.default_rng(args.seed + 9).normal(size=action[test].shape).astype(np.float32),
        }

        for name, act in modes.items():
            xm = (make_x(act) - x_mu) / x_sd
            pred = predict(model, xm, args)
            acc = eval_rank(pred, dz_pca, cand, correct)
            rows.append((k, name, chance, oracle, zero, acc))
            print(f"k={k:03d} {name:16s} chance={chance:.3f} oracle={oracle:.3f} zero={zero:.3f} acc={acc:.3f}")

        xm = (make_x(action[test]) - x_mu) / x_sd
        pred = predict(model, xm, args)
        pred = np.roll(pred, 17, axis=0)
        acc = eval_rank(pred, dz_pca, cand, correct)
        rows.append((k, "pred_shuffled", chance, oracle, zero, acc))
        print(f"k={k:03d} {'pred_shuffled':16s} chance={chance:.3f} oracle={oracle:.3f} zero={zero:.3f} acc={acc:.3f}")

    stem = Path(args.features).stem
    out_md = REPORT / f"v62d_dino_pca_{stem}_{args.input_mode}_seed{args.seed}.md"

    lines = [f"# v62D DINO PCA audit — {stem} — input={args.input_mode} seed={args.seed}\n"]
    lines.append("| pca_dim | system | chance | oracle_pca | zero_delta | acc |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for k, name, chance, oracle, zero, acc in rows:
        lines.append(f"| {k} | `{name}` | {chance:.3f} | {oracle:.3f} | {zero:.3f} | {acc:.3f} |")
    out_md.write_text("\n".join(lines) + "\n")
    print("\n" + out_md.read_text())
    print("[v62D] wrote", out_md)

if __name__ == "__main__":
    main()
