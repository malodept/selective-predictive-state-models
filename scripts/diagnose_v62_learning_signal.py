from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def split_by_state(state_id, seed):
    rng = np.random.default_rng(seed)
    states = np.unique(state_id)
    rng.shuffle(states)
    n = len(states)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_s = set(states[:n_train])
    val_s = set(states[n_train:n_train+n_val])
    test_s = set(states[n_train+n_val:])
    idx = np.arange(len(state_id))
    train = idx[np.array([s in train_s for s in state_id])]
    val = idx[np.array([s in val_s for s in state_id])]
    test = idx[np.array([s in test_s for s in state_id])]
    return train, val, test

def standardize_z(zc, zf, train_idx):
    stack = np.concatenate([zc[train_idx], zf[train_idx]], axis=0)
    mu = stack.mean(axis=(0,1), keepdims=True)
    sd = stack.std(axis=(0,1), keepdims=True) + 1e-6
    return (zc-mu)/sd, (zf-mu)/sd

def build_same_state_candidates(state_id, action_id, rows):
    actions = sorted(np.unique(action_id).tolist())
    table = {}
    for i, s, a in zip(np.arange(len(state_id)), state_id, action_id):
        table[(int(s), int(a))] = int(i)
    cand, correct = [], []
    for r in rows:
        sid = int(state_id[r])
        aid = int(action_id[r])
        c = [table[(sid, a)] for a in actions]
        cand.append(c)
        correct.append(actions.index(aid))
    return np.asarray(cand), np.asarray(correct)

def eval_rank(pred_delta, zc, zf, cand, correct):
    delta = zf - zc
    cand_delta = delta[cand]
    dist = ((cand_delta - pred_delta[:, None, :, :]) ** 2).mean(axis=(2,3))
    return float((dist.argmin(axis=1) == correct).mean())

def eval_rank_flat(pred_delta, cur, fut, cand, correct):
    delta = fut - cur
    cand_delta = delta[cand]
    dist = ((cand_delta - pred_delta[:, None, :]) ** 2).mean(axis=2)
    return float((dist.argmin(axis=1) == correct).mean())

class MLP(nn.Module):
    def __init__(self, din, dout, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(din, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dout),
        )
    def forward(self, x):
        return self.net(x)

def train_mlp(x, y, train_idx, val_idx, test_idx, epochs=200, lr=1e-3, device="cuda"):
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    y = torch.as_tensor(y, dtype=torch.float32, device=device)

    model = MLP(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best = None
    best_val = 1e9

    for ep in range(1, epochs+1):
        model.train()
        perm = torch.as_tensor(np.random.permutation(train_idx), dtype=torch.long, device=device)
        for s in range(0, len(perm), 256):
            ids = perm[s:s+256]
            pred = model(x[ids])
            loss = F.mse_loss(pred, y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if ep % 25 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                val = F.mse_loss(model(x[val_idx]), y[val_idx]).item()
            if val < best_val:
                best_val = val
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"epoch={ep:03d} val={val:.6f}")

    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        pred_test = model(x[test_idx]).detach().cpu().numpy()
    return pred_test, best_val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    d = dict(np.load(args.features))

    zc_raw = d["z_current"]
    zf_raw = d["z_future"]
    state = d["state"].astype(np.float32)
    fut = d["future_state"].astype(np.float32)
    action = d["action"].astype(np.float32)
    state_id = d["state_id"]
    action_id = d["action_id"]

    train_idx, val_idx, test_idx = split_by_state(state_id, args.seed)
    zc, zf = standardize_z(zc_raw, zf_raw, train_idx)

    cand, correct = build_same_state_candidates(state_id, action_id, test_idx)
    chance = 1.0 / cand.shape[1]

    print("# v62 learning-signal diagnostics")
    print("rows", len(state_id), "train", len(train_idx), "val", len(val_idx), "test", len(test_idx))
    print("K", cand.shape[1], "chance", chance)

    # 1. Latent oracle sanity.
    true_delta = zf[test_idx] - zc[test_idx]
    zero_delta = np.zeros_like(true_delta)
    print("oracle_true_latent_delta", eval_rank(true_delta, zc, zf, cand, correct))
    print("zero_latent_delta", eval_rank(zero_delta, zc, zf, cand, correct))

    # 2. Mean latent delta by action.
    mean_by_action = {}
    for a in sorted(np.unique(action_id)):
        ids = train_idx[action_id[train_idx] == a]
        mean_by_action[int(a)] = (zf[ids] - zc[ids]).mean(axis=0)
    pred_mean_action = np.stack([mean_by_action[int(action_id[r])] for r in test_idx])
    print("mean_latent_delta_by_action", eval_rank(pred_mean_action, zc, zf, cand, correct))

    # 3. Physical-state oracle and MLP.
    phys_delta_true = fut[test_idx] - state[test_idx]
    phys_zero = np.zeros_like(phys_delta_true)
    print("oracle_true_physical_delta", eval_rank_flat(phys_delta_true, state, fut, cand, correct))
    print("zero_physical_delta", eval_rank_flat(phys_zero, state, fut, cand, correct))

    # Normalize physical inputs.
    x_phys = np.concatenate([state, action], axis=1)
    x_mu = x_phys[train_idx].mean(axis=0, keepdims=True)
    x_sd = x_phys[train_idx].std(axis=0, keepdims=True) + 1e-6
    x_phys = (x_phys - x_mu) / x_sd

    y_phys = fut - state
    y_mu = y_phys[train_idx].mean(axis=0, keepdims=True)
    y_sd = y_phys[train_idx].std(axis=0, keepdims=True) + 1e-6
    y_phys_n = (y_phys - y_mu) / y_sd

    print("\ntraining physical_state_mlp")
    pred_phys_n, val_phys = train_mlp(
        x_phys, y_phys_n, train_idx, val_idx, test_idx,
        epochs=args.epochs, lr=1e-3, device=args.device
    )
    pred_phys = pred_phys_n * y_sd + y_mu
    print("physical_state_mlp_val", val_phys)
    print("physical_state_mlp_rank", eval_rank_flat(pred_phys, state, fut, cand, correct))

    # 4. Latent MLP from physical state+action to DINO delta.
    y_lat = (zf - zc).reshape(len(zc), -1)
    print("\ntraining latent_mlp_from_physical_state_action")
    pred_lat_flat, val_lat = train_mlp(
        x_phys, y_lat, train_idx, val_idx, test_idx,
        epochs=args.epochs, lr=1e-3, device=args.device
    )
    pred_lat = pred_lat_flat.reshape(len(test_idx), zc.shape[1], zc.shape[2])
    print("latent_mlp_val", val_lat)
    print("latent_mlp_rank", eval_rank(pred_lat, zc, zf, cand, correct))

if __name__ == "__main__":
    main()
