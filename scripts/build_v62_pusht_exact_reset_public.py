from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path
import argparse
import importlib.util
import math
import numpy as np
import pandas as pd
from PIL import Image

import gymnasium as gym
import gym_pusht

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


spec = importlib.util.spec_from_file_location(
    "v60a", "scripts/build_v60a_pusht_public_bridge.py"
)
v60a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v60a)

OUT_ROOT = Path("outputs/counterfactual/pusht_exact_reset_public")
REPORT = Path("reports/tables/protocol/pusht_exact_reset_public/v62")
PAPER = Path("reports/paper_assets_v62_pusht_exact_reset")
for p in [OUT_ROOT, REPORT, PAPER / "tables", PAPER / "figures", PAPER / "text"]:
    p.mkdir(parents=True, exist_ok=True)

MODELS = ["Tiny", "Medium", "Full"]
ENV_ID = "gym_pusht/PushT-v0"


def primitive_targets(agent_xy, block_xy=None, radius=80.0):
    """Five action targets around the block, not around the agent.

    PushT actions are target positions for the agent. If targets are defined
    around the current agent position, the pusher often moves without touching
    the T-block. For an action-conditioned audit, we need primitives that make
    contact likely and create different futures from the same initial state.
    """
    if block_xy is None:
        block_xy = agent_xy

    offsets = np.array([
        [ 0.0,  0.0],        # go toward block center
        [ radius,  0.0],     # approach/right-side target
        [-radius,  0.0],     # approach/left-side target
        [ 0.0,  radius],     # approach/upper target
        [ 0.0, -radius],     # approach/lower target
    ], dtype=np.float32)

    targets = block_xy[None, :] + offsets
    return np.clip(targets, 5.0, 507.0).astype(np.float32)


def set_exact_state(env, state):
    u = env.unwrapped
    u._set_state(np.asarray(state, dtype=np.float64))


def render_rgb(env):
    frame = env.render()
    if frame is None:
        frame = env.unwrapped.render()
    frame = np.asarray(frame)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame.astype(np.uint8)


def rollout_action(env, base_state, target, horizon):
    set_exact_state(env, base_state)
    for _ in range(horizon):
        obs, reward, terminated, truncated, info = env.step(target.astype(np.float32))
        if terminated or truncated:
            break
    return np.asarray(obs, dtype=np.float64), render_rgb(env), dict(info)


def sanity_exact_reset(env, base_state, target, horizon):
    obs1, img1, info1 = rollout_action(env, base_state, target, horizon)
    obs2, img2, info2 = rollout_action(env, base_state, target, horizon)
    state_diff = float(np.abs(obs1 - obs2).max())
    img_diff = float(np.abs(img1.astype(np.int16) - img2.astype(np.int16)).max())
    return state_diff, img_diff


def build_cache(args):
    rng = np.random.default_rng(args.seed)
    env = gym.make(ENV_ID, render_mode="rgb_array")

    current_images = []
    future_images = []
    z_state_rows = []
    z_future_state_rows = []
    actions = []
    state_ids = []
    action_ids = []
    rewards = []
    coverages = []

    sanity_done = False

    for i in range(args.num_states):
        seed_i = int(args.seed * 100000 + i)
        obs, info = env.reset(seed=seed_i)
        base_state = np.asarray(obs, dtype=np.float64)
        base_img = render_rgb(env)
        agent_xy = base_state[:2].astype(np.float32)
        block_xy = base_state[2:4].astype(np.float32)

        targets = primitive_targets(agent_xy, block_xy=block_xy, radius=args.radius)

        if not sanity_done:
            sd, imd = sanity_exact_reset(env, base_state, targets[0], args.horizon)
            print(f"[v62] exact reset sanity state_maxdiff={sd:.6g} image_maxdiff={imd:.6g}")
            sanity_done = True

        for a_id, target in enumerate(targets):
            fut_state, fut_img, fut_info = rollout_action(env, base_state, target, args.horizon)

            current_images.append(base_img)
            future_images.append(fut_img)
            z_state_rows.append(base_state)
            z_future_state_rows.append(fut_state)
            actions.append((target - agent_xy) / 512.0)
            state_ids.append(i)
            action_ids.append(a_id)
            rewards.append(float(fut_info.get("coverage", fut_info.get("reward", 0.0))))
            coverages.append(float(fut_info.get("coverage", 0.0)))

        if (i + 1) % 100 == 0:
            print(f"[v62] generated {i+1}/{args.num_states} states")

    env.close()

    current_images = np.stack(current_images)
    future_images = np.stack(future_images)

    cache_raw = OUT_ROOT / f"v62_raw_images_n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}.npz"
    np.savez_compressed(
        cache_raw,
        current_images=current_images,
        future_images=future_images,
        state=np.asarray(z_state_rows, dtype=np.float32),
        future_state=np.asarray(z_future_state_rows, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        state_id=np.asarray(state_ids, dtype=np.int64),
        action_id=np.asarray(action_ids, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float32),
        coverage=np.asarray(coverages, dtype=np.float32),
    )
    print("[v62] wrote raw cache", cache_raw)
    return cache_raw


def preprocess_batch(images):
    xs = []
    for im in images:
        pil = Image.fromarray(im).resize((224, 224), Image.BICUBIC)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        xs.append(arr.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(xs)).float()


def load_dino(device):
    print("[v62] loading DINOv2 ViT-S/14")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    return model


@torch.no_grad()
def encode_images(images, device, batch_size):
    model = load_dino(device)
    feats = []
    for s in range(0, len(images), batch_size):
        batch = preprocess_batch(images[s:s + batch_size]).to(device)
        out = model.forward_features(batch)
        if isinstance(out, dict):
            patch = out["x_norm_patchtokens"]
        else:
            patch = out
        b, n, c = patch.shape
        side = int(math.sqrt(n))
        patch = patch.reshape(b, side, side, c).permute(0, 3, 1, 2)
        pooled = F.adaptive_avg_pool2d(patch, (4, 4)).permute(0, 2, 3, 1).reshape(b, 16, c)
        feats.append(pooled.detach().cpu().numpy().astype(np.float32))
        print(f"[v62] encoded {min(s + batch_size, len(images))}/{len(images)}")
    return np.concatenate(feats, axis=0)


def ensure_feature_cache(args):
    feature_cache = OUT_ROOT / f"v62_features_n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}.npz"
    if feature_cache.exists() and args.reuse_cache:
        print("[v62] reusing feature cache", feature_cache)
        return feature_cache

    raw_cache = OUT_ROOT / f"v62_raw_images_n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}.npz"
    if not raw_cache.exists() or not args.reuse_cache:
        raw_cache = build_cache(args)

    raw = dict(np.load(raw_cache))
    zc = encode_images(raw["current_images"], args.device, args.dino_batch_size)
    zf = encode_images(raw["future_images"], args.device, args.dino_batch_size)

    np.savez_compressed(
        feature_cache,
        z_current=zc,
        z_future=zf,
        state=raw["state"],
        future_state=raw["future_state"],
        action=raw["action"],
        state_id=raw["state_id"],
        action_id=raw["action_id"],
        reward=raw["reward"],
        coverage=raw["coverage"],
    )
    print("[v62] wrote feature cache", feature_cache)
    return feature_cache


def split_by_state(state_id, seed):
    rng = np.random.default_rng(seed)
    states = np.unique(state_id)
    rng.shuffle(states)
    n = len(states)
    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    train_s = set(states[:n_train])
    val_s = set(states[n_train:n_train + n_val])
    test_s = set(states[n_train + n_val:])

    idx = np.arange(len(state_id))
    train = idx[np.array([s in train_s for s in state_id])]
    val = idx[np.array([s in val_s for s in state_id])]
    test = idx[np.array([s in test_s for s in state_id])]
    return train, val, test


def standardize_z(zc, zf, train_idx):
    stack = np.concatenate([zc[train_idx], zf[train_idx]], axis=0)
    mu = stack.mean(axis=(0, 1), keepdims=True)
    sd = stack.std(axis=(0, 1), keepdims=True) + 1e-6
    return (zc - mu) / sd, (zf - mu) / sd


def train_predictor(name, zc, zf, action, train_idx, val_idx, args):
    device = args.device
    model = v60a.make_model(name, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    xz = torch.as_tensor(zc, dtype=torch.float32)
    xa = torch.as_tensor(action, dtype=torch.float32)
    yd = torch.as_tensor(zf - zc, dtype=torch.float32)

    ds = TensorDataset(
        xz[train_idx],
        xa[train_idx],
        yd[train_idx],
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    best = None
    best_val = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for bz, ba, by in dl:
            bz = bz.to(device)
            ba = ba.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(bz, ba)
            loss = F.mse_loss(pred, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            vz = xz[val_idx].to(device)
            va = xa[val_idx].to(device)
            vy = yd[val_idx].to(device)
            vp = model(vz, va)
            val = float(F.mse_loss(vp, vy).detach().cpu())

        if val < best_val:
            best_val = val
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"[v62] {name} epoch={ep:03d} train={np.mean(losses):.6f} val={val:.6f}")

    model.load_state_dict(best)
    ckpt = OUT_ROOT / f"v62_pusht_exact_{name.lower()}_n{args.num_states}_h{args.horizon}_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "best_val": best_val}, ckpt)
    print("[v62] saved", ckpt)
    return model


@torch.no_grad()
def predict(model, zc, action_rows, rows, device, batch_size):
    z = torch.as_tensor(zc, dtype=torch.float32, device=device)
    a = torch.as_tensor(action_rows, dtype=torch.float32, device=device)
    preds = []
    for s in range(0, len(rows), batch_size):
        rr = torch.as_tensor(rows[s:s + batch_size], dtype=torch.long, device=device)
        pred = model(z[rr], a[s:s + batch_size])
        preds.append(pred.detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_same_state_candidates(state_id, action_id, rows):
    unique_actions = sorted(np.unique(action_id).tolist())
    by_state_action = {}
    for idx, sid, aid in zip(np.arange(len(state_id)), state_id, action_id):
        by_state_action[(int(sid), int(aid))] = int(idx)

    cand = []
    correct = []
    for r in rows:
        sid = int(state_id[r])
        aid = int(action_id[r])
        c = [by_state_action[(sid, a)] for a in unique_actions]
        cand.append(c)
        correct.append(unique_actions.index(aid))
    return np.asarray(cand, dtype=np.int64), np.asarray(correct, dtype=np.int64)


def eval_rank(pred_delta, zc, zf, cand, correct):
    delta = zf - zc
    cand_delta = delta[cand]
    dist = ((cand_delta - pred_delta[:, None, :, :]) ** 2).mean(axis=(2, 3))
    return (dist.argmin(axis=1) == correct).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-states", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--radius", type=float, default=90.0)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--dino-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print("[v62] args", args)

    feature_cache = ensure_feature_cache(args)
    data = dict(np.load(feature_cache))

    zc_raw = data["z_current"]
    zf_raw = data["z_future"]
    action = data["action"].astype(np.float32)
    state_id = data["state_id"]
    action_id = data["action_id"]

    train_idx, val_idx, test_idx = split_by_state(state_id, args.seed)
    zc, zf = standardize_z(zc_raw, zf_raw, train_idx)

    print("[v62] rows", len(state_id), "train", len(train_idx), "val", len(val_idx), "test", len(test_idx))
    print("[v62] states", len(np.unique(state_id)), "actions", sorted(np.unique(action_id).tolist()))

    models = {}
    for name in MODELS:
        models[name] = train_predictor(name, zc, zf, action, train_idx, val_idx, args)

    cand, correct = build_same_state_candidates(state_id, action_id, test_idx)
    chance = 1.0 / cand.shape[1]

    rows_out = []

    modes = {
        "clean_action": action[test_idx],
        "zero_action": np.zeros_like(action[test_idx]),
        "neg_action": -action[test_idx],
        "shuffled_action": action[np.roll(test_idx, 17)],
        "random_action": np.random.default_rng(args.seed + 123).normal(size=action[test_idx].shape).astype(np.float32),
    }

    for mode, act_rows in modes.items():
        row = {"system": mode, "chance": chance}
        for name, model in models.items():
            pred = predict(model, zc, act_rows, test_idx, args.device, args.batch_size)
            row[name] = eval_rank(pred, zc, zf, cand, correct)
        rows_out.append(row)

    row = {"system": "pred_shuffled", "chance": chance}
    for name, model in models.items():
        pred = predict(model, zc, action[test_idx], test_idx, args.device, args.batch_size)
        pred = np.roll(pred, 17, axis=0)
        row[name] = eval_rank(pred, zc, zf, cand, correct)
    rows_out.append(row)

    zero_pred = np.zeros_like(zf[test_idx] - zc[test_idx])
    rows_out.append({
        "system": "zero_delta",
        "chance": chance,
        "Tiny": eval_rank(zero_pred, zc, zf, cand, correct),
        "Medium": eval_rank(zero_pred, zc, zf, cand, correct),
        "Full": eval_rank(zero_pred, zc, zf, cand, correct),
    })

    df = pd.DataFrame(rows_out)
    tag = f"n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}"
    csv = REPORT / f"v62_pusht_exact_reset_{tag}.csv"
    md = REPORT / f"v62_pusht_exact_reset_{tag}.md"
    df.to_csv(csv, index=False)

    lines = [f"# v62 public PushT exact-reset action audit — {tag}\n"]
    lines.append("| system | chance | Tiny | Medium | Full |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            f"| `{r['system']}` | {r['chance']:.3f} | "
            f"{r['Tiny']:.3f} | {r['Medium']:.3f} | {r['Full']:.3f} |"
        )
    md.write_text("\n".join(lines) + "\n")
    print(md.read_text())
    print("[v62] wrote", csv)


if __name__ == "__main__":
    main()
