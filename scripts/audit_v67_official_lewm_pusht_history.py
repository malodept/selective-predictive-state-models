import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path
import argparse
import random
import math

import gymnasium as gym
import numpy as np
import torch
from torchvision.transforms import v2 as T

import stable_worldmodel
import stable_pretraining as spt


def unwrap_env(env):
    e = env
    while hasattr(e, "env"):
        e = e.env
    return e


def set_state(env, state):
    base = unwrap_env(env)
    base._set_state(np.asarray(state, dtype=np.float64))


def make_transform():
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(**spt.data.dataset_stats.ImageNet),
        T.Resize(size=(224, 224)),
    ])


def render_tensor(env, transform, device):
    return transform(env.render()).to(device)


def chunk_action(a2, frameskip=5):
    return np.tile(np.asarray(a2, dtype=np.float32), frameskip)


def unit(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.asarray([1.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


def action_set_from_state(state):
    # state: [agent_x, agent_y, block_x, block_y, block_angle, ...]
    agent = np.asarray(state[:2], dtype=np.float32)
    block = np.asarray(state[2:4], dtype=np.float32)
    u = unit(block - agent)
    perp = np.asarray([-u[1], u[0]], dtype=np.float32)

    acts = np.stack([
        u,
        -u,
        perp,
        -perp,
        np.zeros(2, dtype=np.float32),
    ], axis=0)
    return np.clip(acts, -1.0, 1.0).astype(np.float32)



def weak_actions_from_env(env, rng, k=5, dist_constraint=100.0):
    # Matches stable_worldmodel.envs.pusht.WeakPolicy:
    # sample target around agent, clip target near block, rescale to native [-1, 1] action.
    base = unwrap_env(env)
    actions = []
    tries = 0

    while len(actions) < k and tries < 1000:
        tries += 1

        a = rng.uniform(-1, 1, size=base.action_space.shape).astype(np.float32)

        agent_pos = np.array(
            (base.agent.position.x, base.agent.position.y),
            dtype=np.float32,
        )
        block_pos = np.array(
            (base.block.position.x, base.block.position.y),
            dtype=np.float32,
        )

        target = agent_pos + a * base.action_scale
        target = np.clip(
            target,
            block_pos - dist_constraint,
            block_pos + dist_constraint,
        )

        a = (target - agent_pos) / base.action_scale
        a = np.clip(a, -1, 1).astype(np.float32)

        if all(np.linalg.norm(a - b) > 1e-3 for b in actions):
            actions.append(a)

    while len(actions) < k:
        actions.append(rng.uniform(-1, 1, size=(2,)).astype(np.float32))

    return np.stack(actions, axis=0).astype(np.float32)


def random_action(rng, scale=0.75):
    return rng.uniform(-scale, scale, size=(2,)).astype(np.float32)


def step_chunk(env, a2, frameskip):
    obs = None
    for _ in range(frameskip):
        obs, reward, terminated, truncated, info = env.step(a2.astype(np.float32))
        if terminated or truncated:
            break
    return obs


@torch.no_grad()
def encode_frames(model, pixels_btchw):
    return model.encode({"pixels": pixels_btchw})["emb"]


@torch.no_grad()
def predict_next(model, hist_pixels, action_rows):
    info = model.encode({"pixels": hist_pixels, "action": action_rows})
    pred = model.predict(info["emb"], info["act_emb"])
    return pred[:, -1]


def rank_stats(pred, cand_emb):
    dist = ((cand_emb - pred[:, None, :]) ** 2).mean(dim=-1)  # N,K
    correct = dist[:, 0]
    wrong = dist[:, 1:]

    # Tie-robust rank: rank is 1 + number of wrong candidates strictly closer
    # than the correct one. This avoids arbitrary argsort tie-breaking when two
    # candidate futures are visually/latently indistinguishable.
    eps = 1e-12
    ranks = 1 + (wrong < (correct[:, None] - eps)).sum(dim=1)

    acc = (ranks == 1).float().mean().item()
    mrr = (1.0 / ranks.float()).mean().item()
    mean_rank = ranks.float().mean().item()

    best_wrong = wrong.min(dim=1).values
    margin = (best_wrong - correct).mean().item()

    return {
        "acc": acc,
        "mrr": mrr,
        "mean_rank": mean_rank,
        "margin": margin,
        "dist": dist,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--candidate-mode", type=str, default="oriented", choices=["oriented", "weak"])
    ap.add_argument("--dist-constraint", type=float, default=100.0)
    ap.add_argument("--out", type=str, default="reports/v67_external_lewm/lewm_history_n50.md")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device(args.device)

    ckpt = Path(os.environ["STABLEWM_HOME"]) / "pusht" / "lewm_object.ckpt"
    model = torch.load(ckpt, map_location=device, weights_only=False).to(device).eval()
    model.requires_grad_(False)

    transform = make_transform()
    env = gym.make("swm/PushT-v1", render_mode="rgb_array")

    clean_preds, zero_preds, neg_preds, rand_preds = [], [], [], []
    cand_embs = []
    oracle_preds = []
    current_embs = []

    skipped = 0

    for i in range(args.n):
        obs, info = env.reset(seed=args.seed * 100000 + i)

        # Build real 3-frame history separated by frameskip.
        frames = []
        hist_chunks = []

        frames.append(render_tensor(env, transform, device))

        for _ in range(2):
            a_hist = random_action(rng)
            hist_chunks.append(chunk_action(a_hist, args.frameskip))
            obs = step_chunk(env, a_hist, args.frameskip)
            if obs is None:
                skipped += 1
                break
            frames.append(render_tensor(env, transform, device))

        if len(frames) != 3:
            continue

        current_state = obs["state"].copy()
        hist_pixels = torch.stack(frames, dim=0).unsqueeze(0)  # 1,3,C,H,W

        if args.candidate_mode == "weak":
            set_state(env, current_state)
            cand_actions = weak_actions_from_env(
                env,
                rng,
                k=5,
                dist_constraint=args.dist_constraint,
            )
        else:
            cand_actions = action_set_from_state(current_state)
        correct_idx = i % len(cand_actions)
        correct = cand_actions[correct_idx]
        ordered = [correct] + [
            cand_actions[j] for j in range(len(cand_actions)) if j != correct_idx
        ]

        fut_imgs = []
        for a in ordered:
            set_state(env, current_state)
            step_chunk(env, a, args.frameskip)
            fut_imgs.append(render_tensor(env, transform, device))

        fut = torch.stack(fut_imgs, dim=0).unsqueeze(1)  # K,1,C,H,W
        fut_emb = encode_frames(model, fut)[:, 0, :]  # K,D
        cand_embs.append(fut_emb)

        # Current embedding / oracle diagnostics.
        cur_emb = encode_frames(model, hist_pixels[:, -1:, ...])[:, 0, :]
        current_embs.append(cur_emb[0])
        oracle_preds.append(fut_emb[0])

        def pred_for(last_chunk_np):
            rows = np.stack([hist_chunks[0], hist_chunks[1], last_chunk_np], axis=0)
            rows = torch.as_tensor(rows, device=device).float().unsqueeze(0)
            return predict_next(model, hist_pixels, rows)[0]

        clean_chunk = chunk_action(correct, args.frameskip)
        zero_chunk = np.zeros_like(clean_chunk)
        neg_chunk = chunk_action(-correct, args.frameskip)
        rand_chunk = rng.uniform(-1, 1, size=clean_chunk.shape).astype(np.float32)

        clean_preds.append(pred_for(clean_chunk))
        zero_preds.append(pred_for(zero_chunk))
        neg_preds.append(pred_for(neg_chunk))
        rand_preds.append(pred_for(rand_chunk))

    env.close()

    clean = torch.stack(clean_preds, dim=0)
    zero = torch.stack(zero_preds, dim=0)
    neg = torch.stack(neg_preds, dim=0)
    rand = torch.stack(rand_preds, dim=0)
    oracle = torch.stack(oracle_preds, dim=0)
    current = torch.stack(current_embs, dim=0)
    cand = torch.stack(cand_embs, dim=0)

    perm = torch.randperm(clean.shape[0], device=clean.device)
    shuffled = clean[perm]

    K = cand.shape[1]

    results = {}
    for name, pred in [
        ("oracle_future", oracle),
        ("current_embedding", current),
        ("clean_action", clean),
        ("zero_action", zero),
        ("negative_action", neg),
        ("random_action", rand),
        ("shuffled_prediction", shuffled),
    ]:
        results[name] = rank_stats(pred, cand)

    # diagnostics
    cand_pairwise = torch.cdist(cand, cand).detach().cpu()
    pred_norm = clean.norm(dim=-1).mean().item()
    fut_norm = cand.norm(dim=-1).mean().item()
    clean_to_correct = ((clean - cand[:, 0, :]) ** 2).mean(dim=-1).mean().item()
    clean_to_wrong = ((clean[:, None, :] - cand[:, 1:, :]) ** 2).mean(dim=-1).mean().item()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# v67 official LeWM PushT SPSM history-faithful smoke\n")
    lines.append(f"- n requested: {args.n}")
    lines.append(f"- n used: {clean.shape[0]}")
    lines.append(f"- skipped: {skipped}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- K: {K}")
    lines.append(f"- frameskip/action chunk: {args.frameskip}")
    lines.append(f"- candidate mode: {args.candidate_mode}")
    lines.append(f"- weak dist constraint: {args.dist_constraint}")
    lines.append(f"- checkpoint: {ckpt}")
    lines.append("")
    lines.append("| system | top-1 | MRR | mean rank | margin |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| chance | {1.0 / K:.4f} | - | {(K + 1) / 2:.2f} | 0.0000 |")
    for k, v in results.items():
        lines.append(
            f"| {k} | {v['acc']:.4f} | {v['mrr']:.4f} | "
            f"{v['mean_rank']:.2f} | {v['margin']:.6f} |"
        )

    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.append(f"- mean clean pred norm: {pred_norm:.6f}")
    lines.append(f"- mean future emb norm: {fut_norm:.6f}")
    lines.append(f"- mean MSE clean to correct: {clean_to_correct:.6f}")
    lines.append(f"- mean MSE clean to wrong: {clean_to_wrong:.6f}")
    lines.append(f"- mean candidate pairwise distance: {cand_pairwise.mean().item():.6f}")

    text = "\n".join(lines)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
