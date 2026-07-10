from __future__ import annotations

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from pathlib import Path
import argparse
import numpy as np
import gymnasium as gym
import gym_pusht

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
    img = env.render()
    return np.asarray(obs, dtype=np.float32), np.asarray(img, dtype=np.uint8), float(reward), dict(info)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-states", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--radius", type=float, default=80.0)
    ap.add_argument("--outdir", default="outputs/counterfactual/pusht_exact_reset_public")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="rgb_array")

    current_images = []
    future_images = []
    states = []
    future_states = []
    actions = []
    state_ids = []
    action_ids = []
    rewards = []
    coverages = []

    for i in range(args.num_states):
        obs, info = env.reset(seed=args.seed * 100000 + i)
        base = np.asarray(obs, dtype=np.float32)
        base_img = np.asarray(env.render(), dtype=np.uint8)

        agent_xy = base[:2].astype(np.float32)
        block_xy = base[2:4].astype(np.float32)
        targets = primitive_targets(agent_xy, block_xy, radius=args.radius)

        for aid, target in enumerate(targets):
            fut, fut_img, reward, fut_info = rollout(env, base, target, args.horizon)

            current_images.append(base_img)
            future_images.append(fut_img)
            states.append(base)
            future_states.append(fut)
            actions.append((target - agent_xy) / 512.0)
            state_ids.append(i)
            action_ids.append(aid)
            rewards.append(reward)
            coverages.append(float(fut_info.get("coverage", 0.0)))

        if (i + 1) % 200 == 0:
            print(f"[raw-only] generated {i+1}/{args.num_states}", flush=True)

    env.close()

    tag = f"n{args.num_states}_h{args.horizon}_r{int(args.radius)}_seed{args.seed}"
    out = outdir / f"v62g_raw_images_{tag}.npz"

    np.savez_compressed(
        out,
        current_images=np.asarray(current_images, dtype=np.uint8),
        future_images=np.asarray(future_images, dtype=np.uint8),
        state=np.asarray(states, dtype=np.float32),
        future_state=np.asarray(future_states, dtype=np.float32),
        action=np.asarray(actions, dtype=np.float32),
        state_id=np.asarray(state_ids, dtype=np.int64),
        action_id=np.asarray(action_ids, dtype=np.int64),
        reward=np.asarray(rewards, dtype=np.float32),
        coverage=np.asarray(coverages, dtype=np.float32),
    )

    print("[raw-only] wrote", out)
    print("[raw-only] rows", len(state_ids))

if __name__ == "__main__":
    main()
