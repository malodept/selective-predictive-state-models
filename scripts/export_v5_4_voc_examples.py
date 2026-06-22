from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from train_mixed_hard_delta_transformer import DeltaTransformer


VARIANTS = {
    "id_block2_h36_seed0": {
        "data": "outputs/counterfactual/pybullet_obstacles_rgb_scale/pybullet_obstacle_rgb_v1_5k_seed0_dinov2_vits14_pool4.npz",
        "groups": "outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/mixed_hard_state_disjoint_v0_groups.npz",
    },
    "block2_h72_seed11": {
        "data": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/pybullet_obstacle_rgb_v1_ood_block2_h72_seed11_dinov2_vits14_pool4.npz",
        "groups": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_h72_seed11/mixed_hard_state_disjoint_v0_groups.npz",
    },
    "block2_v18_seed12": {
        "data": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/pybullet_obstacle_rgb_v1_ood_block2_v18_seed12_dinov2_vits14_pool4.npz",
        "groups": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block2_v18_seed12/mixed_hard_state_disjoint_v0_groups.npz",
    },
    "block3_h36_seed13": {
        "data": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/pybullet_obstacle_rgb_v1_ood_block3_h36_seed13_dinov2_vits14_pool4.npz",
        "groups": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h36_seed13/mixed_hard_state_disjoint_v0_groups.npz",
    },
    "block3_h48_seed10": {
        "data": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/pybullet_obstacle_rgb_v1_ood_block3_h48_seed10_dinov2_vits14_pool4.npz",
        "groups": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h48_seed10/mixed_hard_state_disjoint_v0_groups.npz",
    },
    "block3_h72_seed14": {
        "data": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/pybullet_obstacle_rgb_v1_ood_block3_h72_seed14_dinov2_vits14_pool4.npz",
        "groups": "outputs/counterfactual/pybullet_obstacles_rgb_v5_3_ood/block3_h72_seed14/mixed_hard_state_disjoint_v0_groups.npz",
    },
}

MODES = ["full", "action_only", "state_only", "no_context"]


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


@torch.no_grad()
def eval_model(data_path: Path, groups_path: Path, mode: str, checkpoint_path: Path, device: str, batch_groups: int, temperature: float):
    d = np.load(data_path, allow_pickle=True)
    g = np.load(groups_path, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)
    neg_type = g["negative_type"].astype(str)
    action_id = g["action_id"].astype(np.int64)
    action_names = d["action_names"].astype(str)

    if "correct_candidate" in g.files:
        correct_pos_all = g["correct_candidate"].astype(np.int64)
    else:
        correct_pos_all = np.zeros(len(anchor), dtype=np.int64)

    if "split" in g.files:
        test_groups = np.where(g["split"].astype(str) == "test")[0]
    else:
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(anchor))
        n_train = int(0.8 * len(anchor))
        n_val = int(0.1 * len(anchor))
        test_groups = perm[n_train + n_val:]

    moving = []
    for idx in test_groups:
        aid = action_id[idx]
        name = str(action_names[aid]) if 0 <= aid < len(action_names) else str(aid)
        if name != "stay":
            moving.append(idx)
    test_groups = np.array(moving, dtype=np.int64)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    ckpt_args = checkpoint["args"]

    model = DeltaTransformer(
        token_dim=int(checkpoint["token_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        tokens=int(checkpoint["tokens"]),
        model_dim=int(ckpt_args["model_dim"]),
        heads=int(ckpt_args["heads"]),
        layers=int(ckpt_args["layers"]),
        dropout=float(ckpt_args["dropout"]),
        mode=ckpt_args["mode"],
    ).to(device)

    model.load_state_dict(checkpoint["model"])
    model.eval()

    delta = y - z
    rows = []

    for start in range(0, len(test_groups), batch_groups):
        group_ids = test_groups[start:start + batch_groups]
        anchor_rows = anchor[group_ids]
        candidate_rows = cand[group_ids]

        z_current = torch.from_numpy(z[anchor_rows]).float().to(device)
        action = torch.from_numpy(a[anchor_rows]).float().to(device)

        pred_delta = model(z_current, action).detach().cpu().numpy()
        candidate_delta = delta[candidate_rows]

        distances = ((pred_delta[:, None] - candidate_delta) ** 2).mean(axis=(2, 3))
        probs = stable_softmax(-distances / temperature)

        predicted = distances.argmin(axis=1)
        confidence = probs.max(axis=1)

        for j, gid in enumerate(group_ids):
            correct_pos = int(correct_pos_all[gid])
            dist_row = distances[j]
            order = np.argsort(dist_row)
            best = int(order[0])
            second = int(order[1])
            correct_dist = float(dist_row[correct_pos])
            best_dist = float(dist_row[best])
            second_best_dist = float(dist_row[second])
            margin = second_best_dist - correct_dist
            pred_is_correct = int(best == correct_pos)

            aid = int(action_id[gid])
            action_name = str(action_names[aid]) if 0 <= aid < len(action_names) else str(aid)

            rows.append({
                "group_id": int(gid),
                "mode": mode,
                "action_id": aid,
                "action_name": action_name,
                "predicted": best,
                "correct_pos": correct_pos,
                "correct": pred_is_correct,
                "confidence": float(confidence[j]),
                "margin": float(margin),
                "correct_distance": correct_dist,
                "best_distance": best_dist,
                "second_best_distance": second_best_dist,
                "negative_types": ";".join(map(str, neg_type[gid].tolist())),
            })

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_4_voc/voc_examples.csv"))
    p.add_argument("--batch-groups", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.02)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    args.out.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for variant, paths in VARIANTS.items():
        print(f"===== {variant} =====")
        data_path = Path(paths["data"])
        groups_path = Path(paths["groups"])

        for mode in MODES:
            ckpt = Path(f"outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/{mode}_seed0/checkpoint.pt")
            if not ckpt.exists():
                raise FileNotFoundError(ckpt)

            print(f"--- {mode}")
            rows = eval_model(data_path, groups_path, mode, ckpt, args.device, args.batch_groups, args.temperature)
            for r in rows:
                r["variant"] = variant
            all_rows.extend(rows)

    fieldnames = [
        "variant", "group_id", "mode", "action_id", "action_name",
        "predicted", "correct_pos", "correct", "confidence", "margin",
        "correct_distance", "best_distance", "second_best_distance",
        "negative_types",
    ]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(args.out)
    print(f"rows={len(all_rows)}")


if __name__ == "__main__":
    main()
