from __future__ import annotations

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

MODE_TO_CKPT = {
    "small_full": "outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/small_full_seed0/checkpoint.pt",
    "full": "outputs/counterfactual/pybullet_obstacles_rgb_scale/mixed_hard_state_disjoint/delta_transformer/full_seed0/checkpoint.pt",
}


def stable_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


@torch.no_grad()
def eval_model(data_path: Path, groups_path: Path, mode: str, checkpoint_path: Path, device: str = "cuda", batch_groups: int = 128, temperature: float = 0.02):
    d = np.load(data_path, allow_pickle=True)
    g = np.load(groups_path, allow_pickle=True)

    z = d["z_current"].astype(np.float32)
    y = d["z_future"].astype(np.float32)
    a = d["action"].astype(np.float32)

    anchor = g["anchor_indices"].astype(np.int64)
    cand = g["candidate_indices"].astype(np.int64)
    action_id = g["action_id"].astype(np.int64)
    action_names = d["action_names"].astype(str)
    correct_pos_all = g["correct_candidate"].astype(np.int64) if "correct_candidate" in g.files else np.zeros(len(anchor), dtype=np.int64)

    if "split" in g.files:
        test_groups = np.where(g["split"].astype(str) == "test")[0]
    else:
        raise RuntimeError("Expected explicit split in groups file")

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
            correct = int(best == correct_pos)

            aid = int(action_id[gid])
            action_name = str(action_names[aid]) if 0 <= aid < len(action_names) else str(aid)

            rows.append({
                "variant": "",
                "group_id": int(gid),
                "mode": mode,
                "action_id": aid,
                "action_name": action_name,
                "correct": correct,
                "confidence": float(confidence[j]),
                "best_distance": float(dist_row[best]),
                "second_best_distance": float(dist_row[second]),
                "pred_margin": float(dist_row[second] - dist_row[best]),
                "correct_distance": float(dist_row[correct_pos]),
            })

    return rows


def main():
    out = Path("reports/tables/protocol/pybullet_obstacle_rgb_encoder/v5_6_small_full/small_full_voc_examples.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for variant, paths in VARIANTS.items():
        print(f"===== {variant} =====")
        for mode, ckpt in MODE_TO_CKPT.items():
            print(f"--- {mode}")
            rows = eval_model(Path(paths["data"]), Path(paths["groups"]), mode, Path(ckpt), device=device)
            for r in rows:
                r["variant"] = variant
            all_rows.extend(rows)

    fieldnames = [
        "variant", "group_id", "mode", "action_id", "action_name", "correct",
        "confidence", "best_distance", "second_best_distance", "pred_margin", "correct_distance",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(out)
    print("rows", len(all_rows))


if __name__ == "__main__":
    main()
