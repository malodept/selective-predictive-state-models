from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def q_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def q_conj(q):
    q = q_normalize(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b

    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dtype=np.float64)


def q_to_rotmat(q):
    x, y, z, w = q_normalize(q)

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def q_to_rotvec(q):
    q = q_normalize(q)

    # Canonical representation: avoid q and -q discontinuity.
    if q[3] < 0:
        q = -q

    v = q[:3]
    w = q[3]
    nv = np.linalg.norm(v)

    if nv < 1e-10:
        return 2.0 * v

    angle = 2.0 * np.arctan2(nv, w)
    return angle * v / nv


def load_pose_file(raw_root: Path, trajectory_id: str):
    env, difficulty, traj = trajectory_id.split("/")

    candidates = [
        raw_root / env / difficulty / env / difficulty / traj / "pose_left.txt",
        raw_root / env / difficulty / traj / "pose_left.txt",
    ]

    for c in candidates:
        if c.exists():
            poses = np.loadtxt(c, dtype=np.float64)
            if poses.ndim == 1:
                poses = poses[None, :]
            return poses

    matches = list(raw_root.rglob(f"{traj}/pose_left.txt"))
    matches = [m for m in matches if env in str(m) and difficulty in str(m)]

    if not matches:
        raise FileNotFoundError(f"Could not find pose_left.txt for trajectory_id={trajectory_id}")

    poses = np.loadtxt(matches[0], dtype=np.float64)
    if poses.ndim == 1:
        poses = poses[None, :]
    return poses


def relative_se3_action(pose_i, pose_j):
    # Expected TartanAir convention: [tx, ty, tz, qx, qy, qz, qw].
    t_i = pose_i[:3]
    t_j = pose_j[:3]

    q_i = q_normalize(pose_i[3:7])
    q_j = q_normalize(pose_j[3:7])

    R_i = q_to_rotmat(q_i)

    # Relative translation expressed in current camera/body frame.
    t_rel = R_i.T @ (t_j - t_i)

    # Relative rotation q_i^{-1} ⊗ q_j.
    q_rel = q_mul(q_conj(q_i), q_j)
    rotvec = q_to_rotvec(q_rel)

    return np.concatenate([t_rel, rotvec]).astype(np.float32)


def main():
    args = parse_args()

    d = np.load(args.input, allow_pickle=True)
    out = {k: d[k] for k in d.files}

    trajectory_ids = np.asarray(out["trajectory_id"]).astype(str)
    frame_i = out["frame_i"].astype(np.int64)
    frame_j = out["frame_j"].astype(np.int64)

    cache = {}
    actions = np.zeros((len(trajectory_ids), 6), dtype=np.float32)

    for idx, tid in enumerate(trajectory_ids):
        if tid not in cache:
            cache[tid] = load_pose_file(args.raw_root, tid)

        poses = cache[tid]
        i = int(frame_i[idx])
        j = int(frame_j[idx])

        actions[idx] = relative_se3_action(poses[i], poses[j])

        if (idx + 1) % 20000 == 0:
            print(f"processed {idx + 1}/{len(trajectory_ids)}", flush=True)

    out["action_pose_diff_7d"] = out["action"].astype(np.float32)
    out["action"] = actions
    out["action_type"] = np.asarray("relative_se3_translation_rotvec")
    out["action_dim"] = np.asarray(6)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out)

    print(f"Wrote {args.output}")
    print("z_current:", out["z_current"].shape, out["z_current"].dtype)
    print("action:", out["action"].shape, out["action"].dtype)
    print("action mean:", out["action"].mean(axis=0))
    print("action std :", out["action"].std(axis=0))


if __name__ == "__main__":
    main()
