from __future__ import annotations

from pathlib import Path
import math
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pybullet as p
    import pybullet_data
except Exception as e:
    raise RuntimeError("PyBullet is required for v50B object-pushing sanity.") from e


OUT = Path("reports/tables/protocol/pybullet_object_pushing/v50_object_pushing")
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

PAPER = Path("reports/paper_assets_v50_object_pushing_sanity")
PTAB = PAPER / "tables"
PFIG = PAPER / "figures"
PTXT = PAPER / "text"
PTAB.mkdir(parents=True, exist_ok=True)
PFIG.mkdir(parents=True, exist_ok=True)
PTXT.mkdir(parents=True, exist_ok=True)

ACTIONS = {
    "stay": np.array([0.0, 0.0], dtype=np.float32),
    "right": np.array([1.0, 0.0], dtype=np.float32),
    "left": np.array([-1.0, 0.0], dtype=np.float32),
    "forward": np.array([0.0, 1.0], dtype=np.float32),
    "backward": np.array([0.0, -1.0], dtype=np.float32),
}

SIDES = {
    "left": np.array([-1.0, 0.0], dtype=np.float32),
    "right": np.array([1.0, 0.0], dtype=np.float32),
    "back": np.array([0.0, -1.0], dtype=np.float32),
    "front": np.array([0.0, 1.0], dtype=np.float32),
}


def rot2(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


class RichObjectPushingEnv:
    def __init__(
        self,
        seed: int = 51,
        horizon_steps: int = 120,
        action_speed: float = 0.55,
        obj_half_extents=(0.085, 0.045, 0.030),
        pusher_radius: float = 0.040,
    ):
        self.rng = np.random.default_rng(seed)
        self.horizon_steps = horizon_steps
        self.action_speed = action_speed
        self.obj_half_extents = np.array(obj_half_extents, dtype=np.float32)
        self.pusher_radius = pusher_radius

        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=90,
            deterministicOverlappingPairs=1,
            physicsClientId=self.cid,
        )
        self._make_world()

    def close(self):
        try:
            p.disconnect(self.cid)
        except Exception:
            pass

    def _make_world(self):
        p.resetSimulation(physicsClientId=self.cid)
        p.setGravity(0, 0, -9.81, physicsClientId=self.cid)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.9, physicsClientId=self.cid)

        hx, hy, hz = self.obj_half_extents.tolist()
        obj_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            physicsClientId=self.cid,
        )
        obj_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
            rgbaColor=[0.12, 0.28, 0.95, 1.0],
            physicsClientId=self.cid,
        )
        self.obj_id = p.createMultiBody(
            baseMass=0.28,
            baseCollisionShapeIndex=obj_col,
            baseVisualShapeIndex=obj_vis,
            basePosition=[0.0, 0.0, hz],
            physicsClientId=self.cid,
        )
        p.changeDynamics(
            self.obj_id,
            -1,
            lateralFriction=0.85,
            spinningFriction=0.02,
            rollingFriction=0.02,
            linearDamping=0.025,
            angularDamping=0.025,
            physicsClientId=self.cid,
        )

        push_col = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.pusher_radius,
            physicsClientId=self.cid,
        )
        push_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.pusher_radius,
            rgbaColor=[0.95, 0.20, 0.12, 1.0],
            physicsClientId=self.cid,
        )
        self.pusher_id = p.createMultiBody(
            baseMass=0.40,
            baseCollisionShapeIndex=push_col,
            baseVisualShapeIndex=push_vis,
            basePosition=[-0.2, 0.0, self.pusher_radius],
            physicsClientId=self.cid,
        )
        p.changeDynamics(
            self.pusher_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.03,
            rollingFriction=0.03,
            linearDamping=0.01,
            angularDamping=0.01,
            physicsClientId=self.cid,
        )

    def sample_state(self, idx: int) -> dict:
        side_name = list(SIDES.keys())[idx % len(SIDES)]
        side_vec_world = SIDES[side_name]

        obj_xy = self.rng.uniform(-0.10, 0.10, size=2).astype(np.float32)
        obj_yaw = float(self.rng.uniform(-math.pi, math.pi))

        # Make contact non-trivial: tangent offsets and variable gap.
        tangent = np.array([-side_vec_world[1], side_vec_world[0]], dtype=np.float32)
        tangent_offset = float(self.rng.uniform(-0.145, 0.145))
        gap = float(self.rng.uniform(0.018, 0.115))

        # Approximate support radius of the rotated rectangle in the side direction.
        R = rot2(obj_yaw)
        local_dir = R.T @ side_vec_world
        support = float(
            abs(local_dir[0]) * self.obj_half_extents[0]
            + abs(local_dir[1]) * self.obj_half_extents[1]
        )

        start_dist = support + self.pusher_radius + gap
        pusher_xy = obj_xy + side_vec_world * start_dist + tangent * tangent_offset

        # State also contains dynamics parameters for future hidden-dynamics shifts.
        friction = float(self.rng.uniform(0.65, 1.05))
        mass = float(self.rng.uniform(0.22, 0.38))

        return {
            "state_id": idx,
            "side": side_name,
            "object_xy": obj_xy,
            "object_yaw": obj_yaw,
            "pusher_xy": pusher_xy.astype(np.float32),
            "start_dist": start_dist,
            "tangent_offset": tangent_offset,
            "gap": gap,
            "object_mass": mass,
            "object_friction": friction,
        }

    def reset_to_state(self, state: dict):
        obj_xy = np.asarray(state["object_xy"], dtype=np.float32)
        push_xy = np.asarray(state["pusher_xy"], dtype=np.float32)
        hz = float(self.obj_half_extents[2])

        p.changeDynamics(
            self.obj_id,
            -1,
            mass=float(state["object_mass"]),
            lateralFriction=float(state["object_friction"]),
            physicsClientId=self.cid,
        )

        p.resetBasePositionAndOrientation(
            self.obj_id,
            [float(obj_xy[0]), float(obj_xy[1]), hz],
            p.getQuaternionFromEuler([0.0, 0.0, float(state["object_yaw"])]),
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(self.obj_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        p.resetBasePositionAndOrientation(
            self.pusher_id,
            [float(push_xy[0]), float(push_xy[1]), self.pusher_radius],
            [0, 0, 0, 1],
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(self.pusher_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.cid)

    def get_xy(self, body_id: int) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
        return np.array(pos[:2], dtype=np.float32)

    def get_yaw(self, body_id: int) -> float:
        _, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
        return float(p.getEulerFromQuaternion(quat)[2])

    def rollout(self, state: dict, action_name: str) -> dict:
        self.reset_to_state(state)

        direction = ACTIONS[action_name]
        contact_seen = False
        contact_steps = 0
        max_normal_force = 0.0

        obj_xy0 = self.get_xy(self.obj_id)
        obj_yaw0 = self.get_yaw(self.obj_id)
        push_xy0 = self.get_xy(self.pusher_id)

        for _ in range(self.horizon_steps):
            vel = direction * self.action_speed
            p.resetBaseVelocity(
                self.pusher_id,
                [float(vel[0]), float(vel[1]), 0.0],
                [0.0, 0.0, 0.0],
                physicsClientId=self.cid,
            )
            p.stepSimulation(physicsClientId=self.cid)

            cps = p.getContactPoints(self.obj_id, self.pusher_id, physicsClientId=self.cid)
            if cps:
                contact_seen = True
                contact_steps += 1
                # Normal force is tuple index 9 in PyBullet contact point.
                for cp in cps:
                    if len(cp) > 9:
                        max_normal_force = max(max_normal_force, float(cp[9]))

        obj_xy1 = self.get_xy(self.obj_id)
        obj_yaw1 = self.get_yaw(self.obj_id)
        push_xy1 = self.get_xy(self.pusher_id)

        return {
            "object_xy0": obj_xy0,
            "object_xy1": obj_xy1,
            "object_yaw0": obj_yaw0,
            "object_yaw1": obj_yaw1,
            "pusher_xy0": push_xy0,
            "pusher_xy1": push_xy1,
            "object_disp": float(np.linalg.norm(obj_xy1 - obj_xy0)),
            "object_rot": float(abs(math.atan2(math.sin(obj_yaw1 - obj_yaw0), math.cos(obj_yaw1 - obj_yaw0)))),
            "pusher_disp": float(np.linalg.norm(push_xy1 - push_xy0)),
            "contact": bool(contact_seen),
            "contact_steps": int(contact_steps),
            "max_normal_force": max_normal_force,
        }

    def render_rgb(self, width=128, height=128) -> np.ndarray:
        view = p.computeViewMatrix(
            cameraEyePosition=[0.0, -0.05, 1.35],
            cameraTargetPosition=[0.0, 0.0, 0.0],
            cameraUpVector=[0.0, 1.0, 0.0],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=42,
            aspect=float(width) / float(height),
            nearVal=0.02,
            farVal=3.0,
        )
        img = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self.cid,
        )[2]
        rgba = np.asarray(img, dtype=np.uint8).reshape(height, width, 4)
        return rgba[:, :, :3].copy()


def run(n_states=512, seed=51):
    env = RichObjectPushingEnv(seed=seed)
    rows = []
    group_rows = []

    try:
        states = [env.sample_state(i) for i in range(n_states)]

        for s in states:
            futures = {}
            contacts = {}
            disps = {}
            rots = {}

            for action in ACTIONS:
                out = env.rollout(s, action)
                futures[action] = out["object_xy1"]
                contacts[action] = out["contact"]
                disps[action] = out["object_disp"]
                rots[action] = out["object_rot"]

                rows.append({
                    "state_id": s["state_id"],
                    "side": s["side"],
                    "action": action,
                    "gap": s["gap"],
                    "tangent_offset": s["tangent_offset"],
                    "object_mass": s["object_mass"],
                    "object_friction": s["object_friction"],
                    "object_yaw": s["object_yaw"],
                    "contact": int(out["contact"]),
                    "contact_steps": out["contact_steps"],
                    "max_normal_force": out["max_normal_force"],
                    "object_disp": out["object_disp"],
                    "object_rot": out["object_rot"],
                    "pusher_disp": out["pusher_disp"],
                    "object_x0": float(out["object_xy0"][0]),
                    "object_y0": float(out["object_xy0"][1]),
                    "object_x1": float(out["object_xy1"][0]),
                    "object_y1": float(out["object_xy1"][1]),
                    "pusher_x0": float(out["pusher_xy0"][0]),
                    "pusher_y0": float(out["pusher_xy0"][1]),
                    "pusher_x1": float(out["pusher_xy1"][0]),
                    "pusher_y1": float(out["pusher_xy1"][1]),
                })

            pairwise = [
                float(np.linalg.norm(futures[a] - futures[b]))
                for a, b in itertools.combinations(ACTIONS.keys(), 2)
            ]

            contact_count = int(sum(contacts.values()))
            moving_contact_count = int(sum(contacts[a] for a in ["right", "left", "forward", "backward"]))
            strong_count = int(sum(disps[a] > 0.06 for a in ACTIONS))
            rot_count = int(sum(rots[a] > 0.05 for a in ACTIONS))

            group_rows.append({
                "state_id": s["state_id"],
                "side": s["side"],
                "mean_future_xy_spread": float(np.mean(pairwise)),
                "max_future_xy_spread": float(np.max(pairwise)),
                "contact_action_count": contact_count,
                "moving_contact_action_count": moving_contact_count,
                "strong_displacement_action_count": strong_count,
                "rotation_action_count": rot_count,
                "nondegenerate": int(np.max(pairwise) > 0.02),
                "contact_mode_nontrivial": int(0 < moving_contact_count < 4),
            })

        df = pd.DataFrame(rows)
        groups = pd.DataFrame(group_rows)

        # Repeatability.
        rep_rows = []
        for s in states[:32]:
            for action in ACTIONS:
                a = env.rollout(s, action)
                b = env.rollout(s, action)
                rep_rows.append({
                    "state_id": s["state_id"],
                    "action": action,
                    "object_xy_repeat_diff": float(np.max(np.abs(a["object_xy1"] - b["object_xy1"]))),
                    "object_rot_repeat_diff": float(abs(a["object_rot"] - b["object_rot"])),
                    "contact_match": int(a["contact"] == b["contact"]),
                })
        rep = pd.DataFrame(rep_rows)

        df.to_csv(OUT / "v50b_rich_object_pushing_transitions.csv", index=False)
        groups.to_csv(OUT / "v50b_rich_object_pushing_groups.csv", index=False)
        rep.to_csv(OUT / "v50b_rich_object_pushing_repeatability.csv", index=False)

        action_summary = (
            df.groupby("action", as_index=False)
            .agg(
                contact_rate=("contact", "mean"),
                object_disp_mean=("object_disp", "mean"),
                object_disp_median=("object_disp", "median"),
                object_disp_p90=("object_disp", lambda x: float(np.quantile(x, 0.90))),
                object_rot_mean=("object_rot", "mean"),
                contact_steps_mean=("contact_steps", "mean"),
                force_mean=("max_normal_force", "mean"),
            )
        )
        action_summary.to_csv(OUT / "v50b_rich_action_summary.csv", index=False)
        action_summary.to_csv(PTAB / "table_v50b_rich_action_summary.csv", index=False)

        group_summary = pd.DataFrame([{
            "states": len(groups),
            "nondegenerate_frac": float(groups["nondegenerate"].mean()),
            "contact_mode_nontrivial_frac": float(groups["contact_mode_nontrivial"].mean()),
            "mean_contact_action_count": float(groups["contact_action_count"].mean()),
            "mean_moving_contact_action_count": float(groups["moving_contact_action_count"].mean()),
            "contact_count_std": float(groups["moving_contact_action_count"].std()),
            "mean_future_xy_spread": float(groups["mean_future_xy_spread"].mean()),
            "max_future_xy_spread_mean": float(groups["max_future_xy_spread"].mean()),
            "mean_strong_displacement_actions": float(groups["strong_displacement_action_count"].mean()),
            "mean_rotation_actions": float(groups["rotation_action_count"].mean()),
        }])
        group_summary.to_csv(OUT / "v50b_rich_group_summary.csv", index=False)
        group_summary.to_csv(PTAB / "table_v50b_rich_group_summary.csv", index=False)

        repeat_summary = pd.DataFrame([{
            "repeat_tests": len(rep),
            "object_xy_repeat_diff_max": float(rep["object_xy_repeat_diff"].max()),
            "object_xy_repeat_diff_mean": float(rep["object_xy_repeat_diff"].mean()),
            "object_rot_repeat_diff_max": float(rep["object_rot_repeat_diff"].max()),
            "contact_match_rate": float(rep["contact_match"].mean()),
        }])
        repeat_summary.to_csv(OUT / "v50b_rich_repeatability_summary.csv", index=False)
        repeat_summary.to_csv(PTAB / "table_v50b_rich_repeatability_summary.csv", index=False)

        make_figures(env, states, df, groups, action_summary)
        write_markdown(action_summary, group_summary, repeat_summary)
        write_tex(action_summary, group_summary, repeat_summary)
        write_manifest()

        print(OUT / "v50b_rich_object_pushing_sanity.md")
        print((OUT / "v50b_rich_object_pushing_sanity.md").read_text())

    finally:
        env.close()


def make_figures(env, states, df, groups, action_summary):
    acts = list(ACTIONS.keys())
    a = action_summary.set_index("action").loc[acts].reset_index()
    x = np.arange(len(a))
    w = 0.35

    plt.figure(figsize=(7.4, 4.0))
    plt.bar(x - w / 2, a["object_disp_mean"], width=w, label="Mean object displacement")
    plt.bar(x + w / 2, a["contact_rate"], width=w, label="Contact rate")
    plt.xticks(x, a["action"])
    plt.ylabel("Value")
    plt.title("Rich object-pushing action sanity")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for pth in [FIG / "fig_v50b_rich_action_contact_displacement.png", PFIG / "fig_v50b_rich_action_contact_displacement.png"]:
        plt.savefig(pth, dpi=240)
    plt.close()

    plt.figure(figsize=(6.8, 3.8))
    counts = groups["moving_contact_action_count"].value_counts().sort_index()
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Moving actions with contact")
    plt.ylabel("Groups")
    plt.title("Contact-mode diversity per counterfactual group")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for pth in [FIG / "fig_v50b_contact_count_distribution.png", PFIG / "fig_v50b_contact_count_distribution.png"]:
        plt.savefig(pth, dpi=240)
    plt.close()

    plt.figure(figsize=(6.8, 3.8))
    plt.hist(groups["max_future_xy_spread"], bins=35)
    plt.xlabel("Max pairwise future object-xy spread")
    plt.ylabel("Groups")
    plt.title("Rich counterfactual group diversity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for pth in [FIG / "fig_v50b_rich_group_diversity.png", PFIG / "fig_v50b_rich_group_diversity.png"]:
        plt.savefig(pth, dpi=240)
    plt.close()

    fig, axes = plt.subplots(2, len(ACTIONS) + 1, figsize=(10.5, 4.0))
    for row_idx, state in enumerate(states[:2]):
        env.reset_to_state(state)
        axes[row_idx, 0].imshow(env.render_rgb())
        axes[row_idx, 0].set_title(f"current\n{state['side']}", fontsize=8)
        axes[row_idx, 0].axis("off")
        for col_idx, action in enumerate(ACTIONS.keys(), start=1):
            env.rollout(state, action)
            axes[row_idx, col_idx].imshow(env.render_rgb())
            axes[row_idx, col_idx].set_title(action, fontsize=8)
            axes[row_idx, col_idx].axis("off")
    plt.tight_layout()
    for pth in [FIG / "fig_v50b_rich_counterfactual_rgb_grid.png", PFIG / "fig_v50b_rich_counterfactual_rgb_grid.png"]:
        plt.savefig(pth, dpi=220)
    plt.close()


def write_markdown(action_summary, group_summary, repeat_summary):
    gs = group_summary.iloc[0]
    rs = repeat_summary.iloc[0]

    lines = []
    lines.append("# SPSM v50B rich object-pushing sanity\n")
    lines.append("v50B enriches v50A with a rectangular object, object yaw, tangent offsets, variable gaps, mass/friction variation, and non-trivial contact modes.\n")

    lines.append("## Repeatability\n")
    lines.append(f"- repeat tests: `{int(rs['repeat_tests'])}`")
    lines.append(f"- max object xy repeat diff: `{rs['object_xy_repeat_diff_max']:.6g}`")
    lines.append(f"- mean object xy repeat diff: `{rs['object_xy_repeat_diff_mean']:.6g}`")
    lines.append(f"- max object rotation repeat diff: `{rs['object_rot_repeat_diff_max']:.6g}`")
    lines.append(f"- contact repeat match rate: `{rs['contact_match_rate']:.3f}`\n")

    lines.append("## Action summary\n")
    lines.append("| action | contact rate | mean disp | median disp | p90 disp | mean rot | mean contact steps | mean force |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in action_summary.iterrows():
        lines.append(
            f"| `{r['action']}` | {r['contact_rate']:.3f} | {r['object_disp_mean']:.3f} | "
            f"{r['object_disp_median']:.3f} | {r['object_disp_p90']:.3f} | "
            f"{r['object_rot_mean']:.3f} | {r['contact_steps_mean']:.2f} | {r['force_mean']:.3f} |"
        )

    lines.append("\n## Group diversity\n")
    lines.append(f"- states: `{int(gs['states'])}`")
    lines.append(f"- nondegenerate groups: `{gs['nondegenerate_frac']:.3f}`")
    lines.append(f"- contact-mode nontrivial groups, 0 < contact actions < 4: `{gs['contact_mode_nontrivial_frac']:.3f}`")
    lines.append(f"- mean moving contact action count: `{gs['mean_moving_contact_action_count']:.3f}`")
    lines.append(f"- contact count std: `{gs['contact_count_std']:.3f}`")
    lines.append(f"- mean future xy spread: `{gs['mean_future_xy_spread']:.3f}`")
    lines.append(f"- mean max future xy spread: `{gs['max_future_xy_spread_mean']:.3f}`")
    lines.append(f"- mean strong displacement actions: `{gs['mean_strong_displacement_actions']:.3f}`")
    lines.append(f"- mean rotation actions: `{gs['mean_rotation_actions']:.3f}`\n")

    lines.append("## Interpretation\n")
    lines.append("- v50B should replace v50A if repeatability remains exact and contact-mode diversity is non-trivial.")
    lines.append("- The target is not maximum realism yet; it is to create a second exact-intervention regime where failures depend on contact geometry and object pose.")
    lines.append("- If contact-mode diversity is high enough, v50C should generate the actual RGB intervention dataset.")
    lines.append("- If contact-mode diversity is still too simple, increase tangent-offset range, add diagonal actions, or add multiple object shapes.")

    (OUT / "v50b_rich_object_pushing_sanity.md").write_text("\n".join(lines) + "\n")

    claim = """
# v50B rich object-pushing sanity claim

v50B tests whether SPSM-v2 can move beyond obstacle navigation into object interaction. Compared with v50A, it introduces contact-mode diversity through a rectangular object, randomized object yaw, tangent offsets, variable gaps, and mass/friction variation.

The scientific target is a second exact-intervention environment where the failure axes include object pose, contact geometry, pusher-object relation, and later hidden dynamics.
"""
    (PTXT / "v50b_rich_object_pushing_sanity_claim.md").write_text(claim.strip() + "\n")


def write_tex(action_summary, group_summary, repeat_summary):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Action & Contact & Mean disp. & P90 disp. & Mean rot. \\")
    lines.append(r"\midrule")
    for _, r in action_summary.iterrows():
        lines.append(
            f"{r['action']} & {r['contact_rate']:.3f} & {r['object_disp_mean']:.3f} & "
            f"{r['object_disp_p90']:.3f} & {r['object_rot_mean']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{v50B rich object-pushing action sanity. Randomized object pose and pusher offsets create contact-dependent future object states.}")
    lines.append(r"\label{tab:v50b_rich_object_pushing_action_sanity}")
    lines.append(r"\end{table}")
    (PTAB / "table_v50b_rich_action_summary.tex").write_text("\n".join(lines) + "\n")

    gs = group_summary.iloc[0]
    rs = repeat_summary.iloc[0]
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lr}")
    lines.append(r"\toprule")
    lines.append(r"Metric & Value \\")
    lines.append(r"\midrule")
    lines.append(f"States & {int(gs['states'])} \\\\")
    lines.append(f"Nondegenerate groups & {gs['nondegenerate_frac']:.3f} \\\\")
    lines.append(f"Contact-mode nontrivial & {gs['contact_mode_nontrivial_frac']:.3f} \\\\")
    lines.append(f"Mean moving contact actions & {gs['mean_moving_contact_action_count']:.3f} \\\\")
    lines.append(f"Contact-count std & {gs['contact_count_std']:.3f} \\\\")
    lines.append(f"Mean future spread & {gs['mean_future_xy_spread']:.3f} \\\\")
    lines.append(f"Repeat object diff max & {rs['object_xy_repeat_diff_max']:.3e} \\\\")
    lines.append(f"Contact repeat match & {rs['contact_match_rate']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{v50B exact-intervention sanity. The richer object-pushing environment remains deterministic while introducing contact-mode diversity.}")
    lines.append(r"\label{tab:v50b_rich_exact_intervention_sanity}")
    lines.append(r"\end{table}")
    (PTAB / "table_v50b_rich_exact_intervention_sanity.tex").write_text("\n".join(lines) + "\n")


def write_manifest():
    manifest = ["# Paper assets v50B rich object pushing sanity\n", "## Tables\n"]
    for pth in sorted(PTAB.glob("*v50b*")):
        manifest.append(f"- `{pth}`")
    manifest.append("\n## Figures\n")
    for pth in sorted(PFIG.glob("*v50b*")):
        manifest.append(f"- `{pth}`")
    manifest.append("\n## Text\n")
    for pth in sorted(PTXT.glob("*v50b*")):
        manifest.append(f"- `{pth}`")
    (PAPER / "paper_assets_v50b_manifest.md").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    run(n_states=512, seed=51)
