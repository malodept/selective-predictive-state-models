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
    raise RuntimeError("PyBullet is required for v50A object-pushing sanity.") from e


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


class ObjectPushingEnv:
    def __init__(
        self,
        seed: int = 0,
        horizon_steps: int = 120,
        action_speed: float = 0.55,
        cube_half: float = 0.055,
        cube_height: float = 0.055,
        pusher_radius: float = 0.045,
    ):
        self.rng = np.random.default_rng(seed)
        self.horizon_steps = horizon_steps
        self.action_speed = action_speed
        self.cube_half = cube_half
        self.cube_height = cube_height
        self.pusher_radius = pusher_radius

        self.cid = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.cid)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.cid)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / 240.0,
            numSolverIterations=80,
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
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.cid)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.cid)
        p.changeDynamics(self.plane_id, -1, lateralFriction=0.9, physicsClientId=self.cid)

        cube_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_half, self.cube_half, self.cube_height / 2.0],
            physicsClientId=self.cid,
        )
        cube_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_half, self.cube_half, self.cube_height / 2.0],
            rgbaColor=[0.15, 0.35, 0.95, 1.0],
            physicsClientId=self.cid,
        )
        self.cube_id = p.createMultiBody(
            baseMass=0.25,
            baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis,
            basePosition=[0.0, 0.0, self.cube_height / 2.0],
            physicsClientId=self.cid,
        )
        p.changeDynamics(
            self.cube_id,
            -1,
            lateralFriction=0.85,
            spinningFriction=0.02,
            rollingFriction=0.02,
            linearDamping=0.02,
            angularDamping=0.02,
            physicsClientId=self.cid,
        )

        pusher_col = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.pusher_radius,
            physicsClientId=self.cid,
        )
        pusher_vis = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.pusher_radius,
            rgbaColor=[0.95, 0.25, 0.15, 1.0],
            physicsClientId=self.cid,
        )
        self.pusher_id = p.createMultiBody(
            baseMass=0.35,
            baseCollisionShapeIndex=pusher_col,
            baseVisualShapeIndex=pusher_vis,
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
        side_vec = SIDES[side_name]

        obj_xy = self.rng.uniform(-0.10, 0.10, size=2).astype(np.float32)
        obj_yaw = float(self.rng.uniform(-math.pi, math.pi))

        # Gap from object surface to pusher: close enough that one inward action contacts.
        start_dist = float(self.cube_half + self.pusher_radius + self.rng.uniform(0.045, 0.080))
        pusher_xy = obj_xy + side_vec * start_dist

        return {
            "state_id": idx,
            "side": side_name,
            "object_xy": obj_xy,
            "object_yaw": obj_yaw,
            "pusher_xy": pusher_xy.astype(np.float32),
            "start_dist": start_dist,
        }

    def reset_to_state(self, state: dict):
        obj_xy = np.asarray(state["object_xy"], dtype=np.float32)
        push_xy = np.asarray(state["pusher_xy"], dtype=np.float32)

        p.resetBasePositionAndOrientation(
            self.cube_id,
            [float(obj_xy[0]), float(obj_xy[1]), self.cube_height / 2.0],
            p.getQuaternionFromEuler([0.0, 0.0, float(state["object_yaw"])]),
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(self.cube_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        p.resetBasePositionAndOrientation(
            self.pusher_id,
            [float(push_xy[0]), float(push_xy[1]), self.pusher_radius],
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.cid,
        )
        p.resetBaseVelocity(self.pusher_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.cid)

        for _ in range(10):
            p.stepSimulation(physicsClientId=self.cid)

    def get_xy(self, body_id: int) -> np.ndarray:
        pos, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=self.cid)
        return np.array(pos[:2], dtype=np.float32)

    def step_action(self, action_name: str) -> dict:
        direction = ACTIONS[action_name]
        contact_seen = False

        obj_xy0 = self.get_xy(self.cube_id)
        push_xy0 = self.get_xy(self.pusher_id)

        for _ in range(self.horizon_steps):
            vel_xy = direction * self.action_speed
            p.resetBaseVelocity(
                self.pusher_id,
                [float(vel_xy[0]), float(vel_xy[1]), 0.0],
                [0.0, 0.0, 0.0],
                physicsClientId=self.cid,
            )
            p.stepSimulation(physicsClientId=self.cid)

            cps = p.getContactPoints(
                bodyA=self.cube_id,
                bodyB=self.pusher_id,
                physicsClientId=self.cid,
            )
            if len(cps) > 0:
                contact_seen = True

        obj_xy1 = self.get_xy(self.cube_id)
        push_xy1 = self.get_xy(self.pusher_id)

        return {
            "object_xy0": obj_xy0,
            "object_xy1": obj_xy1,
            "pusher_xy0": push_xy0,
            "pusher_xy1": push_xy1,
            "object_disp": float(np.linalg.norm(obj_xy1 - obj_xy0)),
            "pusher_disp": float(np.linalg.norm(push_xy1 - push_xy0)),
            "contact": bool(contact_seen),
        }

    def rollout(self, state: dict, action_name: str) -> dict:
        self.reset_to_state(state)
        out = self.step_action(action_name)
        return out

    def render_rgb(self, width: int = 128, height: int = 128) -> np.ndarray:
        view = p.computeViewMatrix(
            cameraEyePosition=[0.0, -0.05, 1.35],
            cameraTargetPosition=[0.0, 0.0, 0.0],
            cameraUpVector=[0.0, 1.0, 0.0],
        )
        proj = p.computeProjectionMatrixFOV(
            fov=42.0,
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


def run_sanity(n_states: int = 256, seed: int = 50):
    env = ObjectPushingEnv(seed=seed)
    rows = []
    group_rows = []

    try:
        states = [env.sample_state(i) for i in range(n_states)]

        # Main transition sweep.
        for state in states:
            futures = {}
            contacts = {}
            disps = {}

            for action_name in ACTIONS.keys():
                out = env.rollout(state, action_name)

                row = {
                    "state_id": state["state_id"],
                    "side": state["side"],
                    "action": action_name,
                    "start_dist": state["start_dist"],
                    "object_x0": float(out["object_xy0"][0]),
                    "object_y0": float(out["object_xy0"][1]),
                    "object_x1": float(out["object_xy1"][0]),
                    "object_y1": float(out["object_xy1"][1]),
                    "pusher_x0": float(out["pusher_xy0"][0]),
                    "pusher_y0": float(out["pusher_xy0"][1]),
                    "pusher_x1": float(out["pusher_xy1"][0]),
                    "pusher_y1": float(out["pusher_xy1"][1]),
                    "object_disp": out["object_disp"],
                    "pusher_disp": out["pusher_disp"],
                    "contact": int(out["contact"]),
                }
                rows.append(row)

                futures[action_name] = out["object_xy1"]
                contacts[action_name] = out["contact"]
                disps[action_name] = out["object_disp"]

            pairwise = []
            for a, b in itertools.combinations(ACTIONS.keys(), 2):
                pairwise.append(float(np.linalg.norm(futures[a] - futures[b])))

            group_rows.append({
                "state_id": state["state_id"],
                "side": state["side"],
                "max_future_xy_spread": float(np.max(pairwise)),
                "mean_future_xy_spread": float(np.mean(pairwise)),
                "contact_action_count": int(sum(bool(v) for v in contacts.values())),
                "moving_contact_count": int(sum(bool(contacts[a]) for a in ["right", "left", "forward", "backward"])),
                "max_object_disp": float(max(disps.values())),
                "nondegenerate": int(np.max(pairwise) > 0.02),
            })

        df = pd.DataFrame(rows)
        groups = pd.DataFrame(group_rows)

        # Exact reset reproducibility check.
        rep_rows = []
        for state in states[:32]:
            for action_name in ACTIONS.keys():
                out1 = env.rollout(state, action_name)
                out2 = env.rollout(state, action_name)
                rep_rows.append({
                    "state_id": state["state_id"],
                    "action": action_name,
                    "object_xy_repeat_diff": float(np.max(np.abs(out1["object_xy1"] - out2["object_xy1"]))),
                    "pusher_xy_repeat_diff": float(np.max(np.abs(out1["pusher_xy1"] - out2["pusher_xy1"]))),
                    "contact_match": int(out1["contact"] == out2["contact"]),
                })
        rep = pd.DataFrame(rep_rows)

        # Save raw results.
        df.to_csv(OUT / "v50a_object_pushing_transitions.csv", index=False)
        groups.to_csv(OUT / "v50a_object_pushing_groups.csv", index=False)
        rep.to_csv(OUT / "v50a_object_pushing_repeatability.csv", index=False)

        # Summary tables.
        action_summary = (
            df.groupby("action", as_index=False)
            .agg(
                contact_rate=("contact", "mean"),
                object_disp_mean=("object_disp", "mean"),
                object_disp_median=("object_disp", "median"),
                object_disp_p90=("object_disp", lambda x: float(np.quantile(x, 0.90))),
                pusher_disp_mean=("pusher_disp", "mean"),
            )
        )
        action_summary.to_csv(OUT / "v50a_action_summary.csv", index=False)
        action_summary.to_csv(PTAB / "table_v50a_action_summary.csv", index=False)

        group_summary = pd.DataFrame([{
            "states": len(groups),
            "mean_future_xy_spread": float(groups["mean_future_xy_spread"].mean()),
            "median_future_xy_spread": float(groups["mean_future_xy_spread"].median()),
            "max_future_xy_spread_mean": float(groups["max_future_xy_spread"].mean()),
            "nondegenerate_frac": float(groups["nondegenerate"].mean()),
            "mean_contact_action_count": float(groups["contact_action_count"].mean()),
            "mean_moving_contact_count": float(groups["moving_contact_count"].mean()),
            "max_object_disp_mean": float(groups["max_object_disp"].mean()),
        }])
        group_summary.to_csv(OUT / "v50a_group_summary.csv", index=False)
        group_summary.to_csv(PTAB / "table_v50a_group_summary.csv", index=False)

        repeat_summary = pd.DataFrame([{
            "repeat_tests": len(rep),
            "object_xy_repeat_diff_max": float(rep["object_xy_repeat_diff"].max()),
            "object_xy_repeat_diff_mean": float(rep["object_xy_repeat_diff"].mean()),
            "pusher_xy_repeat_diff_max": float(rep["pusher_xy_repeat_diff"].max()),
            "contact_match_rate": float(rep["contact_match"].mean()),
        }])
        repeat_summary.to_csv(OUT / "v50a_repeatability_summary.csv", index=False)
        repeat_summary.to_csv(PTAB / "table_v50a_repeatability_summary.csv", index=False)

        make_figures(env, states, df, groups, action_summary)

        write_tex(action_summary, group_summary, repeat_summary)
        write_markdown(action_summary, group_summary, repeat_summary, groups)
        write_manifest()

        print(OUT / "v50a_object_pushing_env_sanity.md")
        print((OUT / "v50a_object_pushing_env_sanity.md").read_text())

    finally:
        env.close()


def make_figures(env: ObjectPushingEnv, states, df, groups, action_summary):
    # Action displacement/contact bar plot.
    acts = list(ACTIONS.keys())
    tmp = action_summary.set_index("action").loc[acts].reset_index()

    x = np.arange(len(tmp))
    w = 0.35

    plt.figure(figsize=(7.2, 4.0))
    plt.bar(x - w/2, tmp["object_disp_mean"], width=w, label="Mean object displacement")
    plt.bar(x + w/2, tmp["contact_rate"], width=w, label="Contact rate")
    plt.xticks(x, tmp["action"])
    plt.ylabel("Value")
    plt.title("Object-pushing action sanity")
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for path in [
        FIG / "fig_v50a_action_contact_displacement.png",
        PFIG / "fig_v50a_action_contact_displacement.png",
    ]:
        plt.savefig(path, dpi=240)
    plt.close()

    # Group diversity histogram.
    plt.figure(figsize=(6.5, 3.8))
    plt.hist(groups["max_future_xy_spread"], bins=30)
    plt.xlabel("Max pairwise future object-xy spread")
    plt.ylabel("Groups")
    plt.title("Counterfactual group diversity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    for path in [
        FIG / "fig_v50a_counterfactual_group_diversity.png",
        PFIG / "fig_v50a_counterfactual_group_diversity.png",
    ]:
        plt.savefig(path, dpi=240)
    plt.close()

    # Visual counterfactual grid for two states.
    fig, axes = plt.subplots(2, len(ACTIONS) + 1, figsize=(10.5, 4.0))
    shown_states = states[:2]

    for row_idx, state in enumerate(shown_states):
        env.reset_to_state(state)
        axes[row_idx, 0].imshow(env.render_rgb())
        axes[row_idx, 0].set_title(f"current\n{state['side']}", fontsize=8)
        axes[row_idx, 0].axis("off")

        for col_idx, action_name in enumerate(ACTIONS.keys(), start=1):
            env.rollout(state, action_name)
            axes[row_idx, col_idx].imshow(env.render_rgb())
            axes[row_idx, col_idx].set_title(action_name, fontsize=8)
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    for path in [
        FIG / "fig_v50a_counterfactual_rgb_grid.png",
        PFIG / "fig_v50a_counterfactual_rgb_grid.png",
    ]:
        plt.savefig(path, dpi=220)
    plt.close()


def write_tex(action_summary, group_summary, repeat_summary):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Action & Contact & Mean disp. & Median disp. & P90 disp. \\")
    lines.append(r"\midrule")
    for _, r in action_summary.iterrows():
        lines.append(
            f"{r['action']} & {r['contact_rate']:.3f} & {r['object_disp_mean']:.3f} & "
            f"{r['object_disp_median']:.3f} & {r['object_disp_p90']:.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{v50A object-pushing action sanity. The environment produces contact-rich counterfactual futures with action-dependent object displacement.}")
    lines.append(r"\label{tab:v50a_object_pushing_action_sanity}")
    lines.append(r"\end{table}")
    (PTAB / "table_v50a_action_summary.tex").write_text("\n".join(lines) + "\n")

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
    lines.append(f"Mean future spread & {gs['mean_future_xy_spread']:.3f} \\\\")
    lines.append(f"Mean contact actions & {gs['mean_contact_action_count']:.3f} \\\\")
    lines.append(f"Repeat object diff max & {rs['object_xy_repeat_diff_max']:.3e} \\\\")
    lines.append(f"Contact repeat match & {rs['contact_match_rate']:.3f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{v50A exact-intervention sanity. The same reset/action pair is deterministic, while different actions produce non-degenerate future object states.}")
    lines.append(r"\label{tab:v50a_exact_intervention_sanity}")
    lines.append(r"\end{table}")
    (PTAB / "table_v50a_exact_intervention_sanity.tex").write_text("\n".join(lines) + "\n")


def write_markdown(action_summary, group_summary, repeat_summary, groups):
    gs = group_summary.iloc[0]
    rs = repeat_summary.iloc[0]

    lines = []
    lines.append("# SPSM v50A object-pushing exact-intervention sanity\n")
    lines.append("This is the first SPSM-v2 prototype. It tests whether a contact-rich object-pushing environment can support exact-intervention counterfactual groups.\n")

    lines.append("## Setup\n")
    lines.append("- environment: PyBullet DIRECT")
    lines.append("- scene: plane, dynamic cube, dynamic spherical pusher")
    lines.append("- actions: `stay`, `right`, `left`, `forward`, `backward`")
    lines.append("- state sampling: cube xy/yaw plus pusher initialized on a cardinal side")
    lines.append("- protocol: reset exact same state, apply each action, compare future object xy\n")

    lines.append("## Repeatability\n")
    lines.append(f"- repeat tests: `{int(rs['repeat_tests'])}`")
    lines.append(f"- max object xy repeat diff: `{rs['object_xy_repeat_diff_max']:.6g}`")
    lines.append(f"- mean object xy repeat diff: `{rs['object_xy_repeat_diff_mean']:.6g}`")
    lines.append(f"- max pusher xy repeat diff: `{rs['pusher_xy_repeat_diff_max']:.6g}`")
    lines.append(f"- contact repeat match rate: `{rs['contact_match_rate']:.3f}`\n")

    lines.append("## Action summary\n")
    lines.append("| action | contact rate | mean object disp | median object disp | p90 object disp | pusher disp |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _, r in action_summary.iterrows():
        lines.append(
            f"| `{r['action']}` | {r['contact_rate']:.3f} | {r['object_disp_mean']:.3f} | "
            f"{r['object_disp_median']:.3f} | {r['object_disp_p90']:.3f} | {r['pusher_disp_mean']:.3f} |"
        )

    lines.append("\n## Counterfactual group diversity\n")
    lines.append(f"- states: `{int(gs['states'])}`")
    lines.append(f"- nondegenerate groups, max pairwise future xy spread > 0.02: `{gs['nondegenerate_frac']:.3f}`")
    lines.append(f"- mean future xy spread: `{gs['mean_future_xy_spread']:.3f}`")
    lines.append(f"- median future xy spread: `{gs['median_future_xy_spread']:.3f}`")
    lines.append(f"- mean max future xy spread: `{gs['max_future_xy_spread_mean']:.3f}`")
    lines.append(f"- mean contact action count: `{gs['mean_contact_action_count']:.3f}`")
    lines.append(f"- mean moving contact action count: `{gs['mean_moving_contact_count']:.3f}`")
    lines.append(f"- mean max object displacement: `{gs['max_object_disp_mean']:.3f}`\n")

    lines.append("## Interpretation\n")
    lines.append("- If repeat diffs are near zero, exact reset/action replay is deterministic.")
    lines.append("- If nondegenerate group fraction is high, the environment can generate meaningful counterfactual hard negatives.")
    lines.append("- If contact rates vary across actions and states, the environment creates contact-mode structure beyond obstacle navigation.")
    lines.append("- v50B should turn this into a dataset generator with RGB futures, state variables, candidate sets, and DINOv2 features.")
    lines.append("- The target failure axes for SPSM-v2 are object pose, contact mode, pusher-object relation, and later hidden dynamics such as friction or mass.")

    (OUT / "v50a_object_pushing_env_sanity.md").write_text("\n".join(lines) + "\n")

    claim = """
# v50A object-pushing sanity claim

v50A starts SPSM-v2 by moving from obstacle navigation to contact-rich object interaction. The goal is not realism yet, but transfer of the exact-intervention protocol to a new physics regime where failures can involve object pose, contact mode, and pusher-object geometry.

If this environment is deterministic under reset and produces non-degenerate counterfactual futures, it is suitable for v50B dataset generation and the full SPSM predictive-usefulness stack.
"""
    (PTXT / "v50a_object_pushing_sanity_claim.md").write_text(claim.strip() + "\n")


def write_manifest():
    manifest = ["# Paper assets v50A object pushing sanity\n", "## Tables\n"]
    for pth in sorted(PTAB.glob("*")):
        manifest.append(f"- `{pth}`")
    manifest.append("\n## Figures\n")
    for pth in sorted(PFIG.glob("*")):
        manifest.append(f"- `{pth}`")
    manifest.append("\n## Text\n")
    for pth in sorted(PTXT.glob("*")):
        manifest.append(f"- `{pth}`")
    (PAPER / "paper_assets_v50a_manifest.md").write_text("\n".join(manifest) + "\n")


if __name__ == "__main__":
    run_sanity(n_states=256, seed=50)
