"""
Phase B: collect VLA training trajectories from the PyBullet physics environment.
- Real 224x224x3 camera frames (PhysicsEngine.get_camera_image)
- Scripted waypoint expert steering toward the goal
- Instructions from LLMPlanner templates (custom-LLM dogfooding)
Saves episodes to an npz dataset (images uint8, actions, non-visual states, instructions).
"""
import sys, os, time, random, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from pathlib import Path
from sim_environment import RoboticsEnv
from llm_agent import LLMPlanner

GOAL_INSTRUCTIONS = [
    "Navigate to the goal and pick up the red block.",
    "Move the robot to the target position.",
    "Follow the waypoint toward the destination.",
    "Approach the red block and prepare to grasp it.",
    "Drive to the goal location and stop.",
]


def expert_action(env, noise: float = 0.0) -> np.ndarray:
    """Steer the robot toward the goal.
    Physics (diff-drive): P-controller on heading error — action[:2] = (v, ω)
    normalized to the env scales (4 m/s, 9 rad/s). The old steer-VECTOR mapping
    slowed v = cos(bearing) while turning, so episodes never reached the goal.
    Dummy: B-141 diffdrive body-frame P-controller (action[0]=v, action[1]=w)."""
    if env.use_physics and env.physics is not None:
        ee = np.array(env.physics.get_end_effector_position(), dtype=np.float32)
        delta = env._goal[:2] - ee[:2]
        _, orn = env.physics.pb.getBasePositionAndOrientation(env.physics.robot_id)
        heading = env.physics.pb.getEulerFromQuaternion(orn)[2]
        dist = np.linalg.norm(delta)
        bearing = np.arctan2(delta[1], delta[0]) - heading
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))  # wrap [-pi, pi]
        # kinematic drive: brake proportionally to distance but keep a floor
        # ABOVE zero — B-110: `4*max(0, dist-0.2)` asymptotically STOPPED at
        # the 0.2m threshold, so the <0.2 goal bonus never fired and episodes
        # truncated half-stationary. The floor (0.36 m/s) guarantees arrival.
        v = max(1.5 * dist, 0.36)
        w = float(np.clip(6.0 * bearing, -9.0, 9.0))
        action = np.array([v / 4.0, w / 9.0, 0.0, 0.0], dtype=np.float32)
    else:
        # Sep-5 (audit): the OLD dummy branch emitted a WORLD-frame steer
        # [dx,dy,0,0] — since B-141 the dummy step interprets action[0]=forward
        # v and action[1]=turn w, so the robot SPIRALED and never reached the
        # goal (empirically 0/3 episodes). Now mirror the physics branch:
        # body-frame bearing from the real dummy yaw (same controller shape).
        pos = np.array(env._position, dtype=np.float32)
        delta = env._goal[:2] - pos
        heading = env._current_yaw()
        dist = np.linalg.norm(delta)
        bearing = np.arctan2(delta[1], delta[0]) - heading
        bearing = np.arctan2(np.sin(bearing), np.cos(bearing))
        v = max(1.5 * dist, 0.36)
        w = float(np.clip(6.0 * bearing, -9.0, 9.0))
        action = np.array([v / 4.0, w / 9.0, 0.0, 0.0], dtype=np.float32)
    if noise > 0:
        action += np.random.randn(4).astype(np.float32) * noise
    return np.clip(action, -1.0, 1.0)


def make_instruction(llm: LLMPlanner, goal_text: str) -> str:
    """Vary instructions: LLMPlanner plan when available, else goal templates."""
    if llm is not None and random.random() < 0.5:
        from state_representation import TextState, ExecutionState
        plan = llm.plan_next_action(TextState(instruction=goal_text),
                                    ExecutionState("", "Collecting trajectory", np.zeros(512)))
        return f"{goal_text} {plan[:120]}"
    return goal_text


def collect_episode(env, llm, max_steps=200, chunk_explore=0.05):
    obs, _ = env.reset()
    frames = []
    done = False
    step = 0
    while not done and step < max_steps:
        img = obs.visual.get_tensor()                       # [224, 224, 3] float32
        state_nv = obs.get_combined_state()[150528:]        # non-visual 1872-dim
        action = expert_action(env, noise=chunk_explore)
        frames.append({
            "image": (np.clip(img, 0, 1) * 255).astype(np.uint8),
            "state": state_nv.astype(np.float32),
            "action": action,
            "instruction": make_instruction(llm, random.choice(GOAL_INSTRUCTIONS)),
        })
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    return frames


def main():
    ap = argparse.ArgumentParser(description="Collect VLA trajectories (Phase B)")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--use_physics", action="store_true", default=False,
                    help="Run the physics environment (default: dummy kinematics)")
    ap.add_argument("--out", type=str, default="out/vla_data/trajectories.npz")
    ap.add_argument("--explore", type=float, default=0.05,
                    help="expert action noise (data augmentation)")
    args = ap.parse_args()

    env = RoboticsEnv(use_physics=args.use_physics)
    llm = LLMPlanner(use_ambatron=False, use_moe=False)  # template plans (no ckpt yet)

    episodes = []
    t0 = time.time()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path = out_path.with_suffix(".partial.npz")
    if saved_path.exists():  # resume a crashed collection
        try:
            episodes = list(np.load(saved_path, allow_pickle=True)["episodes"])
            print(f"[collect] resumed {len(episodes)} episodes from {saved_path}")
        except Exception:
            episodes = []
    start = len(episodes)
    for ep in range(start, args.episodes):
        frames = collect_episode(env, llm, args.max_steps, args.explore)
        if len(frames) < 10:
            continue
        episodes.append({
            "images": np.stack([f["image"] for f in frames]),
            "states": np.stack([f["state"] for f in frames]),
            "actions": np.stack([f["action"] for f in frames]),
            "instructions": [f["instruction"] for f in frames],
        })
        print(f"[collect] ep {ep+1}/{args.episodes}: {len(frames)} steps | "
              f"physics={env.use_physics}", flush=True)
        np.savez_compressed(saved_path, episodes=np.array(episodes, dtype=object))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, episodes=np.array(episodes, dtype=object))
    saved_path.unlink(missing_ok=True)
    n_frames = sum(len(e["images"]) for e in episodes)
    print(f"[collect] saved {len(episodes)} episodes, {n_frames} frames -> {args.out} "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
