import hashlib
import numpy as np
from gymnasium import Env
from typing import Tuple, Optional, Any
from state_representation import VisualState, GraphState, CodeState, ExecutionState, AgentState, TextState
from physics_engine import PhysicsEngine

class RoboticsEnv(Env):
    metadata = {'render.modes': ['none', 'human']}

    def __init__(self, render_mode: Optional[str] = None, use_physics: bool = False,
                 goal_range=(-1.0, 1.0), seed: Optional[int] = 0):
        super().__init__()
        self._seed = seed if seed is not None else 0
        self.current_step = 0
        self.max_steps = 500
        self.render_mode = render_mode
        self.use_physics = use_physics

        self._rng = np.random.default_rng(self._seed)
        self._position = np.zeros(2, dtype=np.float32)
        self._dummy_yaw = 0.0  # B-141: dummy-mode heading for body-frame semantics
        self._goal = np.array([1.0, 1.0], dtype=np.float32)
        self._prev_distance = np.linalg.norm(self._position - self._goal)
        self._prev_position = self._position.copy()
        self.min_spawn = 1.2
        # B-172: wider goal range for arena extension.
        self.goal_range = goal_range

        self.physics = None
        if self.use_physics:
            # 19th-audit H1b: a broken pybullet (numpy ABI race etc.) used to
            # silently degrade to dummy mode with random-noise "vision" and
            # "succeed" — eval/collect results were written as if physical. An
            # explicit use_physics=True must FAIL LOUD so the operator sees the
            # broken engine instead of trusting dummy-mode numbers.
            self.physics = PhysicsEngine(gui=(render_mode == 'human'), seed=self._seed)
            try:
                self.physics.connect()
                self.physics.setup_scene(robot_urdf="diff_drive.urdf",
                                         goal_position=np.append(self._goal, 0.0))
            except Exception as e:
                import traceback
                raise RuntimeError(
                    f"use_physics=True but the physics engine failed to start: {e} — "
                    f"dummy-mode results are NOT physics. Fix pybullet (see H1: "
                    f"numpy C-API must match the pybullet build) before collecting "
                    f"or evaluating.\n{traceback.format_exc()[-800:]}") from e

    def _proprio_position(self) -> np.ndarray:
        """Task-relevant position: physics end-effector (dummy: internal state)."""
        if self.use_physics and self.physics is not None:
            return np.array(self.physics.get_end_effector_position()[:2], dtype=np.float32)
        return self._position

    def _current_yaw(self) -> float:
        """B-141: heading in radians. Physics mode reads the robot base Euler
        yaw; dummy mode tracks its own diff-drive yaw so body-frame task
        vectors stay meaningful and consistent with physics action semantics."""
        if self.use_physics and self.physics is not None and self.physics.robot_id is not None:
            _, orn_q = self.physics.pb.getBasePositionAndOrientation(self.physics.robot_id)
            # B-152: the hand-rolled quaternion-yaw here used the WRONG
            # denominator (x^2 instead of z^2), i.e. atan2(sin th, 1) for a
            # planar robot -- empirically 35deg reported at a true 135deg.
            # Every task_vector/heading ever recorded carried corrupted yaw
            # at large turn angles (the expert was immune: its own
            # yaw_from_quat in gen_demos is correct). Use pybullet's own
            # Euler extraction -- ground truth, nothing to get wrong.
            return float(self.physics.pb.getEulerFromQuaternion(orn_q)[2])
        return float(self._dummy_yaw)

    def task_vector(self) -> np.ndarray:
        """B-132: GOAL-IN-BODY-FRAME proprioception
        [dx_b/3, dy_b/3, dist/3, sin(yaw), cos(yaw)].
        B-124's world-frame version forced the model to learn atan2/sqrt
        through LoRA (measured R2~0.29 underfit). Body-frame makes the turn
        direction read straight off the two delta signs; dist is provided
        explicitly so braking stays trivially linear."""
        pos = self._proprio_position()
        dx_w, dy_w = self._goal[0] - pos[0], self._goal[1] - pos[1]
        yaw = self._current_yaw()
        cy, sy = np.cos(yaw), np.sin(yaw)
        dx_b = cy * dx_w + sy * dy_w     # forward-axis component
        dy_b = -sy * dx_w + cy * dy_w    # left-axis component
        dist = float(np.hypot(dx_w, dy_w))
        return np.array([dx_b / 3.0, dy_b / 3.0, dist / 3.0,
                         sy, cy], dtype=np.float32)

    def _state_rng(self) -> np.random.Generator:
        """Deterministic RNG seeded from the hidden state (Bug 42 fix):
        observations must be a deterministic function of the MDP state so the
        RL agent can actually learn; before, every component was fresh noise."""
        key = np.round(np.concatenate([self._proprio_position(),
                                       self._goal,
                                       self._proprio_position() - self._prev_position]), 6)
        seed = int(hashlib.md5(key.tobytes()).hexdigest(), 16) % (2 ** 32)
        return np.random.default_rng(seed)

    def _get_observation(self) -> AgentState:
        rng = self._state_rng()
        if self.use_physics and self.physics is not None:
            visual = self.physics.get_visual_state()
        else:
            visual = VisualState(image_array=rng.random((224, 224, 3), dtype=np.float32))

        dummy_code_state = CodeState(
            initial_prompt="Navigate and pick up the red block.",
            current_code_artifact="def move_to(coords): print(f'Moving to {coords}')",
            code_embedding=rng.random(768, dtype=np.float32)
        )

        exec_state = ExecutionState(
            error_log="",
            execution_trace="Success",
            updated_graph_features=rng.random(512, dtype=np.float32)
        )

        dummy_text = TextState(instruction="Navigate and pick up the red block.",
                               embedding=rng.random(512, dtype=np.float32))
        # B-104: the goal IS the task — inject it into the observation.
        # B-123 (the deep fix): position existed ONLY inside md5-seeded random
        # embeddings — cryptographically unlearnable, so BC on these obs
        # collapsed to the dataset-mean action (measured constant output).
        # The coords now carry the ACTUAL task variables:
        #   [0] = goal (x, y, 0)          <- target
        #   [1] = robot (x, y, 0)         <- proprioception
        #   [2] = (sin yaw, cos yaw, 0)   <- heading for bearing control
        pos2 = np.asarray(self._proprio_position(), dtype=np.float32)
        yaw = self._current_yaw()
        coords = np.zeros((10, 3), dtype=np.float32)
        coords[0] = np.append(np.asarray(self._goal, dtype=np.float32), 0.0)
        coords[1] = np.array([pos2[0], pos2[1], 0.0], dtype=np.float32)
        coords[2] = np.array([np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
        dummy_graph = GraphState(coordinates=coords)
        return AgentState(visual=visual, graph=dummy_graph, text=dummy_text, code=dummy_code_state, exec_state=exec_state)

    def _compute_reward(self, action: np.ndarray) -> float:
        if self.use_physics and self.physics is not None:
            ee_pos = self.physics.get_end_effector_position()
            distance = np.linalg.norm(np.array(ee_pos[:2], dtype=np.float32) - self._goal)
        else:
            # B-148: measurement only -- the state advance lives in step()
            distance = np.linalg.norm(self._position - self._goal)

        self._last_distance = distance
        progress = self._prev_distance - distance
        action_penalty = 0.01 * np.linalg.norm(action)
        goal_bonus = 50.0 if distance < 0.2 else 0.0
        # B-60: goal bonus 10 -> 50 — the old -distance + 2*progress shaping
        # rewarded APPROACH but not TERMINAL reaching (the +8.3 physics-RL
        # episodes approached without crossing the 0.2 threshold; the 0/6 eval
        # confirmed). A dominant goal bonus makes goal-reaching the optimum.
        reward = -distance + 2.0 * progress - action_penalty + goal_bonus
        self._prev_distance = distance
        return float(reward)

    def step(self, action: np.ndarray) -> Tuple[AgentState, float, bool, bool, dict]:
        self.current_step += 1
        
        if self.use_physics and self.physics is not None:
            self.physics.set_joint_motor_control(action)
            self.physics.step_simulation(num_steps=10)
        else:
            # B-141: emulate a diff-drive so the dummy MDP matches physics
            # forward(yaw)/turn action semantics (the old code applied
            # action[:2] as WORLD-frame translation, an inconsistent MDP for a
            # physics-trained BC prior). action[0]=forward v, action[1]=turn.
            v = float(np.clip(action[0], -1.0, 1.0)) * (10.0 / 240.0) * 4.0
            # ROUND-172 (N1): the physics wheel-base has REAL yaw drag —
            # measured in pybullet: 60 actions of w_cmd=9 over 10x240Hz yields
            # 2.19 rad (0.875 rad/s = 0.097x the commanded 9.0). The old dummy
            # integrated the UNDAMPED 9.0 -> ~10x faster turning than any
            # physics-trained BC prior (turn semantics diverge 10:1). Use the
            # drag-corrected gain so dummy == physics MDP.
            self._dummy_yaw += float(np.clip(action[1], -1.0, 1.0)) * (10.0 / 240.0) * 0.875
            dyaw = float(self._dummy_yaw)
            self._position = np.clip(
                self._position + np.array([v * np.cos(dyaw), v * np.sin(dyaw)], dtype=np.float32),
                -2.0, 2.0)

        observation = self._get_observation()
        reward = self._compute_reward(action)
        self._prev_position = self._proprio_position()
        terminated = bool(self._last_distance < 0.2)     # goal reached -> real termination
        truncated = bool(self.current_step >= self.max_steps)  # time limit -> bootstrappable
        info = {"action_taken": action.tolist(), "step": self.current_step}
        
        return observation, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[AgentState, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._seed = seed
        self.current_step = 0
        spawn = self._rng.uniform(-1.0, 1.0, 2).astype(np.float32)
        self._position = spawn
        self._dummy_yaw = 0.0
        if self.use_physics and self.physics is not None:
            # B-104: randomize the spawn in PHYSICS mode too (the old
            # reset_robot() hardcoded the origin — the VLA overfit to
            # origin-start; also resets the goal-marker position).
            self.physics.reset_robot(base_position=[spawn[0], spawn[1], 0.1])
            self._prev_position = self._proprio_position()
        else:
            self._prev_position = self._position.copy()
        lo, hi = self.goal_range
        self._goal = self._rng.uniform(lo, hi, 2).astype(np.float32)
        # B-65 curriculum: min spawn distance (default 1.2; the RL trainer can
        # lower it early on so goal episodes actually happen -> RWR distillation)
        # B-167: clamp min_spawn to the geometrically achievable maximum --
        # goals sample from [-1,1]^2 while the robot spawns in [-1,1]^2 too,
        # so distances cap near sqrt(8)=2.83 but P(dist >= d) collapses fast;
        # an unachievable value (e.g. 1.7 with a centered robot) spins this
        # resample loop forever at 100% CPU (two eval runs lost to it).
        max_iter = 10_000
        while np.linalg.norm(self._goal - self._proprio_position()) < self.min_spawn:
            self._goal = self._rng.uniform(lo, hi, 2).astype(np.float32)
            max_iter -= 1
            if max_iter <= 0:
                self.min_spawn = max(0.5, self.min_spawn * 0.9)
                max_iter = 10_000
        if self.use_physics and self.physics is not None:
            # B-106: the marker must be updated AFTER the min_spawn loop —
            # the old order left it on a stale position whenever the loop
            # re-sampled (GraphState said goal A, the camera showed marker B).
            self.physics.update_goal_marker(np.append(self._goal, 0.0))
        self._prev_distance = np.linalg.norm(self._proprio_position() - self._goal)
        
        initial_observation = self._get_observation()
        return initial_observation, {}
