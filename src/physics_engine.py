import numpy as np
from typing import Optional, Tuple, Any
from state_representation import VisualState, AgentState
import math

class PhysicsEngine:
    """
    Lightweight wrapper around PyBullet for physics simulation,
    robot kinematics, and camera rendering.
    """
    def __init__(self, gui: bool = False, seed: Optional[int] = None):
        self.pb = None
        self.gui = gui
        self._seed = seed if seed is not None else 0
        self.robot_id = None
        self.plane_id = None
        self.goal_marker_id = None

        # Camera parameters
        self.camera_yaw = 50
        self.camera_pitch = -30
        self.camera_dist = 2.0
        self.camera_target = [0, 0, 0.5]

        # Robot joint mappings (placeholder for generic URDF)
        self.num_joints = 0
        self._initialized = False

    def connect(self):
        """Initialize PyBullet physics client."""
        try:
            import pybullet as p
            import pybullet_data
            self.pb = p
            # Sep-5 (audit): store the returned client id (the old code called
            # pb.connect() bare, so disconnect() targeted client 0 always —
            # in multi-client parallel envs it could kill the WRONG client or
            # raise NotConnected).
            self.client_id = self.pb.connect(self.pb.GUI if self.gui else self.pb.DIRECT)
            # (--seed is not a pybullet connect option). NOTE: p.setRandomSeed()
            # does NOT exist in PyBullet (bullet3#4613), so determinism comes
            # from deterministicOverlappingPairs + the agent's own per-episode
            # sampling RNG (sim_environment._rng), NOT from any physics seed.
            # determinism is real & sufficient for our eval/noise=0 usage.
            self.pb.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
            self.pb.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.pb.setGravity(0, 0, -9.8)
            self._initialized = True
        except ImportError:
            print("Warning: PyBullet not installed. Please run: pip install pybullet")
            self._initialized = False
    
    def disconnect(self):
        """Disconnect from physics client."""
        if self._initialized and self.pb:
            self.pb.disconnect(getattr(self, "client_id", 0))
            self._initialized = False

    def setup_scene(self, robot_urdf: str = "r2d2.urdf", base_position: np.ndarray = None,
                    goal_position: np.ndarray = None):
        """
        Setup ground plane and load a robot URDF. 
        Defaults to R2D2 if available (built-in PyBullet example).
        B-104: an optional VISIBLE goal marker (red cylinder at the goal) —
        the old scene had NO goal in the world, so the observation contained
        no task-relevant signal and "goal-reaching" was geometric luck.
        """
        if not self._initialized:
            raise RuntimeError("PhysicsEngine not connected. Call connect() first.")
        
        # Load ground plane
        if self.plane_id is None:
            self.plane_id = self.pb.loadURDF("plane.urdf")

        # Load robot — prefer the project URDF dir (data/urdf), fall back to
        # pybullet_data search path.
        import os as _os
        proj_urdf = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "urdf", robot_urdf)
        urdf_path = proj_urdf if _os.path.exists(proj_urdf) else robot_urdf
        pos = base_position if base_position is not None else [0, 0, 0.3]
        self.robot_id = self.pb.loadURDF(urdf_path, pos, useFixedBase=False)
        self.num_joints = self.pb.getNumJoints(self.robot_id)

        if goal_position is not None and self.goal_marker_id is None:
            goal_shape = self.pb.createVisualShape(
                self.pb.GEOM_CYLINDER, radius=0.15, length=0.3,
                rgbaColor=[1.0, 0.1, 0.1, 1.0])
            self.goal_marker_id = self.pb.createMultiBody(
                baseMass=0, baseVisualShapeIndex=goal_shape,
                basePosition=[goal_position[0], goal_position[1], 0.15])

        # Differential-drive detection: revolute joints whose names contain
        # "wheel" are grouped left/right (r2d2: right=2,3; left=6,7).
        self.wheel_joints, self.right_wheels = [], set()
        for i in range(self.num_joints):
            info = self.pb.getJointInfo(self.robot_id, i)
            name = info[1].decode().lower()
            if "wheel" in name and info[2] == self.pb.JOINT_REVOLUTE:
                self.wheel_joints.append(i)
                if name.startswith("right"):
                    self.right_wheels.add(i)
        self.wheel_vel_scale = 4.0   # m/s at action=1 (kinematic drive)
        # calibrated: wheel-contact drag damps yaw ~9x; 9.0 gives ~1:1 response
        self.wheel_ang_scale = 9.0
        self.wheel_dist = 0.4        # ~wheelbase of r2d2 (m)
        self.wheel_force = 20.0      # motor force limit (N) — plenty for the droid
        if self.wheel_joints:
            print(f"[Physics] differential-drive wheels: "
                  f"{self.wheel_joints} (right={sorted(self.right_wheels)})")

        print(f"Robot loaded with {self.num_joints} joints from {robot_urdf}")

    def step_simulation(self, num_steps: int = 1):
        """Advance the simulation."""
        if self._initialized:
            for _ in range(num_steps):
                self.pb.stepSimulation()

    def get_robot_state(self) -> dict:
        """Get base position, orientation, and joint states."""
        if not self._initialized or self.robot_id is None:
            return {}
        pos, ori = self.pb.getBasePositionAndOrientation(self.robot_id)
        joint_states = []
        for i in range(self.num_joints):
            state = self.pb.getJointState(self.robot_id, i)
            joint_states.append({"joint_index": i, "position": state[0], "velocity": state[1]})
        return {
            "base_position": np.array(pos),
            "base_orientation": np.array(ori),
            "joints": joint_states
        }

    def set_joint_motor_control(self, action: np.ndarray):
        """
        Apply the action to the robot.
        Wheeled robots: differential-drive command action[:2] = (linear,
        angular) velocity, applied KINEMATICALLY via resetBaseVelocity in the
        body frame (research: pybullet cylinder-plane contacts cannot produce
        skid-steer yaw — empirically verified; wheel motors at 0 so they don't
        fight the kinematic drive). Physics camera/visuals stay real.
        Non-wheeled: falls back to per-joint torque.
        """
        if not self._initialized or self.robot_id is None:
            return

        if self.wheel_joints:
            v_cmd = float(np.clip(action[0], -1, 1)) * self.wheel_vel_scale
            w_cmd = float(np.clip(action[1], -1, 1)) * self.wheel_ang_scale
            _, orn = self.pb.getBasePositionAndOrientation(self.robot_id)
            yaw = self.pb.getEulerFromQuaternion(orn)[2]
            vx = v_cmd * np.cos(yaw)
            vy = v_cmd * np.sin(yaw)
            self.pb.resetBaseVelocity(self.robot_id, [vx, vy, 0.0], [0.0, 0.0, w_cmd])
            return

        action_len = min(len(action), self.num_joints)
        for i in range(action_len):
            self.pb.setJointMotorControl2(
                self.robot_id,
                i,
                self.pb.TORQUE_CONTROL,
                force=float(action[i])
            )

    def get_camera_image(self, width: int = 224, height: int = 224) -> np.ndarray:
        """
        Render camera view and return RGB image as numpy array.
        B-123: the target now FOLLOWS the robot — the old fixed [0,0,0.5]
        aim meant randomized spawns drove the robot off-frame and every
        frame looked identical (the vision channel carried no signal).
        """
        if not self._initialized:
            dummy = np.zeros((height, width, 3), dtype=np.float32)
            return dummy

        if self.robot_id is not None:
            rpos, _ = self.pb.getBasePositionAndOrientation(self.robot_id)
            self.camera_target = [float(rpos[0]), float(rpos[1]), 0.3]

        view_matrix = self.pb.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self.camera_target,
            distance=self.camera_dist,
            yaw=self.camera_yaw,
            pitch=self.camera_pitch,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = self.pb.computeProjectionMatrixFOV(
            fov=60, aspect=float(width)/float(height),
            nearVal=0.1, farVal=100.0
        )
        _, _, rgba, _, _ = self.pb.getCameraImage(
            width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix
        )
        img = np.array(rgba, dtype=np.uint8)
        img = img[:, :, :3]  # Remove alpha channel
        # Normalize to float32
        img = img.astype(np.float32) / 255.0
        return img

    def update_goal_marker(self, goal_position: np.ndarray):
        """B-104: move the visible goal marker (called on env reset when the
        goal re-samples)."""
        if self._initialized and self.goal_marker_id is not None:
            self.pb.resetBasePositionAndOrientation(
                self.goal_marker_id,
                [float(goal_position[0]), float(goal_position[1]), 0.15],
                [0, 0, 0, 1])

    def reset_robot(self, base_position: np.ndarray = None, base_orientation: np.ndarray = None):
        """Reset robot to initial state."""
        if not self._initialized or self.robot_id is None:
            return
        pos = base_position if base_position is not None else [0, 0, 0.1]
        ori = base_orientation if base_orientation is not None else [0, 0, 0, 1]
        self.pb.resetBasePositionAndOrientation(self.robot_id, pos, ori)
        self.pb.resetBaseVelocity(self.robot_id, [0, 0, 0], [0, 0, 0])  # Bug 94
        for i in range(self.num_joints):
            self.pb.resetJointState(self.robot_id, i, 0.0, 0.0)  # pos + velocity

    def get_end_effector_position(self) -> np.ndarray:
        """World position used for the goal distance (B-104: the old version
        returned the LAST JOINT's link — for diff-drive robots that is the
        left-back WHEEL, offset ~±0.17m from the base, so yaw rotation swung
        the measured point and spins earned spurious progress reward).
        The base position is the stable, rotation-invariant reference."""
        if not self._initialized or self.robot_id is None:
            return np.zeros(3, dtype=np.float32)
        pos, _ = self.pb.getBasePositionAndOrientation(self.robot_id)
        return np.array(pos, dtype=np.float32)

    def get_visual_state(self) -> VisualState:
        """Convenience: returns VisualState from camera."""
        img = self.get_camera_image()
        return VisualState(image_array=img)
