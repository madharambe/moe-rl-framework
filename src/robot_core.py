import numpy as np
from typing import Optional
import pathlib
from sim_environment import RoboticsEnv
from physics_engine import PhysicsEngine
from policy_network import MultimodalPolicy
from llm_agent import LLMPlanner
from state_representation import AgentState, TextState, CodeState, ExecutionState
import torch

class RobotAgent:
    """
    The main robot agent that orchestrates perception, planning (LLM),
    and control (Neural Policy or VLA).
    """
    def __init__(self, 
                 env: Optional[RoboticsEnv] = None,
                 policy: Optional[MultimodalPolicy] = None,
                 llm_planner: Optional[LLMPlanner] = None,
                 use_llm: bool = True,
                 vla_policy: Optional[object] = None,
                 use_vla: bool = False,
                 instruction: str = "Navigate to the goal and pick up the red block.",
                 learning_rate: float = 3e-4,
                 inference_noise: float = 0.0):
        
        self.env = env or RoboticsEnv(use_physics=False)
        # 11th-audit H1: this is the deployment footgun. If VLA mode is on, the
        # env MUST be the physics/camera env — dummy mode fabricates pure random
        # noise as the visual, which a real-camera-trained VLA would silently
        # consume as if it were a real scene. Refuse loudly at construct time
        # instead of deploying on garbage. (B-123/B-160 lesson: noise-vision.)
        if use_vla and self.env.use_physics is False:
            raise ValueError(
                "RobotAgent(use_vla=True) requires env.use_physics=True — VLA "
                "policies are trained on real camera images, NOT dummy-mode "
                "random-noise vision. Pass an env built with use_physics=True "
                "(or omit env and let physics mode default), otherwise the "
                "deployed policy runs on garbage pixels.")
        # Determine action_dim from environment or default to 4
        self.action_dim = 4 
        
        # Core Components
        self.policy = policy or MultimodalPolicy(action_dim=self.action_dim)
        self.vla_policy = vla_policy
        self.use_vla = use_vla
        self.instruction = instruction
        # B-62: ESPD bias correction — the deterministic VLA policy has bias
        # (goal-reaching was noise-assisted in physics RL); a small inference
        # noise recovers it at deploy time. Default 0.0 = unchanged behavior.
        self.inference_noise = inference_noise
        self.llm_planner = llm_planner or LLMPlanner()
        self.use_llm = use_llm

        # Training Components
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Internal State
        self.total_reward = 0
        self.episode_count = 0

    def run_loop(self, max_episodes: int = 100, max_steps: int = 500):
        """
        Main training loop. Uses the policy for low-level control and 
        optionally the LLM for high-level code generation/replanning.
        """
        for episode in range(max_episodes):
            observation, _ = self.env.reset()
            done = False
            step = 0
            self.total_reward = 0
            
            while not done and step < max_steps:
                # 1. Use LLM to generate/refine code if enabled
                if self.use_llm and step % 50 == 0: # Replan every 50 steps
                    self._llm_replan(observation)
                
                # 2. Select action using the neural policy
                action = self._select_action(observation)
                
                # 3. Step the environment
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                self.total_reward += reward
                
                # 4. (Optional) Update policy using self-collected data
                # In a full implementation, we would collect experience and run PPO here.
                # For now, we just step through.
                
                observation = next_observation
                done = terminated or truncated
                step += 1
                
            self.episode_count += 1
            print(f"Episode {episode + 1} completed. Total reward: {self.total_reward:.2f}")

    def _select_action(self, observation: AgentState) -> np.ndarray:
        """
        Uses the multimodal policy or the VLA policy to select an action.
        """
        with torch.no_grad():
            if self.use_vla and self.vla_policy is not None:
                # 11th-audit H2: pass the 5D body-frame task vector so get_action
                # delegates to deterministic_action(task=...), the SAME
                # conditioned decode every trained/eval consumer uses. Before,
                # get_action with task=None ran a stochastic 10-step Euler
                # sample and ignored the proprioceptive task conditioning the
                # policy was trained/validated with — two silent divergences.
                task_vec = getattr(self.env, "task_vector", None)
                task = torch.from_numpy(task_vec()).float() if task_vec else None
                img = torch.from_numpy(observation.visual.get_tensor()).float()
                state = torch.from_numpy(observation.get_combined_state()).float()
                action = self.vla_policy.get_action(img, self.instruction, state,
                                                    task=task)
                action = action.cpu().numpy()
                if self.inference_noise > 0.0:
                    action = np.clip(
                        action + np.random.randn(self.action_dim).astype(np.float32)
                        * self.inference_noise, -1, 1)
                return action
            action, _ = self.policy.get_action(observation, deterministic=False)
        return action

    def _llm_replan(self, observation: AgentState):
        """
        Interact with the LLM to generate or refine the robot's strategy 
        based on the current observation. The plan text becomes the VLA instruction.
        """
        # Create a TextState from current context (placeholder)
        current_goal = TextState(instruction=self.instruction)
        
        # Get execution state from observation
        exec_state = observation.exec_state if observation.exec_state else ExecutionState(
            error_log="",
            execution_trace="Initiating new plan.",
            updated_graph_features=np.zeros(512)
        )
        
        # Query LLM for next plan
        plan = self.llm_planner.plan_next_action(current_goal, exec_state)
        print(f"LLM Plan: {plan}")
        if self.use_vla:
            # rolling window: keep only the latest plan (unbounded growth pushed
            # stale text past the tokenizer's 128-token truncation)
            self.instruction = plan[:120]
        
        # Optionally, generate new code based on the plan
        if "code" in plan.lower() or "function" in plan.lower():
            new_code_state = self.llm_planner.generate_code(current_goal, exec_state.execution_trace)
            print(f"Generated new code snippet. Length: {len(new_code_state.current_code_artifact)} chars")

    def save_policy(self, path: str = None):
        """Saves the current policy to disk. Bug 17 fix: persist whichever
        policy actually makes decisions (VLA vs multimodal fallback)."""
        # ROUND-170: default is a ROOT-anchored path (the old relative
        # 'policy.pt' wrote silently to whatever CWD the robot was launched
        # from — hard to find, easy to lose).
        if path is None:
            path = str(pathlib.Path(__file__).resolve().parent.parent / "out" / "policy.pt")
        target = self.vla_policy if self.use_vla and self.vla_policy is not None else self.policy
        from storage import safe_save  # B-151: atomic SSD-drop-safe save
        safe_save({"policy": target.state_dict(), "use_vla": self.use_vla}, path)
        print(f"Policy saved to {path}")

    def load_policy(self, path: str = None):
        """Loads a policy from disk. ROUND-169: MoEConfig-allowlisted
        weights_only (VLA ckpts embed the config object) + strict=False.
        ROUND-170: default path root-anchored like save_policy."""
        if path is None:
            path = str(pathlib.Path(__file__).resolve().parent.parent / "out" / "policy.pt")
        import torch.serialization as _ser
        from moe_model import MoEConfig as _MoeCfg
        _ser.add_safe_globals([_MoeCfg])
        ck = torch.load(path, map_location="cpu", weights_only=True)
        target = self.vla_policy if ck.get("use_vla") and self.vla_policy is not None else self.policy
        sd = ck.get("policy", ck)
        if hasattr(target, "load_state_dict"):
            # ROUND-170 (F2): plain strict=False — the old task-key FILTER
            # stripped PRESENT trained task weights at serve time (random task
            # injection). strict=False tolerates missing keys from older
            # vintages while keeping present ones.
            _r = target.load_state_dict(sd, strict=False)
            # ROUND-171 (N17): a saved VLA ckpt loaded into a VLA-less agent
            # (or vice versa) leaves most of sd's keys UNEXPECTED -> silently
            # random policy. unexpected_keys ⊆ sd's keys when archs differ.
            _frac = 1 - (len(_r.unexpected_keys) / max(len(sd), 1))
            if _frac < 0.5:
                raise RuntimeError(
                    f"[robot_core] ckpt {path} matches only {_frac:.0%} of keys "
                    f"for the {'VLA' if target is self.vla_policy else 'MLP'} "
                    f"policy — refusing a silently-random policy")
        print(f"Policy loaded to {'VLA' if target is self.vla_policy else 'MLP'} from {path}")


if __name__ == "__main__":
    # Example usage
    print("--- Initializing Robot Agent ---")
    agent = RobotAgent(use_llm=True)
    
    # Run for a few episodes to test
    agent.run_loop(max_episodes=5, max_steps=100)
