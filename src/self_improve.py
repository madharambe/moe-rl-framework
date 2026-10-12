import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import random
import re
import math

from sim_environment import RoboticsEnv
from llm_agent import LLMPlanner, NanoGPTModel, MoEModel
from state_representation import TextState, CodeState, ExecutionState

LLM_TASKS = [
    "Write a Python function to move a robot arm to pick up a block at position (1.0, 2.0, 0.5).",
    "Implement a PID controller for a differential drive robot to follow a waypoint.",
    "Write a function that computes inverse kinematics for a 6-DOF robotic arm.",
    "Implement epsilon-greedy exploration for a Q-learning agent.",
    "Write a reward function for a robot reaching a target position.",
    "Implement a simple path planner using potential fields.",
    "Write a function to transform a depth image into point cloud coordinates.",
    "Implement a reset handler that returns the robot to a safe home position.",
    "Write a joint-space trajectory interpolator (cubic spline).",
    "Implement a collision avoidance check using bounding box overlap.",
    "Write a ROS2 subscriber for joint states and publish velocity commands.",
    "Implement a Kalman filter for robot pose estimation from odometry.",
    "Write a function to compute manipulability ellipsoid for a 7-DOF arm.",
    "Implement a force-torque sensor interface with gravity compensation.",
    "Write a B-spline trajectory generator with velocity/acceleration limits.",
    "Implement a configuration-space A* planner for a mobile robot.",
    "Write a function that detects grasping points from point cloud data.",
    "Implement impedance control for compliant robot manipulation.",
]

TASK_DIFFICULTY = {
    "Write a Python function to move a robot arm to pick up a block at position (1.0, 2.0, 0.5).": 1,
    "Implement a PID controller for a differential drive robot to follow a waypoint.": 2,
    "Write a function that computes inverse kinematics for a 6-DOF robotic arm.": 4,
    "Implement epsilon-greedy exploration for a Q-learning agent.": 1,
    "Write a reward function for a robot reaching a target position.": 2,
    "Implement a simple path planner using potential fields.": 3,
    "Write a function to transform a depth image into point cloud coordinates.": 3,
    "Implement a reset handler that returns the robot to a safe home position.": 1,
    "Write a joint-space trajectory interpolator (cubic spline).": 2,
    "Implement a collision avoidance check using bounding box overlap.": 2,
    "Write a ROS2 subscriber for joint states and publish velocity commands.": 3,
    "Implement a Kalman filter for robot pose estimation from odometry.": 5,
    "Write a function to compute manipulability ellipsoid for a 7-DOF arm.": 6,
    "Implement a force-torque sensor interface with gravity compensation.": 4,
    "Write a B-spline trajectory generator with velocity/acceleration limits.": 5,
    "Implement a configuration-space A* planner for a mobile robot.": 5,
    "Write a function that detects grasping points from point cloud data.": 6,
    "Implement impedance control for compliant robot manipulation.": 7,
}


@dataclass
class LLMSelfImprovementRecord:
    iteration: int
    task: str
    task_difficulty: int
    generated_code: str
    reflection: str
    refined_code: str
    eval_score: float
    refined_score: float
    delta_score: float
    was_trained: bool
    best_score_so_far: float


class ReflectionMemory:
    def __init__(self, max_size: int = 200, similarity_threshold: float = 0.3):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.reflections: List[Dict] = []

    def add(self, task: str, code: str, reflection: str, score: float, difficulty: int):
        embedding = self._embed(f"{task} {code[:200]}")
        self.reflections.append({
            "task": task,
            "code_snippet": code[:300],
            "reflection": reflection,
            "score": score,
            "difficulty": difficulty,
            "embedding": embedding,
        })
        if len(self.reflections) > self.max_size:
            self.reflections.sort(key=lambda x: x["score"])
            self.reflections = self.reflections[-self.max_size:]

    def retrieve(self, task: str, code: str, top_k: int = 3) -> List[str]:
        if not self.reflections:
            return []
        query_emb = self._embed(f"{task} {code[:200]}")
        scored = []
        for r in self.reflections:
            sim = float(np.dot(query_emb, r["embedding"]))
            gate = 1.0 if r["score"] > 3.0 else 0.3
            scored.append((sim * gate, r["reflection"]))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_k] if _ > self.similarity_threshold]

    def _embed(self, text: str) -> np.ndarray:
        words = re.findall(r'\w+', text.lower())
        counts = {}
        for w in words:
            counts[w] = counts.get(w, 0) + 1
        emb = np.zeros(64, dtype=np.float32)
        for i, w in enumerate(set(words)):
            if i >= 64:
                break
            emb[i] = np.tanh(counts[w] / (len(words) + 1))
        return emb / (np.linalg.norm(emb) + 1e-8)


class LLMSelfImprovementLoop:
    def __init__(self,
                 llm: Optional[LLMPlanner] = None,
                 env: Optional[RoboticsEnv] = None,
                 tasks: Optional[List[str]] = None,
                 pop_size: int = 50,
                 retain_top_k: int = 10,
                 tree_search_width: int = 3,
                 device: Optional[str] = None):

        self.llm = llm or LLMPlanner()
        self.env = env or RoboticsEnv(use_physics=False)
        self.tasks = tasks or LLM_TASKS
        self.pop_size = pop_size
        self.retain_top_k = retain_top_k
        self.tree_search_width = tree_search_width

        self.best_code_buffer: List[Tuple[str, str, float, float, str]] = []
        self.history: List[LLMSelfImprovementRecord] = []
        self.best_score = -float("inf")
        self.generation = 0
        self.reflection_memory = ReflectionMemory()

        self._current_difficulty = 1.0
        self._difficulty_window = []

        self.device = torch.device(device or "cuda" if torch.cuda.is_available() else "cpu")

    def _select_task(self) -> Tuple[str, int]:
        ramp = min(1.0, self.generation / 30.0)
        max_diff = max(1, int(1 + ramp * 6))
        pool = [(t, TASK_DIFFICULTY.get(t, 3)) for t in self.tasks if TASK_DIFFICULTY.get(t, 3) <= max_diff]
        if not pool:
            pool = [(t, TASK_DIFFICULTY.get(t, 3)) for t in self.tasks]
        weights = []
        for t, d in pool:
            w = 1.0 / (1.0 + d if d > 0 else 1.0)
            freq = sum(1 for r in self.history if r.task == t[:60]) if self.history else 0
            w *= max(0.1, 1.0 - freq / 5.0)
            weights.append(max(0.01, w))
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        idx = np.random.choice(len(pool), p=weights)
        return pool[idx]

    def evaluate_code(self, code: str, task: str) -> float:
        try:
            has_def = "def " in code
            has_class = "class " in code
            compiles = True
            try:
                compile(code, "<string>", "exec")
            except SyntaxError:
                compiles = False

            score = 0.0
            if has_def:
                score += 1.0
            if has_class:
                score += 0.5
            if compiles:
                score += 2.0
            if "robot" in code.lower() or "arm" in code.lower():
                score += 1.0
            if len(code) > 50:
                score += 0.5
            if "return" in code:
                score += 1.0
            if "import" in code:
                score += 0.5

            keyword_bonus = 0.0
            for kw in ["torch", "numpy", "np", "def ", "class ", "if "]:
                if kw in code:
                    keyword_bonus += 0.3
            score += keyword_bonus

            docstring_bonus = 0.5 if '"""' in code or "'''" in code else 0.0
            score += docstring_bonus

            type_hint_bonus = 0.5 if re.search(r'def \w+\(.*:.*\)\s*->', code) else 0.0
            score += type_hint_bonus

            task_words = set(re.findall(r'\w+', task.lower()))
            code_words = set(re.findall(r'\w+', code.lower()))
            overlap = len(task_words & code_words) / max(len(task_words), 1)
            score += overlap * 1.0

            return min(max(score, 0.0), 12.0)
        except Exception:
            return 0.0

    def _generate_reflection(self, task: str, code: str, score: float) -> str:
        reflections = []
        if score < 3.0:
            reflections.append("The code needs significant improvement.")
        if "def" not in code:
            reflections.append("Add a function definition.")
        if not re.search(r'def \w+\(', code):
            reflections.append("Define proper function signatures with parameters.")
        if score > 6.0:
            reflections.append("The code is well-structured.")
        if len(code) < 100:
            reflections.append("Expand the implementation with more complete logic.")

        relevant_memories = self.reflection_memory.retrieve(task, code)
        if relevant_memories:
            reflections.append(f"Previous lessons: {'; '.join(relevant_memories[:2])}")

        if not reflections:
            reflections.append("Review algorithm correctness and edge cases.")

        return " | ".join(reflections[:4])

    def _refine_code(self, code: str, task: str, reflection: str) -> str:
        lines = code.strip().split('\n')
        if "TODO" in code or "pass" in code:
            lines = [l for l in lines if "TODO" not in l and l.strip() != "pass"]
            if not lines:
                lines = [f"def solve_{task.split()[0].lower() if task.split() else 'task'}():", "    pass"]
        if reflection and "Add a function definition" in reflection and "def " not in code:
            function_name = re.findall(r'(\w+)', task)
            fn = function_name[1] if len(function_name) > 1 else "solve"
            lines = [
                f"def {fn}_handler():",
                "    result = None",
                "    return result",
                "    ",
                *lines,
            ]
        refined = '\n'.join(lines)
        if refined != code:
            refined += f"\n# Refined based on: {reflection[:80]}"
        return refined

    def _reflexion_cycle(self, task: str, max_attempts: int = 3) -> Tuple[str, float, str]:
        best_code = ""
        best_score = -1.0
        best_reflection = ""
        memories = self.reflection_memory.retrieve(task, task)

        for attempt in range(max_attempts):
            goal = TextState(instruction=task)
            exec_state = ExecutionState(
                error_log="",
                execution_trace=f"Reflexion attempt {attempt + 1}/{max_attempts}",
                updated_graph_features=np.zeros(512),
            )
            code_state = self.llm.generate_code(goal, exec_state.execution_trace)
            code = code_state.current_code_artifact

            if memories and attempt > 0:
                mem_note = "\n# Past lessons: " + "; ".join(memories[:2])
                if mem_note not in code:
                    code += mem_note

            score = self.evaluate_code(code, task)
            reflection = self._generate_reflection(task, code, score)

            self.reflection_memory.add(task, code, reflection, score, TASK_DIFFICULTY.get(task, 3))

            if score > best_score:
                best_code, best_score, best_reflection = code, score, reflection

            if score >= 10.0:
                break

        return best_code, best_score, best_reflection

    def _tree_search_refine(self, code: str, task: str) -> Tuple[str, float]:
        candidates = [(code, self.evaluate_code(code, task))]
        for _ in range(self.tree_search_width):
            reflection = self._generate_reflection(task, code, candidates[0][1])
            refined = self._refine_code(code, task, reflection)
            refined_score = self.evaluate_code(refined, task)
            candidates.append((refined, refined_score))

            perturbed = code + f"\n# Alternative approach for: {task[:60]}"
            candidates.append((perturbed, self.evaluate_code(perturbed, task)))

        candidates.sort(key=lambda x: -x[1])
        return candidates[0]

    def _compute_delta_reward(self, prev_score: float, new_score: float) -> float:
        delta = new_score - prev_score
        if prev_score <= 0:
            return new_score * 0.5
        if delta > 0:
            return 2.0 * delta + 0.5 * new_score / max(prev_score, 0.1)
        else:
            return -1.0 * abs(delta) - 0.2

    def _consolidate_best(self):
        if len(self.best_code_buffer) < 3:
            return
        top = self.best_code_buffer[:self.retain_top_k]
        tasks_seen = set()
        deduped = []
        for t, c, s, d, ref in top:
            key = c[:100]
            if key not in tasks_seen:
                tasks_seen.add(key)
                deduped.append((t, c, s, d, ref))
        self.best_code_buffer = deduped[:self.pop_size]

    def save_state(self, path: str = "out/self_improve/state.json"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "best_code_buffer": [(t, c, s, d, ref) for t, c, s, d, ref in self.best_code_buffer],
            "best_score": self.best_score,
            "generation": self.generation,
            "history": [asdict(r) for r in self.history],
            "reflection_memory": self.reflection_memory.reflections,
            "current_difficulty": self._current_difficulty,
        }
        for r in state.get("reflection_memory", []):
            if "embedding" in r:
                if isinstance(r["embedding"], np.ndarray):
                    r["embedding"] = r["embedding"].tolist()
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str = "out/self_improve/state.json"):
        p = Path(path)
        if p.exists():
            with open(p) as f:
                state = json.load(f)
            self.best_code_buffer = []
            for item in state.get("best_code_buffer", []):
                if len(item) == 5:
                    self.best_code_buffer.append(tuple(item))
                elif len(item) == 3:
                    t, c, s = item
                    self.best_code_buffer.append((t, c, s, TASK_DIFFICULTY.get(t, 3), ""))
            self.best_score = state.get("best_score", -float("inf"))
            self.generation = state.get("generation", 0)
            for h in state.get("history", []):
                self.history.append(LLMSelfImprovementRecord(**h))
            for r in state.get("reflection_memory", []):
                if "embedding" in r and isinstance(r["embedding"], list):
                    r["embedding"] = np.array(r["embedding"], dtype=np.float32)
            self.reflection_memory.reflections = state.get("reflection_memory", [])
            self._current_difficulty = state.get("current_difficulty", 1.0)
            print(f"[RESUME] Loaded state: iteration {len(self.history)}, "
                  f"best_score={self.best_score:.2f}, "
                  f"reflections={len(self.reflection_memory.reflections)}")

    def improve(self, num_iterations: int = 50, resume: bool = False) -> List[LLMSelfImprovementRecord]:
        print(f"=== LLM Self-Improvement Loop v2 (ERL+LSE) ===")
        print(f"Tasks: {len(self.tasks)} | Population: {self.pop_size} | "
              f"Retain: {self.retain_top_k} | Tree width: {self.tree_search_width}")

        if resume:
            self.load_state()

        nano = self.llm.nano_model
        moe = self.llm.moe_model
        start = len(self.history)

        for iteration in range(start, num_iterations):
            self.generation = iteration
            task, difficulty = self._select_task()

            goal = TextState(instruction=task)
            exec_state = ExecutionState(
                error_log="",
                execution_trace=f"Iteration {iteration} of {num_iterations}",
                updated_graph_features=np.zeros(512),
            )

            prev_score = 0.0
            if self.best_code_buffer:
                prev_score = max(s for _, _, s, _, _ in self.best_code_buffer[:3])

            final_code, final_score, reflection = self._reflexion_cycle(task, max_attempts=3)

            delta_reward = self._compute_delta_reward(prev_score, final_score)

            self.best_score = max(self.best_score, final_score)

            self.reflection_memory.add(task, final_code, reflection, final_score, difficulty)

            self.best_code_buffer.append((task, final_code, final_score, difficulty, reflection))
            self.best_code_buffer.sort(key=lambda x: -x[2])
            self.best_code_buffer = self.best_code_buffer[:self.pop_size]

            diversity_bonus = 0.0
            if len(self.best_code_buffer) >= 2:
                codes_set = set(c[:80] for _, c, _, _, _ in self.best_code_buffer[:5])
                diversity_bonus = min(1.0, len(codes_set) / 5.0)

            was_trained = False
            if iteration > 0 and iteration % 5 == 0:
                top_k = self.best_code_buffer[:self.retain_top_k]
                if len(top_k) >= 3:
                    if moe and moe.loaded and moe.model is not None:
                        print(f"  [Iter {iteration}] Fine-tuning MoE on {len(top_k)} best samples "
                              f"(diversity={diversity_bonus:.2f})...")
                        self._finetune_moe(moe, top_k, diversity_bonus)
                        was_trained = True
                    elif nano and nano.loaded and nano.model is not None:
                        print(f"  [Iter {iteration}] Fine-tuning nanoGPT on {len(top_k)} best samples "
                              f"(diversity={diversity_bonus:.2f})...")
                        self._finetune_nano(nano, top_k, diversity_bonus)
                        was_trained = True

            record = LLMSelfImprovementRecord(
                iteration=iteration,
                task=task[:60],
                task_difficulty=difficulty,
                generated_code=final_code[:120],
                reflection=reflection[:80],
                refined_code=final_code[:120],
                eval_score=final_score,
                refined_score=final_score,
                delta_score=delta_reward,
                was_trained=was_trained,
                best_score_so_far=self.best_score,
            )
            self.history.append(record)
            if iteration % 5 == 0:
                self.save_state()

            if iteration % 10 == 0 or iteration == start:
                avg_top3 = np.mean([s for _, _, s, _, _ in self.best_code_buffer[:3]]) if self.best_code_buffer else 0
                avg_refined = np.mean([r.refined_score for r in self.history[-10:]]) if len(self.history) >= 10 else final_score
                print(f"  [Iter {iteration:3d}/{num_iterations}] "
                      f"Score={final_score:.2f} | Best={self.best_score:.2f} | "
                      f"AvgTop3={avg_top3:.2f} | AvgRefined={avg_refined:.2f} | "
                      f"Diff={difficulty} | Mem={len(self.reflection_memory.reflections)} | "
                      f"Delta={delta_reward:+.2f}"
                      + (" | TRAINED" if was_trained else ""))

        print(f"=== LLM Self-Improvement Complete ===")
        print(f"Final best score: {self.best_score:.2f}")
        print(f"Best code samples stored: {len(self.best_code_buffer)}")
        print(f"Reflections generated: {len(self.reflection_memory.reflections)}")

        improvement_rate = 0.0
        if len(self.history) > 5:
            deltas = [r.delta_score for r in self.history[5:] if abs(r.delta_score) < 50]
            if deltas:
                improvement_rate = np.mean(deltas)
        print(f"Avg improvement per iteration: {improvement_rate:+.3f}")

        self._consolidate_best()
        return self.history

    def _finetune_nano(self, nano: NanoGPTModel, top_samples: List[Tuple], diversity_weight: float = 0.0):
        if nano.model is None:
            return

        model = nano.model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")

            all_texts = []
            for sample in top_samples:
                task, code, score, difficulty, reflection = sample if len(sample) >= 5 else (*sample, "", "")
                weight = max(0.5, score / 5.0)
                for _ in range(int(weight * 3)):
                    code_with_reflection = f"# Task: {task}\n# Approach: {reflection[:100]}\n{code}"
                    all_texts.append(code_with_reflection)

            if not all_texts:
                return

            random.shuffle(all_texts)
            batch_size = min(4, len(all_texts))
            n_epochs = 3
            block_size = getattr(model.config, 'block_size', 512)

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, len(all_texts), batch_size):
                    batch_texts = all_texts[i:i + batch_size]
                    tokens_list = []
                    for text in batch_texts:
                        ids = enc.encode(text)
                        ids = ids[:block_size]
                        tokens_list.append(ids)

                    max_len = max(len(t) for t in tokens_list)
                    padded = []
                    for t in tokens_list:
                        t = t + [enc.eot_token] * (max_len - len(t))
                        padded.append(t[:block_size])

                    if not padded:
                        continue

                    x = torch.tensor(padded, dtype=torch.long, device=self.device)
                    y = x.clone()

                    logits, loss = model(x, targets=y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                if n_batches > 0:
                    avg = epoch_loss / n_batches
                    print(f"    Fine-tune epoch {epoch + 1}/{n_epochs}: loss={avg:.4f}")

            model.eval()
            nano.model = model
            ckpt_path = "out/code_gpt/ckpt_improved.pt"
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": model.config}, ckpt_path)
            print(f"    Improved model saved to {ckpt_path}")

        except Exception as e:
            print(f"    Fine-tuning failed: {e}")

    def _finetune_moe(self, moe: MoEModel, top_samples: List[Tuple], diversity_weight: float = 0.0):
        if moe.model is None or not moe.vocab:
            return

        model = moe.model
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        try:
            all_texts = []
            for sample in top_samples:
                task, code, score, difficulty, reflection = sample if len(sample) >= 5 else (*sample, "", "")
                weight = max(0.5, score / 5.0)
                for _ in range(int(weight * 3)):
                    text = f"# Task: {task}\n# Approach: {reflection[:100]}\n{code}"
                    all_texts.append(text)

            if not all_texts:
                return

            random.shuffle(all_texts)
            batch_size = min(4, len(all_texts))
            n_epochs = 3
            block_size = getattr(model.config, 'block_size', 512)

            for epoch in range(n_epochs):
                epoch_loss = 0.0
                n_batches = 0

                for i in range(0, len(all_texts), batch_size):
                    batch_texts = all_texts[i:i + batch_size]
                    tokens_list = []
                    for text in batch_texts:
                        ids = moe._tokenize(text)
                        ids = ids[:block_size]
                        tokens_list.append(ids)

                    max_len = max(len(t) for t in tokens_list) if tokens_list else 0
                    if max_len < 2:
                        continue
                    padded = []
                    for t in tokens_list:
                        t = t + [0] * (max_len - len(t))
                        padded.append(t[:block_size])

                    x = torch.tensor(padded, dtype=torch.long, device=self.device)
                    y = x.clone()

                    logits, loss, aux = model(x, targets=y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                if n_batches > 0:
                    avg = epoch_loss / n_batches
                    print(f"    MoE fine-tune epoch {epoch + 1}/{n_epochs}: loss={avg:.4f} (ce={aux['ce_loss']:.4f}, bal={aux['balance_loss']:.4f})")

            model.eval()
            moe.model = model
            ckpt_path = "out/moe_gpt/ckpt_improved.pt"
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": model.config}, ckpt_path)
            print(f"    Improved MoE model saved to {ckpt_path}")

        except Exception as e:
            import traceback
            print(f"    MoE fine-tuning failed: {e}")
            traceback.print_exc()

    def get_best_code(self, n: int = 5) -> List[Tuple[str, str, float, float, str]]:
        return self.best_code_buffer[:n]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RSI LLM Self-Improvement v2")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--nanogpt", action="store_true",
                        help="Enable nanoGPT model (requires trained checkpoint)")
    parser.add_argument("--moe", action="store_true",
                        help="Enable MoE model (requires trained checkpoint)")
    parser.add_argument("--tree_width", type=int, default=3,
                        help="Tree search width for refinement exploration")

    args = parser.parse_args()

    print("--- RSI LLM Self-Improvement v2 (ERL + LSE) ---")
    llm = LLMPlanner(use_nanogpt=args.nanogpt, use_moe=args.moe)
    loop = LLMSelfImprovementLoop(llm=llm, tree_search_width=args.tree_width, device=args.device)
    history = loop.improve(num_iterations=args.iterations, resume=args.resume)
    loop.save_state()
    print(f"Completed {len(history)} iterations.")
    print(f"Best score: {loop.best_score:.2f}")
    print(f"Top-3 samples:")
    for i, (t, c, s, d, ref) in enumerate(loop.get_best_code(3)):
        print(f"  {i+1}. [{t[:40]}] score={s:.2f} diff={d}")
