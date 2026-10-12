import csv
import os
from collections import deque
from pathlib import Path
from typing import Optional


class EvalLogger:
    def __init__(self, log_dir: str = None, algo_name: str = "sac", window: int = 10, use_tensorboard: bool = False):
        # ROUND-170: root-anchored default (a relative 'logs' wrote to whatever
        # CWD launched the VLA job).
        if log_dir is None:
            log_dir = str(Path(__file__).resolve().parent.parent / "logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.algo_name = algo_name
        self.window = window
        self.csv_path = self.log_dir / f"{algo_name}_metrics.csv"
        self._file_handle = None
        self._writer = None
        self._needs_header = not self.csv_path.exists()

        self.episode_rewards = deque(maxlen=window)
        self.episode_lengths = deque(maxlen=window)

        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb" / algo_name))
            except Exception:
                pass

    def log_episode(self, episode: int, reward: float, length: int, **extra):
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)

        row = {
            "episode": episode,
            "reward": reward,
            "length": length,
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards),
            "avg_length": sum(self.episode_lengths) / len(self.episode_lengths),
        }
        row.update(extra)

        if self._writer is None:
            self._file_handle = open(self.csv_path, "a", newline="")
            self._writer = csv.DictWriter(self._file_handle, fieldnames=list(row.keys()))
            if self._needs_header:
                self._writer.writeheader()
                self._needs_header = False

        # B-112: the schema froze on the first row — a later log_episode with
        # a new key raised ValueError hours into a run. Mutating fieldnames
        # mid-file would make the CSV ragged (breaks read_csv), so drop
        # unknown keys with a one-time warning instead.
        extra_keys = [k for k in row.keys() if k not in self._writer.fieldnames]
        if extra_keys:
            if not getattr(self, "_warned_keys", set()):
                print(f"[eval-logger] ignoring unplanned column(s): "
                      f"{extra_keys} (add them to the first log call)")
                self._warned_keys = set(extra_keys)
            row = {k: v for k, v in row.items() if k in self._writer.fieldnames}
        self._writer.writerow(row)
        self._file_handle.flush()

        if self.tb_writer:
            self.tb_writer.add_scalar("reward", reward, episode)
            self.tb_writer.add_scalar("avg_reward", row["avg_reward"], episode)
            self.tb_writer.add_scalar("episode_length", length, episode)
            for k, v in extra.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(k, v, episode)

    def log_update(self, step: int, **metrics):
        if self.tb_writer:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f"train/{k}", v, step)
        import json as _json
        with open(self.log_dir / f"{self.algo_name}_train.jsonl", "a") as f:
            f.write(_json.dumps({"step": step, **metrics}) + "\n")

    def print_summary(self):
        avg_r = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0
        avg_l = sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0
        print(f"  Avg Reward: {avg_r:.2f} | Avg Length: {avg_l:.1f}")

    def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._writer = None
            self._file_handle = None
        if self.tb_writer:
            self.tb_writer.close()
