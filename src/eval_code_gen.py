"""Code-completion eval for the mini/430m models (B-40).
Samples function-header prompts, generates completions, scores them with
reference-free signals: compile rate, structure (def/return/import), length.
Usage: python3 src/eval_code_gen.py --ckpt out/ambatron_mini/ckpt_best.pt
                                   [--size mini|430m] [--n 20] [--max_tokens 80]
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import torch.serialization as ser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--size", type=str, default="mini", choices=["mini", "plus", "xl", "430m"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max_tokens", type=int, default=80)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    from moe_model import (MoEGPT, MoEConfig, create_ambatron_mini_config,
                           create_ambatron_plus_config, create_ambatron_xl_config,
                           create_ambatron_430m_config)
    from tokenizers import Tokenizer
    ser.add_safe_globals([MoEConfig])

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck.get("config")
    # 205t: vocab-derived tokenizer (the fixed 16K bpe mismatched every v3
    # 32K champion — the same bug class as eval_agentic 204ah-21).
    _v3 = int(getattr(cfg, "vocab_size", 0)) >= 32768
    tok = Tokenizer.from_file(str(pathlib.Path(__file__).resolve().parent.parent /
                                   ("v3-bank/tokenizer_v3.json" if _v3 else
                                    "data/processed/bpe_tokenizer/tokenizer.json")))
    if cfg is None or not isinstance(cfg, MoEConfig):
        cfg = (create_ambatron_xl_config() if args.size == "xl" else
               create_ambatron_plus_config() if args.size == "plus" else
               create_ambatron_mini_config() if args.size == "mini" else
               create_ambatron_430m_config())
    model = MoEGPT(cfg)
    model.load_state_dict(ck["model_state"], strict=False)
    model.eval()
    model.to(args.device)
    print(f"[eval] loaded {args.ckpt} ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)")

    prompts = [
        "def quicksort(arr):\n",
        "def fibonacci(n):\n",
        "def train_ppo(env, policy, steps):\n",
        "class RobotController:\n    def __init__(self, max_speed):\n",
        "def inverse_kinematics(x, y, z):\n",
        "def pid_controller(setpoint, measured, dt):\n",
        "def read_imu_data(sensor):\n",
        "def plan_path(start, goal, obstacles):\n",
        "def compute_reward(position, goal):\n",
        "def kalman_filter(z, x_hat, P):\n",
        "def gripper_pick(target_pose):\n",
        "def navigate_to(pose, speed):\n",
        "def interpolate_trajectory(waypoints, t):\n",
        "def obstacle_distance(scan):\n",
        "def log_telemetry(state):\n",
        "def execute_tool_call(name, arguments):\n",
        "def parse_json_response(text):\n",
        "def save_checkpoint(model, path):\n",
        "def load_dataset(path, batch_size):\n",
        "def compute_velocity_error(target, current):\n",
    ][: args.n]

    n_ok = n_def = n_ret = n_import = 0
    samples = []
    with torch.no_grad():
        for p in prompts:
            ids = tok.encode(p).ids
            idx = torch.tensor([ids], dtype=torch.long, device=args.device)
            out = model.generate(idx, max_new_tokens=args.max_tokens,
                                 temperature=0.7, top_k=40)
            gen = tok.decode(out[0, len(ids):].tolist())
            samples.append((p, gen))
            # compile the wrapped completion
            import textwrap
            wrapped = "def _generated():\n" + "\n".join(
                ("    " + ln) if ln.strip() else ln for ln in
                textwrap.dedent(gen).split("\n"))
            try:
                compile(wrapped, "<eval>", "exec")
                n_ok += 1
            except SyntaxError:
                try:
                    compile(gen, "<eval>", "exec")
                    n_ok += 1
                except SyntaxError:
                    pass
            if "def " in gen:
                n_def += 1
            if "return" in gen:
                n_ret += 1
            if "import" in gen:
                n_import += 1

    print(f"[eval] n={len(samples)} | compile-ok {n_ok}/{len(samples)} "
          f"| def {n_def} | return {n_ret} | import {n_import}")
    print("\n--- samples ---")
    for p, g in samples[:5]:
        print(f">>> {p}{g[:140]!r}\n")


if __name__ == "__main__":
    main()