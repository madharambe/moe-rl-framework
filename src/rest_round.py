"""B-74: ReST round 1 — execution-feedback rejection filtering.

Generate N candidate tool-answers per held-out query with the CURRENT model,
EXECUTE the generated code, and keep ONLY the candidates whose tool result is
CORRECT. The winners become new SFT records (assistant=tool-call, tool=result,
assistant=verified answer) appended to data/processed/rest_round1.jsonl.

This is the CodeRL/RLEF + ReST recipe on our own stack: the execute_python
tool IS execution feedback. Run AFTER the GPU is free:

  .venv/bin/python3 -u src/rest_round.py --ckpt out/ambatron_tool_v7/ckpt.pt \
      --n_samples 8 --n_queries 24 --out data/processed/rest_round1.jsonl

Then: train_tool_sft.py on tool_sft.jsonl + rest_round1.jsonl (parity + replay 0.02).
"""
import argparse, sys, pathlib, json, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tool_use import (ToolUseMoE, ToolRegistry, _execute_python, _search_knowledge, generate_batch)

QUERIES = [
    ("What is the sum of squares of 1..10?", ("execute_python", 385)),
    ("Compute the 11th Fibonacci number and report it.", ("execute_python", 55)),
    ("Check whether 1729 is divisible by 7.", ("execute_python", "yes")),
    ("Sort [3,1,4,1,5] and report the second element.", ("execute_python", 1)),
    ("What is the sum of cubes of 1..7?", ("execute_python", 784)),
    ("Compute the 14th Fibonacci number and report it.", ("execute_python", 233)),
    ("Check whether 931 is divisible by 19.", ("execute_python", "yes")),
    ("Sort [7,2,9,4,6] and report the third element.", ("execute_python", 6)),
    ("What is the sum of squares of 1..8?", ("execute_python", 204)),
    ("Compute the 10th Fibonacci number and report it.", ("execute_python", 34)),
    ("Check whether 847 is divisible by 11.", ("execute_python", "yes")),
    ("Sort [5,9,2,7,1] and report the fourth element.", ("execute_python", 7)),
    ("What is the 7th prime number?", ("execute_python", 17)),
    ("Compute the sum 1+2+...+20.", ("execute_python", 210)),
    ("Check whether 1001 is divisible by 13.", ("execute_python", "yes")),
    ("Sort [8,3,6,2,9] and report the smallest element.", ("execute_python", 2)),
    ("What is the sum of cubes of 1..5?", ("execute_python", 225)),
    ("Compute the 15th Fibonacci number and report it.", ("execute_python", 377)),
    ("Check whether 153 is divisible by 9.", ("execute_python", "yes")),
    ("Sort [4,1,7,3,8] and report the largest element.", ("execute_python", 8)),
    ("What is the sum of squares of 1..6?", ("execute_python", 91)),
    ("Check whether 357 is divisible by 17.", ("execute_python", "yes")),  # 221 renumbered (19th-audit B-2: same-number near-dup of hard "Is 221 divisible by 17?")
    ("Sort [6,2,9,5,3] and report the first element.", ("execute_python", 2)),
]


def _check_call(name, args, ans):
    """Return (passed: bool, result_str: str) — B-211: return the executed
    result so the caller can reuse it for the winning record instead of
    executing the model's code TWICE (validation + training-record population;
    the double-exec baked the exact raw call result into training data only if
    the sandbox is deterministic, which arithmetic is but the registry's
    side effects may not be)."""
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    r = reg.call(name, args)
    rs = str(r).strip().lower()
    # ROUND-170 (MED): a TIMEOUT/TRUNCATED/SANDBOX-ERROR observation must not
    # win — `print(385); time.sleep(1e9)` returned "[timed out after 10s] 385",
    # whose last number (385) matched the answer and trained the give-up shape.
    if any(m in rs for m in ("[timed out", "[truncated", "sandbox",
                             "traceback", "exited with code")):
        return False, str(r)
    if ans == "yes":
        # B-196/B-202: only an EXPLICIT boolean assertion counts. Reject bare
        # numeric "0"/"0.0" — a model emitting `print(0)` for ANY "divisible
        # by?" query gamed the ReST winner filter to 100% without computing a % b.
        return rs in ("true", "yes"), str(r)
    if ans == "no":
        # B-202: symmetric negative case so an "always-yes" policy is penalized.
        return rs in ("false", "no"), str(r)
    nums = [int(n) for n in re.findall(r"[-+]?\d+", str(r))]
    return (bool(nums) and ans == nums[-1]), str(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--n_queries", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--out", default="data/processed/rest_round1.jsonl")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--compiled", action="store_true", default=True,  # B-189: default True to match deploy
                    help="B-94: use the TSCG compiled tool-schema prompt so the "
                         "winners match the v13's training format")
    ap.add_argument("--no_batch", action="store_true",
                    help="ROUND-183: disable batched generation (default ON).")
    ap.add_argument("--bf16", action="store_true",
                    help="load the model in bfloat16 (~half VRAM) so the ReST "
                         "run fits the flywheel's 2GB VRAM gate (avoids VRAM OOM)")
    args = ap.parse_args()

    import torch
    import torch.serialization as ser
    from moe_model import MoEConfig, MoEGPT
    ser.add_safe_globals([MoEConfig])
    from train_tool_sft import _tokenize
    from gpu_accel import resolve_device

    # B-169: the runtime registry emits a DIFFERENT compiled schema than
    # v13's frozen training bundle (timeout_s param, 2 vs 6 tools) -- the
    # same OOD trap B-146 caught in gen_arith_sft. Winners trained on the
    # runtime prompt DEGRADED the champion (Round-134 candidate B, 4%).
    # Use the frozen snapshot for BOTH generation context and records.
    import json as _json
    with open(pathlib.Path(__file__).resolve().parent.parent
              / "data/processed/tool_sft_compiled.jsonl") as _f:
        FROZEN_SYSTEM = _json.loads(_f.readline())["segments"][0][1]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = MoEGPT(ck["config"])
    model.load_state_dict(ck["model_state"], strict=False)
    if args.bf16:
        model.to(torch.bfloat16)
    model.to(resolve_device())
    model.eval()
    # 205aa: vocab-derived tokenizer (the fixed 16K stream mismatched v3 champions).
    tok = _tokenize("", "v3" if int(getattr(ck["config"], "vocab_size", 0)) >= 32768 else None)
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    # ROUND-183: batch-path self-test — run BOTH paths on 3 math queries at
    # temperature 0 (argmax decode is deterministic) and require IDENTICAL
    # completions. Any divergence -> fall back to sequential.
    if not args.no_batch:
        try:
            # ROUND-183b: parity holds ONLY for IDENTICAL-LENGTH batched rows
            # (right-padding changes the causal attention of shorter rows) —
            # the production batch is [q]*n_samples (the SAME prompt), so the
            # selftest batches one prompt x4 vs sequential x4.
            _q = "What is the sum of cubes of 1..5?"
            _tu1 = ToolUseMoE(model, lambda t: tok.encode(t).ids,
                              lambda i: tok.decode(i), reg,
                              compiled_schema=args.compiled)
            _seq = [_tu1.generate(_q, max_tokens=24, max_tool_calls=1,
                                  temperature=0.0) for _ in range(4)]
            _tu2 = ToolUseMoE(model, lambda t: tok.encode(t).ids,
                              lambda i: tok.decode(i), reg,
                              compiled_schema=args.compiled)
            _bat = generate_batch(_tu2, [_q] * 4, max_tokens=24,
                                  max_tool_calls=1, temperature=0.0)
            if _seq == _bat:
                print("[rest] batch-path selftest: OK (matches sequential)", flush=True)
            else:
                print("[rest] batch-path selftest MISMATCH — falling back to "
                      "sequential", flush=True)
                args.no_batch = True
        except Exception as _e:
            print(f"[rest] batch-path selftest error ({_e}) — falling back to "
                  f"sequential", flush=True)
            args.no_batch = True

    queries = QUERIES[:args.n_queries]
    winners = []
    checked = correct = 0
    tu = ToolUseMoE(model, lambda t: tok.encode(t).ids,
                    lambda i: tok.decode(i), reg,
                    compiled_schema=getattr(args, "compiled", False))
    # ROUND-183f: PER-QUERY batches only. Cross-query batching (183c) right-
    # padded shorter prompts with EOS — the model then conditioned on
    # "(prompt) EOS EOS..." and emitted no valid tool calls (ReST: 0 winners),
    # and the huge chunked rounds made ReST so slow every cycle crossed the
    # 40-min HUNG window (the 'silent deaths' at 18:2x-19:1x). Same-length
    # rows (one query x n_samples) = no padding = full parity.
    _idx = 0
    for q, (tool, ans) in queries:
        if not args.no_batch:
            _outs = generate_batch(tu, [q] * args.n_samples, max_tokens=80,
                                   max_tool_calls=2,
                                   temperature=args.temperature)
            _trajs = list(tu.last_trajectories)
        for s in range(args.n_samples):
            if args.no_batch:
                out = tu.generate(q, max_tokens=80, max_tool_calls=2,
                                  temperature=args.temperature)
            else:
                out = _outs[s]
            # B-XXX: generate() returns ONLY the final completion (the answer
            # AFTER tool executions), which never contains the tool-call JSON —
            # parsing `out` yielded call=None on every sample => 0 winners =>
            # the tool self-improvement ReST loop silently did nothing. The
            # actual tool call lives in tu.last_trajectory (list of
            # {"name","arguments","observation"}).
            call = None
            _traj = (_trajs[s] if not args.no_batch else tu.last_trajectory)
            for _step in _traj:
                if _step.get("name") and "arguments" in _step:
                    call = {"name": _step["name"], "arguments": _step["arguments"]}
                    break
            if call is None:
                continue
            checked += 1
            # M5: the winner must also have called the EXPECTED tool. A
            # search_knowledge/other-tool call that merely returns a string
            # containing the target number would otherwise pass the answer
            # filter and train an off-topic schema.
            if call.get("name") != tool:
                continue
            ok, res = _check_call(call.get("name", ""), call.get("arguments", {}), ans)
            if ok:
                correct += 1
                schema = FROZEN_SYSTEM
                winners.append({
                    "segments": [
                        ("system", schema),
                        ("user", f"User: {q}\n"),
                        ("assistant", f"Assistant: use tool:{json.dumps(call)}"),
                        ("tool", f"result: {res[:500]}\n"),
                        ("assistant", f"Assistant: The result is {ans}."),
                    ],
                    "domain": "math",
                })
            if (checked % 20) == 0:
                print(f"[rest] {checked} checked | {correct} correct ({correct/max(checked,1)*100:.0f}%)",
                      flush=True)
    print(f"[rest] DONE: {checked} checked | {correct} correct | {len(winners)} winners "
          f"({correct/max(checked,1)*100:.0f}% correct-rate)")
    with open(args.out, "w") as f:
        for r in winners:
            f.write(json.dumps(r) + "\n")
    print(f"[rest] wrote {len(winners)} -> {args.out}")


if __name__ == "__main__":
    main()
