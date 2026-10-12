#!/usr/bin/env python3
"""ROUND-199: self-contained HumanEval runner (lm_eval 0.4.13's humaneval/mbpp
tasks depend on PyPI packages that were REMOVED in 2023 — this uses only the
HF datasets package + our trust_remote_code wrapper + the execution sandbox.

Faithful to the canonical protocol:
  - prompt = canonical docstring prompt
  - generation: temperature 0.2, bf16, completion-only via the wrapper
  - completion trimmed at the first top-level def/class/if/while/for
  - pass@1: run the problem's unit tests in the sandbox (model-generated code
    executes inside tool_use's jail — verification is execution, not parsing)

Usage:
  .venv/bin/python3 src/eval_humaneval.py --ckpt out/ambatron_tool_v14/ckpt.pt [--limit 8] [--device cuda]
"""
import argparse, sys, os, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--limit", type=int, default=16)
    ap.add_argument("--hf_export", default=str(ROOT / "hf" / "v14"))
    ap.add_argument("--temp", type=float, default=0.2)
    # 19th-audit B-5a: explicit device — on this card the eval must be able to
    # run CPU-only (or wait for a GPU window) instead of hardcoding cuda.
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # dataset fetch needs the hub ON; the model stays local via local_files_only
    import torch
    import torch.serialization as ser
    from moe_model import MoEConfig
    ser.add_safe_globals([MoEConfig])
    import torch.serialization as _ser

    # export the ckpt to the HF wrapper (fresh, cache-purged)
    from export_hf import main as _export
    sys.argv = ["export_hf", "--ckpt", args.ckpt, "--out", str(args.hf_export)]
    _export()
    import shutil
    cache = pathlib.Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(args.hf_export), trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(str(args.hf_export), trust_remote_code=True)
    model.to(args.device)

    from datasets import load_dataset
    # 19th-audit B-5b: offline fallback — the hub flaked before (net drop =
    # uncaught crash each run). Try the cache first, then the hub, then die
    # loudly with a hint instead of a raw stack.
    try:
        ds = load_dataset("openai/openai_humaneval", split="test")
    except Exception:
        try:
            ds = load_dataset("openai/openai_humaneval", split="test", download_mode="reuse_dataset_if_exists")
        except Exception as e:
            raise SystemExit(f"[humaneval] cannot load openai_humaneval (offline?) — "
                             f"pre-fetch with: python -c \"from datasets import load_dataset; "
                             f"load_dataset('openai/openai_humaneval')\" ({e})")

    from tool_use import _execute_python

    n_ok = n_total = 0
    for row in list(ds)[: args.limit]:
        prompt = row["prompt"]
        test = row["test"]
        entry = row["entry_point"]

        ids = tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], dtype=torch.long, device="cpu").to(args.device)
        with torch.no_grad():
            out = model.generate(inp, max_new_tokens=256, temperature=args.temp)
        completion = tok.decode(out[0, len(ids):].tolist(), skip_special_tokens=True)

        # canonical trimming: stop at the first top-level block starter
        cut = None
        # 204m: TOP-LEVEL-only trim — the old `\n\s*(while|for|if)` fired on a
        # leading newline + indented control flow (cut=0 -> empty completion)
        # and dropped function tails. Only a column-0 block starter stops.
        for m in re.finditer(r"\n(?:def |class |if __name__|while |for |if )",
                             "\n" + completion):
            cut = m.start()
            break
        comp = (completion[:cut - 1] if cut is not None else completion).strip()
        # 19th-audit (CRIT-FP): the canonical humaneval `test` field is ONLY
        # the `def check(candidate): ...` DEFINITION — nothing invokes it. The
        # old `prompt + comp + test` ran a syntactically-valid completion's
        # module body and reported "code executed successfully" for EVERY
        # valid snippet, check or no check = structural false pass@1. Invoke
        # check(entry) exactly like the canonical harness.
        # 204m: the sentinel — the sandbox's success marker only appears with
        # EMPTY stdout, so printing completions (the tool-SFT norm) got a
        # structural FAIL despite check() passing. Print the sentinel AFTER
        # check() and gate on it.
        code = prompt + comp + "\n" + test + f"\ncheck({entry})\nprint('__RSI_OK__')\n"

        try:
            res = _execute_python(code)
        except Exception as e:
            res = f"error: {e}"
        rl = res.lower()
        BAD = ("[timed out", "[truncated", "traceback", "error", "exited with code", "sandbox")
        # 204m: gate on the POST-check sentinel (stdout-independent).
        ok = ("__rsi_ok__" in rl) and not any(b in rl for b in BAD)
        n_total += 1
        n_ok += int(ok)
        print(f"[humaneval] {'PASS' if ok else 'FAIL'} {entry[:40]}", flush=True)

    print(f"[humaneval] {args.ckpt} | pass@1 {n_ok}/{n_total} "
          f"= {n_ok / max(n_total, 1) * 100:.1f}% (temp {args.temp}, limit {args.limit}, device {args.device})")


if __name__ == "__main__":
    main()