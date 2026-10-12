#!/usr/bin/env python3
"""ROUND-204ac-b6.8: AGENTIC file-envelope eval (fully wired B-72 pattern).

Same philosophy as eval_agentic.py: inject a synthetic error into the FIRST
tool call and check the model RETRIES with a corrected call and converges.
Here the envelope = the file tools (read_file/grep/glob/list_dir), the
agentic-on-workspace meter for the file tools. Scores: retried-after-error,
correct-answer, retry-AND-correct. Current champions score ~0 (wrong
envelope); the meter matters once v3+E1 gives the model the file tools.

Usage:
  .venv/bin/python3 src/eval_agentic_files.py --ckpt <ckpt> [--repeat 2]
      [--device cpu|cuda] [--workspace data/agent_workspace]
"""
import argparse, json, os, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import torch

from tool_use import ToolUseMoE, ToolRegistry, TOOL_CALL_START
from agentic_tools import agentic_registry

CASES = [
    ("List the files inside the sub directory and report them.",
     ("list_dir", "a.txt")),
    ("Read the file sub/a.txt and report its content.",
     ("read_file", "hello world")),
    ("Find which file in sub contains the word func and report the filename.",
     ("grep", "main.go")),
    ("Use glob to find the .go files in sub and report the first one.",
     ("glob", "main.go")),
]

SYNTHETIC_ERROR = "Error: injection: tool execution failed, retry with correct arguments."


class FailOnceRegistry(ToolRegistry):
    def __init__(self, base, inject_name, error):
        super().__init__()
        self._tools = base._tools
        self.inject_name = inject_name
        self.error = error
        self.failed_yet = False

    def call(self, name, arguments):
        if name == self.inject_name and not self.failed_yet:
            self.failed_yet = True
            return self.error
        return super().call(name, arguments)


def _tok_for(cfg):
    from tokenizers import Tokenizer
    base = pathlib.Path(__file__).resolve().parent.parent
    if getattr(cfg, "vocab_size", 0) >= 32768:
        return Tokenizer.from_file(str(base / "v3-bank" / "tokenizer_v3.json"))
    from train_tool_sft import _tokenize
    return _tokenize("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--queries-json", type=str, default="",
                    help="205k: arbitrary query list (seed finder)")
    ap.add_argument("--dump-dir", type=str, default="",
                    help="205r: dump (query, expected, out, trajectory) for "
                         "every CORRECT eval (gate-consolidation demos)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--workspace", default="data/agent_workspace")
    args = ap.parse_args()

    base = pathlib.Path(__file__).resolve().parent.parent
    ws = base / args.workspace
    (ws / "sub").mkdir(parents=True, exist_ok=True)
    if not (ws / "sub" / "a.txt").exists():
        (ws / "sub" / "a.txt").write_text("hello world")
    if not (ws / "sub" / "main.go").exists():
        (ws / "sub" / "main.go").write_text("func main() {}")
    os.environ["RSI_AGENT_WORKSPACE"] = str(ws)
    os.environ["RSI_AGENT_WRITE"] = "0"

    import torch.serialization as ser
    from moe_model import MoEConfig, MoEGPT
    ser.add_safe_globals([MoEConfig])

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    model = MoEGPT(cfg)
    model.load_state_dict(ck["model_state"], strict=False)
    from gpu_accel import resolve_device
    model.to(resolve_device(force=args.device))
    model.eval()

    tok = _tok_for(cfg)
    reg = agentic_registry()

    retries = correct = converged = 0
    n = 0
    cases = CASES
    if args.queries_json:
        import json as _json
        cases = [(r["query"], (r.get("tool", "bash"), r.get("expected")))
                 for r in _json.load(open(args.queries_json))]
    for _ in range(args.repeat):
        for q, (tool, ans) in cases:
            fret = FailOnceRegistry(reg, tool, SYNTHETIC_ERROR)
            tu = ToolUseMoE(model, lambda t: tok.encode(t).ids,
                            lambda i: tok.decode(i), fret)
            out = tu.generate(q, max_tokens=200, max_tool_calls=3,
                              temperature=0.3)
            n += 1
            if fret.failed_yet:
                retried = len(getattr(tu, "last_trajectory", [])) >= 2
                if retried:
                    retries += 1
            # 205k: OBSERVATION-BASED correctness (the old whole-out scan
            # counted answers echoed in errored-call arguments — the same
            # gaming class fixed in the agentic meter at 205c).
            _obs = None
            for _e in (getattr(tu, "last_trajectory", []) or []):
                _o = str(_e.get("observation") or "")
                if any(_m in _o.lower() for _m in ("traceback", "_error",
                        "sandbox", "[timed out", "[truncated",
                        "loop detected")) or "Error" in _o or "Exception" in _o:
                    continue
                if not (str(_e.get("arguments") or {})):
                    continue
                _obs = _o
            ans_s = str(ans).lower()
            _ok = _obs is not None and re.search(r"(?<![A-Za-z0-9_'])" +
                                                 re.escape(ans_s) +
                                                 r"(?![A-Za-z0-9_'])", _obs.lower())
            if _ok:
                correct += 1
                if retried:
                    converged += 1
                if args.dump_dir:
                    import os as _os
                    _os.makedirs(args.dump_dir, exist_ok=True)
                    with open(_os.path.join(args.dump_dir, "demos.jsonl"), "a") as fd:
                        fd.write(json.dumps({"query": q, "expected": str(ans),
                                             "out": out, "ok": True,
                                             "trajectory": getattr(tu, "last_trajectory", [])}) + "\n")
    print(f"[agentic-files] {args.ckpt} | n={n} | envelope={list(reg._tools)}")
    print(f"  retried-after-error: {retries}/{n} ({retries/n*100:.0f}%)")
    print(f"  correct-answer:      {correct}/{n} ({correct/n*100:.0f}%)")
    print(f"  retry-AND-correct:   {converged}/{n} ({converged/n*100:.0f}%)")


if __name__ == "__main__":
    main()
