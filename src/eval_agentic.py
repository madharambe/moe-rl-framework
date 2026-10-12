"""B-72: agentic / double-check eval.

The honest test of the Section-0 self-improving loop: give the model a query,
let it emit a tool call, INJECT a synthetic error as the result (the first call
to a tool fails), then check whether the model RETRIES with a corrected call and
converges on the right answer. A model that stops, hallucinates, or loops after
the error has NOT learned the "try -> check -> correct" behavior.

Usage:
  python3 src/eval_agentic.py --ckpt out/ambatron_tool_v7/ckpt.pt [--repeat 3]
"""
import argparse, sys, pathlib, json, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from tool_use import (ToolUseMoE, ToolRegistry, _execute_python, _search_knowledge,
                      TOOL_CALL_START)

AGENTIC_CASES = [
    ("What is the sum of squares of 1..10?", ("execute_python", 385)),
    ("Compute the 11th Fibonacci number and report it.", ("execute_python", 55)),
    ("Check whether 1729 is divisible by 7.", ("execute_python", "yes")),
    ("Sort [3,1,4,1,5] and report the second element.", ("execute_python", 1)),
    ("What is the sum of cubes of 1..7?", ("execute_python", 784)),
    ("Compute the 14th Fibonacci number and report it.", ("execute_python", 233)),
]

SYNTHETIC_ERROR = (
    "Traceback (most recent call last):\n"
    '  File "<string>", line 1, in <module>\n'
    "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
)


def toks_check(out: str, ans, query: str = ""):
    """B-84/205c: count/sum queries ("how many", "sum of") return a SINGLE
    number -> LAST-number match (any-position lets degraded "1,,,,5" outputs
    game the verdict). Sort-element queries legitimately print the sorted
    list -> any-position stays."""
    import re as _re
    toks = [int(n) for n in _re.findall(r"[-+]?\d+", out)]
    if isinstance(ans, int):
        if _re.search(r"how many|sum of|total", query):
            return bool(toks) and toks[-1] == ans
        return ans in toks
    return bool(_re.search(r"(?<![A-Za-z0-9_'])" + _re.escape(str(ans).lower()) +
                           r"(?![A-Za-z0-9_'])", out.lower()))


class FailOnceRegistry(ToolRegistry):
    """Delegates to a real registry but makes the FIRST call to one tool fail."""

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--device", type=str, default="cuda")
    # 205a: --no-inject = plain registry (first-try precision check; the
    # served path has no injected error). Default = fail-once (meter).
    ap.add_argument("--no-inject", action="store_true")
    # 205a: evaluate an ARBITRARY query list through the exact fail-once loop
    # (the E2-GRPO seed finder) + per-query verdicts.
    ap.add_argument("--queries-json", type=str, default="")
    ap.add_argument("--verbose", action="store_true")
    # 205a: dump (query, out, trajectory) for every CORRECT eval — the
    # self-generated demonstration pool for the SFT leg of the flywheel.
    ap.add_argument("--dump-dir", type=str, default="")
    # 205h: few-shot code exemplars injected into the prompt (in-context
    # imitation; the model's tool-loop code-writing lifts WITHOUT weight
    # changes — SFT on the tool format regressed the retry mode twice).
    ap.add_argument("--fewshot", type=str, default="")
    args = ap.parse_args()
    FEWSHOT = ""
    if args.fewshot:
        FEWSHOT = args.fewshot.replace('\\n', '\n')

    cases = AGENTIC_CASES
    if args.queries_json:
        with open(args.queries_json) as fq:
            qc = json.load(fq)
        cases = [(r["query"], (r.get("mode", "execute_python"),
                               r.get("expected"))) for r in qc]

    import torch
    import torch.serialization as ser
    from moe_model import MoEConfig
    ser.add_safe_globals([MoEConfig])

    from lora_adapter import add_lora, freeze_base
    from gpu_accel import resolve_device
    from train_tool_sft import _tokenize

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    from moe_model import create_ambatron_430m_config, MoEGPT
    cfg = ck["config"]
    model = MoEGPT(cfg)
    model.load_state_dict(ck["model_state"], strict=False)
    model.to(resolve_device())
    model.eval()

    # 204ah-21 (REAL BUG): the old _tokenize("") = the 16K corpus tokenizer —
    # a v3 base (32K vocab) got a MISMATCHED stream (the same class banked for
    # GRPO at b17.2.3); every "agentic 0/N" verdict for v3 arms was structurally
    # void (only the files-meter, which uses _tok_for(cfg), was valid for v3).
    tok = _tokenize("", "v3" if int(getattr(cfg, "vocab_size", 0)) >= 32768 else None)
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")

    import re as _re
    retries = correct = converged = 0
    n = 0
    for _ in range(args.repeat):
        for q, (tool, ans) in cases:
            fret = reg if args.no_inject else FailOnceRegistry(reg, tool, SYNTHETIC_ERROR)
            tu = ToolUseMoE(model, lambda t: tok.encode(t).ids,
                            lambda i: tok.decode(i), fret)
            if FEWSHOT:
                _orig_build = tu._build_prompt
                tu._build_prompt = lambda uq, _b=_orig_build: _b(uq) + FEWSHOT
            out = tu.generate(q, max_tokens=200, max_tool_calls=3,
                              temperature=0.3)

            n += 1
            # 205c: honest verdict = the OBSERVATION of an executed call
            _obs = None
            for _e in (getattr(tu, "last_trajectory", []) or []):
                if _e.get("name") != "execute_python":
                    continue
                _o = str(_e.get("observation") or "")
                if any(_m in _o.lower() for _m in ("traceback", "_error",
                        "sandbox", "[timed out", "[truncated",
                        "loop detected")) or "Error" in _o or "Exception" in _o:
                    continue
                # 205e: reject the trivial-literal gaming mode ("1,1",
                # "1,2" — valid Python tuples that execute but are NOT
                # computation): the code must contain a function call.
                _code = str((_e.get("arguments") or {}).get("code", ""))
                if not ("(" in _code and any(ch.isalpha() for ch in _code.split("(")[0])) or len(_code.strip()) < 8:
                    continue
                _obs = _o
            _ok = _obs is not None and toks_check(_obs, ans, q)
            if args.verbose:
                print(f"[ae-v] correct={_ok} retry={getattr(fret, 'failed_yet', False)} | {q}")
            if args.dump_dir and _obs is not None:
                os.makedirs(args.dump_dir, exist_ok=True)
                with open(os.path.join(args.dump_dir, "demos.jsonl"), "a") as fd:
                    fd.write(json.dumps({"query": q, "expected": str(ans),
                                         "out": out, "ok": _ok,
                                         "trajectory": getattr(tu, "last_trajectory", [])}) + "\n")
            # retry detected: the injected error happened AND a second call
            # followed. B-105: the old check counted "use tool:" in the LAST
            # iteration's completion only — the metric was structurally zero.
            if getattr(fret, "failed_yet", False):
                retried = len(getattr(tu, "last_trajectory", [])) >= 2
                if retried:
                    retries += 1
            if _ok:
                correct += 1
                if retried:
                    converged += 1
    print(f"[agentic-eval] {args.ckpt} | n={n}")
    print(f"  retried-after-error: {retries}/{n} ({retries/n*100:.0f}%)")
    print(f"  correct-answer:      {correct}/{n} ({correct/n*100:.0f}%)")
    print(f"  retry-AND-correct:   {converged}/{n} ({converged/n*100:.0f}%)")


if __name__ == "__main__":
    main()
