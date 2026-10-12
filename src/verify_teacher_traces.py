#!/usr/bin/env python3
"""ROUND-186 (teacher distillation): verify teacher traces with EXECUTION.

Teacher (opencode/frontier) traces are claims, not truth — every trace's code
must execute in the sandbox and produce the expected answer. Only verified
records pass. Format (jsonl):
    {"query": "...", "tool": "execute_python", "arguments": {"code": "..."},
     "answer": 42, "domain": "code"}
Usage: src/verify_teacher_traces.py teacher_traces.jsonl -o verified.jsonl
"""
import argparse, json, sys, io
sys.path.insert(0, __file__.rsplit("/", 1)[0] or ".")
from tool_use import ToolRegistry, _execute_python, _search_knowledge


def check(tool, arguments, answer):
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    try:
        r = str(reg.call(tool, arguments)).strip()
    except Exception as e:
        return False, f"exec error: {e}"
    if any(m in r.lower() for m in ("[timed out", "[truncated", "sandbox",
                                    "traceback", "exited with code", "error")):
        return False, r[:120]
    if isinstance(answer, bool):
        return (r.lower() in ("true", "false") and r.lower() == str(answer).lower()), r[:120]
    if isinstance(answer, str):
        ans = answer.lower().strip()
        rl = r.lower().strip()
        import re as _re
        if len(ans) <= 3:
            return ans == rl or _re.match(r"^\W*" + _re.escape(ans) + r"\W*$", rl) is not None, r[:120]
        return ans in rl, r[:120]
    nums = [float(n) for n in __import__("re").findall(r"[-+]?\d*\.?\d+", r)]
    return (bool(nums) and abs(nums[-1] - answer) < 1e-6), r[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()
    ok, bad = 0, 0
    with open(args.out, "w") as wf:
        for line in open(args.inp):
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            passed, why = check(e.get("tool", "execute_python"),
                                e.get("arguments", {}), e.get("answer"))
            if passed:
                wf.write(json.dumps(e) + "\n")
                ok += 1
            else:
                bad += 1
                print(f"[teacher] REJECTED {e.get('query','')[:60]} -> {why}", flush=True)
    print(f"[teacher] verified={ok} rejected={bad} -> {args.out}")
    raise SystemExit(0 if ok >= 1 else 1)


if __name__ == "__main__":
    main()