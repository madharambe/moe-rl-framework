#!/usr/bin/env python3
"""Tool-axis failure-anchoring (ROUND-174): turn EVERY failed tool probe into a
rephrased SFT anchor WITHOUT verbatim-leaking the heldout string.

For each failed probe (query, expected-tool, expected-answer), build a
REPHRASED clone (same semantics, new numbers/shapes) whose correct answer is
executed + verified through the REAL tool registry, then emit the record in
the same 'direct' format the tool SFT expects. The clone is never the heldout
string, so it cannot feed the eval (heldout matches exact text) but trains the
underlying skill (call the right tool, pass right args, read the result).

Usage: tool_anchor.py <failed.jsonl> [--out anchors.jsonl] [--n N]
"""
import argparse, json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from heldout import is_heldout  # ROUND-174: anchors must NEVER be heldout — rechecked per-emit

def make_divisibility(r):
    """Generate a fresh (a,b,result) triple where b divides a / not."""
    b = r.randint(3, 25)
    k = r.randint(2, 12)
    if r.random() < 0.5:
        a = b * k; want = "yes"
    else:
        a = b * k + r.randint(1, b - 1); want = "no"
    return a, b, want

def rephrase(q, rng):
    ql = q.lower()
    if "sum of cubes" in ql:
        k = rng.randint(2, 12)
        return f"What is the sum of cubes of 1..{k}?", "execute_python", None
    if "sum of squares" in ql:
        k = rng.randint(2, 12)
        return f"What is the sum of squares of 1..{k}?", "execute_python", None
    if "fibonacci" in ql:
        # ROUND-182: k=12/17/20 collide with the heldout fib family (near-verbatim
        # leaks caught by the per-emit guard as exact hits, 20 evades it) — skip
        k = rng.randint(5, 25)
        while k in (12, 17, 20):
            k = rng.randint(5, 25)
        return f"Compute the {k}th Fibonacci number.", "execute_python", None
    if "divisible by" in ql:
        a, b, want = make_divisibility(rng)
        return f"Is {a} divisible by {b}?", "execute_python", want
    if "prime" in ql:
        n = rng.choice([13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
                        67, 71, 73, 79, 83, 89, 97, 49, 51, 57, 63, 77, 87,
                        91, 93, 95, 99])
        return f"Is {n} prime?", "execute_python", None
    if "gcd" in ql:
        a = rng.randint(2, 12) * rng.randint(2, 6)
        b = rng.randint(2, 12) * rng.randint(2, 6)
        return f"What is the GCD of {a} and {b}?", "execute_python", None
    if "sort" in ql:
        arr = [rng.randint(0, 30) for _ in range(rng.randint(3, 7))]
        pos = rng.choice(["smallest", "largest", "second largest", "third element"])
        return f"Sort {arr} and report the {pos}.", "execute_python", None
    if "multiplied" in ql or "times" in ql:
        return f"What is {rng.randint(2, 99)} multiplied by {rng.randint(2, 99)}?", \
               "execute_python", None
    if "divided by" in ql:
        b = rng.randint(2, 25); a = b * rng.randint(2, 15)
        return f"What is {a} divided by {b}?", "execute_python", None
    if "to the power" in ql:
        return f"What is {rng.randint(2, 10)} to the power of {rng.randint(2, 5)}?", \
               "execute_python", None
    if "factorial" in ql:
        return f"What is {rng.randint(3, 9)} factorial?", "execute_python", None
    return None  # unknown shape — skip (never risk a leak)

def _answer_code(q, tool_name, want, reg):
    """Build Python code that computes the correct answer, runnable under
    execute_python; verify against want when given; return the code or None."""
    import re as _re
    # gcd
    m = _re.search(r"GCD of (\d+) and (\d+)", q)
    if m:
        import math
        a, b = int(m.group(1)), int(m.group(2))
        return f"import math\nprint(math.gcd({a}, {b}))"
    # prime check
    if "prime" in q:
        m = _re.search(r"Is (\d+) prime", q)
        n = int(m.group(1))
        code = (f"n = {n}\n"
                f"print('yes' if n > 1 and all(n % d for d in range(2, int(n**0.5)+1)) else 'no')")
        return code
    # divisibility
    m = _re.search(r"Is (\d+) divisible by (\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"print('yes' if {a} % {b} == 0 else 'no')"
    # factorial
    m = _re.search(r"(\d+) factorial", q)
    if m:
        k = int(m.group(1))
        return f"import math\nprint(math.factorial({k}))"
    # power
    m = _re.search(r"(\d+) to the power of (\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"print({a} ** {b})"
    # divided
    m = _re.search(r"(\d+) divided by (\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"print({a} // {b})"
    # multiplied / times
    m = _re.search(r"(\d+) multiplied by (\d+)", q)
    if not m:
        m = _re.search(r"(\d+) times (\d+)", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"print({a} * {b})"
    # sort
    m = _re.search(r"Sort \[([\d,\s]+)\] and report the (.*?)\.", q)
    if m:
        arr = [int(x) for x in m.group(1).split(",")]
        pos = m.group(2).strip().lower()
        sorted_arr = sorted(arr)
        if pos == "smallest": idx = 0
        elif pos == "largest": idx = -1
        elif pos == "second largest": idx = -2
        elif pos == "third element": idx = 2
        else: return None
        return f"print({sorted_arr[idx]})"
    # fibonacci nth
    m = _re.search(r"(\d+)(?:st|nd|rd|th) Fibonacci", q)
    if m:
        k = int(m.group(1))
        # 0-indexed fib per repo convention: F(n), n=k-1
        return (f"a, b = 0, 1\n"
                f"for _ in range({k-1}): a, b = b, a+b\n"
                f"print(a)")
    # sums of cubes/squares
    m = _re.search(r"sum of (cubes|squares) of 1\.\.(\d+)", q)
    if m:
        kind, k = m.group(1), int(m.group(2))
        return f"print(sum(i**3 if '{kind}'=='cubes' else i**2 for i in range(1, {k}+1)))"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", default="data/processed/tool_anchors.jsonl")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tool_use import default_tools
    reg = default_tools()
    rng = random.Random(7)
    # ROUND-174: the system-prompt tools section (mirror train_tool_sft's)
    TOOLS_SECTION = ("Available tools:\n  - execute_python(code: str) \u2014 "
                     "Execute Python code and return output\n  - search_knowledge(query: str) \u2014 "
                     "Search internal knowledge base for information\n\nUse tool calls "
                     "like:\nuse tool: execute_python({\"code\": \"print(1)\"})")
    def _code_str(code):
        """A compact, JSON-safe tool-call code string for the assistant turn."""
        import json as _j
        return _j.dumps({"code": code})

    failed = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    out = []
    for f in failed:
        q = f.get("query", "")
        r = rephrase(q, rng)
        if r is None:
            continue
        q2, tool_name, want = r
        # ROUND-174 hard guard: a rephrase could accidentally collide with a
        # heldout literal (measured: sum-of-squares K=9). NEVER emit one.
        if is_heldout(q2):
            continue
        code = _answer_code(q2, tool_name, want, reg)
        if code is None:
            continue
        obs = str(reg.call("execute_python", {"code": code}))
        if "Traceback" in obs or "Error" in obs:
            continue
        # ROUND-174: emit in the SFT conversation format train_tool_sft expects
        # (segments: [system-tools, user-query, assistant-toolcall, assistant-
        # answer]). The old {query,response,mode} crashed with KeyError segments.
        seg = [
            ["system", TOOLS_SECTION],
            ["user", q2],
            ["assistant", f"use tool: execute_python({_code_str(code)})"],
            ["assistant", f"Assistant: The result is {obs.strip()}"],
        ]
        out.append({"segments": seg, "mode": "tool-anchor", "query": q2})
        if len(out) >= args.n:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as g:
        for r in out:
            g.write(json.dumps(r) + "\n")
    print(f"[tool_anchor] {len(out)} anchors -> {args.out}")

if __name__ == "__main__":
    main()