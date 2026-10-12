"""B-72: agentic flywheel -> SFT converter.

Turns the agent's OWN multi-step tool trajectories (out/agent_trajectories.jsonl)
into full AGENTIC episodes for the Section-0 objective (self-improving model):
keep the ENTIRE loop (attempts, errors, retries, successes) so the model learns
"read the result -> correct on error -> converge", plus a VERIFIED final answer
extracted from the last successful call (not the noisy raw completion), plus a
per-domain tag so the model can specialize by usage (robotics/law/math/...).

Usage:
  python3 src/agentic_to_sft.py                # dry-run report
  python3 src/agentic_to_sft.py --append       # append to data/processed/agentic_sft.jsonl
"""
import argparse, json, re
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FLYWHEEL = BASE / "out" / "agent_trajectories.jsonl"
OUT = BASE / "data" / "processed" / "agentic_sft.jsonl"

MAX_CALLS = 8


def _tools_section():
    from tool_use import ToolRegistry, _execute_python, _search_knowledge
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    try:
        from github_tools import build_github_tools
        build_github_tools(reg)
    except Exception:
        pass
    return reg


ERROR_HINTS = ("error", "traceback", "failed", "exception", "not found",
               "does not", "syntax", "no relevant", "invalid",
               # ROUND-5 (F-11): the sandbox/timeout shapes flywheel_to_sft got
               # (B3) were never mirrored here — an all-timeout episode was
               # emitted as "[timed out after Ns]" training signal.
               "timed out", "sandbox process exited", "loop detected",
               "truncated")


def _is_error(obs: str) -> bool:
    o = obs.lower()
    return any(h in o for h in ERROR_HINTS)


def _last_good(calls):
    for c in reversed(calls):
        if not _is_error(str(c.get("observation", ""))):
            return c
    return None


def _verified_answer(c) -> str:
    """A clean, verifiable final answer from the last good call's observation.
    B-114: the old version guessed m[-1] from ANY numeric observation — for
    'Sorted: [1, 3, 4]' asked about the second element it labeled '4'. Now:
    single-number observations are trusted; multi-number ones keep the raw
    observation text (never a guessed number)."""
    if c is None:
        return None
    obs = str(c.get("observation", "")).strip()
    name = c.get("name", "")
    if not obs or obs.lower() in ("code executed successfully (no output)", "no output"):
        return None
    if name == "execute_python":
        m = re.findall(r"[-+]?\d+", obs)
        if len(m) == 1:
            return f"Assistant: The result is {m[0]}."
        # multi-number: too ambiguous to guess — keep the observation itself
        return f"Assistant: {obs[:200]}"
    return f"Assistant: {obs[:200]}"


def _domain(query: str, calls) -> str:
    q = query.lower()
    toks = " ".join(str(c.get("name", "")) for c in calls).lower()
    if re.search(r"\d", q) and any(w in q for w in ("sum", "compute", "calculat",
                                                    "fibonacci", "count", "number",
                                                    "divisible", "square", "cube")):
        return "math"
    if any(w in q for w in ("robot", "vla", "nav", "sensor", "motor", "imu",
                            "physics", "goal")):
        return "robotics"
    if any(w in q for w in ("law", "legal", "court", "contract", "statute")):
        return "law"
    if any(w in q for w in ("code", "python", "sort", "list", "function",
                            "script", "bug", "refactor")) or "github" in toks:
        return "code"
    return "general"


def convert_entries(entries, reg):
    known = set(reg._tools)
    section = reg.get_tools_section()
    records, rejected = [], 0
    # 14th-audit P0: heldout-probe filter — never let eval queries leak to training.
    from heldout import is_heldout  # ROUND-174: shared 32-set
    if not is_heldout or not callable(is_heldout):
        raise RuntimeError("heldout guard unavailable")
    for e in entries:
        query = e.get("query", "")
        if is_heldout(query or ""):
            rejected += 1
            continue
        calls = e.get("calls", [])
        if not query or not calls:
            rejected += 1
            continue
        calls = calls[:MAX_CALLS]
        segs = [("system", section), ("user", f"User: {query}\n")]
        ok = True
        for c in calls:
            name = c.get("name", "")
            args = c.get("arguments")
            obs = c.get("observation", "")
            if name not in known or not isinstance(args, dict):
                ok = False
                break
            segs.append(("assistant",
                         f"Assistant: use tool:{json.dumps({'name': name, 'arguments': args})}"))
            segs.append(("tool", f"result: {str(obs)[:500]}\n"))
        if not ok:
            rejected += 1
            continue
        # B-114: reject episodes where EVERY call errored — the old flow
        # labeled them "Assistant: Done.", teaching give-up after failure
        answer = _verified_answer(_last_good(calls))
        if answer is None:
            rejected += 1
            continue
        segs.append(answer)
        records.append({"segments": segs, "domain": _domain(query, calls)})
    return records, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    if not FLYWHEEL.exists():
        print(f"[agentic] {FLYWHEEL} not found")
        return
    entries = [json.loads(l) for l in open(FLYWHEEL, encoding="utf-8") if l.strip()]
    reg = _tools_section()
    records, rejected = convert_entries(entries, reg)
    from collections import Counter
    doms = Counter(r["domain"] for r in records)
    print(f"[agentic] {len(entries)} trajectories -> {len(records)} episodes "
          f"(rejected {rejected}) | domains: {dict(doms)}")
    if args.append and records:
        # ROUND-172 (D1): dedup by record hash against the existing file
        # (agentic_sft.jsonl measured 333/201 = 40% dupes from re-append)
        import hashlib as _hl
        _seen = set()
        if OUT.exists():
            for _l in open(OUT, encoding="utf-8"):
                if _l.strip():
                    try:
                        _seen.add(_hl.md5(_l.strip().encode("utf-8")).hexdigest())
                    except Exception:
                        pass
        _added = 0
        with open(OUT, "a", encoding="utf-8") as f:
            for r in records:
                _line = json.dumps(r)
                _h = _hl.md5(_line.encode("utf-8")).hexdigest()
                if _h in _seen:
                    continue
                _seen.add(_h)
                f.write(_line + "\n")
                _added += 1
        print(f"[agentic] appended {_added} NEW (deduped {len(records) - _added}) -> {OUT}")
    else:
        print("[agentic] dry-run: pass --append to write (sample below)")
        for s in records[0]["segments"][:5]:
            print("  ", str(s)[:110])


if __name__ == "__main__":
    main()
