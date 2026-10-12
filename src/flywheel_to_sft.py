"""
Flywheel -> SFT converter (B-46): turns the agent's own tool trajectories
(out/agent_trajectories.jsonl) into SFT records, reject-filtered.

This is the RSI synthetic-data engine (phi-1 / TinyStories-style: self-generated
data matched to the model): only trajectories with a WELL-FORMED tool call to a
KNOWN tool survive; garbage/loop-detected calls are rejected.

Records match the exact build_tool_sft format (system section from the real
registry, User:/Assistant:/result: markers), so train_tool_sft.py consumes them
directly.

Usage: python3 src/flywheel_to_sft.py [--min-trajectories 1] [--append]
  (--append: append to data/processed/tool_sft.jsonl; default: dry-run report)
"""
import argparse, json, sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FLYWHEEL = BASE / "out" / "agent_trajectories.jsonl"
# ROUND-204q: the OUT moved to the ROOT-side v2-bank (data/processed = the
# SSD symlink — 0-bytes free; and the flywheel append must feed the SAME
# pool the trainer reads: v2-bank/tool_sft_flywheel.jsonl joins the builder).
OUT = BASE / "v2-bank" / "tool_sft_flywheel.jsonl"


def _tools_section():
    from tool_use import ToolRegistry, _execute_python, _search_knowledge
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    # 8th-wave B (ghost-tool class): the deployed envelope advertises ONLY
    # the 2 tools — the github-tool registration created training records for
    # tools the live agent rejects as "Error: unknown tool".
    return reg


def convert_entries(entries, reg):
    """Return (records, rejected) — reject-filtered SFT records."""
    known = set(reg._tools)
    section = reg.get_tools_section(compiled=True)  # 204m: the deployed/pool envelope parity
    records, rejected = [], 0
    # 14th-audit P0: skip any trajectory whose query is an eval-HELDOUT probe, so
    # self-generated eval queries can never leak back into training (eval ->
    # agent_trajectories -> this converter -> SFT -> eval inflates). Lowercase both.
    from heldout import is_heldout  # ROUND-174: shared 32-set (single source)
    for e in entries:
        query = e.get("query", "")
        if is_heldout(query):
            rejected += 1
            continue
        calls = e.get("calls", [])
        if not query or not calls:
            rejected += 1
            continue
        if len(calls) > 3:  # loop-ish / degenerate multi-call
            rejected += 1
            continue
        # build the segment chain: system, user, assistant(call), tool(result)
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
        # final assistant answer: paraphrase the authoritative tool result so
        # the model imitates a REAL answer, not the old degenerate "Done."
        # (a "Done." final turn taught the model to acknowledge instead of
        # answering). Mirrors the ReST-winner format "The result is ...".
        # 18th-audit: drop tool-ERROR observations (loop-detected / exception /
        # timeout / "error") from the final answer — teaching "The result is
        # {'error': 'Loop detected'}" trains degenerate give-up output.
        _err_markers = ("error", "loop detected", "exception", "traceback",
                        "timeout", "timed out", "failed", "denied", "unavailable",
                        # Sep-5 (B3 audit): search_knowledge MISS observations and
                        # sandbox-exit/truncation messages are real failure shapes
                        # the original list blind-spotted — otherwise an episode
                        # whose only "answer" is a knowledge-miss got emitted as
                        # "The result is No relevant knowledge found." (dogfoods
                        # the give-up output the rejection exists to kill).
                        "no relevant", "sandbox process exited", "truncated")
        last_obs = ""
        for c in calls:
            obs = str(c.get("observation", "")).strip()
            if obs and not any(m in obs.lower() for m in _err_markers):
                last_obs = obs
        if last_obs:
            segs.append(("assistant",
                         f"Assistant: The result is {last_obs[:300]}"
                         f"{'...' if len(last_obs) > 300 else ''}"))
        else:
            # no healthy observation: reject the episode rather than emit a
            # give-up "Done." (an all-error trajectory is not training signal)
            rejected += 1
            continue
        records.append({"segments": segs})
    return records, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true",
                    help="append the converted records to tool_sft.jsonl")
    args = ap.parse_args()

    if not FLYWHEEL.exists():
        print(f"[flywheel] {FLYWHEEL} not found")
        return
    entries = [json.loads(l) for l in open(FLYWHEEL, encoding="utf-8") if l.strip()]
    reg = _tools_section()
    records, rejected = convert_entries(entries, reg)
    print(f"[flywheel] {len(entries)} trajectories -> {len(records)} usable "
          f"(rejected {rejected})")
    if args.append and records:
        # ROUND-172 (D1): the old append re-wrote the ENTIRE trajectory log
        # every run — tool_sft_compiled.jsonl measured 1115 lines / 404 unique
        # (64% dupes), skewing every downstream SFT record frequency. Dedup by
        # record content-hash against the existing file (no sidecar state; both
        # converters can share the source without fighting over offsets).
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
        print(f"[flywheel] appended {_added} NEW records (deduped "
              f"{len(records) - _added} existing) -> {OUT}")
    elif records:
        print("[flywheel] dry-run: pass --append to write (records shown below)")
        for r in records[:2]:
            print("  ", r["segments"][:3])


if __name__ == "__main__":
    main()