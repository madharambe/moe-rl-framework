#!/usr/bin/env python3
"""ROUND-204i: convert the verified TEACHER TRACES (the exec-verified
18-probe pool — query/tool/arguments/answer) into SFT segments in the
deployed tool format, so the distilled/teacher data joins the SFT pool as
first-class training records (the objective's distillation axis: only
execution-verified wins get absorbed).

  python3 src/teacher_to_sft.py --in data/processed/teacher_traces.jsonl \
      --out data/processed/teacher_traces_sft.jsonl

Records gain the full conversation shape:
  system (deployed envelope) / user (query) / assistant json-call /
  user ("The result is ...") / assistant (the answer).
"""
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

SYS = ("Available tools:\n  - execute_python(code: str, timeout_s: float) \u2014 Execute Python code and return output\n"
       "  - search_knowledge(query: str) \u2014 Search internal knowledge base for information\n"
       "To use a tool, respond with:\nuse tool:{\"name\": \"<tool>\", \"arguments\": {...}}")


def to_segments(rec):
    q = rec.get("query", "")
    tool = rec.get("tool", "execute_python")
    args = rec.get("arguments", {})
    code = args.get("code", "")
    answer = rec.get("answer")
    a_text = str(answer) if answer is not None else "done"
    return [
        ["system", SYS],
        ["user", q],
        ["assistant", 'use tool:{"name": "%s", "arguments": {"code": %s}}'
                       % (tool, json.dumps(code))],
        ["user", "The result is:"],
        ["assistant", a_text],
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", default=str(REPO / "data/processed/teacher_traces_sft.jsonl"))
    args = ap.parse_args()

    from heldout import is_heldout
    n = purged = 0
    with open(args.out, "w") as fo:
        for line in open(args.src):
            rec = json.loads(line)
            try:
                rec["segments"] = to_segments(rec)
            except Exception:
                purged += 1
                continue
            txt = " ".join(str(s[1]) for s in rec["segments"])
            if is_heldout(txt):
                purged += 1
                continue
            fo.write(json.dumps(rec) + "\n")
            n += 1
    print(f"[teacher_to_sft] {n} traces converted (purged {purged}) -> {args.out}")


if __name__ == "__main__":
    main()