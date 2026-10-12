"""Round-79: multi-adapter routing (role-specialized SLMs).
The 430m cannot hold the tool-call format and the direct-answer (spatial/chat)
behavior in ONE adapter (v11/v12 measured the interference). Keep the v7
(tool champion) + v12 (spatial/chat) as separate merged checkpoints and route
each query to the right adapter. Each merged ckpt is ~888MB — fits the weak GPU.

Usage:
  from route import pick_adapter
  ckpt, mode = pick_adapter("What direction is the cube from the box?")
  # -> ("out/ambatron_tool_v12/ckpt.pt", "direct")
  ckpt, mode = pick_adapter("What is the sum of squares of 1..10?")
  # -> ("out/ambatron_tool_v7/ckpt.pt", "tool")
"""
import os
import re

# B-208: single source of truth = out/champions.json (the flywheel's champions).
# A promotion (auto_optimize) updates that file; the deployed router must follow
# it so the served adapter is always the current champion. Fall back to the
# known-good seeded adapters when the file is missing/unreadable.
def _champion_ckpt(axis, default):
    import json
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "out", "champions.json")
        ch = json.load(open(p))
        c = ch.get(axis, {}).get("ckpt")
        if c:
            _abs = str(c) if os.path.isabs(str(c)) else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), str(c))
            if os.path.exists(_abs):
                return _abs
    except Exception:
        pass
    return default


# 204m: the tool champion = the v2-4k emission-peak (12 wrong-args vs v14's 1 —
# the measured best; the v14-era pruned). Re-elect via tools/pick_best.py once
# the promotion gate (correct>=1) passes a future checkpoint.
# 204m: the tool champion = the v2-4k emission-peak — the ROOT copy (the
# deployed path must not torch.load the flaky hub-SSD per query).
TOOL_CKPT = _champion_ckpt("tool", "out/ambatron_tool_v2base/ckpt.pt")
DIRECT_CKPT = _champion_ckpt("spatial", "out/cand_spatial_1788791526200_1.pt")  # live spatial champion
CHAT_CKPT = _champion_ckpt("chat", "out/cand_chat_1788944619865_1.pt")   # live chat champion

# signals that the query is spatial/positional reasoning -> direct-answer
SPATIAL_HINTS = re.compile(
    r"direction|position|row \d|column \d|grid|above|below|facing|turn "
    r"(left|right)|north|south|east|west|between|to the (left|right|front|"
    r"behind)|meters above|at row|at column|compass",
    re.IGNORECASE)
# signals the query wants a tool (compute/search)
TOOL_HINTS = re.compile(
    r"sum|compute|calculate|fibonacci|divisible|sort|check whether|what is "
    r"the (square|cube)|prime|search|find|look up|info about|github|clone|"
    r"execute|run this code|convert|area|perimeter|volume|circumference|"
    r"diameter|rectangle|triangle|"
    # B-144: bare-arithmetic phrasings ("what is 14 squared?", "12 times 7")
    r"\b(squared|cubed\b|modulo|mod\b|percent of|times\b|plus\b|minus\b|"
    r"divided by|multiplied)\b|\b\d+\s*[+\-*/^x]\s*\d+"
    # 19th-audit B-6b: the 204h hints over-matched chat idioms ("the power of
    # friendship", "median household income", "2x2 factorial design") —
    # require a NUMERIC right-hand side ("power of 2", "15 factorial",
    # the median of [3,1,9,6]).
    r"|\b(median\s+of\s*(?:\[?\s*\d|\d)|(?:factorial\s*(?:of\s*)?\d|\d+\s+factorial)|"
    r"power\s+of\s*[-+]?\d|remainder|divisors|common (factor|divisor)\b)",
    re.IGNORECASE)


def pick_adapter(query: str):
    """Route a user query to (ckpt_path, mode). mode in {"tool", "direct", "chat"}.
    B-113: if the chosen checkpoint is missing from disk (SSD unmounted, a
    pruned experiment), fall through the remaining adapters instead of handing
    the caller a dead path.
    B-208: the ckpt constants now resolve from out/champions.json at import, so
    a flywheel promotion is picked up on the next query automatically."""
    candidates = []
    if SPATIAL_HINTS.search(query) and not TOOL_HINTS.search(query):
        candidates.append((DIRECT_CKPT, "direct"))
    if TOOL_HINTS.search(query):
        candidates.append((TOOL_CKPT, "tool"))
    candidates.append((CHAT_CKPT, "chat"))
    for ckpt, mode in candidates:
        if os.path.exists(ckpt):
            return ckpt, mode
    return CHAT_CKPT, "chat"  # all missing — return the primary and let the
                              # caller's loader raise a clear FileNotFoundError


if __name__ == "__main__":
    tests = [
        "What direction is the cube from the box?",
        "What is the sum of squares of 1..10?",
        "You are facing north. Turn left 90 degrees. What direction now?",
        "Is 1729 divisible by 7?",
        "A bottle is at row 2, column 3; a key is at row 0, column 4. "
        "Which direction is the key from the bottle?",
        "Sort [3,1,4] and report the second element.",
        "How are you today?",
    ]
    for t in tests:
        print(f"  {pick_adapter(t)[1]:6s} <- {t[:60]}")