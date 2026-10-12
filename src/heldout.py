"""Shared heldout-query registry — single source of truth for the eval
heldout set (CASES_HELDOUT in eval_tool_use.py). Consumers (tool_use,
flywheel_to_sft, agentic_to_sft, query) import this instead of maintaining
their own lists, so the set can never drift again (it did in ROUND-174 and
contaminated 245 trajectory entries).

Contains: full-text arithmetic/verification probes + keyword rules for the
knowledge/currency probes. Use is_heldout(q) everywhere.
"""

from eval_tool_use import CASES_HELDOUT
from hard_suite import gen as _hard_gen
# the hard-suite generator computes its own expected answers (no typos)
HARD_CASES = _hard_gen()

# literal probes (lowercase) that must never log/train
_LITERAL: frozenset[str] = frozenset(
    q.lower() for q, exp in CASES_HELDOUT if exp[0] != "search_knowledge" and exp[0] is not None
)

# keyword rules for the knowledge probes
# ROUND-204g-audit: the HARD suite (29 execution-verified probes) leaked
# into agent_trajectories.jsonl (228 entries!) because it wasn't part of this
# registry — any eval suite must be here or converters let it train again.
_HARD_LITERAL: frozenset[str] = frozenset(
    q.lower() for q in (h[0] for h in HARD_CASES)
)

# 19th-audit B-2: a verb-normalized layer. The EXACT-match purge (228) could
# not see paraphrases ("Is 221 divisible by 17?" vs "Check whether 221 is
# divisible by 17." — same number/modulus/answer, one verb swapped), which
# leaked hard probes into training THROUGH the rest/gym pools. Normalize:
# lowercase, drop leader verbs (is/are/check/what), strip non-alnum — the
# remaining fingerprint is the same for an exact probe and its paraphrase.
import re as _re_norm

_LEADER_RE = _re_norm.compile(
    r"^(?:check\s+(?:whether\s+)?|is\s+|are\s+|what\s+is\s+|what\s+are\s+|"
    r"please\s+(?:check\s+)?(?:whether\s+)?|verify\s+(?:whether\s+)?)+", _re_norm.I)


def _norm_fp(q: str) -> str:
    # verb-morphology: strip the standalone verbs on WHITESPACE boundaries
    # (before the non-alnum collapse), so "221 is divisible by 17" and
    # "221 divisible by 17" fingerprint identically. Interior "is" in words
    # like "distance" is untouched by \b boundaries.
    s = _LEADER_RE.sub("", (q or "").lower())
    s = _re_norm.sub(r"\bis\b|\bare\b|\bcheck\b|\bwhether\b", "", s)
    return _re_norm.sub(r"[\W_]+", "", s)


_NORM_LITERAL: frozenset[str] = frozenset(
    _norm_fp(q) for q in _LITERAL) | frozenset(_norm_fp(q) for q in _HARD_LITERAL)

_KEYWORDS: tuple[str, ...] = (
    # 19th-audit: -boundary classes — the raw "ppo"/"sac" substrings
    # banned "hippo", "appointment" etc. from logging (verified FPs).
    r"\bppo\b", r"\bsac\b", "mixture of experts", "gating in a mixture",
    "currency of south korea",
)


def is_heldout(q: str) -> bool:
    ql = (q or "").lower()
    if ql in _LITERAL or ql in _HARD_LITERAL:
        return True
    # 19th-audit B-2: paraphrased-probe containment on the normalized
    # fingerprint, gated to probes that LOOK like verification queries (the
    # divergence must be < 25% — a long knowledge query won't false-positive).
    if len(ql) < 200:
        _fp = _norm_fp(ql)
        if len(_fp) >= 8 and _fp in _NORM_LITERAL:
            return True
        # 19th-audit (6th-wave P1): REVERSE containment — a probe's
        # fingerprint CONTAINED in a longer query ("Compute the 12th Fibonacci
        # number and report it." supersets the heldout "What is the 12th
        # Fibonacci number?") escaped the exact-set check; 320 rows logged +
        # 11 rows survived into the pool purge. A heldout fingerprint as a
        # SUBSTRING of the query fingerprint = the same probe embedded.
        # 8th-wave B: the digit-BOUNDARY guard — the reverse-containment must
        # not fire when the embedded fingerprint starts/ends inside a NUMBER
        # ("50 divisible by 5" contains the fp "0divisibleby5" — a FP class).
        if len(_fp) >= 16:
            for _h in _NORM_LITERAL:
                if not (8 <= len(_h) <= len(_fp) - 4):
                    continue
                _pos = _fp.find(_h)
                if _pos < 0:
                    continue
                _left = _fp[_pos - 1:_pos] if _pos > 0 else ""
                _right = _fp[_pos + len(_h): _pos + len(_h) + 1]
                if (_h[0].isdigit() and _left.isdigit()):
                    continue
                if (_h[-1].isdigit() and _right.isdigit()):
                    continue
                return True
    import re as _re_kw
    return any(_re_kw.search(k, ql) if k.startswith("\\b") or k.startswith(r"\b")
               else k in ql for k in _KEYWORDS)


HELDOUT_LITERAL = _LITERAL
HELDOUT_KEYWORDS = _KEYWORDS
