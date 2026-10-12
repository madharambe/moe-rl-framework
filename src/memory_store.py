#!/usr/bin/env python3
"""ROUND-204af: the HUMAN-MIND memory axis (the user's design directive:
"only remember the important parts, just write down the not-so-important
parts").

A serve-time memory store with a human-memory contract:
  * IMPORTANT content -> `remember()`: kept in a bounded, in-window state
    (the "in-mind" list — the analog of what a human holds in working
    memory; capped, FIFO-evicted).
  * NOT-SO-IMPORTANT content -> `scribble()`: appended to the session's
    notes file (an external notebook — JSONL, never grows the context
    window; written down, not remembered).
  * `recall()`: flips the notebook when relevance demands it (keyword-
    overlap retrieval, the analog of a human consulting their notes).

v1 salience = an HONEST, documented heuristic (explicit markers, data/
code shapes, questions, repetition) with reason-tags so the scoring is
auditable. A learned salience-filter is a natural future flywheel product
(the notes themselves are SFT-able); v1 stays transparent.

Zero-GPU, thread-safe, append-only notes (atomic line writes), designed for
the serve loop (gui_server/query) behind an --memory flag (default OFF).
"""
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path

_CAP_SIZE = 64            # the in-mind working-memory cap
_MARKER_RE = re.compile(
    r"\b(remember|important|critical|key|note|keep in mind|don't forget|"
    r"essential|core|vital|mandatory|watchers|never|always)\b", re.I)
_QUESTION_RE = re.compile(r"\?")
_CODE_RE = re.compile(r"```|\bdef\s+\w+|\bclass\s+\w+|\bif\s+\w+\s*:|"
                       r"\$(?:LS|H|CMD|[A-Z]+)\b|\b\d{3,}\b")
_QUOTE_RE = re.compile(r"[\"'][A-Za-z0-9 _-]{6,}[\"']")


def salience(text: str) -> tuple[float, list[str]]:
    """Score a text segment for lasting-value. Returns (score, why) where
    why = the matched tags (honest + auditable)."""
    if not text or not isinstance(text, str):
        return 0.0, []
    n = len(text)
    tags = []
    score = 0.0
    if _MARKER_RE.search(text):
        score += 0.5
        tags.append("explicit-marker")
    if _QUESTION_RE.search(text):
        score += 0.15
        tags.append("question")
    if _CODE_RE.search(text) and n > 40:
        score += 0.3
        tags.append("code-or-datum")
    if _QUOTE_RE.search(text):
        score += 0.1
        tags.append("quoted")
    if n > 200:
        score += 0.2
        tags.append("long")
    words = re.findall(r"[a-z]{4,}", text.lower())
    if words and len(words) > 8:
        freq = len(words) / max(1, len(set(words)))
        if freq > 1.6:               # heavy lexical repetition = an anchor
            score += 0.15
            tags.append("repeated")
    return min(score, 0.999), sorted(set(tags))


class MemoryStore:
    """The human-mind memory. IMPORTANT -> in-mind; the rest -> the notes.

    Notes file = JSONL, append-only, atomic line writes, thread-safe.
    remember() = bounded FIFO (the working-memory analog).
    recall()  = keyword-overlap notebook lookup (v1; tf-vs-idf-free on
    purpose — honest and cheap; a learned retriever is future work)."""

    def __init__(self, notes_path: str | Path | None = None, cap: int = _CAP_SIZE):
        self.cap = cap
        self._lock = threading.Lock()
        self._mind: deque[dict] = deque(maxlen=cap)
        self._notes_path = Path(notes_path) if notes_path else None
        if self._notes_path:
            self._notes_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- the two lanes ----
    def process(self, role: str, text: str, force: str | None = None):
        """Route a segment: 'remember' | 'scribble' | None (auto).
        Auto: score >= 0.4 -> remember; else -> scribble (when a notes
        path exists; otherwise dropped, like a human not bothering)."""
        if not text or not text.strip():
            return
        if force == "remember":
            self.remember(text, role)
            return
        if force == "scribble":
            self.scribble(text, role)
            return
        score, why = salience(text)
        if score >= 0.4:
            self.remember(text, role, score, why)
        elif self._notes_path is not None:
            self.scribble(text, role, score, why)

    def remember(self, text: str, role: str = "assistant",
                 score: float | None = None, why: list[str] | None = None):
        """The working-memory lane: bounded FIFO (evictions = the human
        forgetting the least-recent). Always survives in-window."""
        with self._lock:
            if score is None:
                score, why = salience(text)
            self._mind.append({"t": time.time(), "role": role, "text": text,
                               "score": round(score, 3), "why": why or []})

    def scribble(self, text: str, role: str = "assistant",
                 score: float | None = None, why: list[str] | None = None):
        """The notebook lane: append-only line to the session notes JSONL.
        Never grows the context window; consulted on recall()."""
        if self._notes_path is None:
            return
        if score is None:
            score, why = salience(text)
        row = {"t": time.time(), "role": role,
               "score": round(score, 3), "why": why or [],
               "text": text[:2048]}
        with self._lock:
            with open(self._notes_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

    def recall(self, query: str, k: int = 8) -> list[dict]:
        """Human-style recall: BOTH lanes — the in-mind list (working
        memory) and the notebook (the written-down parts). Keyword-overlap
        scoring, newest ties first. v1 = cheap + honest (a learned
        retriever = future)."""
        qw = set(re.findall(r"[a-z]{4,}", (query or "").lower()))
        if not qw:
            return []
        hits = []
        with self._lock:
            for r in self._mind:
                rw = set(re.findall(r"[a-z]{4,}", (r.get("text") or "").lower()))
                ov = len(qw & rw)
                if ov >= 2:
                    hits.append((ov, r["t"], r))
            if self._notes_path is not None and self._notes_path.exists():
                try:
                    with open(self._notes_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                r = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            rw = set(re.findall(r"[a-z]{4,}",
                                                (r.get("text") or "").lower()))
                            ov = len(qw & rw)
                            if ov >= 2:
                                hits.append((ov, r["t"], r))
                except OSError:
                    pass
        hits.sort(key=lambda x: (-x[0], -x[1]))
        return [h[2] for h in hits[:k]]

    def mind(self) -> list[dict]:
        with self._lock:
            return list(self._mind)

    def stats(self) -> dict:
        n_notes = 0
        if self._notes_path and self._notes_path.exists():
            try:
                with open(self._notes_path, encoding="utf-8") as f:
                    n_notes = sum(1 for _ in f)
            except OSError:
                pass
        return {"in_mind": len(self._mind), "cap": self.cap,
                "notes_file": str(self._notes_path or ""), "notes_rows": n_notes}


def _self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        store = MemoryStore(Path(td) / "notes.jsonl")
        # important: explicit marker + datum
        store.process("user", "remember the helper is compute_area(x, y); it's critical for the whole chapter")
        # unimportant: filler
        store.process("assistant", "ok sounds good, we can adjust the color later if you want, minor tweaks only")
        # margin: long + code + question
        store.process("assistant", "def compute_area(x, y):\n    return x * y\n\nDoes the formula hold for negative inputs too? It should.")
        mind = store.mind()
        assert len(mind) >= 2, f"expected >=2 in mind, got {len(mind)}"
        hits = store.recall("how does compute_area work", k=3)
        assert hits and "compute_area" in hits[0].get("text", ""), "recall missed the important note"
        s = store.stats()
        assert s["notes_rows"] >= 1, "the filler should have been scribbled"
        print(f"[mem] SELFTEST OK — in_mind={s['in_mind']} notes={s['notes_rows']} "
              f"recalled={[h['text'][:28] for h in hits]}")


if __name__ == "__main__":
    _self_test()