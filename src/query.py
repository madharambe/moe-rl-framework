"""B-144: query.py -- THE deployment entry point.
One command that routes a query through pick_adapter and answers using the
same inference path each evaluator trusts:

  python3 src/query.py "What is the sum of squares of 1..7?"
  python3 src/query.py --interactive      # REPL; /quit exits
  python3 src/query.py "..." --max_tokens 80 --temperature 0.7

Tool-mode queries run through ToolUseMoE (compiled schema, tool execution);
direct/chat modes generate straight from their adapters. Every interaction
appends to out/agent_trajectories.jsonl for the flywheel (tool mode does
this natively; chat/direct are appended here).
"""
import argparse, sys, pathlib, os, re as _re
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# module-level (testable): the deterministic arithmetic fast-path (B-166) and
# its partial-answer guard (Sep-5 audit + the 19th-audit remainder guard).
_ARITH = _re.compile(
    # 204aa: the prefix is OPTIONAL — the bare form ("18 plus 9") is as real
    # as "what is 18 plus 9"; the operand validation still rejects the
    # phrase-class ("2019 plus a year" -> operand 'a' -> None).
    r"(?:what(?:'s| is|s| are)|compute|calculate)?\s*(-?[\d,]+)\s*"
    r"(squared|cubed|times|multiplied by|x|plus|minus|divided by|modulo|mod|% of)\s*(-?[\d,]*)",
    _re.IGNORECASE)


def _try_arith(q: str):
    """B-166: deterministic arithmetic fast-path. Three adapter
    generations (narrow/wide/frozen-prompt, 15k steps) all proved the
    430m cannot reliably emit tool-call JSON for arithmetic -- but this
    transform needs no model at all. Compute directly; return None to
    fall through to normal routing."""
    m = _ARITH.search(q)
    if not m:
        # ROUND-204z: the BARE form ("18 plus 9" / "17 times 2" / "whats 8
        # divided by 2") — the regex needs a WHAT/COMPUTE/CALCULATE prefix,
        # so a just-the-numbers query fell through to the tool-loop, which
        # emitted raw-stream garbage instead of computing. The fast-path
        # makes the bare math first-class (the deployed GUI's most common
        # probe class — the honest measurement of the "flexible" ask).
        _bare = _re.search(
            r"^\s*(-?\d[\d,]*)\s*(squared|cubed|times|multiplied by|x|"
            r"\btimes\b|plus|minus|divided by|\bmod(?:ulo)?\b|\bmod\b)"
            r"\s*(-?\d[\d,]*)?\s*$", q, _re.I)
        if _bare:
            _a = _num(_bare.group(1))
            _b = _num(_bare.group(3)) if _bare.group(3) else None
            _op = _bare.group(2).lower()
            if _op in ("squared", "cubed"):
                return f"[arith] {_a ** (2 if _op == 'squared' else 3)}"
            if _op == "times" or _op == "multiplied by" or _op == "x":
                if _b is None:
                    return None
                return f"[arith] {_a * _b}"
            if _op == "plus":
                if _b is None:
                    return None
                return f"[arith] {_a + _b}"
            if _op == "minus":
                if _b is None:
                    return None
                return f"[arith] {_a - _b}"
            if _op == "divided by":
                if _b in (None, 0):
                    return None
                return f"[arith] {_a / _b:.8g}"
            if "mod" in _op:
                if _b in (None, 0):
                    return None
                return f"[arith] {_a % _b}"
        return None
    # Sep-5 (audit): a SECOND operator in the sentence means the regex
    # matched only the first pair and would return a PARTIAL answer
    # ("3 times 4 minus 2" -> 12 instead of 10). Refuse (fall through to
    # model routing) when another operator token appears after the match.
    _rest = q[m.end():].lower()
    if _re.search(r"(?:squared|cubed|times|multiplied\s+by|\bm[ai]nus\b|"
                  r"divided\s+by|\bmod(?:ulo)?\b|\bto\s+the\s+power\b|"
                  r"%\s+of|\bx\b|remainder)", _rest):
        return None
    a = float(m.group(1).replace(",", ""))
    op = m.group(2).lower()
    b_s = m.group(3).replace(",", "")
    # ROUND-170 (MED): a MISSING operand ("what is 5 times") must NOT fall
    # through to b=0 (the old `(-?[\d,]*)` allowed empty -> returned a
    # confident WRONG answer like "[arith] 0"). squared/cubed are unary and
    # legal without an operand; binary ops require one.
    # ROUND-172 (F1, verified): a DANGLING operator ("what is 5 minus -")
    # made float("-") raise an uncaught ValueError. Validate the operand.
    if op not in ("squared", "cubed"):
        if not b_s or not _re.fullmatch(r"-?\d+", b_s):
            return None
    b = float(b_s) if b_s else 0.0
    try:
        if op == "squared":
            r = a * a
        elif op == "cubed":
            r = a ** 3
        elif op in ("times", "multiplied by", "x"):
            r = a * b
        elif op == "plus":
            r = a + b
        elif op == "minus":
            r = a - b
        elif op == "divided by":
            r = a / b
        elif op in ("modulo", "mod"):
            r = a % b
        else:  # "% of"
            r = a / 100.0 * b
    except ZeroDivisionError:
        return "[arith] undefined (division by zero)"
    r_i = int(r)
    out = str(r_i) if abs(r - r_i) < 1e-9 else f"{r:.4f}".rstrip("0").rstrip(".")
    return f"[arith] {out}"


def load_adapter(ckpt, compiled=False):
    import torch
    import torch.serialization as ser
    from moe_model import MoEGPT, MoEConfig
    ser.add_safe_globals([MoEConfig])
    from tokenizers import Tokenizer
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    # 204ah-27: the tokenizer must match the CKPT's OWN vocab — the old fixed
    # 16K bpe tokenizer mismatched every v3 (32K) champion (the same bug class
    # as the GRPO/eval_agentic fixes).
    _v3 = int(getattr(ck["config"], "vocab_size", 0)) >= 32768
    tok = Tokenizer.from_file(str(ROOT / ("v3-bank/tokenizer_v3.json" if _v3
                                          else "data/processed/bpe_tokenizer/tokenizer.json")))
    m = MoEGPT(ck["config"]).eval()
    # 205ab: strict=False — the v3 champions are head-TIED; their saved
    # model_state omits lm_head.weight and the strict load crashed every
    # served query (round-169 fixed the evals; the served path missed it).
    m.load_state_dict(ck["model_state"], strict=False)
    m.to(torch.bfloat16).to("cuda")
    return m, tok


def main():
    ap = argparse.ArgumentParser(description="Route a query through the deployed VLA/LLM stack")
    ap.add_argument("query", nargs="*", help="the query text")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=120)
    # 205a: meter-sweet-spot 0.3 (0.7 degenerates the v3 value region).
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import torch
    from route import pick_adapter

    loaded = {}  # ckpt -> model (reuse adapters across queries)

    def _is_heldout(q):
        # ROUND-174: shared 32-set registry (was a drifting per-file 20-set —
        # the expansion leaked 245 trajectories).
        from heldout import is_heldout
        return is_heldout(q)

    def answer(q):
        arith = _try_arith(q)
        if arith is not None:
            print(arith)
            if not _is_heldout(q):
                try:
                    with open(ROOT / "out/agent_trajectories.jsonl", "a") as f:
                        import json as _json
                        f.write(_json.dumps({"query": q, "mode": "arith",
                                             "response": arith}) + "\n")
                except Exception:
                    pass
            return arith
        ckpt, mode = pick_adapter(q)
        if ckpt not in loaded:
            from tool_use import ToolRegistry, ToolUseMoE, _execute_python, _search_knowledge
            m, tok = load_adapter(ckpt)
            if mode == "tool":
                reg = ToolRegistry()
                reg.register(_execute_python, name="execute_python",
                             description="Execute Python code and return output")
                reg.register(_search_knowledge, name="search_knowledge",
                             description="Search internal knowledge base for information")
                # 205s: compiled_schema=False — the meter shape. The
                # compiled prompt degenerates v3 rollouts ("3/3/3/3"
                # loops, measured 204ah-34); serving must match the eval.
                tu = ToolUseMoE(m, lambda t: tok.encode(t).ids,
                                lambda i: tok.decode(i), reg,
                                compiled_schema=False)
                loaded[ckpt] = ("tool", tu)
            else:
                loaded[ckpt] = ("gen", (m, tok))
        kind, obj = loaded[ckpt]
        if kind == "tool":
            tu = obj
            # 205u: guided JSON decoding for the tool-call emission
            result = tu.generate(q, max_tokens=args.max_tokens,
                                 temperature=args.temperature, guided=True)
            print(f"[{mode}] {result}")
            return result
        m, tok = obj
        p = f"You are a helpful assistant.\nUser: {q}\nAssistant:"
        ids = tok.encode(p).ids
        idx = torch.tensor([ids], dtype=torch.long, device=next(m.parameters()).device)
        with torch.no_grad():
            out = m.generate(idx, max_new_tokens=args.max_tokens,
                             temperature=args.temperature, top_k=40)
        gen = tok.decode(out[0, len(ids):].tolist()).strip()
        print(f"[{mode}] {gen}")
        # B-147: the docstring promised flywheel logging for chat/direct but
        # the code never did it -- the objective's usage->trajectories loop
        # needs real interactions recorded, whatever the mode.
        if not _is_heldout(q):
            try:
                logf = ROOT / "out" / "agent_trajectories.jsonl"
                with open(logf, "a") as f:
                    import json as _json
                    f.write(_json.dumps({"query": q, "mode": mode,
                                         "response": gen[:1000]}) + "\n")
            except Exception:
                pass
        return gen

    if args.interactive:
        print("[query] interactive mode — /quit exits, adapters load lazily per route")
        while True:
            try:
                q = input("query> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q == "/quit":
                break
            try:
                answer(q)
            except Exception as e:
                print(f"[query] error: {e}")
    else:
        q = " ".join(args.query).strip()
        if not q:
            print('usage: query.py "your question"  |  or --interactive')
            return
        answer(q)


if __name__ == "__main__":
    main()
def _try_arith_o(q):
    """The GUI/CLI-friendlier arith: no [arith]-prefix, the answer only,
    returns None when the regex doesn't fire (the caller decides the fallback).
    The GUI must not ship the internal envelope tokens in the visible reply —
    the ROUND-201x fix (my "hello → what is ..." earlier sent raw internal
    [arith] labels into the chat bubbles; the kind field carries the info)."""
    r = _try_arith(q)
    if r is None:
        return None
    return r[len("[arith] "):] if r.startswith("[arith] ") else r
