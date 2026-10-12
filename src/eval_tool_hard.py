#!/usr/bin/env python3
"""HARD-TOOL EVAL (ROUND-174): execution-verified hardest cases + trap floors.

Every expected answer is COMPUTED at import (hard_suite.gen) — no hand-typed
constants to typo. Scoring mirrors eval_tool_use.score_one:
  - the model must CALL the expected tool (no-call = fail)
  - the OBSERVATION (not the text) must contain the correct answer
  - wrong-args/wrong-tool = fail (vs "correct")
Extra floors: empty-call traps, all-"no" divisibility, prose-only answers.

README: this is the HARDEST suite we have. A model passing >50% here has a
genuine execution loop, not memorization.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# (b8.3 audit: the old raw "/tmp" insert on the import path let a stray
#  /tmp file shadow src modules — removed; src is already on the path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_tokens", type=int, default=80)
    # 19th-audit B4: the POOL and the DEPLOYED query.py path speak the
    # compiled schema (short signature, no timeout_s) — the old verbose
    # section (compiled=False) diverged from both: the SFT never saw the
    # verbose form, so the eval measured a format the model never learned.
    # compiled=True = the deployment-truthful gate.
    ap.add_argument("--top_k", type=int, default=40,
                    help="204m: the window-hypothesis knob")
    ap.add_argument("--compiled", type=int, default=1,
                    help="use the compiled system-section (deployed format; "
                         "also matches the SFT pool)")
    args = ap.parse_args()

    import torch
    import torch.serialization
    from moe_model import MoEGPT, MoEConfig
    torch.serialization.add_safe_globals([MoEConfig])
    from tokenizers import Tokenizer
    import tool_use as tu

    from hard_suite import CASES

    tok = Tokenizer.from_file(
        str(Path(__file__).resolve().parent.parent / "data/processed/bpe_tokenizer/tokenizer.json"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = MoEGPT(ck["config"]).eval()
    miss, unexp = model.load_state_dict(ck["model_state"], strict=False)
    if miss and not set(miss) <= {"lm_head.weight"}:
        raise RuntimeError(f"arch mismatch: {miss}")
    if args.bf16:
        model.to(torch.bfloat16)
    model.to(args.device)

    reg = tu.default_tools()
    touse = tu.ToolUseMoE(model,
                          tokenizer_func=lambda x: tok.encode(x).ids,
                          detokenizer_func=lambda ids: tok.decode(ids),
                          tool_registry=reg,
                          compiled_schema=bool(args.compiled))

    torch.manual_seed(args.seed)
    correct = wrong_args = wrong_tool = no_call = errored = 0
    for _case in CASES:
        q, (tool, want) = _case[0], _case[1]
        # 19th-audit B9: temperature 0.5 + the seeded RNG — the library's
        # default 0.7/top_k-40 made single-shot pass@1 flaky across repeats.
        out = touse.generate(q, max_tokens=args.max_tokens,
                             max_tool_calls=2, temperature=0.5,
                             top_k=args.top_k)
        traj = getattr(touse, "last_trajectory", []) or []
        call = None
        call_errored = False
        for e in traj:
            if e.get("name") == tool:
                _o = str(e.get("observation") or "").lower()
                if any(m in _o for m in ("traceback", "_error", "sandbox", "[timed out",
                                         "[truncated", "loop detected")) or \
                   __import__("re").search(r"[a-z]+(?:error|exception)\b|\"error\"\s*:", _o):
                    # 204m: a call that RAN but RAISED is its own class —
                    # neither no-call nor wrong-args (the old conflation hid
                    # the whole "6k = 29 no-call" verdict's composition).
                    errored += 1
                    call = e
                    call_errored = True
                    break
                call = e
                break
        # 204m: no expected-tool call -> distinguish the emission classes
        # instead of collapsing them (the 6th-wave B2): a wrong-tool call
        # and an unparseable emission are NOT "no-call".
        if call is None:
            if traj:
                _any_name = any(e.get("name") for e in traj)
                if _any_name:
                    wrong_tool += 1
                else:
                    no_call += 1
            else:
                no_call += 1
            continue
        if call_errored:
            continue
        r = str(call.get("observation"))
        rl = r.lower()
        # EXECUTION-VERIFIED: numbers / yes-no / strings
        if isinstance(want, str):
            # 19th-audit B-4: yes/no must use the explicit-boolean gate
            # (mirrors eval_tool_use.score_one) — substring "yes" passes in
            # "yesterday" and "no" in "note"/"north" (the always-yes/no class
            # the main eval killed is still gameable on this suite).
            _w = want.lower()
            rl = rl or ""
            if _w in ("yes", "true"):
                ok = bool(__import__("re").search(r"\b(?:yes|true)\b", rl))
            elif _w in ("no", "false"):
                ok = bool(__import__("re").search(r"\b(?:no|false)\b", rl)) \
                    and not __import__("re").search(r"\b(?:yes|true)\b", rl)
            else:
                ok = _w in rl
        elif isinstance(want, float):
            # ROUND-182: N5 float handling (mirrors eval_tool_use.py:177) — a
            # "4.5" observation previously parsed as [4,5] and never matched
            fl = [float(n) for n in __import__("re").findall(r"[-+]?\d*\.?\d+", r)]
            ok = bool(fl) and fl[-1] == want
        else:
            nums = [int(n) for n in __import__("re").findall(r"[-+]?\d+", r)]
            ok = bool(nums) and nums[-1] == want
        if ok:
            correct += 1
        elif call.get("name") == tool:
            wrong_args += 1
        else:
            wrong_tool += 1

    total = correct + wrong_args + wrong_tool + no_call + errored
    print(f"\n[tool-hard] {args.ckpt} | n={total}")
    print(f"  correct   : {correct} ({correct/max(total,1)*100:.0f}%)")
    print(f"  wrong-args: {wrong_args}")
    print(f"  wrong-tool: {wrong_tool}")
    print(f"  no-call   : {no_call}")
    print(f"  errored   : {errored}")
    return correct

if __name__ == "__main__":
    sys.exit(main())