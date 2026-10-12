"""Reusable tool-use reliability benchmark (B-45).
Usage: python3 src/eval_tool_use.py --ckpt out/ambatron_mini/tool_sft.pt [--n_repeat 1]
       python3 src/eval_tool_use.py --model moegpt --ckpt out/ambatron_430m_v2/ckpt_best.pt
Scores: format (parseable call OR clean direct answer), tool emission,
right-tool, and strictly-correct (right tool + right args + right result).
Run n_repeat times and report means — a single-shot 8-query sample was too
noisy (round 28-30 lesson)."""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json, re


CASES = [
    ("What is the sum of squares of 1..10?", ("execute_python", 385)),
    ("Compute the 11th Fibonacci number and report it.", ("execute_python", 55)),
    ("Check whether 1729 is divisible by 7.", ("execute_python", "yes")),
    ("Sort [3,1,4,1,5] and report the second element.", ("execute_python", 1)),
    ("Explain PPO in one sentence.", ("search_knowledge", None)),
    ("Explain SAC in one sentence.", ("search_knowledge", None)),
    ("What is a mixture of experts?", ("search_knowledge", None)),
    ("What religion is predominant in Indonesia?", (None, "islam")),
]

# B-69 held-out suite: queries NOT in the build_tool_sft templates (the
# template suite above is memorizable — a leaked eval). Same tools, fresh
# numbers/objects: the honest skill measure.
CASES_HELDOUT = [
    # B-202: parameterized to instances NEVER in rest_round QUERIES / tool_arith /
    # tool_sft training data (the old 4 execute_python probes were verbatim in
    # training -> measured memorization, not generalization).
    ("What is the sum of cubes of 1..9?", ("execute_python", 2025)),     # (45)^2
    ("Compute the 20th Fibonacci number and report it.", ("execute_python", 4181)),  # F(19)=4181 (repo convention: nth -> F(n-1))
    ("Check whether 731 is divisible by 17.", ("execute_python", "yes")),  # 17*43
    ("Sort [4,8,1,7,3] and report the third element.", ("execute_python", 4)),
    # B-202 held-out NEGATIVE divisibility: penalizes the "always-yes" hack
    # (a model emitting `print(0)` for any "divisible by?" query).
    ("Check whether 13 is divisible by 7.", ("execute_python", "no")),     # 13%7=6
    ("Is 25 divisible by 4?", ("execute_python", "no")),                   # 25%4=1
    # 11th-audit: the old knowledge probes (TD3/BC/transformer block/Japan
    # currency) were verbatim in training data, contaminating the held-out
    # suite and self-perpetuating via agent_trajectories -> flywheel_to_sft.
    # Replaced with fresh, zero-hit probes on the knowledge path (search_knowledge
    # matches ppo/sac/moe/gqa keywords) and a fresh no-tool knowledge question.
    ("What is SAC in one sentence.", ("search_knowledge", "Soft Actor-Critic")),
    ("Explain PPO versus a random baseline.", ("search_knowledge", "Proximal Policy")),
    ("What does the gating in a Mixture of Experts do.", ("search_knowledge", "sparse activation")),
    ("Name the currency of South Korea.", (None, "won")),
    # ROUND-174 (Sep-7): EXPAND the heldout suite from 10 -> 30 so the tool
    # axis has discrimination room. The old 10-probe suite made promotion
    # impossible past 10/10 (and noisy at sample size); these additions are
    # fresh, unambiguous, and NOT in any training/rest data. All answers
    # verified by direct computation.
    ("What is the sum of cubes of 1..8?", ("execute_python", 1296)),       # (36)^2
    ("What is the sum of squares of 1..9?", ("execute_python", 285)),      # 9*10*19/6
    ("Compute the 12th Fibonacci number.", ("execute_python", 89)),        # F(11)=89
    ("Compute the 17th Fibonacci number.", ("execute_python", 987)),       # F(16)=987
    ("Is 377 divisible by 13?", ("execute_python", "yes")),                # 13*29
    ("Is 217 divisible by 7?", ("execute_python", "yes")),                 # 7*31
    ("Is 143 divisible by 11?", ("execute_python", "yes")),                # 11*13
    ("Is 91 divisible by 8?", ("execute_python", "no")),                   # 91%8=3
    ("Is 399 divisible by 19?", ("execute_python", "yes")),                # 19*21=399 => yes
    ("Sort [5,2,9,1,7] and report the smallest.", ("execute_python", 1)),
    ("Sort [6,3,8,2] and report the second largest.", ("execute_python", 6)),
    ("What is 13 multiplied by 23?", ("execute_python", 299)),       # 13*23=299 (204i-B: fingerprint-scanned clean of the heldout set)
    ("What is 34 times 12?", ("execute_python", 408)),                     # 34*12
    ("What is 196 divided by 14?", ("execute_python", 14)),                # 14*14
    ("What is 423 divided by 9?", ("execute_python", 47)),                 # 9*47
    ("What is 8 to the power of 3?", ("execute_python", 512)),             # 2^9
    ("What is 6 factorial?", ("execute_python", 720)),                     # 1..6
    ("What is the 5th prime number?", ("execute_python", 11)),             # 2,3,5,7,11
    ("What is the 8th prime number?", ("execute_python", 19)),             # ...19
    ("What is the GCD of 54 and 36?", ("execute_python", 18)),             # 18
    ("Is 97 prime?", ("execute_python", "yes")),                           # 97 prime
    ("Is 119 prime?", ("execute_python", "no")),                           # 7*17
]


def score_one(tu, reg, q, exp):
    out = tu.generate(q, max_tokens=60, max_tool_calls=2)
    # B-105: score from the RECORDED trajectory, not the final completion text.
    # generate() returns only the last segment (after the final answer with no
    # "use tool:" marker), so _parse_tool_call(out) saw no call and every
    # tool-use query scored as "no-call"/0 — the benchmark read ~0%. Mirror
    # eval_agentic.py: walk last_trajectory for the first call to the expected
    # tool and verify its observation. Fall back to parsing `out` only when the
    # trajectory is empty (a genuine direct/no-call answer).
    traj = getattr(tu, "last_trajectory", []) or []
    call = None
    obs = None
    if exp[0] is None:
        # no tool expected: any recorded call counts as a wrong call
        if traj:
            e = traj[0]
            call = {"name": e.get("name"), "arguments": e.get("arguments", {})}
            obs = e.get("observation")
    else:
        for e in traj:
            if e.get("name") == exp[0]:
                # Sep-5 (audit): an ERRORED call (exception/traceback/timeout/
                # sandbox) must not be scored as a "correct" execution — the old
                # loop took the FIRST matching call; a `raise` after `print(385)`
                # still parsed the number and rewarded the failure. Skip error
                # observations so a later successful recovery gets credit.
                _o = str(e.get("observation") or "")
                if any(m in _o.lower() for m in
                       ("traceback", "_error", "sandbox", "[timed out",
                        "[truncated", "loop detected")) or \
                   re.search(r"[A-Za-z]+(?:Error|Exception)\b", _o):
                    continue
                call = {"name": e.get("name"), "arguments": e.get("arguments", {})}
                obs = e.get("observation")
                break
    if call is None:
        if traj:
            # B-XXX: the model DID call a tool but not the expected one — that
            # is a WRONG-TOOL failure, not a no-call (the old path parsed `out`
            # -> None -> "no-call", collapsing two distinct failure modes).
            e = traj[0]
            call = {"name": e.get("name"), "arguments": e.get("arguments", {})}
            obs = e.get("observation")
        else:
            call = tu._parse_tool_call(out)
            if call is None:
                # no tool call -> the direct answer. For a (None, expected) case the
                # expected answer text is checked against the completion (B-101: the
                # old scoring marked every correct direct answer "no-call",
                # structurally undercounting the honest suite's direct-answer half).
                # only a genuine direct-answer case (exp[0] is None, no tool expected)
                # may score "correct" from bare text — a divisibility/verification case
                # (exp[0] == "execute_python", expected "yes"/"no") MUST call the tool,
                # 3rd-audit fix + ROUND-5: substring AND \b both false-positive on refusals
                # ("I won't speculate" contains "won"; \b treats the apostrophe
                # as a boundary so \bwon\b ALSO matches inside won't). Use an
                # apostrophe-excluded boundary class so "won" (currency) never
                # matches won't/won't-you.
                if exp[0] is None and exp[1] is not None and isinstance(exp[1], str):
                    if re.search(r"(?<![A-Za-z0-9_'])" + re.escape(exp[1]) +
                                 r"(?![A-Za-z0-9_'])", out.lower()):
                        return "correct"
                    return "no-call"
                # ROUND-174: tool-EXPECTED case with NO parseable call -> the
                # old code fell through to call.get() with call==None and
                # crashed the whole eval run. Score honestly as no-call.
                return "no-call"
        name, args = call.get("name", "?"), call.get("arguments", {})
        r = reg.call(name, args)
    else:
        name, args = call.get("name", "?"), call.get("arguments", {})
        r = obs
    if exp[0] is None:
        # no tool expected (e.g. general-knowledge query) -> a tool call is wrong
        return "wrong-tool"
    ok_tool = name in reg._tools and name == exp[0]
    ok_args = True
    if exp[1] is None:
        if name == "search_knowledge" and "No relevant" in str(r):
            ok_args = False
    elif exp[1] == "yes":
        # B-77/B-202: only an EXPLICIT boolean assertion counts. A bare numeric
        # "0"/"0.0" must NOT be accepted as "divisible" — that let a model
        # output `print(0)` for ANY "divisible by?" query and game the reward to
        # 100%. Require the tool result to be exactly True/yes.
        ok_args = str(r).strip().lower() in ("true", "yes")
    elif exp[1] == "no":
        # B-202: symmetric negative case so an "always-yes" policy is penalized.
        ok_args = str(r).strip().lower() in ("false", "no")
    elif isinstance(exp[1], str):
        ok_args = exp[1].lower() in str(r).lower()
    else:
        # B-161 consistency: LAST number only -- a printed sorted list
        # merely containing the target no longer scores correct.
        nums = [int(n) for n in re.findall(r"[-+]?\d+", str(r))]
        # ROUND-174 (N5): float division prints "12.0" -> nums=[12,0] -> the
        # last number (0) wrongly failed the int compare. Strip a trailing .0
        # so "196/14 -> 14.0" scores correct.
        _rstr = str(r).strip()
        if _rstr.endswith(".0") and _rstr[:-2].lstrip("-").isdigit():
            nums = [int(_rstr[:-2])]
        ok_args = bool(nums) and nums[-1] == exp[1]
    if ok_tool and ok_args:
        return "correct"
    if ok_tool:
        return "wrong-args"
    return "wrong-tool"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--repeat", type=int, default=3,
                    help="repeat the whole suite this many times (sampling noise)")
    ap.add_argument("--heldout", action="store_true",
                    help="use the B-69 held-out suite (fresh queries, no template leak)")
    ap.add_argument("--compiled", action="store_true",
                    help="B-75: use the TSCG-style compiled tool-schema prompt (A/B vs default)")
    ap.add_argument("--bf16", action="store_true",
                    help="load the model in bfloat16 (~half VRAM) to run alongside other GPU apps")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    import torch
    torch.manual_seed(0)  # 14th-audit M4: fixed seed for champion/candidate comparability
    import torch.serialization as ser
    from moe_model import MoEGPT, MoEConfig
    from tokenizers import Tokenizer
    from tool_use import ToolRegistry, ToolUseMoE, _execute_python, _search_knowledge
    ser.add_safe_globals([MoEConfig])

    tok = Tokenizer.from_file(
        str(pathlib.Path(__file__).resolve().parent.parent
            / "data/processed/bpe_tokenizer/tokenizer.json"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model = MoEGPT(ck["config"]).eval()
    # ROUND-174 (golden-rule mini-smoke caught this): strict load crashes on
    # every checkpoint — _model_sd pops the TIED lm_head.weight by design, so
    # strict=False + tolerate it (as train_tool_sft does).
    _miss, _unexp = model.load_state_dict(ck["model_state"], strict=False)
    if _miss and not set(_miss) <= {"lm_head.weight"}:
        raise RuntimeError(f"base ckpt arch mismatch: {_miss}")
    if args.bf16:
        model.to(torch.bfloat16)
    model.to(args.device)
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    tu = ToolUseMoE(model, lambda t: tok.encode(t).ids, lambda i: tok.decode(i), reg,
                    compiled_schema=args.compiled)

    totals = {"correct": 0, "wrong-args": 0, "wrong-tool": 0, "no-call": 0}
    n = 0
    suite = CASES_HELDOUT if args.heldout else CASES
    for _ in range(args.repeat):
        for q, exp in suite:
            totals[score_one(tu, reg, q, exp)] += 1
            n += 1
    print(f"[tool-eval] {args.ckpt} | n={n}")
    for k, v in totals.items():
        print(f"  {k:11s}: {v} ({v/n*100:.0f}%)")
    print(f"  emit-rate: {(totals['correct']+totals['wrong-args']+totals['wrong-tool'])/n*100:.0f}% "
          f"| correct: {totals['correct']/n*100:.0f}%")
    # B-217 (5th audit): an explicit parseable N/M line — auto_optimize's
    # `correct\s*: (\d+)` was scraping the PERCENTAGE off the line above and
    # treating it as a /8 score (a ~43 vs 3 comparison). The evaluator + the
    # flywheel gate must agree on the count AND the suite size.
    print(f"correct: {totals['correct']}/{n}")


if __name__ == "__main__":
    main()