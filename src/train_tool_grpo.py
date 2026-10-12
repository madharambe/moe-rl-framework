"""Round-82: GRPO / RLVR for tool-use (the 2605.30478 recipe on our stack).

Sample G completions per query with the CURRENT policy (v13, LoRA'd), score
each by REAL EXECUTION (parse the tool call, run it, compare the result), then
update with the group-relative advantage (GRPO) + KL-to-reference. The queries
are FRESH execution-verifiable ones (distinct from the held-out eval suite —
no leak). The reference (frozen) is the --base checkpoint; the policy = LoRA on
top; only the LoRA trains.

Usage:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 src/train_tool_grpo.py \
      --base out/ambatron_tool_v13/ckpt.pt --steps 10 --group 8 \
      --out out/ambatron_tool_v15/ckpt.pt
"""
import argparse, json, sys, pathlib, re, math, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
from tool_use import ToolUseMoE, ToolRegistry, _execute_python, _search_knowledge
from eval_agentic import FailOnceRegistry, SYNTHETIC_ERROR

FILES_MODE = False
FILES_TOOLS = {"read_file", "grep", "glob", "list_dir", "write_file", "bash"}

# Fresh execution-verifiable queries — NOT in the held-out eval suite.
QUERIES = [
    ("What is the sum of squares of 1..12?", "execute_python", 650),
    ("Compute the 15th Fibonacci number and report it.", "execute_python", 377),
    ("Check whether 979 is divisible by 11.", "execute_python", "yes"),
    ("Sort [9,3,6,1,8] and report the second element.", "execute_python", 3),
    ("What is the sum of cubes of 1..6?", "execute_python", 441),
    ("What is the 13th prime number?", "execute_python", 41),
    ("Compute the sum 1+2+...+15.", "execute_python", 120),
    ("Check whether 1001 is divisible by 13.", "execute_python", "yes"),
    ("Compute the 9th Fibonacci number and report it.", "execute_python", 21),   # 12th was heldout near-verbatim
    ("Sort [4,9,2,7,5] and report the largest element.", "execute_python", 9),
    ("What is the sum of squares of 1..8?", "execute_python", 204),
    ("What is the 7th prime number?", "execute_python", 17),
    ("Check whether 576 is divisible by 24.", "execute_python", "yes"),
    ("Sort [8,1,5,3,6] and report the first element.", "execute_python", 1),
    ("What is the sum of cubes of 2..6?", "execute_python", 440),   # was 1..9 (heldout)
    ("Compute the 16th Fibonacci number and report it.", "execute_python", 610),
    ("What is the sum 1+2+...+10?", "execute_python", 55),
    ("Sort [6,4,8,2,9] and report the third element.", "execute_python", 6),
    ("What is the 11th prime number?", "execute_python", 31),
    ("Check whether 361 is divisible by 19.", "execute_python", "yes"),  # 323 renumbered (B-2: near-dup of hard "Is 323 prime?")
    ("Check whether 247 is divisible by 13.", "execute_python", "yes"),  # 143 renumbered (19th-audit: 143/13 was itself a hard-suite pair!)
    ("Is 57 divisible by 3?", "execute_python", "yes"),                  # was 25/4 (heldout)
    ("What is the sum of squares of 1..5?", "execute_python", 55),
    ("Sort [3,7,1,9,4] and report the smallest element.", "execute_python", 1),
    ("Compute the 13th Fibonacci number and report it.", "execute_python", 144),
    ("What is the sum of cubes of 1..4?", "execute_python", 100),
]

# 205d: shaped ladder — correct 1.0 > valid-exec 0.5 > wrong-call 0.1 >
# no-call 0.0 (the retry->valid-call path gets a graded gradient; retry-100%
# policies otherwise see zero signal for call VALIDITY).
# 205q: ToolRL-decomposed ladder (Qian et al.): R_format + R_name + R_args +
# R_value. name-ok = parseable call with the SCHEMA tool name (0.35) —
# pulls the served-era "read" drift toward "read_file" without execution;
# valid-exec = executed w/ clean observation (0.5); correct = answer (1.0).
REWARD = {"correct": 1.0, "valid-exec": 0.5, "name-ok": 0.35,
          "wrong-call": 0.1, "no-call": 0.0}


def score_one(tu, reg, q, exp, out):
    """Score the GIVEN completion `out` (B-102: the old signature ignored its
    arguments and RE-GENERATED a fresh completion inside — the rewards were
    attached to the wrong rollouts, the GRPO collapsed/zeroed). B-105: prefer
    the recorded trajectory (tu.last_trajectory) to recover the actual tool
    call + observation; the returned `out` is only the final segment and lost
    the call, so parsing it scored every rollout as no-call/0."""
    traj = getattr(tu, "last_trajectory", []) or []
    # 205bz: FILES-mode grading = the FILES METER's semantics (the
    # observation-based 'the answer appears in a SUCCESSFUL call's
    # observation' rule from eval_agentic_files — the agentic numeric
    # grader below cannot score file tools at all).
    _file_tools = {"read_file", "glob", "grep", "list_dir", "bash"}
    if FILES_MODE:
        _obs = None
        _file_hit = False
        for _e in traj:
            _n = str(_e.get("name") or "")
            if _n in _file_tools:
                _file_hit = True
                _o = str(_e.get("observation") or "")
                if any(_m in _o.lower() for _m in ("traceback", "_error",
                        "sandbox", "[timed out", "[truncated", "loop detected")) \
                        or "Error" in _o or "Exception" in _o:
                    continue
                if not (str(_e.get("arguments") or {})):
                    continue
                _obs = _o
        if _obs is None:
            # 205cm: the exec-reflex (execute_python-only rollouts) must NOT
            # earn the 0.5 valid-exec in files mode — that ties every sample
            # (no differential = no learning). The reflex = wrong-call (0.1),
            # steering toward the file tools.
            if traj and not _file_hit:
                return "wrong-call"
            return "valid-exec" if _file_hit else "no-call"
        ans_s = str(exp).lower()
        if re.search(r"(?<![A-Za-z0-9_'])" + re.escape(ans_s) +
                     r"(?![A-Za-z0-9_'])", _obs.lower()):
            return "correct"
        return "wrong-call"
    call = None
    for e in traj:
        if e.get("name") == "execute_python":
            # ROUND-5 (F-6): skip ERRORED calls (print+raise was rewarded as
            # 'correct' — the error observation still carried the printed
            # number). A later successful recovery wins; all-error -> scored
            # below as a fail.
            _o = str(e.get("observation") or "")
            if any(m in _o.lower() for m in
                   ("traceback", "_error", "sandbox", "[timed out",
                    "[truncated", "loop detected")) or \
               re.search(r"[A-Za-z]+(?:Error|Exception)\b", _o):
                continue
            # 205e: reject the trivial-literal gaming mode ("1,1" tuples):
            # the call must contain a real function call in its code.
            _code = str((e.get("arguments") or {}).get("code", ""))
            if not ("(" in _code and any(ch.isalpha() for ch in _code.split("(")[0])) or len(_code.strip()) < 8:
                continue
            call = e
            break
    if call is not None:
        name = call.get("name", "")
        r = str(call.get("observation"))
        # 205d shape: an EXECUTED call (valid but possibly wrong answer) is
        # strictly better than an errored attempt — the retry-100% policy
        # needs a graded path to validity before correctness.
        _executed = True
        _call_valid = True
    else:
        _executed = False
        _call_valid = False
        call = tu._parse_tool_call(out)
        if call is None:
            return "no-call"
        name, args = call.get("name", "?"), call.get("arguments", {})
        r = str(reg.call(name, args))
        _executed = True
        _call_valid = True
    if name != "execute_python":
        if FILES_MODE and name in FILES_TOOLS:
            # name-level first (ToolRL R_name): a parseable call whose NAME
            # matches the schema gets 0.35 even before any execution
            if _executed and name in FILES_TOOLS:
                _n_ok = True
            else:
                _n_ok = False
            if not _n_ok:
                return "name-ok" if _executed else "wrong-call"
            # 205p: ToolRL-style decomposed reward (Qian et al., UIUC —
            # "ToolRL: Reward is All Tool Learning Needs"): score FORMAT +
            # TOOL-NAME + PARAMS + VALUE separately. The tools section names
            # ARE the schema; the model's served-era "read" vs registry
            # "read_file" mismatch gets a graded pull toward the schema.
            ans_s = str(exp).lower()
            _obs_ok = _executed and not any(em in r.lower() for em in
                     ("traceback", "_error", "sandbox", "[timed out",
                      "[truncated", "loop detected")) and                      "Error" not in r and "Exception" not in r
            _ok = _obs_ok and bool(re.search(r"(?<![A-Za-z0-9_'])" +
                                  re.escape(ans_s) +
                                  r"(?![A-Za-z0-9_'])", r.lower()))
            if _ok:
                return "correct"
            # gradations (mapped to the ladder): executed obs w/ real args
            if _obs_ok:
                return "valid-exec"
            if _executed:
                return "wrong-call"
            return "wrong-call"
        return "wrong-call"
    if exp == "yes":
        # B-196/B-202: only an EXPLICIT boolean assertion counts. Reject bare
        # numeric "0"/"0.0" — a model emitting `print(0)` for ANY "divisible
        # by?" query gamed the reward to 100% without computing a % b.
        rs = r.strip().lower()
        ok = rs in ("true", "yes")
    elif exp == "no":
        # B-202: symmetric negative case so an "always-yes" policy is penalized.
        rs = r.strip().lower()
        ok = rs in ("false", "no")
    else:
        # 205a/205c: meter-exact int rule — ANY-position for sort-element
        # answers (the champion prints the sorted list), LAST-number for
        # count/sum queries (single-answer semantics; any-position let
        # degraded "1,,,,5" outputs game the count verdicts).
        nums = [int(n) for n in re.findall(r"[-+]?\d+", r)]
        if re.search(r"how many|sum of|total", q):
            ok = bool(nums) and nums[-1] == exp
        else:
            ok = bool(nums) and exp in nums
    if ok:
        return "correct"
    if _executed:
        # 205d: EXPICITLY-executed call (wrong answer) = the valid-exec shape
        return "valid-exec"
    return "wrong-call"


def completion_logps(model, tok, prompt, completion, device):
    """Per-token log-probs of the policy over the completion (tokens shifted)."""
    block = model.config.block_size
    ids = tok.encode(prompt + completion).ids
    comp = tok.encode(completion).ids
    if len(ids) > block:
        ids = ids[-block:]  # keep the completion tail fully (B-78 lesson)
        # completion survives intact at the tail; recompute the boundary from the
        # window instead of the untruncated prompt (else n_prompt>len(ids) → empty
        # slice → NaN). Only reached when prompt+completion > block.
        n_prompt = max(1, len(ids) - len(comp))
    else:
        # B-180: encode prompt alone (BPE boundary at the join can differ); this is
        # exact only when prompt+completion fits in `block`.
        n_prompt = len(tok.encode(prompt).ids)
    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, _, _ = model(x)
    logp = F.log_softmax(logits.float(), dim=-1)
    comp_logp = logp[0, n_prompt - 1:, :].gather(1, y[0, n_prompt - 1:].unsqueeze(1)).squeeze(1)
    return comp_logp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--steps", type=int, default=10, help="GRPO rounds")
    ap.add_argument("--group", type=int, default=8, help="samples per query (G)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04, help="KL-to-ref coefficient")
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--n_epochs", type=int, default=2, help="update epochs per round")
    ap.add_argument("--max_norm", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", type=str, default="cuda")
    # 205a: E2-GRPO — rollout temperature (GRPO3's hard 0.7 killed the
    # champion's rare-correct mode: temp 0.7 samples 0% correct in every
    # round -> zero-reward no-op). And --queries-json: restrict the round
    # to the execution-verified CORRECT-greedy subset (nonzero mass).
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--files", action="store_true",
                    help="205k: FILES-envelope mode — register the file tools, "
                         "score word-boundary answers in executed observations "
                         "(the honest files gate crossed by dual2).")
    ap.add_argument("--fail-once", action="store_true",
                    help="205a: match the AGENTIC METER objective — inject one "
                         "synthetic error per rollout and reward retry+converge "
                         "(GRPO trained on plain execution never transferred to "
                         "the meter: metric-vs-objective mismatch).")
    ap.add_argument("--queries-json", type=str, default="")
    args = ap.parse_args()

    from moe_model import MoEConfig, MoEGPT
    import torch.serialization as ser
    ser.add_safe_globals([MoEConfig])
    from train_tool_sft import _tokenize
    from lora_adapter import add_lora, freeze_base, merge_lora, remove_lora, LoRALinear

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.base, map_location="cpu", weights_only=True)
    # 204ah-11 (L1 wiring): the tokenizer must match the BASE CKPT's own vocab
    # — the old _tokenize("") always loaded the 16K corpus tokenizer, so a v3
    # base (32K vocab) GRPO'd against a mismatched 16K stream (banked b17.2.3).
    _v3_base = int(getattr(ck["config"], "vocab_size", 0)) >= 32768
    tok = _tokenize("", "v3" if _v3_base else None)
    ref = MoEGPT(ck["config"]).eval()
    ref.load_state_dict(ck["model_state"], strict=False)  # ROUND-172 E1: tolerate omitted ties/EMA-era keys
    policy = MoEGPT(ck["config"]).eval()
    policy.load_state_dict(ck["model_state"], strict=False)
    add_lora(policy, r=16, alpha=32)
    freeze_base(policy)
    if device.type == "cuda":
        # B-83: cast the WHOLE model to bf16 (the selective Linear-only cast
        # leaves fp32 modules like QKNorm/ValueResidual in the bf16 graph ->
        # mixed-dtype matmuls outside the autocast). Restore the LoRA A/B
        # masters to fp32 (optimizer precision) — the LoRALinear.forward casts
        # them to x.dtype for the compute.
        for m in (ref, policy):
            m.to(dtype=torch.bfloat16)
            for mod in m.modules():
                if isinstance(mod, LoRALinear):
                    mod.A.to(dtype=torch.float32)
                    mod.B.to(dtype=torch.float32)
    ref.to(device)
    policy.to(device)

    global FILES_MODE
    if args.files:
        # 205o: files-mode registry = the files envelope ONLY (the eval
        # prompt lists 6 file tools; the mixed 8-tool section degenerates
        # files rollouts — arms were trained on pure-files sections).
        FILES_MODE = True
        from eval_agentic_files import agentic_registry
        reg = agentic_registry()
    else:
        reg = ToolRegistry()
        reg.register(_execute_python, name="execute_python",
                     description="Execute Python code and return output")
        reg.register(_search_knowledge, name="search_knowledge",
                     description="Search internal knowledge base for information")

    if args.queries_json:
        with open(args.queries_json) as fq:
            loaded_q = json.load(fq)
        if isinstance(loaded_q, list) and loaded_q and isinstance(loaded_q[0], dict):
            QUERIES = [(r["query"], r.get("mode", "execute_python"),
                        r.get("expected", ""),
                        r.get("fail_once", True)) for r in loaded_q]
        elif isinstance(loaded_q, list):
            QUERIES = [(q, m, a, True) for q, m, a in loaded_q]
        else:
            QUERIES = loaded_q
        print(f"[grpo] query file: {len(QUERIES)} queries (from {args.queries_json})")

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    # 205a: compiled_schema=False - the AGENTIC METER (eval_agentic.py)
    # builds ToolUseMoE with the DEFAULT (non-compiled) tool section; a
    # compiled prompt trains a DIFFERENT distribution than the flywheel
    # measures (measured: compiled rollouts degenerate "3/3/3/3" loops).
    tu = ToolUseMoE(policy, lambda t: tok.encode(t).ids, lambda i: tok.decode(i),
                    reg, compiled_schema=False)

    for step in range(args.steps):
        batches = []
        n_correct = 0
        loss = None
        for q, tool, ans, _fo in QUERIES:
            outs = []
            rewards = []
            for _ in range(args.group):
                if args.fail_once or _fo:
                    fresh = FailOnceRegistry(reg, tool, SYNTHETIC_ERROR)
                    tu.tools = fresh
                out = tu.generate(q, max_tokens=160, max_tool_calls=3,
                                  temperature=args.temperature)
                if args.files and step == 0 and _ == 0 and q == QUERIES[0][0]:
                    _pb = tu._build_prompt(q)
                    print(f"[grpo-files] PROMPT ({len(_pb)}): {_pb[:260]!r}", flush=True)
                    print(f"[grpo-files] TOKENS: {tok.encode(_pb).ids[:40]}", flush=True)
                    print(f"[grpo-files] sample out: {out[:200]!r}", flush=True)
                    print(f"[grpo-files] sample traj: {str(getattr(tu, 'last_trajectory', []))[:200]}", flush=True)
                # score immediately so tu.last_trajectory belongs to THIS rollout
                # (generate() overwrites it, so scoring the list after the fact
                # mis-attributes every-but-the-last sample's tool-call/reward).
                label = score_one(tu, reg, q, ans, out)
                outs.append(out)
                rewards.append(REWARD[label])
                n_correct += int(label == "correct")
            rmean = sum(rewards) / len(rewards)
            _rstd = (sum((r - rmean) ** 2 for r in rewards) / len(rewards)) ** 0.5 + 1e-4
            # 205aj Dr. GRPO (arXiv 2503.20783): the std-normalizer is an
            # OPTIMIZATION BIAS (difficulty bias; our high-correct-rate, small-
            # sigma groups are the worst case). Mean-only advantages by
            # default; GRPO_DR_STD=1 restores the biased normalizer.
            rstd = _rstd if os.environ.get("GRPO_DR_STD", "0") == "1" else 1.0
            for out, r in zip(outs, rewards):
                # B-102: keep the negative samples too (the old `r <= 0:
                # continue` never pushed the policy away from no-call — the
                # transition the RL is meant to teach).
                prompt = tu._build_prompt(q)
                old = completion_logps(policy, tok, prompt, out, device)
                refp = completion_logps(ref, tok, prompt, out, device)
                batches.append({"prompt": prompt, "completion": out,
                                "adv": (r - rmean) / rstd, "old": old, "ref": refp})
        n = max(1, len(batches))
        for _ in range(args.n_epochs):
            for b in batches:
                block = policy.config.block_size
                ids = tok.encode(b["prompt"] + b["completion"]).ids
                comp = tok.encode(b["completion"]).ids
                if len(ids) > block:
                    ids = ids[-block:]  # keep the completion tail (B-78)
                    # recompute boundary from the window (bug: used untruncated
                    # prompt length → n_prompt>len(ids) → NaN when truncated)
                    n_prompt = max(1, len(ids) - len(comp))
                else:
                    # B-180: encode prompt alone (BPE boundary mismatch if we
                    # subtract completion from the joined encoding).
                    n_prompt = len(tok.encode(b["prompt"]).ids)
                x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
                y = torch.tensor([ids[1:]], dtype=torch.long, device=device)
                opt.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits, _, _ = policy(x)
                else:
                    logits, _, _ = policy(x)
                logp = F.log_softmax(logits.float(), dim=-1)
                newp = logp[0, n_prompt - 1:, :].gather(
                    1, y[0, n_prompt - 1:].unsqueeze(1)).squeeze(1)
                # B-197: clamp the logp GAP before exp — with a small RL dataset
                # the policy memorizes and exp(newp-old) blows up to inf -> NaN
                # gradients (the loss-spike contributor). Standard GRPO impls
                # bound the ratio / logp gap.
                ratio = torch.exp((newp - b["old"]).clamp(-20.0, math.log(2.0)))
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                # B-179: correct PPO/GRPO clipped surrogate = min(ratio*A, clip*A).
                # The old `min(ratio, clipped) * A` drops the LOWER clip when A<0
                # (min over scalar-mult picks the actual ratio, not the 1-eps floor),
                # allowing ratio->0 collapse on negative-advantage samples.
                pg = -torch.min(ratio * b["adv"], clipped * b["adv"])
                # B-102: canonical GRPO KL (DeepSeek-Math 2402.03300):
                # exp(ref - new) - (ref - new) - 1, ref detached — the old
                # form exp(new)*(new-ref) let gradients flow through the exp
                # and pushed the WRONG direction (it maximized the divergence).
                kl = torch.exp(b["ref"].detach() - newp) - \
                    (b["ref"].detach() - newp) - 1
                loss = (pg + args.beta * kl).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], args.max_norm)
                opt.step()
        acc = n_correct / (len(QUERIES) * args.group)
        loss_str = f"{loss.item():.4f}" if loss is not None else "n/a (no positives)"
        print(f"[grpo] round {step} | sampled {n} | accuracy {acc:.2%} | "
              f"loss {loss_str}", flush=True)

    from pathlib import Path
    import torch as _t
    merge_lora(policy)
    remove_lora(policy)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from storage import safe_save as _ss, ensure_disk as _ed
    _ed(out, 3.0, context="grpo final save")
    _ss({"model_state": policy.state_dict(), "config": ck["config"],
         "args": vars(args)}, out)
    print(f"[grpo] done -> {out}")


if __name__ == "__main__":
    main()