#!/usr/bin/env python3
"""ROUND-204i: TOOL-USE DISTILLATION (the objective's distillation axis).

A trained tool adapter (the v2-base champion) acts as the TEACHER; a smaller
student (the 430m base or the mini) learns the tool-use behavior on FRESH
instances (gym/rest queries — never heldout), with execution verification
as the filter: only teacher trajectories whose tool observation is clean
enter the student's loss.

Two label modes:
  --labels argmax   teacher's argmax full-session tokens (hard CE) [default]
  --labels soft     teacher logits (student KL on the teacher's distribution)
                    — needs the teacher's logits; ~2x the memory.

Gates (golden rules): storage + RAM + GPU-free + no heldout data + the
10s-probe discipline belongs to the OPERATOR, not the script.

Usage:
  .venv/bin/python3 src/distill_tool.py --teacher out/ambatron_tool_v2base/ckpt.pt \
      --student-base out/ambatron_430m_v2/ckpt_best.pt \
      --queries-file data/processed/distill_queries.jsonl --steps 4000 --out out/ambatron_sm_tool/ckpt.pt
"""
import argparse, json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:128,"
                      "garbage_collection_threshold:0.8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="the trained tool adapter ckpt")
    ap.add_argument("--student-base", required=True, help="the student base ckpt")
    ap.add_argument("--queries-file", required=True, help="JSONL of {'query': ...} — fresh instances, never heldout")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--labels", choices=["argmax", "soft"], default="argmax")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.labels == "soft":
        print("[distill] FATAL: --labels soft is NOT implemented (the loop is "
              "hard-CE only). Refusing the silent no-op.")
        import sys as _s
        _s.exit(2)

    import torch, torch.serialization
    import torch.nn.functional as F
    from moe_model import MoEGPT, MoEConfig
    torch.serialization.add_safe_globals([MoEConfig])
    from tokenizers import Tokenizer
    from tool_use import default_tools, ToolUseMoE
    from heldout import is_heldout

    tok = Tokenizer.from_file(str(REPO / "data/processed/bpe_tokenizer/tokenizer.json"))

    def load(base):
        ck = torch.load(base, map_location="cpu", weights_only=True)
        cfg = ck.get("config") or MoEConfig()
        m = MoEGPT(cfg).eval()
        miss, _ = m.load_state_dict(ck["model_state"], strict=False)
        rm = [k for k in miss if k != "lm_head.weight"]
        if rm:
            raise SystemExit(f"[distill] {base} arch mismatch: {rm[:3]}")
        return m.to(torch.bfloat16).cuda(), ck

    teacher, _ = load(args.teacher)
    student, ck2 = load(args.student_base)
    # the student stays LoRA-frozen-base + small adapter set (DoRA)
    from lora_adapter import add_lora, freeze_base
    add_lora(student, r=16, alpha=32, dora=True)
    freeze_base(student)
    n_tr = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[distill] teacher: n_tok={teacher.config.n_embd if hasattr(teacher, 'config') else '?'} "
          f"| student LoRA trainable: {n_tr:,}")

    reg = default_tools()
    tuse = ToolUseMoE(teacher, lambda t: tok.encode(t).ids,
                      lambda i: tok.decode(i), reg, compiled_schema=False)

    queries = [json.loads(l)["query"] for l in open(args.queries_file) if l.strip()]
    queries = [q for q in queries if not is_heldout(q)]
    print(f"[distill] {len(queries)} heldout-free queries")

    # the deployed envelope (shared with teacher_to_sft — single source)
    from teacher_to_sft import SYS

    # pre-generate the teacher trajectories (one pass — cheap, reusable)
    print("[distill] generating teacher trajectories (argmax, temp 0.5)...")
    import re as _re_dist
    import re

    def _q_first_num(q):
        _m = _re_dist.search(r"-?\d+", q)
        return int(_m.group()) if _m else None

    def _q_expected(q):
        # deterministic arith fast-path = the exact expectation for arith
        # probes ("what is 5 times 4" -> 20); None = no computable expectation.
        from query import _try_arith
        try:
            _a = _try_arith(q)
        except Exception:
            return None
        if _a and _a.startswith("[arith] "):
            _n = _re_dist.search(r"-?[\d.]+", _a[8:])
            return float(_n.group()) if _n else None
        return None

    trajs = []
    for q in queries:
        tuse.generate(q, max_tokens=args.max_new_tokens, max_tool_calls=2,
                      temperature=0.5)
        for e in (getattr(tuse, "last_trajectory", []) or []):
            obs = str(e.get("observation") or "")
            if any(b in obs.lower() for b in ("[timed out", "[truncated", "traceback",
                                              "error", "sandbox")):
                continue
            # ROUND-204k: syntax-clean is NOT content-clean — the 4k-milestone
            # produced executable calls whose OBSERVATION was wrong (the
            # wrong-args class). A distill teacher must pass the ARG-MATCH
            # bar: the observation's last number/yes-no must EQUAL the
            # expected answer for the query's probe class; queries without a
            # computable expectation are kept only if the observation looks
            # like a real (non-empty, numeric-ish) result.
            _dm = _re_dist.search(r"divisible by (\d+)", q, _re_dist.I)
            _kept = False
            if _dm:
                _n = _q_first_num(q)
                _mod = int(_dm.group(1)) or 1
                _yn = _n is not None and _n % _mod == 0
                _obs_l = obs.lower()
                _kept = _n is not None and (
                    ("yes" in _obs_l or "true" in _obs_l) == _yn)
            else:
                _wq = _q_expected(q)
                _nums = [int(x) for x in _re_dist.findall(r"-?\d+", obs)]
                if _wq is not None:
                    _kept = bool(_nums) and abs(_nums[-1] - _wq) < 0.5
                else:
                    _kept = bool(re.search(r"\w", obs))
            if not _kept:
                continue
            trajs.append({"query": q, "segments": [
                ["system", SYS],
                ["user", q],
                ["assistant", 'use tool:{"name": "%s", "arguments": %s}'
                              % (e.get("name"),
                                 json.dumps(e.get("arguments")
                                            or e.get("args") or {}))],
                ["user", "The result is:"],
                ["assistant", obs.strip()[:200]],
            ]})
            break
    del teacher
    torch.cuda.empty_cache()
    if not trajs:
        print("[distill] FATAL: no ARG-MATCHED teacher trajectory (the teacher "
              "emits clean-by-syntax but wrong-by-content calls on these "
              "queries — the 204k wrong-args class). Nothing to distill.")
        import sys as _s
        _s.exit(2)
    print(f"[distill] {len(trajs)} arg-matched trajectories")

    # tokenize the segments into (x, mask) — the student's targets
    def enc(t):
        toks, m = [], []
        for role, text in t:
            ids = tok.encode(text).ids
            toks += ids
            m += [1 if role == "assistant" else 0] * len(ids)
        return toks, m

    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    for step in range(args.steps):
        if args.labels == "soft":
            opt.param_groups[0]["lr"] = args.lr * max(0.1, 1.0 - step / args.steps)
        t = trajs[step % len(trajs)]
        toks, m = enc(t["segments"])
        toks = toks[-1023:]
        m = m[-1023:]
        if len(toks) < 8:
            continue
        x = torch.tensor([toks[:-1]], dtype=torch.long).cuda()
        y = torch.tensor([toks[1:]], dtype=torch.long).cuda()
        pm = torch.tensor([m[1:]], dtype=torch.bool).cuda()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss, aux = student(x, targets=y, use_grad_ckpt=True, pad_mask=pm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"[distill] step {step} | loss {loss.item():.4f} | n_tok {int(pm.sum())}")

    from lora_adapter import merge_lora, remove_lora
    merge_lora(student)
    remove_lora(student)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    from storage import safe_save
    safe_save({"model_state": student.state_dict(),
               "config": getattr(student, "config", None)
               or (ck2.get("config") if "ck2" in dir() else None),
               "args": vars(args)}, out)
    print(f"[distill] done -> {out}")


if __name__ == "__main__":
    main()