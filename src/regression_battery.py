#!/usr/bin/env python3
"""ROUND-187 (B4/E11/E12): regression battery — after EVERY tool decision, a
fixed battery re-measures the (possibly new) champion across all axes so
silent drift is caught (promotion currently only re-measures the /32 fresh).

Battery (all exist already):
  - tool heldout /32 (eval_tool_use --heldout)
  - hard suite /29 (eval_tool_hard)
  - chat honest /15 + quality (eval_chat_honest)
  - spatial /40 (eval_spatial)

Output: one CSV row appended to .state/cycle_metrics.csv:
  ts,cand,tool_32,hard_29,chat_15,chat_q,spatial_40

Usage: src/regression_battery.py --ckpt <ckpt.pt> [--tag tag] [--csv path]
Called automatically by auto_optimize on PROMOTIONS (report-only; never blocks).
"""
import argparse, csv, io, json, re, subprocess, sys, time

ROOT = __file__.rsplit("/", 1)[0]
sys.path.insert(0, ROOT)


def run_eval(cmd, pattern, timeout_s=900):
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             timeout=timeout_s).stdout
    except Exception as e:
        return f"ERR {e}"
    m = re.search(pattern, out)
    return m.group(1) if m else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)          # the TOOL candidate/champ
    ap.add_argument("--tag", default="manual")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    csv_path = args.csv or f"{ROOT}/../.state/cycle_metrics.csv"
    champs = json.load(open(f"{ROOT}/../out/champions.json"))

    tool = run_eval(f"{sys.executable} -u {ROOT}/eval_tool_use.py --ckpt {args.ckpt} "
                    f"--heldout --compiled --bf16 --repeat 1 2>&1 | tail -1",
                    r"correct[^0-9]*([0-9]+)/([0-9]+)")
    hard = run_eval(f"{sys.executable} -u {ROOT}/eval_tool_hard.py --ckpt {args.ckpt} "
                    f"--bf16 2>&1 | grep 'no-call'", r"no-call\s*:\s*([0-9]+)")
    # drift axes: the EXISTING champions (chat + spatial) — battery doesn't
    # re-roll candidates, it guards the whole system from silent regression.
    chat = run_eval(f"{sys.executable} -u {ROOT}/eval_chat_honest.py --ckpt "
                    f"{champs['chat']['ckpt']} --bf16 --repeat 1 2>&1 | "
                    f"grep 'helpful-honest'", r"helpful-honest ([0-9]+)/([0-9]+) \| quality ([0-9]+)/([0-9]+)")
    spatial = run_eval(f"{sys.executable} -u {ROOT}/eval_spatial.py --ckpt "
                       f"{champs['spatial']['ckpt']} --bf16 2>&1 | grep 'spatial-eval'",
                       r"correct ([0-9]+)/([0-9]+)")

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    row = [ts, args.tag, args.ckpt.rsplit("/", 1)[-1],
           tool or "?", hard or "?", chat or "?", spatial or "?"]
    header = ["ts", "tag", "cand", "tool_32", "hard_no_call", "chat_15_q", "spatial"]
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    print(f"[battery] {' | '.join(row[1:])} -> {csv_path}")
    raise SystemExit(0)


if __name__ == "__main__":
    import os
    main()