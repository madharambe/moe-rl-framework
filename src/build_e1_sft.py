#!/usr/bin/env python3
"""ROUND-204ac-b5.11: E1 trajectory pool -> SFT records (dedupe + pass-gate).

Consumes the REAL-trajectory reservoirs and emits SFT records in the exact
build_tool_sft segment format ({"segments":[(role,text)...]}, loss masked to
assistant only), so train_tool_sft.py can consume them directly once the v3
base exists.

Sources (all read-only):
  A) opencode db  (session_message: assistant tool parts + user intents)
  B) dsh sessions (session.jsonl.zstd: user/message + tool/call + tool/result)
  C) RSI flywheel out/agent_trajectories.jsonl (query/calls entries)

Pipeline per episode: group steps after an intent -> pass-gate each step
(status completed/success + error-marker-free output) -> drop all-error
episodes -> dedupe (intent+tool+args hash) -> heldout-probe guard -> emit.

NOTE on tools: tool names are PRESERVED as used (bash/read/edit/grep/webfetch/
websearch/glob/write/...). The live serving envelope currently registers only
execute_python + search_knowledge; registering the shell/file/web toolset on
the serving side is an E1-launch-time decision, banked separately.

Usage:
  .venv/bin/python3 src/build_e1_sft.py            # report + emit
  .venv/bin/python3 src/build_e1_sft.py --no-emit  # report only
"""
import argparse, hashlib, json, re, sqlite3, sys
from collections import Counter
from pathlib import Path

HOME = Path.home()
BASE = Path(__file__).resolve().parent.parent
OPENCODE_DB = HOME / ".local" / "share" / "opencode" / "opencode.db"
DSH = HOME / ".dsh" / "sessions"
RSI_LOG = BASE / "out" / "agent_trajectories.jsonl"
OUT = BASE / "v3-bank" / "e1_tool_sft.jsonl"
MODE = "verifier"  # set from --pool-mode (B3 arms)
OUT_TRD = BASE / "v3-bank" / "e1_tool_sft_trd.jsonl"
OUT_NAIVE = BASE / "v3-bank" / "e1_tool_sft_naive.jsonl"

MAX_OBS = 500
MAX_CALLS = 12

ERR_MARKERS = ("traceback", "loop detected", "exception",
               "permission denied", "no such file", "command not found",
               "sandbox process exited", "is not recognized",
               "fatal:", "segmentation fault")

def _err_like(text):
    t = text.lower()
    return any(m in t for m in ERR_MARKERS)

def _m(session, seq, tool, args):
    return hashlib.md5(f"{session}|{seq}|{tool}|{str(args)[:400]}".encode()).hexdigest()

def _emit_episode(ep, seen, records, stats):
    intent = ep.get("intent") or ""
    steps = ep.get("steps") or []
    fallback = ep.get("fallback") or ""
    if not intent or not steps:
        stats["drop_no_ctx"] += 1
        return
    try:
        from heldout import is_heldout
        if is_heldout(intent):
            stats["drop_heldout"] += 1
            return
    except Exception:
        pass
    # B3 three-arm ablation (ROUND-204ac-b7): verifier (pass-only, the ship
    # path) vs naive (all steps INCLUDING error observations — naive-trace
    # SFT) vs trd (retry-converged episodes only — the prefix-failure
    # avoidance arm; error-call then a later corrected call).
    if MODE == "trd":
        flags = [ok for _, _, _, ok in steps]
        err_idx = [i for i, f in enumerate(flags) if not f]
        ok_idx = [i for i, f in enumerate(flags) if f]
        if not any(i < j for i in err_idx for j in ok_idx):
            stats["drop_no_retry"] += 1
            return
    segs = [("user", "User: " + intent.strip() + "\n")]
    healthy_obs = None
    last_obs = None
    for name, args, obs, ok in steps:
        key = _m(ep.get("session", ""), ep.get("seq", 0), name, args)
        if key in seen:
            continue
        seen.add(key)
        obs_s = str(obs)
        if MODE == "verifier" and (not ok or _err_like(obs)):
            continue
        segs.append(("assistant", "Assistant: use tool:" +
                     json.dumps({"name": name, "arguments": args}) + "\n"))
        segs.append(("tool", "result: " + obs_s[:MAX_OBS] + "\n"))
        if not healthy_obs and MODE == "verifier":
            healthy_obs = obs_s
        if MODE != "verifier":
            last_obs = obs_s
    if MODE == "verifier":
        if not healthy_obs:
            if fallback.strip():
                answer = fallback.strip()[:300]
                segs.append(("assistant", "Assistant: " + answer +
                             ("..." if len(answer) == 300 else "")))
                records.append({"segments": segs})
                stats["emitted_fallback"] += 1
                return
            stats["drop_allempty"] += 1
            return
        answer = healthy_obs[:300]
    else:
        raw = last_obs if (last_obs or "").strip() else fallback
        if not raw or not raw.strip():
            stats["drop_allempty"] += 1
            return
        answer = raw.strip()[:300]
    segs.append(("assistant", "Assistant: The result is " + answer +
                 ("..." if len(answer) == 300 else "")))
    records.append({"segments": segs})
    stats["emitted"] += 1

def scan_opencode(records, seen, stats):
    con = sqlite3.connect("file:" + str(OPENCODE_DB) + "?mode=ro", uri=True)
    cur = con.cursor()
    cur.execute("select session_id, time_created, type, data from session_message "
                "where type in ('user','assistant') order by session_id, time_created, seq")
    cur_session = None
    cur_intent = None
    steps_ep = []
    last_text = ""
    seq = 0
    for sid, t, typ, data in cur:
        try:
            o = json.loads(data)
        except Exception:
            continue
        if sid != cur_session:
            if cur_session is not None:
                _emit_episode({"session": cur_session, "seq": seq, "intent": cur_intent,
                              "steps": steps_ep, "fallback": last_text},
                              seen, records, stats)
            cur_session, cur_intent, steps_ep, seq, last_text = sid, None, [], 0, ""
        if typ == "user":
            txt = ""
            if isinstance(o, dict):
                txt = o.get("text") or ""
                if isinstance(txt, str) is False:
                    txt = ""
            if isinstance(txt, str) and txt.strip():
                _emit_episode({"session": sid, "seq": seq, "intent": cur_intent,
                              "steps": steps_ep, "fallback": last_text},
                              seen, records, stats)
                steps_ep = []
                cur_intent = txt
                last_text = ""
                seq += 1
        elif typ == "assistant":
            for part in o.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "text":
                    if (part.get("text") or "").strip():
                        last_text = (part.get("text") or "").strip()
                    continue
                if not isinstance(part, dict) or part.get("type") != "tool":
                    continue
                st = part.get("state") or {}
                name = part.get("name") or "?"
                inp = st.get("input") or {}
                out = ((st.get("metadata") or {}).get("output") or "")[:MAX_OBS]
                ok = st.get("status") in ("completed", "success")
                if cur_intent is None:
                    stats["steps_no_intent"] += 1
                steps_ep.append((name, inp, out, ok))
    if cur_session is not None:
        _emit_episode({"session": cur_session, "seq": seq, "intent": cur_intent,
                      "steps": steps_ep, "fallback": last_text},
                      seen, records, stats)
    con.close()

def scan_dsh(records, seen, stats):
    import io
    import zstandard as zstd
    for f in sorted(DSH.rglob("session.jsonl.zstd")):
        sid = f.parent.name
        try:
            with open(f, "rb") as fh:
                text = io.TextIOWrapper(zstd.ZstdDecompressor().stream_reader(fh),
                                        encoding="utf-8", errors="replace")
                calls = {}
                ep_steps = []
                cur_intent = None
                for line in text:
                    try:
                        ev = json.loads(line)
                    except Exception:
                        continue
                    t = ev.get("type")
                    d = ev.get("data") or {}
                    if t == "user/message":
                        content = d.get("content") or []
                        cur_intent = " ".join(
                            p.get("text", "") for p in content
                            if isinstance(p, dict) and p.get("type") == "text").strip()
                        _emit_episode({"session": sid, "seq": 0, "intent": cur_intent,
                              "steps": ep_steps}, seen, records, stats)
                        ep_steps = []
                    elif t == "tool/call":
                        calls[d.get("callId")] = (d.get("name") or "?",
                                                  d.get("arguments") or {})
                    elif t == "tool/result":
                        cid = None
                        text2 = ""
                        try:
                            cid = d["message"]["content"][0].get("toolCallId")
                            text2 = d["message"]["content"][0]["content"][0].get("text", "")
                        except Exception:
                            pass
                        if cid in calls:
                            name, args = calls.pop(cid)
                            if cur_intent is not None:
                                ep_steps.append((name, args, text2,
                                                 not _err_like(text2)))
                _emit_episode({"session": sid, "seq": 0, "intent": cur_intent,
                              "steps": ep_steps}, seen, records, stats)
        except Exception as e:
            stats["dsh_warn"] += 1

def scan_rsi(records, seen, stats):
    try:
        for line in open(RSI_LOG, encoding="utf-8"):
            try:
                e = json.loads(line)
            except Exception:
                continue
            query = e.get("query", "")
            if not query:
                continue
            calls = e.get("calls") or []
            steps = []
            for c in calls[:MAX_CALLS]:
                name = c.get("name", "")
                args = c.get("arguments")
                obs = c.get("observation", "")
                steps.append((name, args if isinstance(args, dict) else {},
                              obs, not _err_like(str(obs))))
            _emit_episode({"session": "rsi", "seq": 0, "intent": query,
                              "steps": steps}, seen, records, stats)
    except Exception as e:
        stats["rsi_warn"] += 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-emit", action="store_true")
    ap.add_argument("--pool-mode", choices=["verifier", "naive", "trd"],
                    default="verifier",
                    help="B3 ablation arm: verifier (pass-only, ship) | naive "
                         "(all steps incl errors) | trd (retry-converged only)")
    args = ap.parse_args()
    global MODE, OUT
    MODE = args.pool_mode
    if args.pool_mode == "trd":
        OUT = OUT_TRD
    elif args.pool_mode == "naive":
        OUT = OUT_NAIVE
    records, seen = [], set()
    stats = Counter()
    stats["emitted"] = 0
    scan_opencode(records, seen, stats)
    scan_dsh(records, seen, stats)
    scan_rsi(records, seen, stats)
    print("== E1 pool -> SFT ==")
    print("  emitted records   : " + str(stats["emitted"]) +
          " (fallback-answered " + str(stats.get("emitted_fallback", 0)) + ")")
    print("  drops             : no_ctx " + str(stats["drop_no_ctx"]) +
          " | heldout " + str(stats["drop_heldout"]) +
          " | allempty " + str(stats["drop_allempty"]))
    print("  dedupe seen       : " + str(len(seen)))
    if not args.no_emit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print("  EMITTED           : " + str(OUT) + " (" + str(len(records)) + " records)")

if __name__ == "__main__":
    main()
