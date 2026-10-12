#!/usr/bin/env python3
"""ROUND-204v: the interactive chat GUI server — real inference to test the
deployed stack manually (the champions + the tool loop), the same code-path
as the CLI (query.py's answer logic: route.pick_adapter + the arith fast-path
+ the ToolUseMoE compiled-loop), server-shaped.

  .venv/bin/python3 src/gui_server.py [--port 8080] [--device cpu|cuda] [--host 127.0.0.1]

Endpoints:
  GET  /                  the chat UI (static/gui/index.html)
  GET  /api/models        champions + the loaded state
  POST /api/chat          {"session": ..., "message": ..., "mode": auto|chat|tool|direct}
"""
import argparse, json, os, secrets, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
# (b8.3 audit: the old raw "/tmp" insert on the import path let a stray
#  /tmp file shadow src modules — removed)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",
                      "expandable_segments:True,max_split_size_mb:128")

MODES = ("auto", "chat", "tool", "direct")
LOCK = threading.Lock()
SYS_PROMPT = "You are a helpful assistant."


def build_answer_fns(device: str):
    """Lazy per-mode loaders + the CLI-parity answer function."""
    import torch, torch.serialization
    from moe_model import MoEGPT, MoEConfig
    torch.serialization.add_safe_globals([MoEConfig])
    from tokenizers import Tokenizer
    import tool_use as tu
    from route import pick_adapter, TOOL_CKPT, CHAT_CKPT, DIRECT_CKPT
    from query import _try_arith

    _tok_cache: dict = {}

    def _tok_for(c):
        # 204ah-27: tokenizer from the CKPT's OWN vocab (the old fixed 16K bpe
        # tokenizer mismatched v3 32K champions).
        k = "v3" if int(getattr(c["config"], "vocab_size", 0)) >= 32768 else "bpe"
        if k not in _tok_cache:
            _tok_cache[k] = Tokenizer.from_file(str(REPO / (
                "v3-bank/tokenizer_v3.json" if k == "v3"
                else "data/processed/bpe_tokenizer/tokenizer.json")))
        return _tok_cache[k]
    cache: dict = {}

    def load(ckpt_key):
        ckpt = ckpt_key
        c = torch.load(ckpt, map_location="cpu", weights_only=True)
        model = MoEGPT(c["config"]).eval()
        model.load_state_dict(c["model_state"], strict=False)
        model.to(torch.bfloat16 if device == "cuda" else torch.float32)
        model.to(device)
        return model, _tok_for(c)

    def answer(session, message, mode, history=None):
        q = (message or "").strip()
        if not q:
            return {"reply": "", "mode": "none"}
        m = mode
        if m == "auto":
            m = pick_adapter(q)[1]
        # 204w: the multi-turn roll — the last-6 turns prepended in the
        # deployed marker shape (User:/Assistant:) so the model sees its own
        # previous answers (the CLI-session-clamp's server-side twin).
        _roll = []
        for _r, _t in (history or [])[-6:]:
            _roll.append(f"{'User' if _r == 'user' else 'Assistant'}: {_t}")
        _ctx = "\n".join(_roll) + "\n" if _roll else ""
        if m == "tool":
            # CLI-parity: the arith fast-path first, then the compiled tool loop
            ar = _try_arith(q)
            if ar:
                # 204z: NO hardcoded display markers — the [arith] protocol
                # string stays internal; the GUI serves the bare answer with a
                # kind tag for the frontend.
                return {"reply": ar[len("[arith] "):], "kind": "arith",
                        "mode": "tool-arith"}
            key = "tool"
            if key not in cache:
                _m, _tk = load(TOOL_CKPT)
                reg = tu.default_tools()
                # 205s: compiled_schema=False — the meter shape (the
                # compiled prompt degenerates v3 rollouts; serving must
                # match the eval's measured-good prompt).
                cache[key] = tu.ToolUseMoE(
                    _m, lambda t, _t=_tk: _t.encode(t).ids,
                    lambda i, _t=_tk: _t.decode(i), reg, compiled_schema=False)
            out = cache[key].generate(_ctx + "User: " + q, max_tokens=120,
                                      max_tool_calls=3, temperature=0.3,
                                      guided=True)
            # surface the trajectory steps too (the observation visibility)
            traj = getattr(cache[key], "last_trajectory", []) or []
            steps = [{"tool": e.get("name"), "args": e.get("arguments") or e.get("args"),
                      "obs": str(e.get("observation") or "")[:300]}
                     for e in traj]
            if not steps:
                # 204y/204z: no hardcoded prefixes — the raw text + a kind tag.
                return {"reply": str(out)[:600], "kind": "no-call",
                        "note": "the model did not call a tool",
                        "steps": [], "mode": "tool-no-call"}
            return {"reply": str(out)[:2000], "steps": steps,
                    "kind": "tool", "mode": "tool"}
        if m == "chat":
            key = "chat"
            if key not in cache:
                cache[key] = load(CHAT_CKPT)
            model, tk = cache[key]
            # 204x: the champion's EVAL template (the honest-eval's exact
            # shape) + the repetition guards — the raw "User: hello" with no
            # penalty degenerated into the "sats sats sats" loop (the classic
            # low-context repetition-degeneracy).
            _prompt = _ctx + SYS_PROMPT + "\n User: " + q + "\nAssistant:"
            ids = tk.encode(_prompt).ids
            import torch as _t
            idx = _t.tensor([ids], dtype=_t.long).to(device)
            with _t.no_grad():
                out = model.generate(idx, max_new_tokens=120, temperature=0.3,
                                     top_k=20, repetition_penalty=1.15,
                                     no_repeat_ngram=4)
            return {"reply": tk.decode(out[0][len(ids):].tolist(),
                                       skip_special_tokens=True)[:2000],
                    "kind": "chat", "mode": "chat"}
        # direct mode
        key = "direct"
        if key not in cache:
            cache[key] = load(DIRECT_CKPT)
        model, tk = cache[key]
        ids = tk.encode(q).ids
        import torch as _t
        idx = _t.tensor([ids], dtype=_t.long).to(device)
        with _t.no_grad():
            out = model.generate(idx, max_new_tokens=120, temperature=0.5,
                                 top_k=40)
        return {"reply": tk.decode(out[0][len(ids):].tolist(),
                                   skip_special_tokens=True)[:2000],
                "mode": "direct"}

    return answer, cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--system", default="You are a helpful assistant.",
                    help="the chat-mode system prompt (default = the "
                         "champion's eval-template; override freely)")
    args = ap.parse_args()

    answer, cache = build_answer_fns(args.device)
    global SYS_PROMPT
    SYS_PROMPT = args.system
    sessions: dict = {}   # session -> [("user"/"assistant", text)]: the multi-turn roll

    from flask import Flask, jsonify, request, send_from_directory, Response
    app = Flask(__name__, static_folder=None)
    from flask_cors import CORS
    CORS(app)

    GUI = REPO / "static" / "gui"

    @app.get("/")
    def index():
        return send_from_directory(GUI, "index.html")

    @app.get("/api/models")
    def models():
        from route import TOOL_CKPT, CHAT_CKPT, DIRECT_CKPT
        return jsonify({
            "tool": str(TOOL_CKPT), "chat": str(CHAT_CKPT),
            "direct": str(DIRECT_CKPT),
            "loaded": list(cache.keys()),
            "device": args.device,
        })

    @app.post("/api/chat")
    def chat():
        body = request.get_json(force=True, silent=True) or {}
        sess = str(body.get("session", "default"))[:64]
        mode = str(body.get("mode", "auto"))[:16]
        if mode not in MODES:
            mode = "auto"
        q = str(body.get("message", ""))
        if not q.strip():
            return jsonify({"reply": "", "mode": mode})
        t0 = time.time()
        with LOCK:
            try:
                _h = sessions.get(sess, [])
                r = answer(sess, q, mode, list(_h))
                _h.append(("user", q))
                _h.append(("assistant", str(r.get("reply", ""))))
                sessions[sess] = _h[-12:]
            except Exception as e:
                import traceback
                return jsonify({"reply": f"[error] {type(e).__name__}: {e}",
                                "mode": mode,
                                "trace": traceback.format_exc()[-600:]}), 500
        r["seconds"] = round(time.time() - t0, 2)
        return jsonify(r)

    # static assets (css/js if the page references any)
    @app.get("/static/<path:filename>")
    def static_files(filename: str):
        return send_from_directory(GUI, filename)

    # ---- OpenAI-compatible surface (the opencode v1 + DeepSeek Harness
    # ---- custom-provider contract): /v1/models + /v1/chat/completions.
    # ---- Protocol flags the harness docs demand of self-hosted servers:
    # ----   supportsDeveloperRole: false  (we accept role "system", never
    # ----                                "developer") and
    # ----   maxTokensField: max_tokens    (we read max_tokens, never
    # ----                                max_completion_tokens).
    _MODEL_MODES = {
        "ambatron-tool": "tool",
        "ambatron-chat": "chat",
        "ambatron-direct": "direct",
        "ambatron-auto": "auto",
    }

    @app.get("/v1/models")
    def oai_models():
        return jsonify({
            "object": "list",
            "data": [{"id": mid, "object": "model",
                      "created": 0, "owned_by": "ambatron"}
                     for mid in _MODEL_MODES],
        })

    @app.post("/v1/chat/completions")
    def oai_chat():
        global SYS_PROMPT
        body = request.get_json(force=True, silent=True) or {}
        msgs = body.get("messages") or []
        if not msgs or not isinstance(msgs, list):
            return jsonify({"error": {"message": "messages required",
                                      "type": "invalid_request_error",
                                      "code": 400}}), 400
        last_user = ""
        system_txt = ""
        for _m in msgs:
            _c = _m.get("content") or ""
            if isinstance(_c, list):
                _c = " ".join(str(p.get("text", "")) for p in _c
                              if isinstance(p, dict))
            if _m.get("role") in ("system", "developer"):
                system_txt += str(_c) + "\n"
            elif _m.get("role") == "user":
                last_user = str(_c)
        if not last_user.strip():
            return jsonify({"error": {"message": "no user message",
                                      "type": "invalid_request_error",
                                      "code": 400}}), 400
        mid = str(body.get("model", "ambatron-auto"))[:64]
        mode = _MODEL_MODES.get(mid, "auto")
        tid = "chatcmpl-" + secrets.token_hex(12)
        with LOCK:
            try:
                _old_sys = SYS_PROMPT
                if system_txt.strip():
                    SYS_PROMPT = system_txt[:2000]
                try:
                    r = answer("openai", last_user, mode, None)
                finally:
                    SYS_PROMPT = _old_sys
            except Exception as e:
                import traceback
                return jsonify({"error": {
                    "message": f"{type(e).__name__}: {e}",
                    "type": "server_error", "code": 500,
                    "trace": traceback.format_exc()[-600:]}}), 500
        content = str(r.get("reply", ""))[:4096] if isinstance(r, dict) else ""
        steps = (r.get("steps") or []) if isinstance(r, dict) else []
        tool_calls = None
        finish = "stop"
        if steps and r.get("kind") == "tool":
            # the internal loop already executed the tools; surface them as
            # function calls on the message (single-turn contract: the final
            # content is the loop's result, this tells the client what ran).
            tool_calls = [{"id": f"call_{i}", "type": "function",
                           "function": {"name": s.get("tool", "execute_python"),
                                        "arguments": json.dumps({
                                            "query": last_user[:500]})}}
                          for i, s in enumerate(steps[:3])]
        return jsonify({
            "id": tid, "object": "chat.completion",
            "created": int(time.time()),
            "model": mid,
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": content or None,
                                     **({"tool_calls": tool_calls}
                                        if tool_calls else {})},
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
        })

    print(f"[gui] http://{args.host}:{args.port}  (device={args.device})")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()