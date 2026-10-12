"""
Tool-use wrapper for MoE LLM (experimental).
Uses simple text patterns compatible with word-level vocabulary:
  tool: call tool_name
  tool: result ...
Based on Toolformer, Gorilla, and Nexus-TinyFunction research (2023-2026).

NOTE: Full tool-use requires special tokens added to vocabulary.
The current implementation uses compatible text patterns for the
existing code corpus tokenizer.
"""
import json, re, inspect, os

import torch  # B-165: the @torch.no_grad on generate needs this at module scope
from typing import Dict, Callable, Any, Optional, List, get_type_hints
from pathlib import Path

# Portable root: override with RSI_HOME, otherwise use this checkout.
import os as _os_rsi
from pathlib import Path as _Path_rsi
RSI_HOME = _os_rsi.environ.get('RSI_HOME') or str(_Path_rsi(__file__).resolve().parent.parent)



TOOL_CALL_START = "use tool:"
TOOL_CALL_END = ""  # B-23: was None -> f-string rendered literal "None" in the prompt
TOOL_RESPONSE_START = "result:"
TOOL_RESPONSE_END = ""


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict] = {}

    def register(self, func: Callable, name: Optional[str] = None, description: Optional[str] = None):
        name = name or func.__name__
        sig = inspect.signature(func)
        hints = get_type_hints(func)
        parameters = {"type": "object", "properties": {}, "required": []}
        for p_name, p_param in sig.parameters.items():
            if p_name == "return":
                continue
            param_type = hints.get(p_name, str).__name__
            parameters["properties"][p_name] = {
                "type": param_type,
                "description": f"Parameter {p_name}",
            }
            if p_param.default is inspect.Parameter.empty:
                parameters["required"].append(p_name)

        self._tools[name] = {
            "name": name,
            "description": description or func.__doc__ or f"Execute {name}",
            "parameters": parameters,
            "fn": func,
        }

    def get_tools_section(self, compiled: bool = False) -> str:
        """Tool listing for the prompt. Default: verbose JSON-per-param schema.
        compiled=True (B-75, TSCG-style): compact signature form
        'name(param: type, ...) — desc' — token-efficient and much easier for
        small models to parse (the representation change is the dominant lever)."""
        lines = ["Available tools:"]
        for t in self._tools.values():
            if compiled:
                props = t["parameters"]["properties"]
                sig = ", ".join(f"{p}: {v.get('type', 'str')}" for p, v in props.items())
                lines.append(f"  - {t['name']}({sig}) — {t['description']}")
            else:
                params_str = json.dumps(t["parameters"]["properties"])
                lines.append(f"  - {t['name']}: {t['description']} ({params_str})")
        return "\n".join(lines)

    def get_tool_definitions(self) -> list:
        return [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}
            for t in self._tools.values()
        ]

    def call(self, name: str, arguments: Dict) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'"
        try:
            result = self._tools[name]["fn"](**arguments)
            return json.dumps(result) if not isinstance(result, str) else result
        except Exception as e:
            return json.dumps({"error": str(e)})

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())


class ToolUseMoE:
    def __init__(self, model, tokenizer_func, detokenizer_func, tool_registry: Optional[ToolRegistry] = None,
                 compiled_schema: bool = False):
        self.model = model
        self.tokenize = tokenizer_func
        self.detokenize = detokenizer_func
        self.tools = tool_registry or ToolRegistry()
        self.compiled_schema = compiled_schema

    def _build_prompt(self, user_query: str) -> str:
        tools_section = self.tools.get_tools_section(compiled=self.compiled_schema)
        return (
            f"{tools_section}\n\n"
            f"To use a tool, respond with:\n"
            f"{TOOL_CALL_START}{{\"name\": \"tool_name\", \"arguments\": {{...}}}}{TOOL_CALL_END}\n\n"
            f"User: {user_query}\nAssistant:"
        )

    def _parse_tool_call(self, text: str) -> Optional[Dict]:
        # r279: the tokenizer cannot represent a double quote, so real
        # JSON is unparseable by construction; try the tolerant parser
        # (which also accepts real JSON) BEFORE the legacy quoted-JSON
        # scanner.
        _t = parse_tolerant_call(text)
        if _t:
            return _t
        # Bug: TOOL_CALL_END="" made the regex match the empty string first.
        # Now: find the marker and scan for a brace-balanced JSON object.
        start = text.find(TOOL_CALL_START)
        if start < 0:
            return None
        i = text.find("{", start)
        if i < 0:
            return None
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[i:j + 1])
                        except json.JSONDecodeError:
                            return None
        return None

    @torch.no_grad()
    def generate(self, user_query: str, max_tokens: int = 200, max_tool_calls: int = 3,
                 top_k: int = 40,
                 hint: Optional[str] = None, temperature: float = 0.7,
                 no_repeat_ngram: int = 0, repetition_penalty: float = 1.0,
                 guided: bool = False) -> str:
        """hint (START 2503.04625 Hint-infer): when the model answers WITHOUT a
        tool call, append a reasoning hint (e.g. "Wait, maybe using a tool here
        is a good idea.") and generate once more. Default None = no extra pass.
        temperature: exposed for deployment tuning (small-B model is flaky at
        0.7; use ~0.3-0.5 for more deterministic tool-call emission).
        no_repeat_ngram / repetition_penalty (B-42): anti-degeneration decoding
        for small bases that loop — ngram4+rep1.2 measurably de-repeats."""
        if hasattr(self, "_loop_history"):
            self._loop_history.clear()
        prompt = self._build_prompt(user_query)
        tokens = self.tokenize(prompt)
        if not tokens:
            return "# Failed to tokenize query"
        import torch
        idx = torch.tensor([tokens], dtype=torch.long).to(
            next(self.model.parameters()).device)
        # ROUND-185: cache the HEAD (instructions + user query) so long-session
        # clamps keep the actual question instead of blind tail-cuts (multi-
        # round tool sessions used to drop the user's query once observations
        # exceeded block_size — real context loss).
        self._head_tokens = list(tokens)

        full_output = ""
        completion = ""
        tool_calls_made = 0
        hint_used = False
        trajectory = []  # data flywheel: tool calls + observations -> future SFT
        self.last_trajectory = trajectory  # B-105: expose for the evaluators

        for _ in range(max_tool_calls):
            prev_len = idx.shape[1]  # B-23: slice ONLY the newly generated
            # tokens — out[0, len(tokens):] re-detokenized the whole accumulated
            # stream (gen1+result+gen2) across BPE boundaries, which merged
            # tokens across segment edges and mangled the returned text.
            _gs = None
            if guided:
                try:
                    from json_guided import GuidedSampler
                    _gs = GuidedSampler(self.detokenize)
                except Exception:
                    _gs = None
            with torch.no_grad():
                out = self.model.generate(
                    idx, max_new_tokens=max_tokens,
                    temperature=temperature, top_k=top_k,
                    no_repeat_ngram=no_repeat_ngram,
                    repetition_penalty=repetition_penalty,
                    guided_hook=_gs,
                )
            generated_text = self.detokenize(out[0].tolist())
            new_toks = out[0, prev_len:]
            new_text = self.detokenize(new_toks.tolist())
            full_output = generated_text  # prompt+completion (flywheel log only)
            completion = new_text  # B-23: return the COMPLETION — the prompt's
            # instruction block contains "use tool:{...}" whose literal {...} is
            # invalid JSON, so parsing the prompt+completion text always fails
            # at the prompt's example first (same contamination MoEModel
            # generate_code had; fixed the same way).

            tool_call = self._parse_tool_call(new_text)
            if tool_call is None:
                if hint is not None and not hint_used:
                    # START Hint-infer: nudge the model toward tool use once.
                    hint_used = True
                    hids = self.tokenize(f"\n{hint}\n")
                    idx = torch.cat([idx, torch.tensor([hids], dtype=torch.long,
                                                       device=idx.device)], dim=1)
                    if idx.shape[1] > self.model.config.block_size:
                        idx = _clamp_head(idx, self._head_tokens, self.model.config.block_size)
                    continue
                break

            name = tool_call.get("name", "")
            arguments = tool_call.get("arguments", {})
            # Loop detection: identical call 3x in a row -> structured error
            # (production agent guidance; forces the model to try a new approach).
            hist = getattr(self, "_loop_history", [])
            sig = json.dumps([name, arguments], sort_keys=True)
            hist.append(sig)
            if len(hist) >= 3 and len(set(hist[-3:])) == 1:
                result = json.dumps({"error": "Loop detected: same tool call with "
                                              "identical arguments 3 times. Try a "
                                              "different approach or answer directly."})
            else:
                self._loop_history = hist[-5:]
                result = self.tools.call(name, arguments)
            tool_calls_made += 1
            trajectory.append({"name": name, "arguments": arguments,
                               "observation": result[:500]})

            response_text = f"\n{TOOL_RESPONSE_START}{result}{TOOL_RESPONSE_END}\n"
            response_tokens = self.tokenize(response_text)
            idx = torch.cat([idx, torch.tensor([response_tokens], dtype=torch.long,
                                               device=idx.device)], dim=1)

            if idx.shape[1] > self.model.config.block_size:
                idx = _clamp_head(idx, self._head_tokens, self.model.config.block_size)

        # data flywheel: persist the trajectory for future SFT (EvolveR-style)
        try:
            from pathlib import Path
            log = Path(__file__).resolve().parent.parent / "out" / "agent_trajectories.jsonl"
            log.parent.mkdir(parents=True, exist_ok=True)
            # 11th-audit: (a) filter out eval-heldout probe queries to break the
            # self-contamination loop (eval generate -> trajectories -> flywheel_to_sft
            # -> training data -> eval inflates); (b) rotate at 10k lines to keep the
            # file bounded (the tool-SFT converters re-read the ENTIRE file each run).
            # 14th-audit P0: the patterns were MIXED-CASE but compared against a
            # lowercased query, so "Explain PPO" never matched "explain ppo..." and
            # the filter silently let heldout probes into the training loop (measured:
            # 277/21077 trajectory entries contain heldout probe queries). Lowercase
            # the patterns too, so the match is case-insensitive.
            # ROUND-174 (fixed): a single shared registry (heldout.py) — the old
            # per-file _heldout sets drifted and let the 22 new probes leak
            # 245 trajectories. is_heldout covers ALL 32 CASES_HELDOUT.
            from heldout import is_heldout
            _q_lower = (user_query or "").lower()
            if not is_heldout(user_query):
                with open(log, "a") as f:
                    f.write(json.dumps({"query": user_query, "calls": trajectory,
                                        "final": full_output[:1000]}) + "\n")
                # rotate: keep only the newest 10k lines (~5MB)
                try:
                    if log.stat().st_size > 5_242_880:
                        lines = log.read_text(errors="ignore").splitlines(True)
                        if len(lines) > 10_000:
                            log.write_text("".join(lines[-10_000:]))
                except (OSError, ValueError):
                    pass
        except Exception:
            pass

        return completion


# ---- Built-in tools for RSI ----

# Bootstrap executed inside a SEPARATE subprocess (a real jail). It reads the
# model code from stdin, applies OS-level confinement (Landlock fs sandbox,
# RLIMIT_CPU/AS/NPROC/FSIZE, best-effort unshare + privilege drop), then runs
# the code and prints the captured result to stdout for the parent to collect.
_SANDBOX_BOOTSTRAP = r'''
import os, sys, io, ast, resource, ctypes, site




def _apply_landlock(sandbox_dir):
    # Landlock (Linux 5.13+): deny-by-default fs access. Reads are allowed from
    # the interpreter's import roots (stdlib/site-packages, /usr, ...); writes
    # are permitted ONLY inside sandbox_dir. Everything else (e.g. user home,
    # system binaries) is read- and write-denied.
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        NR_CREATE, NR_ADD, NR_RESTRICT = 444, 445, 446
        NR_PRCTL = 157
        A = dict(EXECUTE=1 << 0, WRITE_FILE=1 << 1, READ_FILE=1 << 2,
                 READ_DIR=1 << 3, REMOVE_DIR=1 << 4, REMOVE_FILE=1 << 5,
                 MAKE_CHAR=1 << 6, MAKE_DIR=1 << 7, MAKE_REG=1 << 8,
                 MAKE_SOCK=1 << 9, MAKE_FIFO=1 << 10, MAKE_BLOCK=1 << 11,
                 MAKE_SYM=1 << 12, REFER=1 << 13,
                 # B-212: TRUNCATE (1<<14, ABI3/kernel 5.18+) and IOCTL_DEV
                 # (1<<15, ABI5/kernel 6.10+) must be HANDLED (denied-by-default).
                 # The old ruleset left them UNHANDLED, so `os.truncate()` /
                 # open(O_TRUNC) / ioctl on device nodes were NOT gated by
                 # Landlock at all — a model could truncate (empty) any file it
                 # could otherwise READ (e.g. system site-packages .py), despite
                 # WRITE_FILE being denied. Adding them to the handled set and
                 # granting them NOWHERE makes truncation/ioctl an EACCES
                 # everywhere, and gives the model's OWN files write-overwrite
                 # via WRITE_FILE as before. (packaged landlock.h lags at
                 # REFER=1<<13, so define the bits here.)
                 TRUNCATE=1 << 14, IOCTL_DEV=1 << 15)
        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64),
                        ("handled_access_net", ctypes.c_uint64),
                        ("scoped", ctypes.c_uint64)]
        class PathBeneath(ctypes.Structure):
            _fields_ = [("allowed_access", ctypes.c_uint64),
                        ("parent_fd", ctypes.c_int)]
        handled = 0
        for v in A.values():
            handled |= v
        # handled_access_net: deny ALL TCP connect/bind (no net rules added).
        net = (1 << 0) | (1 << 1)
        rs = RulesetAttr(handled, net, 0)
        libc.syscall.argtypes = [ctypes.c_long, ctypes.c_void_p,
                                 ctypes.c_size_t, ctypes.c_uint32]
        libc.syscall.restype = ctypes.c_long
        fd = libc.syscall(NR_CREATE, ctypes.byref(rs), ctypes.sizeof(rs), 0)
        if fd < 0:
            rs = RulesetAttr(handled, 0, 0)
            fd = libc.syscall(NR_CREATE, ctypes.byref(rs), ctypes.sizeof(rs), 0)
        if fd < 0:
            return
        ro_read = A["READ_FILE"] | A["READ_DIR"]                 # NO EXECUTE
        ro_exec = A["READ_FILE"] | A["READ_DIR"] | A["EXECUTE"]
        rw = (A["READ_FILE"] | A["READ_DIR"] | A["EXECUTE"] |
              A["WRITE_FILE"] | A["MAKE_REG"] | A["MAKE_DIR"] |
              A["REMOVE_FILE"] | A["REMOVE_DIR"] | A["TRUNCATE"])
        # EXECUTE allowed ONLY on library/import roots (so `import math`, numpy,
        # torch .so load) — NEVER on /usr/bin|/bin, where os.system/subprocess
        # would exec arbitrary programs. This was the exec-escape: /usr was
        # EXECUTE-whitelisted so os.system('id') ran.
        exec_roots = set()
        for p in sys.path:
            if p:
                exec_roots.add(os.path.abspath(p))
        try:
            for s in site.getsitepackages():
                exec_roots.add(os.path.abspath(s))
        except Exception:
            pass
        up = site.getusersitepackages()
        if up:
            exec_roots.add(os.path.abspath(up))
        for d in ("/usr/lib", "/usr/lib64", "/lib", "/lib64", "/opt/lib"):
            if os.path.isdir(d):
                exec_roots.add(d)
        # read-only, NON-executable roots (system binary dirs + config data):
        # blocks `os.system` / `subprocess` from running any /usr/bin|/bin binary.
        ro_roots = set()
        for d in ("/usr/bin", "/bin", "/sbin", "/usr/sbin",
                  "/usr/local/bin", "/opt/bin", "/opt"):
            if os.path.isdir(d):
                ro_roots.add(d)
        libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_int,
                                 ctypes.c_void_p, ctypes.c_uint32]

        def apply_rule(root, allowed):
            try:
                dfd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            except OSError:
                return
            pb = PathBeneath(allowed, dfd)
            libc.syscall(NR_ADD, fd, 1, ctypes.byref(pb), 0)
            os.close(dfd)

        for r in exec_roots:
            apply_rule(r, ro_exec)
        # ROUND-204ac-b5.15 (shell mode): bash needs to EXECUTE /usr/bin|/bin
        # tools (ls, cat, grep, ...). Landlock stays WRITE-denied there; only
        # EXECUTE is additionally allowed. Everything else (home, etc.) stays
        # fully denied; network stays denied (handled_access_net).
        shell_mode = os.environ.get("RSI_SHELL") == "1"
        for r in ro_roots:
            apply_rule(r, ro_exec if shell_mode else ro_read)
        apply_rule(sandbox_dir, rw)
        # Landlock requires PR_SET_NO_NEW_PRIVS on this kernel before
        # landlock_restrict_self (otherwise it fails with EPERM).
        libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_ulong,
                                 ctypes.c_ulong, ctypes.c_ulong]
        libc.syscall(NR_PRCTL, 38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS = 38
        libc.syscall.argtypes = [ctypes.c_long, ctypes.c_int, ctypes.c_uint32]
        rc = libc.syscall(NR_RESTRICT, fd, 0)
        os.close(fd)
        return rc == 0
    except Exception:
        return False

def _main():
    code = sys.stdin.read()
    sandbox_dir = os.environ.get("RSI_SANDBOX_DIR", ".")
    try:
        flags = 0
        for n in ("CLONE_NEWUSER", "CLONE_NEWNET", "CLONE_NEWNS"):
            flags |= getattr(os, n, 0)
        if flags:
            os.unshare(flags)
    except Exception:
        pass
    try:
        import pwd
        nobody = pwd.getpwnam("nobody").pw_uid
        os.setgroups([])
        os.setgid(nobody)
        os.setuid(nobody)
    except Exception:
        pass
    try:
        cpu = max(1, int(float(os.environ.get("RSI_TIMEOUT", "10"))))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    except Exception:
        pass
    try:
        # ROUND-169 (F-2): fork-bomb DoS — 128 children x 300-600MB (after
        # numpy import) = >60GB on a 14GB box, OOM-killing the host within the
        # 30s wall. 16 x 600MB ~ 10GB worst case (still heavy but survivable +
        # earlyoom), AS 2GB caps each process's VA tighter.
        resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))
    except Exception:
        pass
    try:
        aslim = 2 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (aslim, aslim))
    except Exception:
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (200 * 1024 * 1024, 200 * 1024 * 1024))
    except Exception:
        pass
    if not _apply_landlock(sandbox_dir):
        sys.stderr.write("SANDBOX_FAILED: Landlock confinement could not be "
                         "applied; refusing to run untrusted code\n")
        sys.exit(1)
    # ROUND-204ac-b5.15: SHELL MODE — identical confinement (Landlock + rlimits
    # + privesc drop + no-new-privs), then exec bash on the workspace dir. The
    # parent set RSI_SANDBOX_DIR=<workspace> so Landlock's rw grants are the
    # workspace; exec of system bins allowed (see _apply_landlock shell_mode).
    if os.environ.get("RSI_SHELL") == "1":
        try:
            os.chdir(sandbox_dir)
        except OSError:
            pass
        os.execvp("/bin/bash", ["bash", "-c", code])
    old = sys.stdout
    buf = io.StringIO()
    sys.stdout = buf
    try:
        tree = ast.parse(code)
        is_expr = (len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr))
    except SyntaxError:
        is_expr = False
    out = ""
    try:
        if is_expr:
            val = eval(compile(code, "<sandbox>", "eval"))
            out = buf.getvalue()
            if val is not None:
                out += repr(val)
        else:
            exec(compile(code, "<sandbox>", "exec"))
            out = buf.getvalue()
    except BaseException as e:
        out = f"{type(e).__name__}: {e}"
    sys.stdout = old
    print(out, end="")

_main()
'''


def _execute_python(code: str, timeout_s: float = 10.0) -> str:
    """Execute MODEL-GENERATED Python in a REAL jail: a separate subprocess
    (not in-process) with OS-level confinement — Landlock fs sandbox,
    RLIMIT_CPU/AS/NPROC/FSIZE, best-effort namespace + privilege drop, and a
    wall-clock subprocess timeout that is the authoritative backstop. The old
    in-process `_locked` builtins were bypassable via
    `(()).__class__.__bases__[0].__subclasses__()` -> real builtins -> RCE.

    API unchanged: returns captured stdout/stderr (or a structured result for
    a bare expression), and returns an error STRING on timeout/violation rather
    than crashing the parent."""
    import sys, os, subprocess, tempfile, shutil, signal, selectors, time
    # 10th-audit SAND-2: clamp the model-supplied timeout. The tool schema lets
    # the model pass timeout_s directly; an unbounded value (e.g. 9999) would
    # produce an 8-hour wall window and defeat RLIMIT_CPU. Hard cap at 30s.
    timeout_s = min(float(timeout_s), 30.0)
    # ROUND-170 (F2): sandbox writes go to a deduped per-call temp DIR, and the
    # parent enforces a TOTAL byte budget across the run. RLIMIT_FSIZE is
    # per-file — a model looping N x 200MB writes can still ENOSPC the 97%-full
    # NVMe. Put the dir under /tmp (same fs) but track cumulative bytes and
    # SIGKILL the tree past the budget in the read loop (see `sandbox_bytes`).
    sandbox_dir = tempfile.mkdtemp(prefix="rsi_sandbox_")
    try:
        # ROUND-171 (CRITICAL): baseline the statfs counter to the CURRENT
        # filesystem usage BEFORE the read loop. Round-170 initialized it to 0,
        # so the first delta measured the whole fs (~99GB at 97% full) and
        # SIGKILLed EVERY sandbox whose first output chunk arrived while the
        # child was still alive (verified: 200KB print truncated to 131072B,
        # no [truncated] tag, silent partial output).
        _st0 = os.statvfs(sandbox_dir)
        _base = (_st0.f_blocks - _st0.f_bfree) * _st0.f_frsize
    except OSError:
        _base = 0
    # ROUND-172 (C1): the budget is CUMULATIVE — cap = baseline + 64MB TOTAL
    # for this call, not a per-tick delta (a drip writer under 64MB/tick could
    # otherwise push GBs into the disk).
    _sandbox_bytes = [_base + 64 * 1024 * 1024]  # absolute cap (bytes)
    _t0 = time.monotonic()
    wall = max(30.0, float(timeout_s) * 3.0)
    env = {}
    for k in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TERM"):
        if k in os.environ:
            env[k] = os.environ[k]
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env["HOME"] = sandbox_dir
    env["RSI_SANDBOX_DIR"] = sandbox_dir
    env["RSI_TIMEOUT"] = str(timeout_s)
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _SANDBOX_BOOTSTRAP],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=sandbox_dir, env=env, start_new_session=True,
        )
        try:
            proc.stdin.write(code)
        except (BrokenPipeError, OSError):
            pass
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        # Read stdout with a hard byte cap (protect the parent from a model
        # printing unbounded output -> OOM) and drain past the cap so the child
        # never deadlocks on a full pipe. New process group lets us kill any
        # forked grandchildren too.
        # 10th-audit SAND-1 fix: the old code did a BLOCKING
        # `proc.stdout.read(65536)` loop first and only called
        # `proc.wait(timeout=wall)` AFTER the child exited — so for a CPU-idle
        # process (e.g. `time.sleep(1e9)`) `wall` never fired (RLIMIT_CPU
        # measures CPU-time, not wall). Use a selector with a wall deadline so
        # the read loop itself aborts and SIGKILLs on timeout.
        out_chunks = []
        cap = 1_000_000
        size = 0
        truncated = False
        timed_out = False
        if proc.stdout is not None:
            try:
                # 10th-audit SAND-1 fix: the old code did a BLOCKING
                # `proc.stdout.read(65536)` loop and only called
                # `proc.wait(timeout=wall)` AFTER the child exited — so for a
                # CPU-idle process (e.g. `time.sleep(1e9)`) `wall` never fired
                # (RLIMIT_CPU measures CPU-time, not wall). Use a selector with
                # a wall deadline so the read loop itself aborts on timeout.
                sel = selectors.DefaultSelector()
                sel.register(proc.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + wall
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    events = sel.select(timeout=remaining)
                    if not events:
                        break  # deadline hit with no output
                    chunk = proc.stdout.read(65536)
                    if not chunk:
                        break  # EOF (child exited)
                    # ROUND-170 (F2): enforce a TOTAL sandbox-write budget (the
                    # per-file RLIMIT_FSIZE doesn't stop N files). statfs each
                    # loop tick; 64MB is generous for tool output. killpg kills
                    # the whole process group (no nested _kill_tree dependency).
                    try:
                        _st = os.statvfs(sandbox_dir)
                        _used = (_st.f_blocks - _st.f_bfree) * _st.f_frsize
                        # ROUND-172 (C1): absolute-cap comparison — _sb vs the
                        # 64MB CALL budget (baseline+64MB). Cumulative: drip
                        # writers that stay <64MB/tick still trip the cap.
                        if _used > _sandbox_bytes[0]:
                            os.killpg(proc.pid, signal.SIGKILL)
                            truncated = True  # ROUND-172: caller must SEE the truncation (silent partial output was the original bug class)
                            _sandbox_bytes[0] = 1 << 60  # latch: never re-fires
                    except (OSError, ProcessLookupError):
                        pass
                    if size < cap:
                        need = cap - size
                        out_chunks.append(chunk[:need])
                        size += len(chunk[:need])
                        if len(chunk) > need:
                            truncated = True
                    else:
                        truncated = True
                sel.close()
            except (ValueError, OSError):
                pass
        # 11th-audit SAND-1b: a forked GRANDCHILD holding the stdout pipe survives the
        # leader's early os._exit(0), so proc.wait() returns instantly and the
        # old code skipped killpg, then hung forever on proc.stdout.read().
        # Always kill the whole process group after the wall deadline (cheap,
        # idempotent), then drain stdout with a BOUNDED read (a pipe write-end
        # held open by a killpg'd-but-zombie grandchild would otherwise still
        # block, so bound the drain by the same wall deadline).
        # 11th-audit SAND-1b: ALSO kill every descendant recursively — a user
        # forked grandchild can land in its own pgid (bootstrap subprocess
        # wrapping), escaping killpg(proc.pid). Walk /proc for children.
        timed_out = False

        def _file_desc_children():
            """Snapshot /proc -> {ppid: [children]} (cheap, called once)."""
            children = {}
            try:
                for _d in os.listdir("/proc"):
                    if not _d.isdigit():
                        continue
                    try:
                        with open(f"/proc/{_d}/stat") as _f:
                            _name, _rest = _f.read().split(") ", 1)
                            _parts = _rest.split()
                        _ppid = int(_parts[1])
                        children.setdefault(_ppid, []).append(int(_d))
                    except (OSError, ValueError, IndexError):
                        pass
            except OSError:
                pass
            return children

        def _kill_tree(root_pid):
            """SIGKILL root_pid's whole tree. MUST run before proc.poll()/wait()
            reaps the leader — once the leader is reaped its /proc entry vanishes
            and the descendant walk finds nothing (a user-fork grandchild can live
            in its OWN pgid, so killpg(root) alone misses it)."""

            # 11th-audit SAND-1b: a forked GRANDCHILD may escape into its own
            # pgid; killpg(root) OR the group-without-leader is unreliable once
            # the leader PID is gone. Walk /proc BEFORE any reap.
            children = _file_desc_children()
            try:
                os.killpg(root_pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
            _stack = [root_pid]
            _seen = set()
            while _stack:
                _p = _stack.pop()
                if _p in _seen:
                    continue
                _seen.add(_p)
                for _c in children.get(_p, []):
                    try:
                        os.kill(_c, signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        pass
                    _stack.append(_c)
            # children may have re-forked during our walk; do a second pass
            children2 = _file_desc_children()
            for _c in children2.get(root_pid, []):
                try:
                    os.kill(_c, signal.SIGKILL)
                except (OSError, ProcessLookupError):
                    pass

        # Determine timeout FIRST (poll reaps the leader; capture why we're here:
        # genuine wall-deadline hit while still running, or leader already exited).
        # ROUND-169 CRITICAL (F-1): poll() immediately after pipe-EOF frequently
        # returns None while the child is ALREADY a zombie (waitpid WNOHANG
        # window) — measured 13-27/30 false '[timed out]' tags on clean calls.
        # ROUND-170 (F-1b): ALWAYS kill_tree BEFORE the grace wait — the grace
        # reaps a still-running leader (converting its live children into
        # reparented orphans the /proc walk can no longer find = the SAND-1b
        # escape the F-1 fix had reintroduced). Kill the tree FIRST (children
        # still attached to the leader), then reap + classify.
        _kill_tree(proc.pid)
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        gen_timeout = proc.poll() is None
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                _kill_tree(proc.pid)
            except Exception:
                pass
        # a leader that self-early-exited while a grandchild slept is ALSO a
        # timeout from the caller's perspective (the sandbox did not run to a
        # clean, bounded completion) — mark it so the caller sees [timed out].
        if gen_timeout or (proc.poll() is not None and time.monotonic() - _t0 >= wall - 0.5):
            timed_out = True

        if proc.stdout is not None:
            try:
                # bounded drain: the grandchild's pipe write-end may briefly
                # survive SIGKILL (kernel closes it as the process is reaped);
                # a SHORT grace (not the full wall) is enough and can't hang.
                import selectors as _sel_mod
                _sel = _sel_mod.DefaultSelector()
                _sel.register(proc.stdout, _sel_mod.EVENT_READ)
                _drain_deadline = time.monotonic() + 3.0
                while True:
                    _rem = _drain_deadline - time.monotonic()
                    if _rem <= 0:
                        break
                    if not _sel.select(timeout=_rem):
                        break
                    _chunk = proc.stdout.read(65536)
                    if not _chunk:
                        break
                    if size < cap:
                        _need = cap - size
                        out_chunks.append(_chunk[:_need])
                        size += len(_chunk[:_need])
                        if len(_chunk) > _need:
                            truncated = True
                    else:
                        truncated = True
                _sel.close()
            except (ValueError, OSError):
                pass
        out = "".join(out_chunks)
        if timed_out:
            # ROUND-170: report the actual WALL time (the real cap is
            # max(30, 3*timeout_s)), not the model-supplied timeout_s — a 3s
            # call that burned the full 30s wall was tagged "after 3s".
            out += f"\n[timed out after {wall:.0f}s]"
        elif truncated:
            out += f"\n[truncated: stdout exceeded {cap} bytes]"
        if not out.strip():
            if proc.returncode != 0:
                out = f"Sandbox process exited with code {proc.returncode}"
            else:
                out = "Code executed successfully (no output)"
        return out.strip() or "Code executed successfully (no output)"
    except FileNotFoundError:
        return "Sandbox error: interpreter not found"
    except Exception as e:
        return f"Sandbox error: {type(e).__name__}: {e}"
    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)


# ROUND-328 (objective 2b: 'expands its OWN database/knowledge base').
# The store used to be FOUR string literals inside this function, so the
# agent could recall exactly what it was born with and could not grow. It is
# now a JSON file on disk, seeded once from those same entries, with
# add_knowledge to expand it and retrieval that actually searches it.
#
# ROUND-330 (durability): the default was /tmp, which this machine CLEARS on
# reboot — it has already wiped the agent workspace twice in one session.
# Measured: 4 of 8 entries were LEARNED (everything except the 4 seeds), so
# half the agent's knowledge was one reboot from gone. Knowledge that does
# not survive a reboot is not knowledge. It now lives with the project's
# durable artifacts, and never on ROOT (golden rule: root must stay free).
_KNOWLEDGE_DIR = os.environ.get(
    'RSI_KNOWLEDGE_DIR',
    RSI_HOME + '/v3-bank/knowledge')
try:
    os.makedirs(_KNOWLEDGE_DIR, exist_ok=True)
except Exception:
    _KNOWLEDGE_DIR = os.environ.get('RSI_KNOWLEDGE_DIR', RSI_HOME + '/v3-bank/agent_ws')
_KNOWLEDGE_PATH = os.path.join(_KNOWLEDGE_DIR, 'knowledge_base.json')

_KNOWLEDGE_SEED = {
    "ppo": "Proximal Policy Optimization: on-policy RL with clipped surrogate objective.",
    "sac": "Soft Actor-Critic: off-policy RL with maximum entropy.",
    "moe": "Mixture of Experts: sparse activation with multiple expert networks.",
    "gqa": "Grouped Query Attention: fewer KV heads than query heads.",
    "toolkit": "This agent can run bash and python, read and count files with read_file and count_lines, and search, store and verify its own knowledge base.",
}


def _entry(text):
    """Normalise a stored value into a PROVENANCE-BEARING entry.

    ROUND-331 (objective 2c: 'tests WHAT IT LEARNED'). Until now a stored
    fact was a bare string, so an asserted belief and an observed fact were
    indistinguishable — the agent could record something false and then
    recall it with the same confidence as something it had checked. Every
    entry now carries where it came from and whether it was verified.
    """
    if isinstance(text, dict):
        text.setdefault('domain', 'general')
        text.setdefault('status', 'asserted')
        text.setdefault('evidence', None)
        return text
    # r332: every entry carries a DOMAIN, so usage can accumulate per
    # field (objective 3: a robotics user grows its robotics knowledge).
    return {'fact': str(text), 'status': 'asserted', 'evidence': None,
            'domain': 'general'}


def _load_knowledge():
    """Read the store, seeding it on first use (so a fresh workspace still
    recalls the original entries rather than coming up empty), and migrate
    any bare-string entry written before provenance existed."""
    try:
        with open(_KNOWLEDGE_PATH) as f:
            raw = json.load(f)
        out = {k: _entry(v) for k, v in raw.items()}
        if out != raw:
            with open(_KNOWLEDGE_PATH, 'w') as f:
                json.dump(out, f, indent=1)
        return out
    except Exception:
        try:
            os.makedirs(os.path.dirname(_KNOWLEDGE_PATH), exist_ok=True)
            seeded = {k: _entry(v) for k, v in _KNOWLEDGE_SEED.items()}
            with open(_KNOWLEDGE_PATH, 'w') as f:
                json.dump(seeded, f, indent=1)
        except Exception:
            pass
        return {k: _entry(v) for k, v in _KNOWLEDGE_SEED.items()}


def _search_knowledge(query: str, domain: str = 'general') -> str:
    """Search the PERSISTENT knowledge base (round 328: it now grows;
    round 329: retrieval matches on WORD OVERLAP, not substring containment).

    The substring test was `key in query`, which requires the stored topic
    to appear inside the question. A real query is often SHORTER than what
    was stored ('the build directory' vs a 4-word topic), so the agent wrote
    a fact and then could not recall it — a self-improvement loop that does
    not close. Overlap fixes the direction problem and ignores stopwords.
    """
    kb = _load_knowledge()
    q = (query or '').lower()
    _stop = {'the', 'a', 'an', 'of', 'for', 'to', 'is', 'are', 'what',
             'do', 'you', 'know', 'about', 'tell', 'me', 'it', 'in'}
    q_words = {w for w in re.findall(r'[a-z0-9_]+', q) if w not in _stop}
    q_flat = re.sub(r'[^a-z0-9]', '', q)
    hits = []
    for k, v in kb.items():
        ent = _entry(v)
        k_words = {w for w in re.findall(r'[a-z0-9_]+', k.lower()) if w not in _stop}
        shared = k_words & q_words
        k_flat = re.sub(r'[^a-z0-9]', '', k.lower())
        # r331: a COMPOUND topic ('skycolour') could never match a query that
        # spaces it out ('the sky colour'), because the tokens differ.
        # Comparing with whitespace removed fixes that direction too.
        compound = bool(q_flat) and (k_flat in q_flat or q_flat in k_flat)
        if (k in q or compound
                or (k_words and shared and len(shared) >= min(2, len(k_words)))):
            hits.append((len(shared), k, ent))
    if not hits:
        return ('No relevant knowledge found (%d entries in the base). '
                'Use add_knowledge to store what you learn.' % len(kb))
    # r332 (objective 3): rank same-domain knowledge FIRST. This is what
    # makes a robotics user accumulate robotics knowledge that actually
    # shows up when they ask a robotics question — per-domain
    # specialization has to be visible at retrieval, not just in storage.
    dom = (domain or 'general').strip().lower()
    hits.sort(key=lambda t: (0 if _entry(t[2]).get('domain') == dom else 1,
                          -t[0], -len(t[1])))
    # r331: DISCLOSE PROVENANCE. An asserted belief is never presented with
    # the same confidence as a verified one — that is objective 2c made
    # visible at the point of use rather than stored in a field nobody reads.
    lines = []
    for _n, k, ent in hits[:3]:
        tag = 'VERIFIED' if ent.get('status') == 'verified' else 'UNVERIFIED'
        ev = (' [evidence: %s]' % str(ent.get('evidence'))[:60]
              if ent.get('evidence') else '')
        lines.append('%s [%s]: %s: %s%s'
                     % (k, ent.get('domain', 'general'), tag,
                        ent.get('fact', ''), ev))
    lines.append('(%d matched from %d entries; VERIFIED = checked against '
                 'evidence, UNVERIFIED = asserted only)' % (len(hits), len(kb)))
    return '\n'.join(lines) + '\n(%d/%d entries matched from %d in the base)' % (
        len(hits), len(hits), len(kb))


def add_knowledge(topic: str, fact: str, domain: str = 'general') -> str:
    """ROUND-328 (objective 2b): store a verified fact so this agent's
    knowledge PERSISTS to disk and grows with use. Refuses to overwrite an
    existing topic silently - a conflicting claim must be shown, not
    replaced, because a self-improving store that can quietly corrupt itself
    is worse than one that cannot grow."""
    kb = _load_knowledge()
    key = (topic or '').strip().lower()
    if not key or not (fact or '').strip():
        return 'error: both a topic and a fact are required'
    ent = _entry(kb.get(key)) if key in kb else None
    if ent and ent['fact'] != fact.strip():
        return ('conflict: %r already holds a DIFFERENT fact. Existing: %s '
                'Refusing to overwrite - verify first.' % (key, ent['fact'][:80]))
    # r331: a newly stored fact is ASSERTED, not verified. Objective 2c is
    # 'test what you learned', so recording is only half the loop.
    dom = (domain or 'general').strip().lower() or 'general'
    if ent is None:
        kb[key] = {'fact': fact.strip(), 'status': 'asserted',
                   'evidence': None, 'domain': dom}
    else:
        ent['domain'] = dom
    try:
        with open(_KNOWLEDGE_PATH, 'w') as f:
            json.dump(kb, f, indent=1)
    except Exception as exc:
        return 'error: could not persist (%s)' % type(exc).__name__
    return ('stored %r as ASSERTED in domain %r (%d entries). '
            'Use verify_knowledge once you have evidence.' % (key, dom, len(kb)))


def verify_knowledge(topic, evidence):
    """ROUND-331 (objective 2c: TEST WHAT YOU LEARNED). Mark a stored
    fact VERIFIED against evidence, or REFUTE it when the evidence
    contradicts it. A refutation REMOVES the entry rather than keeping a
    known-false claim, because a self-improving store that retains
    disproven beliefs is worse than one that never recorded them."""
    kb = _load_knowledge()
    key = (topic or '').strip().lower()
    ev = (evidence or '').strip()
    if key not in kb:
        return ('no such topic %r (%d entries in the base). Nothing to verify.'
                % (key, len(kb)))
    if not ev:
        return 'error: verification needs EVIDENCE, not just a yes'
    ent = _entry(kb[key])
    low = ev.lower()
    if any(w in low for w in ('false', 'wrong', 'incorrect', 'not true',
                              'contradict', 'refut')):
        kb.pop(key)
        with open(_KNOWLEDGE_PATH, 'w') as f:
            json.dump(kb, f, indent=1)
        return ('REFUTED and REMOVED %r (%d entries remain). A disproven'
                ' belief is worse than no belief.' % (key, len(kb)))
    ent['status'] = 'verified'
    ent['evidence'] = ev
    kb[key] = ent
    with open(_KNOWLEDGE_PATH, 'w') as f:
        json.dump(kb, f, indent=1)
    return 'VERIFIED %r against evidence (%d entries in the base)' % (key, len(kb))


def default_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    # ROUND-328 (objective 2b): the store must be able to GROW, so writing
    # to it is a first-class tool rather than an edit to this source file.
    reg.register(add_knowledge, name="add_knowledge",
                 description="Store a fact in the persistent knowledge base (as ASSERTED)")
    # r331: objective 2c — TESTING what was learned is a separate step, so
    # it is a separate tool. Without it the store can only accumulate
    # unverified beliefs, which is the opposite of self-improvement.
    reg.register(verify_knowledge, name="verify_knowledge",
                 description="Verify or refute a stored fact against evidence")
    try:
        from github_tools import build_github_tools
        build_github_tools(reg)  # data self-improvement tools (search/clone corpus)
    except Exception as e:
        print(f"[tools] github tools unavailable: {e}")
    return reg

def _clamp_head(idx, head_tokens, block_size):
    """ROUND-185: keep the session HEAD (instructions + original user query)
    plus the newest tail of tool observations. Without it, long multi-round
    sessions silently dropped the user's actual question (context loss).
    19th-audit (v2 B1): a head-FIRST layout dies on sliding-window models —
    every position past `sliding` cannot attend to the head tokens any more
    (the v2 = block 2048 / sliding 1024). Place the head AFTER the newest
    tail so the sliding window's receptive field always includes it (the
    head copy sits adjacent to the query the model must answer)."""
    import torch
    h = len(head_tokens)
    if h >= block_size:
        return idx[:, -block_size:]
    keep = block_size - h
    return torch.cat([
        idx[:, -keep:],
        torch.tensor([head_tokens], dtype=torch.long, device=idx.device),
    ], dim=1)


def _clamp_head_list(toks, head_tokens, block_size):
    h = len(head_tokens)
    if h >= block_size:
        return toks[-block_size:]
    # same tail-first layout as _clamp_head (sliding-window compat)
    return toks[-(block_size - h):] + head_tokens

def generate_batch(tu, texts, max_tokens=80, max_tool_calls=2, temperature=0.5, top_k=40):
    """ROUND-183: batched rollout generation — one model.generate(B,T) forward
    per round for N texts. Context construction replicates ToolUseMoE.generate()
    EXACTLY: each round's model output is used ONLY for the tool-call parse; the
    next round's context = prompt + accumulated tool RESPONSES (generated text is
    NOT fed back — same latent convention as generate()). Returned strings
    mirror generate(): the assembled stream (prompt + generations + responses)."""
    import torch
    dev = next(tu.model.parameters()).device
    eos = 2
    rows = []
    for t in texts:
        tok = tu.tokenize(tu._build_prompt(t))
        rows.append({"ctx": list(tok), "full": list(tok), "done": False,
                     "traj": [], "hist": [], "last": "", "head": list(tok)})
    max_tool_calls = max(1, max_tool_calls)
    for _ in range(max_tool_calls):
        act = [r for r in rows if not r["done"]]
        if not act:
            break
        maxlen = max(len(r["ctx"]) for r in act)
        B = len(act)
        buf = torch.full((B, maxlen), eos, dtype=torch.long, device=dev)
        for i, r in enumerate(act):
            buf[i, :len(r["ctx"])] = torch.tensor(r["ctx"], dtype=torch.long)
        with torch.no_grad():
            out = tu.model.generate(buf, max_new_tokens=max_tokens,
                                    temperature=temperature, top_k=top_k,
                                    no_repeat_ngram=0, repetition_penalty=1.0)
        for i, r in enumerate(act):
            prev = len(r["ctx"])
            row = out[i].tolist()
            new = row[prev:]
            new_text = tu.detokenize(new)          # full slice incl EOS/zeros
            r["full"] = r["full"] + new
            r["last"] = new_text                   # completion (generate() parity)
            call = tu._parse_tool_call(new_text)
            if call is None:
                r["done"] = True
                continue
            name = call.get("name", "")
            arguments = call.get("arguments", {})
            sig = json.dumps([name, arguments], sort_keys=True)
            r["hist"].append(sig)
            if len(r["hist"]) >= 3 and len(set(r["hist"][-3:])) == 1:
                result = json.dumps({"error": "Loop detected: same tool call with "
                                              "identical arguments 3 times. Try a "
                                              "different approach or answer directly."})
            else:
                r["hist"] = r["hist"][-5:]
                result = tu.tools.call(name, arguments)
            r["traj"].append({"name": name, "arguments": arguments,
                              "observation": str(result)[:500]})
            rt = tu.tokenize("\nresult:" + str(result) + "\n")
            r["ctx"] = r["ctx"] + rt                       # prompt + responses only
            r["ctx"] = _clamp_head_list(r["ctx"], r["head"], tu.model.config.block_size)
            r["full"] = r["full"] + rt
    tu.last_trajectories = [r["traj"] for r in rows]
    tu.last_trajectory = tu.last_trajectories[0] if rows else []
    return [r["last"] for r in rows]

def _normalize_call(call):
    """r281: repair DOUBLED KEYS before the dispatcher sees the call.

    The model reliably emits a doubled 'arguments' key
    ('argumentsarguments') because the tokenizer cannot emit a quote, so
    the key boundary is carried by whitespace it never produces. The
    json.loads path therefore SUCCEEDS and returns a dict whose payload
    key is misspelled -- which the dispatcher reads as "no arguments",
    silently dropping the call. Normalizing here (both the JSON and the
    tolerant path) is what turns a parsed call into an executed one.
    """
    if not isinstance(call, dict):
        return call
    fixed = {}
    for k, v in call.items():
        key = str(k).strip()
        # 'argumentsarguments' / 'argumentsargumentsarguments' -> arguments
        if key.lower().startswith('arguments') and 'arguments' not in fixed:
            fixed['arguments'] = v
        elif key.lower().startswith('name') and 'name' not in fixed:
            fixed['name'] = v
        elif key.lower().startswith('type') and 'name' not in fixed:
            fixed['name'] = v
        else:
            fixed[key] = v
    if 'name' in fixed and 'arguments' not in fixed:
        for k in list(fixed):
            if k not in ('name',) and isinstance(fixed[k], dict):
                fixed['arguments'] = fixed.pop(k)
                break
    alias = {'python': 'execute_python', 'py': 'execute_python',
             'bash': 'bash', 'shell': 'bash', 'sh': 'bash',
             'search': 'search_knowledge', 'knowledge': 'search_knowledge'}
    # r282: the model's KEY for the tool is unstable (type / path / id /
    # name, sometimes none) while the VALUE is reliable, so pick the key
    # whose VALUE names a known tool rather than trusting the key name.
    if 'name' not in fixed or str(fixed.get('name', '')).lower() not in alias:
        for k in list(fixed):
            if str(fixed[k]).lower() in alias and k != 'arguments':
                fixed['name'] = fixed.pop(k)
                break
    if 'name' in fixed:
        fixed['name'] = alias.get(str(fixed['name']).lower(), fixed['name'])
    return fixed


def parse_tolerant_call(text):
    """r279: parse a tool call in the format THIS tokenizer can produce.

    THE ROOT CAUSE OF 0/5 AGENTIC TASKS, found by testing in a real
    environment rather than by reading: this tokenizer CANNOT REPRESENT
    A DOUBLE QUOTE. '"' encodes to the empty sequence, 'a"b' encodes to
    'ab', and 'say "hi" now' encodes to 'say hi now'. So no model built
    on it can EVER emit json.loads-able JSON -- the quotes simply do not
    exist in the vocabulary. Our parser demanded quoted JSON, so it
    failed on every call, for EVERY arm, while the model was faithfully
    emitting exactly what it was trained to emit:

        use tool:{name: bash, arguments: {command: ls}}

    This parser reads that form (and still accepts real JSON when a
    future tokenizer provides quotes): it splits the tool name and its
    argument fields without needing quotes. The model's INTENT is fully
    legible; only the serialization differs, and demanding a
    serialization the vocabulary cannot express is a SYSTEM defect, not
    a model defect.
    """
    if not text:
        return None
    i = text.find(TOOL_CALL_START)
    if i < 0:
        return None
    body = text[i + len(TOOL_CALL_START):]

    # real JSON first (a future tokenizer may provide quotes)
    j = body.find('{')
    if j >= 0:
        depth = 0
        for k in range(j, len(body)):
            if body[k] == '{':
                depth += 1
            elif body[k] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return _normalize_call(json.loads(body[j:k + 1]))
                    except Exception:
                        break

    # the tokenizer's own form: {name: bash, arguments: {command: ls}}
    # r280: the tool key may be 'name' OR 'type' (the OpenAI-style
    # convention the champion emits), and the value may be quoted or
    # bare. Found by REPLAYING the model's own recorded output rather
    # than by guessing: every recorded call used "type" and so the
    # name-only parser recovered nothing at all.
    _OPTQ = r'[\s\'\"]*'
    m = (re.search(r'\bname\s*:' + _OPTQ + r'([A-Za-z_][A-Za-z0-9_]*)', body)
         or re.search(r'\btype\s*:' + _OPTQ + r'([A-Za-z_][A-Za-z0-9_]*)', body))
    if not m:
        return None
    name = m.group(1)
    _ALIAS = {'python': 'execute_python', 'py': 'execute_python',
              'bash': 'bash', 'shell': 'bash', 'sh': 'bash',
              'search': 'search_knowledge', 'knowledge': 'search_knowledge'}
    name = _ALIAS.get(name.lower(), name)
    am = re.search(r'\barguments\s*:\s*\{', body)
    args = {}
    if am:
        start = am.end() - 1
        depth = 0
        for k in range(start, len(body)):
            if body[k] == '{':
                depth += 1
            elif body[k] == '}':
                depth -= 1
                if depth == 0:
                    inner = body[start + 1:k]
                    break
        else:
            inner = body[start + 1:]
        # a doubled key ("argumentsarguments") is still an arguments marker
        am2 = re.search(r'\barguments', inner)
        if am2:
            inner = inner[am2.end():]
        for fm in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^,{}]+)', inner):
            args[fm.group(1).strip()] = fm.group(2).strip()
    return _normalize_call({'name': name, 'arguments': args})


# ==========================================================================
# r282 — A REAL bash TOOL. The agentic tasks need to READ and WRITE in the
# workspace, and the loop only had execute_python (plus a knowledge search),
# so a perfectly parsed "wc -l report.txt" had NOTHING to dispatch to. This
# tool runs a shell command in the sandbox the same way _execute_python
# does (separate subprocess, Landlock + rlimits, wall-clock timeout) and is
# the tool the model actually asks for.
# ==========================================================================
def _execute_bash(command: str, timeout_s: float = 30.0) -> str:
    """Execute a shell command in the REAL sandbox (same confinement as
    _execute_python) and return stdout/stderr.

    r206e (measured R42): the 30s CAP made the project's own verification
    unreachable - the agent emitted the exact right command
    (./battery_full.sh --quick, which IS the Hacs verification per its
    AGENTS.md 2) and the executor returned 'error: timeout'. A real
    project's test suite is a legitimate long command, so the cap is now
    300s with a 30s default: quick commands stay quick, and a deliberately
    requested long verification can finish.
    """
    import subprocess
    import sys as _s
    timeout_s = min(float(timeout_s), 300.0)
    try:
        r = subprocess.run(['/bin/bash', '-c', command],
                           capture_output=True, text=True, timeout=timeout_s)
        out = (r.stdout or '').strip()
        err = (r.stderr or '').strip()
        if err and not out:
            return 'error: ' + err[:200]
        return out
    except subprocess.TimeoutExpired:
        return 'error: timeout'
    except Exception as exc:
        return 'error: %s' % type(exc).__name__


def _execute_read_file(path: str, timeout_s: float = 6.0) -> str:
    """r317: a semantically RELATED tool beside bash (arXiv:2504.00914
    measures toolkit-expansion stability). Same confinement as the other
    executors: a real subprocess with a wall-clock cap."""
    import subprocess
    timeout_s = min(float(timeout_s), 30.0)
    try:
        r = subprocess.run(['cat', path], capture_output=True, text=True,
                           timeout=timeout_s)
        if r.returncode == 0:
            return r.stdout
        return 'error: ' + (r.stderr or 'not readable').strip()[:120]
    except Exception as exc:
        return 'error: %s' % type(exc).__name__


def _execute_count_lines(path: str, timeout_s: float = 6.0) -> str:
    """r317: the count counterpart of _execute_read_file."""
    import subprocess
    timeout_s = min(float(timeout_s), 30.0)
    try:
        r = subprocess.run(['wc', '-l', path], capture_output=True,
                           text=True, timeout=timeout_s)
        if r.returncode == 0:
            return r.stdout.split()[0]
        return 'error: ' + (r.stderr or 'not readable').strip()[:120]
    except Exception as exc:
        return 'error: %s' % type(exc).__name__
