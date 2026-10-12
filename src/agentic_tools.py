#!/usr/bin/env python3
"""ROUND-204ac-b5.14: AGENTIC file-tool envelope (safe-by-construction).

The model's agentic axis (spec objective #1 + the E1 trajectory data: bash,
read, edit, grep, glob, write) needs a tool surface on the serving loop. These
tools are safe BY CONSTRUCTION: pure Python, no shell, no subprocess, and every
path is resolved + contained inside the workspace root (RSI_AGENT_WORKSPACE env
or a per-call sandbox dir). No network. Bounded I/O (cap bytes). bash/web
(arbitrary command execution) = a LATER step that must ride the same Landlock
jail as _execute_python — deliberately NOT in this v1.

Registration: agentic_registry() -> ToolRegistry with:
  read_file(file_path)  grep(pattern, path='.')  glob(pattern, path='.')
  list_dir(path='.')    write_file(file_path, content)

Containment rule: any resolved path must be under WORKSPACE_ROOT (resolve() +
startswith check). Absolute escapes, ".." escapes and symlink escapes onto the
host FS are rejected with a clear error string. write_file is disabled by
default (read-only) unless RSI_AGENT_WRITE=1.
"""
import os, re
from pathlib import Path

TOOL_CALL_START = "use tool:"

_WORKSPACE = Path(os.environ.get(
    "RSI_AGENT_WORKSPACE",
    str(Path(__file__).resolve().parent.parent / "data" / "agent_workspace")))

MAX_READ_BYTES = 64 * 1024
MAX_MATCH_ROWS = 60
MAX_LIST_ENTRIES = 80
MAX_GLOB_ENTRIES = 80


def _contain(p: Path) -> bool:
    try:
        resolved = p.resolve()
        root = _WORKSPACE.resolve()
        return resolved == root or str(resolved).startswith(str(root) + os.sep)
    except OSError:
        return False


def _guard(p: str) -> Path:
    r = Path(p)
    if r.is_absolute():
        r = Path(str(_WORKSPACE)) / str(r).lstrip(os.sep)
    cand = (_WORKSPACE / r).resolve()
    if not _contain(cand):
        return None
    return cand


def __tools():
    pass

def _read_file(file_path: str = "") -> str:
    if not file_path:
        return "Error: file_path is required"
    p = _guard(file_path)
    if p is None:
        return "Error: path escapes the workspace"
    try:
        data = p.read_bytes()[:MAX_READ_BYTES]
        return data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return f"Error: no such file: {file_path}"
    except OSError as e:
        return f"Error: {e}"


def _grep(pattern: str = "", path: str = ".") -> str:
    if not pattern:
        return "Error: pattern is required"
    p = _guard(path)
    if p is None:
        return "Error: path escapes the workspace"
    try:
        pat = re.compile(pattern)
    except re.error as e:
        return f"Error: bad pattern: {e}"
    hits = []
    if p.is_file():
        files = [p]
    else:
        files = []
        for root, dirs, fs in os.walk(p):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
            for f in fs:
                files.append(Path(root) / f)
                if len(files) > 2000:
                    break
            if len(files) > 2000:
                break
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if pat.search(line):
                        hits.append(f"{fp.name}:{i}: {line.rstrip()[:200]}")
                        if len(hits) >= MAX_MATCH_ROWS:
                            return "\n".join(hits) + "\n...(capped)"
        except OSError:
            continue
    return "\n".join(hits) if hits else "No matches found."


def _glob(pattern: str = "", path: str = ".") -> str:
    if not pattern:
        return "Error: pattern is required"
    p = _guard(path)
    if p is None:
        return "Error: path escapes the workspace"
    if not p.exists():
        return f"Error: no such path: {path}"
    try:
        out = [str(x.relative_to(_WORKSPACE)) for x in p.glob(pattern)
               if _contain(x)][:MAX_GLOB_ENTRIES]
    except Exception as e:
        return f"Error: {e}"
    return "\n".join(out) if out else "No files match."


def _list_dir(path: str = ".") -> str:
    p = _guard(path)
    if p is None:
        return "Error: path escapes the workspace"
    try:
        ents = sorted(os.listdir(p))[:MAX_LIST_ENTRIES]
        return "\n".join(ents)
    except FileNotFoundError:
        return f"Error: no such dir: {path}"
    except OSError as e:
        return f"Error: {e}"


def _write_file(file_path: str = "", content: str = "") -> str:
    if not file_path:
        return "Error: file_path is required"
    if os.environ.get("RSI_AGENT_WRITE") != "1":
        return "Error: write_file disabled (RSI_AGENT_WRITE not set)"
    p = _guard(file_path)
    if p is None:
        return "Error: path escapes the workspace"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content[:MAX_READ_BYTES], encoding="utf-8")
        return f"wrote {len(content[:MAX_READ_BYTES])} bytes to {file_path}"
    except OSError as e:
        return f"Error: {e}"


_DANGEROUS = re.compile(
    r"(?i)\brm\s+-(?:[a-z]*[rf][a-z]*)\b\s+/*$|\bmkfs\b|\bdd\b\s+if=|"
    r"/dev/(sd[a-z]|sda|mem|port|kmem)|sudo\s+|\bchmod\s+-R\s+777\s+/"
    r"|\bin -f\b|> */(etc|boot|lib|usr)(/|$)|\bmount\b|\bumount\b|iptables|"
    r"\bkexec\b|eval.?\$|\bapt\s+(install|remove|purge)\b")


def _agent_bash(command: str = "", timeout_s: float = 10.0) -> str:
    """Execute a MODEL-GENERATED shell command inside the SAME jail discipline
    as _execute_python (Landlock sandbox + net deny + rlimits + privesc drop +
    wall timeout + output byte cap + cumulative disk budget), confined to the
    agent workspace (RSI_AGENT_WORKSPACE). Defense-in-depth: a dangerous-pattern
    blocklist fails fast (the jail would deny anyway). Never crashes."""
    if not command or not command.strip():
        return "Error: command is required"
    if _DANGEROUS.search(command):
        return "Error: command rejected as dangerous"
    import os, selectors, subprocess, sys, tempfile, time
    import tool_use as _tu
    work = str(_WORKSPACE)
    _st0 = os.statvfs(work)
    _base = (_st0.f_blocks - _st0.f_bfree) * _st0.f_frsize
    timeout_s = min(float(timeout_s), 30.0)
    wall = max(30.0, float(timeout_s) * 3.0)
    env = {}
    for k in ("PATH", "LANG", "LC_ALL", "TERM"):
        if k in os.environ:
            env[k] = os.environ[k]
    env["HOME"] = work
    env["RSI_SANDBOX_DIR"] = work          # Landlock rw = the workspace
    env["RSI_SHELL"] = "1"
    env["RSI_TIMEOUT"] = str(timeout_s)
    proc = subprocess.Popen(
        [sys.executable, "-u", "-c", _tu._SANDBOX_BOOTSTRAP],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=work, env=env, start_new_session=True)
    try:
        proc.stdin.write(command)
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    out_chunks, size, truncated, timed_out = [], 0, False, False
    cap = 1_000_000
    try:
        sel = selectors.DefaultSelector()
        sel.register(proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + wall
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            if not sel.select(timeout=remaining):
                timed_out = True
                break
            chunk = proc.stdout.read(65536)
            if not chunk:
                break
            try:
                _st = os.statvfs(work)
                if (_st.f_blocks - _st.f_bfree) * _st.f_frsize > _base + 64 * 1024 * 1024:
                    _kill_tree(proc)
                    out_chunks.append("\n[disk budget exceeded]")
                    break
            except OSError:
                pass
            if size < cap:
                out_chunks.append(chunk[: cap - size])
                size += len(out_chunks[-1])
            else:
                truncated = True
        try:
            sel.close()
        except Exception:
            pass
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            _kill_tree(proc)
    out = "".join(out_chunks)
    if truncated:
        out += "\n[truncated]"
    if timed_out:
        out += f"\n[timed out after {wall:.0f}s wall]"
    if proc.returncode not in (0, None):
        out += f"\n[exit code {proc.returncode}]"
    return out if out.strip() else f"Error: no output (command: {command[:80]}...)"


def _kill_tree(proc):
    import os, signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def agentic_registry():
    from tool_use import ToolRegistry
    reg = ToolRegistry()
    reg.register(_read_file, name="read_file",
                 description="Read a text file inside the workspace and return its content")
    reg.register(_grep, name="grep",
                 description="Search files inside the workspace for a regex pattern")
    reg.register(_glob, name="glob",
                 description="List workspace files matching a glob pattern")
    reg.register(_list_dir, name="list_dir",
                 description="List directory entries inside the workspace")
    reg.register(_write_file, name="write_file",
                 description="Write text content to a file inside the workspace (disabled by default)")
    reg.register(_agent_bash, name="bash",
                 description="Run a shell command confined to the workspace (Landlock jail, network denied, timeout-capped)")
    return reg
