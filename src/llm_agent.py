import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional, List
from pathlib import Path
from state_representation import TextState, ExecutionState, CodeState

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

try:
    from moe_model import MoEGPT, MoEConfig
    _HAS_MOE = True
except ImportError:
    _HAS_MOE = False


class NanoGPTModel:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_DIR / "code_gpt.py"
        self.model = None
        self.vocab = {}
        self.loaded = False

    def load(self, checkpoint_path: str = None):
        # ROUND-171: root-anchored default (CWD-relative was silent-offline)
        if checkpoint_path is None:
            checkpoint_path = str(Path(__file__).resolve().parent.parent / "out" / "code_gpt" / "ckpt.pt")
        try:
            import sys
            root = Path(__file__).resolve().parent.parent
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            import torch.serialization as ser
            from config.code_gpt import CodeGPTConfig
            ser.add_safe_globals([CodeGPTConfig])

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.config = checkpoint.get("config", CodeGPTConfig())
            self.model = self._build_model(self.config)
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"])
            self.model.eval()
            self.loaded = True
            print(f"[NanoGPT] Loaded from {checkpoint_path}")
        except Exception as e:
            import traceback
            print(f"[NanoGPT] Could not load model: {e}")
            traceback.print_exc()

    def _build_model(self, config):
        class GPT(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.config = config
                self.transformer = nn.ModuleDict(dict(
                    wte=nn.Embedding(config.vocab_size, config.n_embd),
                    wpe=nn.Embedding(config.block_size, config.n_embd),
                    drop=nn.Dropout(config.dropout),
                    h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
                    ln_f=nn.LayerNorm(config.n_embd),
                ))
                self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

            def forward(self, idx, targets=None):
                B, T = idx.shape
                pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
                tok_emb = self.transformer.wte(idx)
                pos_emb = self.transformer.wpe(pos)
                x = self.transformer.drop(tok_emb + pos_emb)
                for block in self.transformer.h:
                    x = block(x)
                x = self.transformer.ln_f(x)
                logits = self.lm_head(x)
                loss = None
                if targets is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                return logits, loss

            def generate(self, idx, max_new_tokens=100, temperature=1.0):
                for _ in range(max_new_tokens):
                    idx_cond = idx[:, -self.config.block_size:]
                    logits, _ = self(idx_cond)
                    logits = logits[:, -1, :] / temperature
                    probs = torch.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, idx_next), dim=1)
                return idx

        class ValueEmbeddings(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.value_embeds = nn.Parameter(torch.randn(config.n_layer, config.n_embd) * 0.02)
                self.gates = nn.Parameter(torch.ones(config.n_layer))

            def forward(self, x, layer_idx):
                return x + self.gates[layer_idx] * self.value_embeds[layer_idx]

        class Block(nn.Module):
            def __init__(self, config, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx
                self.ln1 = nn.LayerNorm(config.n_embd)
                self.attn = CausalSelfAttention(config)
                self.ln2 = nn.LayerNorm(config.n_embd)
                self.mlp = nn.Sequential(
                    nn.Linear(config.n_embd, 4 * config.n_embd),
                    nn.GELU(),
                    nn.Linear(4 * config.n_embd, config.n_embd),
                    nn.Dropout(config.dropout),
                )
                self.value_embed = ValueEmbeddings(config)

            def forward(self, x):
                x = x + self.attn(self.ln1(x))
                x = x + self.mlp(self.ln2(x))
                x = self.value_embed(x, self.layer_idx)
                return x

        class CausalSelfAttention(nn.Module):
            def __init__(self, config):
                super().__init__()
                assert config.n_embd % config.n_head == 0
                self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
                self.c_proj = nn.Linear(config.n_embd, config.n_embd)
                self.n_head = config.n_head
                self.n_embd = config.n_embd
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

            def forward(self, x):
                B, T, C = x.shape
                q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
                k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
                att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1) ** 0.5))
                att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
                att = torch.softmax(att, dim=-1)
                y = att @ v
                y = y.transpose(1, 2).contiguous().view(B, T, C)
                return self.c_proj(y)

        return GPT(config)


    def _tokenize(self, text: str) -> list:
        try:
            from tokenizers import Tokenizer
            if getattr(self, "_tok", None) is None:
                self._tok = Tokenizer.from_file(
                    str(DATA_PROCESSED / "bpe_tokenizer" / "tokenizer.json"))
            return self._tok.encode(text).ids
        except Exception:
            return [0]

    def _detokenize(self, ids: list) -> str:
        try:
            if getattr(self, "_tok", None) is not None:
                return self._tok.decode(ids)
        except Exception:
            pass
        return ""


    def generate_code(self, prompt: str, max_tokens: int = 200) -> str:
        """NanoGPTModel lacked generate_code (AttributeError on use). Mirror of
        MoEModel.generate_code: completion-only (prompt stripped)."""
        if not self.loaded or self.model is None:
            return "# NanoGPT model not loaded."
        try:
            import torch
            tokens = self._tokenize(prompt)
            if not tokens:
                tokens = [0]
            idx = torch.tensor([tokens], dtype=torch.long, device=next(self.model.parameters()).device)
            with torch.no_grad():
                out = self.model.generate(idx, max_new_tokens=max_tokens,
                                          temperature=0.8)  # nano GPT: no top_k param
            return self._detokenize(out[0][len(tokens):].tolist())
        except Exception as e:
            return f"# Generation failed: {e}"


class MoEModel:
    def __init__(self, config_path: Optional[Path] = None):
        self.model = None
        self.loaded = False
        self.config = None
        self.tokenizer = None

    def load(self, checkpoint_path: str = "out/moe_gpt/ckpt.pt"):
        if not _HAS_MOE:
            print("[MoE] moe_model module not available")
            return
        try:
            from tokenizers import Tokenizer
            tok_path = DATA_PROCESSED / "bpe_tokenizer" / "tokenizer.json"
            if tok_path.exists():
                self.tokenizer = Tokenizer.from_file(str(tok_path))

            import torch.serialization
            from moe_model import MoEConfig
            torch.serialization.add_safe_globals([MoEConfig])

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.config = checkpoint.get("config", MoEConfig())
            self.model = MoEGPT(self.config)
            # Bug 47 fix: train_ambatron saves under "model_state"; accept both
            # (previously the load silently no-op'd -> random weights).
            state = checkpoint.get("model_state", checkpoint.get("model"))
            if state is not None:
                self.model.load_state_dict(state)
            self.model.eval()
            self.loaded = True
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[MoE] Loaded from {checkpoint_path} ({n_params:,} params)")
        except Exception as e:
            import traceback
            print(f"[MoE] Could not load model: {e}")
            traceback.print_exc()

    def _tokenize(self, text: str) -> list:
        if self.tokenizer:
            return self.tokenizer.encode(text).ids
        return [0]

    def _detokenize(self, ids: list) -> str:
        if self.tokenizer:
            return self.tokenizer.decode(ids)
        return ""

    def generate_code(self, prompt: str, max_tokens: int = 200) -> str:
        if not self.loaded or self.model is None:
            return "# MoE model not loaded."
        try:
            tokens = self._tokenize(prompt)
            if not tokens:
                tokens = [0]
            idx = torch.tensor([tokens], dtype=torch.long, device=next(self.model.parameters()).device)
            with torch.no_grad():
                out = self.model.generate(idx, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
            # Eval-contamination fix: return ONLY the completion, not prompt+completion
            # (the old return included the input prompt, inflating code-quality heuristics)
            return self._detokenize(out[0][len(tokens):].tolist())
        except Exception as e:
            return f"# Generation failed: {e}"


class LLMPlanner:
    def __init__(self, model_name: str = "placeholder", use_nanogpt: bool = False,
                 use_moe: bool = False, use_ambatron: bool = False,
                 ambatron_ckpt: str = ""):
        # Bug-11 fix: ckpt_best.pt only exists after a COMPLETED run; fall back
        # through ckpt.pt -> ckpt_ema.pt so a stopped run still wires up.
        if not ambatron_ckpt:
            _root = Path(__file__).resolve().parent.parent  # ROUND-170: CWD-independent
            for cand in ("out/ambatron_430m_v2/ckpt_best.pt",
                         "out/ambatron_430m_v2/ckpt_improved.pt",
                         "out/ambatron_430m_v2/ckpt.pt",
                         "out/ambatron_430m_v2/ckpt_ema.pt"):
                if (_root / cand).exists():
                    ambatron_ckpt = str(_root / cand)
                    break
        self.model_name = model_name
        self.nano_model = None
        self.moe_model = None

        if use_nanogpt:
            self.nano_model = NanoGPTModel()
            _root2 = Path(__file__).resolve().parent.parent
            ckpt = next((str(_root2 / "out" / "code_gpt" / c) for c in
                         ("ckpt.pt", "ckpt_improved.pt", "ckpt_best.pt")
                         if (_root2 / "out" / "code_gpt" / c).exists()), None)
            if ckpt:
                self.nano_model.load(ckpt)
        if use_moe or use_ambatron:
            self.moe_model = MoEModel()
            if use_ambatron:
                ckpt = ambatron_ckpt
            else:
                _root3 = Path(__file__).resolve().parent.parent
                ckpt = str(_root3 / "out" / "moe_gpt" / "ckpt.pt")
            if Path(ckpt).exists():
                self.moe_model.load(ckpt)

        nano_status = "loaded" if self.nano_model and self.nano_model.loaded else "offline"
        moe_status = "loaded" if self.moe_model and self.moe_model.loaded else "offline"
        print(f"[INFO] LLMPlanner: {model_name} "
              f"(nanogpt={nano_status}, moe={moe_status})")

    def plan_next_action(self, goal: TextState, exec_state: ExecutionState) -> str:
        if self.moe_model and self.moe_model.loaded:
            prompt = f"Goal: {goal.instruction}\nExecution: {exec_state.execution_trace[:200]}\nPlan:"
            return self.moe_model.generate_code(prompt, max_tokens=100)
        if self.nano_model and self.nano_model.loaded:
            prompt = f"Goal: {goal.instruction}\nExecution: {exec_state.execution_trace[:200]}\nPlan:"
            return self.nano_model.generate_code(prompt, max_tokens=100)

        if exec_state.error_log:
            plan = f"[RECOVERY] Previous error detected: {exec_state.error_log[:50]}... " \
                   f"Consider checking environment state and retrying with adjusted parameters."
        else:
            plan = "[PROCEED] Execution on track. Continue with current strategy and monitor for anomalies."

        if exec_state.updated_graph_features is not None:
            norm = float(np.linalg.norm(exec_state.updated_graph_features))
            plan += f" Graph stability metric: {norm:.3f}."

        return plan

    def generate_code(self, goal: TextState, execution_trace: str, max_tokens: int = 300) -> CodeState:
        if self.moe_model and self.moe_model.loaded:
            prompt = f"# Goal: {goal.instruction}\n# Context: {execution_trace[:200]}\n"
            generated = self.moe_model.generate_code(prompt, max_tokens=300)
            code_embedding = np.random.randn(768).astype(np.float32)
            return CodeState(
                initial_prompt=goal.instruction,
                current_code_artifact=generated,
                code_embedding=code_embedding,
            )
        if self.nano_model and self.nano_model.loaded:
            prompt = f"# Goal: {goal.instruction}\n# Context: {execution_trace[:200]}\n"
            generated = self.nano_model.generate_code(prompt, max_tokens=300)
            code_embedding = np.random.randn(768).astype(np.float32)
            return CodeState(
                initial_prompt=goal.instruction,
                current_code_artifact=generated,
                code_embedding=code_embedding,
            )

        code_template = '''
# Generated for goal: {goal}
# Based on execution trace: {trace_summary}

def execute_mission():
    print("Executing mission...")
    return True

if __name__ == "__main__":
    success = execute_mission()
    print("Mission status: " + ("SUCCESS" if success else "FAILED"))
'''

        code_artifact = code_template.format(
            goal=goal.instruction[:100],
            trace_summary=execution_trace[:100] if execution_trace else "Initial run"
        ).strip()

        code_embedding = np.random.randn(768).astype(np.float32)

        return CodeState(
            initial_prompt=goal.instruction,
            current_code_artifact=code_artifact,
            code_embedding=code_embedding,
        )

    def refine_code(self, code_state: CodeState, feedback: str) -> CodeState:
        refined_code = code_state.current_code_artifact + f"\n# FEEDBACK: {feedback}\n"
        return CodeState(
            initial_prompt=code_state.initial_prompt,
            current_code_artifact=refined_code,
            code_embedding=code_state.code_embedding,
        )
