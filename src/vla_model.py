"""
Vision-Language-Action model built on our custom MoEGPT/Ambatron backbone.
Blueprint: SmolVLA (arXiv 2506.01844) + pi0 (flow-matching action expert).

Pipeline:
  [state token] + [SigLIP image tokens (pixel-shuffle)] + [BPE instruction tokens]
      -> MoEGPT backbone (layers up to feature_layer, layer-skipping) -> features
      -> FlowActionExpert (interleaved cross/self attention) -> action chunk
Training: conditional flow matching  L = ||v_theta(x_t, t, cond) - (a1 - a0)||^2
Inference: 10-step Euler integration of the velocity field.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List

from moe_model import MoEGPT, MoEConfig, create_ambatron_430m_config
from state_representation import AgentState

DATA_PROCESSED = __import__("pathlib").Path(__file__).resolve().parent.parent / "data" / "processed"
BPE_TOKENIZER_PATH = DATA_PROCESSED / "bpe_tokenizer" / "tokenizer.json"

FLOW_STEPS = 10


def _bpe_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(BPE_TOKENIZER_PATH))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] in [0, 1]
        t = t.to(dtype=self.freqs.dtype)
        args = t[:, None] * self.freqs[None, :]  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class VisionTower(nn.Module):
    """SigLIP vision encoder -> pixel-shuffle token reduction -> MLP projector.
    use_pretrained=False uses a random SigLIP (smoke tests); True downloads weights.
    """
    def __init__(self, out_dim: int, use_pretrained: bool = True,
                 image_size: int = 224, patch_size: int = 16,
                 pixel_shuffle: int = 2, hidden: int = 768):
        super().__init__()
        from transformers import SiglipModel, SiglipConfig
        if use_pretrained:
            self.siglip = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
        else:
            cfg = SiglipConfig(
                vision_config=dict(hidden_size=hidden, image_size=image_size,
                                   patch_size=patch_size, num_hidden_layers=12,
                                   num_attention_heads=12, intermediate_size=3072))
            self.siglip = SiglipModel(cfg)
        self.pixel_shuffle = pixel_shuffle
        enc_dim = hidden * (pixel_shuffle ** 2)  # space-to-depth: channels scale by r^2
        self.n_tokens = (image_size // patch_size // pixel_shuffle) ** 2
        self.projector = nn.Sequential(
            nn.Linear(enc_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.image_size = image_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, H, W, 3] uint8/float in [0,1] -> [B, n_tokens, out_dim]"""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.shape[-1] == 3 and images.shape[-2] != 3:  # NHWC -> NCHW
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.shape[-2] != self.image_size:
            images = F.interpolate(images, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        # B-103: SigLIP input normalization (the model was fed raw [0,1] pixels
        # since day one — siglip expects (x/255 - 0.5)/0.5; with [0,1] inputs
        # that is (x - 0.5)/0.5). The frozen tower saw a permanently shifted
        # distribution, degrading every downstream result.
        images = (images - 0.5) / 0.5
        out = self.siglip.vision_model(images)
        feats = out.last_hidden_state                    # [B, seq, C] or [B, H, W, C]
        if feats.dim() == 3:
            B, seq, C = feats.shape
            side = int(round(seq ** 0.5))
            feats = feats.reshape(B, side, side, C)
        B, h, w, C = feats.shape
        r = self.pixel_shuffle
        feats = feats.permute(0, 3, 1, 2).contiguous()          # [B, C, h, w]
        feats = feats.view(B, C, h // r, r, w // r, r)
        feats = feats.permute(0, 1, 3, 5, 2, 4).reshape(
            B, C * r * r, h // r, w // r)                        # [B, C*r^2, h/r, w/r]
        feats = feats.permute(0, 2, 3, 1).reshape(B, -1, C * r * r)  # [B, n, C*r^2]
        return self.projector(feats)


class FlowActionExpert(nn.Module):
    """Flow-matching action expert (SmolVLA/pi0 style).
    Alternating cross-attention (to VLM features) + causal self-attention + SwiGLU.
    Trained with conditional flow matching; decoded via Euler integration.
    """
    def __init__(self, cond_dim: int, action_dim: int, chunk_len: int = 50,
                 expert_dim: Optional[int] = None, n_blocks: int = 6,
                 n_heads: int = 8, time_dim: int = 64, tanh_output: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.tanh_output = tanh_output
        self.dim = expert_dim or int(0.75 * cond_dim)
        self.time_dim = time_dim
        self.n_heads = n_heads
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Linear(time_dim, self.dim)
        self.action_proj = nn.Linear(action_dim, self.dim)
        self.cond_proj = nn.Linear(cond_dim, self.dim)
        # B-125: DIRECT additive task-proprioception injection. The v4 attempt
        # appended a projected task TOKEN to cond — cross-attention diluted it
        # to nothing among hundreds of backbone tokens (measured: output did
        # not vary with the task at all). Additive broadcast mirrors the
        # time-embedding path, which demonstrably shapes every output.
        self.task_mlp = nn.Sequential(
            nn.Linear(5, self.dim), nn.GELU(), nn.Linear(self.dim, self.dim))
        # B-129: DIRECT linear skip from task vector to the output actions.
        # Measured: the deep-path gradient to any task module vanishes to
        # ~1e-10 through six attention blocks (attention dilution), so every
        # injection variant learned nothing. This skip's gradient is O(1)
        # by construction and fits the near-linear bearing->action map.
        self.task_head = nn.Linear(5, self.action_dim)
        self.pos = nn.Parameter(torch.zeros(1, max(chunk_len, 64), self.dim))
        nn.init.normal_(self.pos, std=0.02)

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ExpertBlock(self.dim, n_heads))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim),
            nn.GELU(), nn.Linear(self.dim, action_dim))

    def _cond_causal_mask(self, n_cond: int, n_act: int, device, dtype):
        """Action tokens attend to all cond tokens + past action tokens (causal)."""
        mask = torch.zeros(n_act, n_cond + n_act, device=device, dtype=torch.bool)
        mask[:, :n_cond] = True
        for i in range(n_act):
            mask[i, n_cond:n_cond + i + 1] = True
        return mask

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                task_vec: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_t: [B, chunk, act_dim] noisy actions at time t; t: [B] in [0,1];
        cond: [B, n_cond, cond_dim] VLM features -> [B, chunk, act_dim] velocity
        task_vec: [B, 4] additive proprioception (B-125)."""
        dt = next(self.parameters()).dtype
        x_t = x_t.to(dtype=dt)
        t = t.to(dtype=dt)
        cond = cond.to(dtype=dt)
        B, chunk, _ = x_t.shape
        h = self.action_proj(x_t) + self.time_proj(self.time_embed(t)).unsqueeze(1)
        if task_vec is not None:
            h = h + self.task_mlp(task_vec.to(dtype=dt)).unsqueeze(1)
        h = h + self.pos[:, :chunk]
        cond_h = self.cond_proj(cond)
        full = torch.cat([cond_h, h], dim=1)
        mask = self._cond_causal_mask(cond_h.shape[1], chunk, x_t.device, x_t.dtype)
        for block in self.blocks:
            full = block(full, mask)
        out = self.head(full[:, -chunk:])
        if task_vec is not None:
            # B-129: the vanishing-gradient-proof shortcut
            out = out + self.task_head(task_vec.to(dtype=out.dtype)).unsqueeze(1)
        return out

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, chunk_len: Optional[int] = None,
               steps: int = FLOW_STEPS) -> torch.Tensor:
        """Euler integration of the velocity field from x_0 ~ N(0, I)."""
        chunk = chunk_len or self.chunk_len
        B = cond.shape[0]
        dt = next(self.parameters()).dtype
        x = torch.randn(B, chunk, self.action_dim, device=cond.device, dtype=dt)
        dt_step = 1.0 / steps
        for i in range(steps):
            t = torch.full((B,), (i + 0.5) * dt_step, device=cond.device, dtype=dt)
            v = self.forward(x, t, cond)
            x = x + v * dt_step
        if self.tanh_output:
            x = torch.tanh(x)
        return x


class ExpertBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        hidden = int(dim * 2 * 2 / 3)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.attn_dropout = nn.Dropout(0.05)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        # cross-attention: action tokens (tail) attend to cond tokens (head)
        # mask is [n_act, n_cond + n_act] -> n_cond = cols - rows
        n_cond = causal_mask.shape[1] - causal_mask.shape[0]
        q, kv = x[:, n_cond:], x
        return self._forward_sdpa(x, q, kv, causal_mask, n_cond)

    def _forward_sdpa(self, x, q, kv, causal_mask, n_cond):
        dim, nh = self.cross_attn.embed_dim, self.cross_attn.num_heads
        hd = dim // nh

        def split(t):
            B, T, D = t.shape
            return t.view(B, T, nh, hd).transpose(1, 2)

        # cross-attention (q = action tokens, kv = all tokens)
        qw, kw, vw = self.cross_attn.in_proj_weight.chunk(3)
        qb, kb, vb = self.cross_attn.in_proj_bias.chunk(3)
        qc = split(F.linear(q, qw, qb))
        kc = split(F.linear(kv, kw, kb))
        vc = split(F.linear(kv, vw, vb))
        scores = (qc @ kc.transpose(-1, -2)) / math.sqrt(hd)
        am = causal_mask.unsqueeze(0).unsqueeze(0)  # [1,1,n_act,n_cond+n_act]
        scores = scores.masked_fill(~am, float("-inf"))
        y = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype) @ vc
        B, T, _ = q.shape
        y = y.transpose(1, 2).reshape(B, T, dim)
        x = x.clone()
        x[:, n_cond:] = self.cross_norm(x[:, n_cond:] + self.cross_attn.out_proj(y))
        x[:, n_cond:] = self.attn_dropout(x[:, n_cond:])

        # causal self-attention over the full sequence (cond rows attend to cond)
        T = x.shape[1]
        full_mask = torch.zeros(T, T, device=x.device, dtype=torch.bool)
        full_mask[:n_cond, :n_cond] = True
        full_mask[n_cond:, :] = causal_mask
        qs, ks, vs = [split(F.linear(x, w, b))
                      for w, b in zip(self.self_attn.in_proj_weight.chunk(3),
                                      self.self_attn.in_proj_bias.chunk(3))]
        scores = (qs @ ks.transpose(-1, -2)) / math.sqrt(hd)
        scores = scores.masked_fill(~full_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        y = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype) @ vs
        y = y.transpose(1, 2).reshape(B, T, dim)
        x = self.self_norm(x + self.self_attn.out_proj(y))
        x = x + self.ffn(x)
        return x


class VLAPolicy(nn.Module):
    """Vision-Language-Action policy: SigLIP + MoEGPT backbone + FlowActionExpert.

    get_action(state, instruction) -> np action, compatible with RoboticsEnv/RobotAgent.
    """
    def __init__(self,
                 backbone_config: Optional[MoEConfig] = None,
                 action_dim: int = 4,
                 chunk_len: int = 50,
                 feature_layer: Optional[int] = None,
                 freeze_backbone: bool = True,
                 freeze_vision: bool = True,
                 use_pretrained_siglip: bool = True,
                 vision_hidden: int = 768,
                 state_dim: int = 1872,
                 expert_blocks: int = 6,
                 max_instruction_tokens: int = 128):
        super().__init__()
        self.config = backbone_config or create_ambatron_430m_config()
        self.backbone = MoEGPT(self.config)
        self.n_layer = self.config.n_layer
        self.feature_layer = feature_layer or max(1, self.n_layer // 2)
        self.n_embd = self.config.n_embd
        self.chunk_len = chunk_len
        self.max_instruction_tokens = max_instruction_tokens
        self.action_dim = action_dim
        self.freeze_backbone = freeze_backbone
        self.freeze_vision = freeze_vision
        self.tokenizer = None  # lazy

        self.vision = VisionTower(self.n_embd, use_pretrained=use_pretrained_siglip,
                                  hidden=vision_hidden)
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, self.n_embd), nn.GELU(), nn.Linear(self.n_embd, self.n_embd))
        self.feature_proj = nn.Linear(self.n_embd, self.n_embd)
        # B-124: direct task-proprioception token. The backbone is frozen and
        # never learned to decode coordinates; 6k LoRA steps cannot rewire it
        # (measured: constant dataset-mean actions). This fresh trainable
        # projection gives the flow expert DIRECT linear access to
        # [goal_dx, goal_dy, sin(yaw), cos(yaw)] as one extra cond token.
        # 19th-audit H5: THIS LAYER IS AN ORPHAN — it appears in NO forward
        # path (the live path is the 5-D task_mlp/task_head). Every saved ckpt
        # carries random garbage weights for it (key-match noise on load).
        # Kept intentionally: removing it breaks strict load_state_dict for
        # every pre-audit ckpt. Do NOT wire it into forward — use task_proj
        # conds only after re-training from scratch.
        self.task_proj = nn.Linear(4, self.n_embd)
        self.expert = FlowActionExpert(self.n_embd, action_dim, chunk_len,
                                       n_blocks=expert_blocks)

        self._set_frozen()

    def _set_frozen(self):
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
        if self.freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad_(False)

    def _tokenize_instruction(self, texts: List[str]) -> List[List[int]]:
        if self.tokenizer is None:
            self.tokenizer = _bpe_tokenizer()
        ids = []
        for t in texts:
            toks = self.tokenizer.encode(t).ids[:self.max_instruction_tokens]
            if not toks:
                toks = [0]  # <PAD>, not <UNK>: pad tokens must be inert
            ids.append(toks)
        return ids

    def _backbone_features_from_emb(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone's first `feature_layer` layers (layer skipping)."""
        x = self.backbone.drop(x)
        v_first = None
        for i, layer in enumerate(self.backbone.layers[:self.feature_layer]):
            use_vr = getattr(layer, "use_value_residual", False)
            if i == 0 and use_vr:
                out = layer(x, None, None)
                x = out if isinstance(out, torch.Tensor) else out[0]
                v_first = layer.v_local
                continue
            out = layer(x, None, v_first)
            x = out if isinstance(out, torch.Tensor) else out[0]
        return self.backbone.norm(x)

    @staticmethod
    def _prepare_state(states: torch.Tensor) -> torch.Tensor:
        """Raw AgentState vectors embed the visual pixels (150,528); the VLA has
        its own vision tower, so keep only the non-visual modalities (1,872)."""
        if states.shape[-1] >= 152400:
            return states[..., 150528:]
        return states

    def _build_sequence(self, images, instructions, states):
        """[state(1) | vision(n_vis) | instruction(T)] embeddings."""
        dt = next(self.parameters()).dtype
        if images.is_floating_point():
            images = images.to(dtype=dt)
        if states.is_floating_point():
            states = states.to(dtype=dt)
        vis = self.vision(images)                          # [B, n_vis, n_embd]
        st = self.state_proj(self._prepare_state(states)).unsqueeze(1)  # [B, 1, n_embd]
        ids = self._tokenize_instruction(instructions)
        B = vis.shape[0]
        max_len = max(len(i) for i in ids)
        if max_len > 0:
            padded = torch.full((B, max_len), 0, dtype=torch.long, device=images.device)  # <PAD>
            for i, tids in enumerate(ids):
                padded[i, :len(tids)] = torch.tensor(tids, dtype=torch.long, device=images.device)
            emb = self.backbone.token_embed(padded)
        else:
            emb = torch.zeros(B, 0, self.n_embd, device=images.device)
        return torch.cat([st, vis, emb], dim=1)            # [B, 1+n_vis+T, n_embd]

    def forward(self, images: torch.Tensor, instructions: List[str],
                states: torch.Tensor, actions: torch.Tensor,
                t: Optional[torch.Tensor] = None,
                task: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """BC forward: returns (flow_velocity, flow_loss_target)."""
        full = self._build_sequence(images, instructions, states)
        feats = self._backbone_features_from_emb(full)
        # Bug 40 fix: no detach — freeze_backbone only sets requires_grad_(False)
        # on base weights, so LoRA adapters + state_proj actually receive
        # gradients. (The old detach silently killed LoRA: all B matrices were
        # still exactly 0.0 in the BC checkpoint.)
        cond = self.feature_proj(feats)

        if t is None:
            t = torch.rand(actions.shape[0], device=images.device)
        a0 = torch.randn_like(actions)
        x_t = (1 - t[:, None, None]) * a0 + t[:, None, None] * actions
        v_pred = self.expert(x_t, t, cond, task_vec=self._task_vec(task))
        return v_pred, (actions - a0)

    def flow_loss(self, images, instructions, states, actions,
                  task: Optional[torch.Tensor] = None) -> torch.Tensor:
        v_pred, target = self.forward(images, instructions, states, actions,
                                      task=task)
        return F.mse_loss(v_pred, target)

    def _cond_features(self, images, instructions, states) -> torch.Tensor:
        full = self._build_sequence(images, instructions, states)
        feats = self._backbone_features_from_emb(full)
        return self.feature_proj(feats)

    def _task_vec(self, task: Optional[torch.Tensor]):
        """B-125: cast the [B,5] task-proprioception vector to the model dtype/device."""
        if task is None:
            return None
        return task.to(next(self.parameters()).dtype)

    def deterministic_action(self, images, instructions, states,
                             chunk_len: Optional[int] = None,
                             n_euler_steps: int = 1,
                             task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Greedy action: integrate the learned velocity field from the MEAN
        of the starting distribution (x = 0 at t = 0 — a0 ~ N(0, I)) to the
        action endpoint. Deterministic -> TD3-compatible.
        NOT no_grad: the TD3 actor update backprops through it; rollout callers
        (train_vla_rl.py rollout, target calc; robot_core; BC eval) wrap in no_grad.
        B-111: n_euler_steps > 1 runs proper ODE integration (pi0 uses ~10);
        the default of 1 preserves the exact behavior the deployed checkpoints
        (phys17 et al.) were trained and validated under."""
        cond = self._cond_features(images, instructions, states)
        chunk = chunk_len or self.chunk_len
        B = cond.shape[0]
        dt_dtype = next(self.parameters()).dtype
        x = torch.zeros(B, chunk, self.action_dim, device=cond.device,
                        dtype=dt_dtype)
        n = max(1, int(n_euler_steps))
        h = 1.0 / n
        tv = self._task_vec(task)
        for i in range(n):
            t = torch.full((B,), i * h, device=cond.device, dtype=dt_dtype)
            v = self.expert(x, t, cond, task_vec=tv)
            x = x + h * v
        v = x
        if self.expert.tanh_output:
            v = torch.tanh(v)
        return v[:, 0].float()

    @torch.no_grad()
    def get_action(self, image: torch.Tensor, instruction: str,
                   state: Union[torch.Tensor, AgentState],
                   chunk_len: Optional[int] = None,
                   task: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Returns the first action of the decoded chunk: [action_dim].
        B-151: pass `task` (the 5D body-frame vector) -- without it this API
        silently produced UNCONDITIONED actions while every live consumer
        uses deterministic_action(task=...). With task given we delegate to
        the same conditioned path the eval/DAgger loops use."""
        if task is not None:
            self.eval()  # ROUND-169: deployment path — no dropout in decode
            # ROUND-170 + ROUND-171: the task branch must fully mirror the
            # non-task preprocessing. Round-170 fixed the str->List instruction
            # and 1-D state; Round-171 (audit, empirically reproduced) found
            # the 1-D [5] TASK was still never unsqueezed (crash in
            # task_mlp: [192,1] vs batch 4) AND nothing moved image/state/task
            # to the model device (robot_core feeds CPU tensors -> CUDA crash).
            dev = next(self.parameters()).device
            if isinstance(state, AgentState):
                state = torch.from_numpy(state.get_combined_state()).float()
            if not isinstance(state, torch.Tensor):
                state = torch.as_tensor(state, dtype=torch.float32)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = self._prepare_state(state)
            if not isinstance(task, torch.Tensor):
                task = torch.as_tensor(task, dtype=torch.float32)
            if task.dim() == 1:
                task = task.unsqueeze(0)
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            # ROUND-172 (N3): a uint8 image (or 0..255 values) must be
            # normalized BEFORE VisionTower (B-103 assumes [0,1]; uint8 would
            # silently become -1..509). Mirror dagger_round's clamp logic.
            if (image.dtype != torch.float32 or image.max() > 1.5) and image.numel() > 0:
                image = image.float() / 255.0
            # ROUND-172 (N2): deterministic_action returns [B,4]; RobotAgent/
            # env.step consume a 1-D (4,) action — squeeze the batch dim here so
            # the task branch matches the no-task branch's (4,) contract.
            return self.deterministic_action(image.to(dev), [instruction], state.to(dev),
                                             task=task.to(dev),
                                             chunk_len=chunk_len)[0]
        self.eval()
        if isinstance(state, AgentState):
            state = torch.from_numpy(state.get_combined_state()).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = self._prepare_state(state)
        img = image if image.dim() == 4 else image.unsqueeze(0)
        dev = next(self.parameters()).device  # Bug-11 fix: cross-device crash
        dt = next(self.parameters()).dtype
        img = img.to(device=dev, dtype=dt)
        state = state.to(device=dev, dtype=dt)
        vis = self.vision(img)
        st = self.state_proj(state.to(vis.device)).unsqueeze(1)
        ids = self._tokenize_instruction([instruction])
        tid = torch.tensor(ids[0], dtype=torch.long, device=vis.device)
        emb = self.backbone.token_embed(tid).unsqueeze(0)
        full = torch.cat([st, vis, emb], dim=1)
        feats = self._backbone_features_from_emb(full)
        cond = self.feature_proj(feats)
        chunk = self.expert.sample(cond, chunk_len)
        return chunk[0, 0].float()


def create_vla_small_config(vocab_size: int = 16384):
    """Small MoEGPT backbone for smoke tests (fast on CPU)."""
    return MoEConfig(
        vocab_size=vocab_size, block_size=512, n_layer=6, n_head=4, n_kv_head=2,
        n_embd=256, n_experts=4, n_experts_per_token=2, use_shared_expert=True,
        moe_layers=(1, 3, 5), sliding_window=256, dropout=0.0)
