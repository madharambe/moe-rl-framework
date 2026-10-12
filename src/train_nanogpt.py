import sys, os, math, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path
from config.code_gpt import CodeGPTConfig
from gpu_accel import resolve_device


def load_corpus(data_dir="data/processed"):
    data_path = Path(data_dir) / "code_corpus.npz"
    if not data_path.exists():
        print(f"Corpus not found at {data_path}. Run data_scraper.py first.")
        sys.exit(1)

    data = np.load(data_path, allow_pickle=True)
    samples = data["samples"]
    vocab = dict(data["vocab"])
    print(f"Loaded {len(samples)} samples, vocab={len(vocab)}")
    return samples, vocab


def build_model(config, vocab_size):
    config.vocab_size = vocab_size

    class ValueEmbeddings(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.value_embeds = nn.Parameter(torch.randn(config.n_layer, config.n_embd) * 0.02)
            self.gates = nn.Parameter(torch.ones(config.n_layer) * 0.1)

        def forward(self, x, layer_idx):
            return x + self.gates[layer_idx] * self.value_embeds[layer_idx]

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
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = torch.softmax(att, dim=-1)
            y = att @ v
            y = y.transpose(1, 2).contiguous().view(B, T, C)
            return self.c_proj(y)

    class Block(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()
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
            self.layer_idx = layer_idx

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            x = self.value_embed(x, self.layer_idx)
            return x

    return GPT(config).to(resolve_device())


def tokenize_text(text, vocab, block_size):
    import re
    tokens = re.findall(r"[a-zA-Z_]\w*|[^\w\s]", text.lower())
    ids = [vocab.get(t, 1) for t in tokens]
    ids = ids[:block_size]
    return ids


def get_lr(iteration, config):
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    if iteration > config.lr_decay_iters:
        return config.min_lr
    decay_ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    cos_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    lr_floor_ratio = 0.15
    return config.min_lr + (config.learning_rate - config.min_lr) * max(cos_decay, lr_floor_ratio)


def get_block_size(iteration, config, curriculum_stages=3):
    if not hasattr(config, 'curriculum') or not config.curriculum:
        return config.block_size
    total = config.lr_decay_iters if hasattr(config, 'lr_decay_iters') else config.max_iters
    stage_size = total // curriculum_stages
    sizes = [max(128, config.block_size // 3),
             max(256, config.block_size // 2),
             config.block_size]
    stage = min(iteration // stage_size, curriculum_stages - 1)
    frac = (iteration % stage_size) / max(stage_size, 1)
    if stage >= curriculum_stages - 1:
        return sizes[-1]
    return int(sizes[stage] + (sizes[stage + 1] - sizes[stage]) * frac)


def train(config, model, samples, vocab, resume: bool = False, compile_model: bool = True):
    device = resolve_device()
    model.train()

    if compile_model and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead", dynamic=True)
            print("[Compile] Model compiled with reduce-overhead")
        except Exception as e:
            print(f"[Compile] Failed: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                   weight_decay=config.weight_decay,
                                   betas=(config.beta1, config.beta2))

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    block_size = config.block_size
    batch_size = config.batch_size
    total_tokens = 0
    start_iter = 0
    best_loss = float('inf')

    if resume:
        latest = sorted(out_dir.glob("ckpt_*.pt")) + [out_dir / "ckpt.pt"]
        latest = [p for p in latest if p.exists()]
        if latest:
            ckpt = torch.load(latest[-1], map_location=device)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            if "iteration" in ckpt:
                start_iter = ckpt["iteration"] + 1
            if "total_tokens" in ckpt:
                total_tokens = ckpt["total_tokens"]
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"])
            if "best_loss" in ckpt:
                best_loss = ckpt["best_loss"]
            print(f"[RESUME] Continuing from iteration {start_iter} (loaded {latest[-1].name}, best_loss={best_loss:.4f})")

    n_grammar_terms = 4 if hasattr(config, 'multi_token_prediction') and config.multi_token_prediction else 1

    for iteration in range(start_iter, config.max_iters):
        lr = get_lr(iteration, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        curr_block_size = get_block_size(iteration, config)
        curr_block_size = min(curr_block_size, block_size)

        batch_ids = np.random.randint(0, len(samples), batch_size)
        x_batch, y_batch = [], []
        for sid in batch_ids:
            ids = tokenize_text(samples[sid], vocab, curr_block_size)
            if len(ids) < 2 + n_grammar_terms:
                continue
            x_batch.append(ids[:-n_grammar_terms])
            y_batch.append(ids[n_grammar_terms:])

        if not x_batch:
            continue

        max_len = min(max(len(t) for t in x_batch), curr_block_size - n_grammar_terms)
        x_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        y_padded = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        for i, (x, y) in enumerate(zip(x_batch, y_batch)):
            x_padded[i, :len(x)] = torch.tensor(x[:max_len], dtype=torch.long)
            y_padded[i, :len(y)] = torch.tensor(y[:max_len], dtype=torch.long)

        logits, loss = model(x_padded, targets=y_padded)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        total_tokens += x_padded.numel()
        current_loss = loss.item()
        best_loss = min(best_loss, current_loss)

        if iteration % config.log_interval == 0:
            print(f"Iter {iteration:6d} | Loss {current_loss:.4f} | "
                  f"LR {lr:.2e} | BS {curr_block_size:4d} | "
                  f"Tokens {total_tokens:,}")
        if iteration % config.save_interval == 0 and iteration > 0:
            ckpt_path = out_dir / f"ckpt_{iteration}.pt"
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration,
                "config": config,
                "loss": current_loss,
                "best_loss": best_loss,
                "total_tokens": total_tokens,
            }, ckpt_path)
            print(f"  Saved {ckpt_path}")

    final_path = out_dir / "ckpt.pt"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": config.max_iters - 1,
        "config": config,
        "best_loss": best_loss,
        "total_tokens": total_tokens,
    }, final_path)
    print(f"Training complete. Best loss: {best_loss:.4f}. Model saved to {final_path}")
    model.eval()
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NanoGPT Training v2 (Curr+ValEmb+Multi)")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument("--curriculum", action="store_true", help="Enable progressive context window")
    parser.add_argument("--multi_token", type=int, default=1,
                        help="Number of future tokens to predict (1=standard, 2-3=multi-token)")
    args = parser.parse_args()

    config = CodeGPTConfig()
    config.curriculum = args.curriculum
    config.multi_token_prediction = args.multi_token > 1

    samples, vocab = load_corpus(config.data_dir)
    model = build_model(config, len(vocab) + 3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    model = train(config, model, samples, vocab, resume=args.resume, compile_model=args.compile)


if __name__ == "__main__":
    main()
