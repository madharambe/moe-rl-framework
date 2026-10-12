"""
Fine-tune MoE LLM on tool-use data.
Generates synthetic instruction → tool_call → response pairs from code corpus.
"""
import sys, os, json, re, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from pathlib import Path
from moe_model import MoEConfig, MoEGPT
from gpu_accel import resolve_device
from tool_use import TOOL_CALL_START, TOOL_CALL_END, TOOL_RESPONSE_START, TOOL_RESPONSE_END

DATA_PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_corpus():
    data_path = DATA_PROCESSED / "code_corpus.npz"
    data = np.load(data_path, allow_pickle=True)
    return data["samples"], dict(data["vocab"])


def tokenize_text(text, vocab, block_size):
    toks = re.findall(r"[a-zA-Z_]\w*|[^\w\s]", text.lower())
    ids = [vocab.get(t, 1) for t in toks]
    return ids[:block_size]


def extract_functions(samples):
    functions = []
    for s in samples:
        matches = re.findall(r"def (\w+)\s*\(([^)]*)\)(?:\s*->\s*([^:]+))?:\s*(.*?)(?=\ndef |\Z)", s, re.DOTALL)
        for name, args, ret, body in matches:
            body_clean = body.strip()[:200]
            if len(body_clean) > 20:
                functions.append({"name": name, "args": args, "ret": ret, "body": body_clean})
    return functions


def generate_tool_data(samples, vocab, num_samples=2000, block_size=512):
    functions = extract_functions(samples)
    print(f"Extracted {len(functions)} functions from corpus")

    tool_data = []
    instructions = [
        "Write a function to", "Implement", "Create a function that",
        "I need code for", "Help me implement",
    ]
    intros = ["Here is the implementation:", "You can use this:", "Try this code:", ""]

    for _ in range(num_samples):
        if functions and random.random() < 0.5:
            fn = random.choice(functions)
            instr = f"{random.choice(instructions)} {fn['name'].replace('_', ' ')}"
            tool_call = json.dumps({"name": "generate_code", "arguments": {"task": instr}})
            response = f"def {fn['name']}({fn['args']})"
            if fn['ret']:
                response += f" -> {fn['ret']}"
            response += ":\n    " + fn['body'].replace("\n", "\n    ")
        else:
            sample = random.choice(samples)
            lines = [l for l in sample.split("\n") if l.strip() and not l.strip().startswith("#")]
            if not lines:
                continue
            code_line = random.choice(lines)[:120]
            instr = f"Explain this code: {code_line}"
            tool_call = json.dumps({"name": "explain_code", "arguments": {"code": code_line}})
            response = f"This code {code_line.lower().strip()[:60]}..."

        text = (
            f"User: {instr}\n"
            f"Assistant: {TOOL_CALL_START}{tool_call}{TOOL_CALL_END}\n"
            f"{TOOL_RESPONSE_START}{response}{TOOL_RESPONSE_END}\n"
        )
        ids = tokenize_text(text, vocab, block_size)
        if len(ids) >= 5:
            tool_data.append(ids)

    print(f"Generated {len(tool_data)} tool-use training sequences")
    return tool_data


def train_tools(config, model, tool_data, vocab, base_iters=200):
    device = resolve_device()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    block_size = config.block_size
    batch_size = min(8, len(tool_data))
    n_epochs = 3

    for epoch in range(n_epochs):
        random.shuffle(tool_data)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(tool_data), batch_size):
            batch = tool_data[i:i + batch_size]
            max_len = min(max(len(t) for t in batch), block_size - 1)
            x = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            y = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)

            for j, ids in enumerate(batch):
                ids = ids[:block_size]
                x[j, :len(ids) - 1] = torch.tensor(ids[:-1], dtype=torch.long)
                y[j, :len(ids) - 1] = torch.tensor(ids[1:], dtype=torch.long)

            logits, loss, aux = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.4f}, ce={aux['ce_loss']:.4f}")

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "best_loss": min(avg_loss, 99),
        "total_tokens": 0,
    }, out_dir / "ckpt_tools.pt")
    print(f"Tool-use model saved to {out_dir / 'ckpt_tools.pt'}")
    model.eval()
    return model


def main():
    samples, vocab = load_corpus()
    config = MoEConfig()
    config.vocab_size = len(vocab) + 3

    # Load base MoE checkpoint
    base_ckpt = Path(config.out_dir) / "ckpt.pt"
    if base_ckpt.exists():
        ckpt = torch.load(base_ckpt, map_location="cpu", weights_only=False)
        model = MoEGPT(config)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        print(f"Loaded base checkpoint (best_loss={ckpt.get('best_loss', 'N/A')})")
    else:
        print("No base checkpoint found, training from scratch")
        model = MoEGPT(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    tool_data = generate_tool_data(samples, vocab, num_samples=2000)
    model = train_tools(config, model, tool_data, vocab)


if __name__ == "__main__":
    main()
