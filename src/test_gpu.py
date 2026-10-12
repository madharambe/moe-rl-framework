"""GPU smoke test — only runs when executed directly."""
import sys, pathlib
if __name__ == "__main__":
    import sys, subprocess
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from moe_model import create_ambatron_430m_config, MoEGPT
    import torch

    device = torch.device("cuda")
    config = create_ambatron_430m_config(batch_size=2, max_iters=2)
    model = MoEGPT(config).to(device)
    model.train()

    # Check all params on GPU
    for name, p in model.named_parameters():
        if p.device.type != "cuda":
            print(f"CPU param: {name}")
            break
    else:
        print(f"All params on GPU")

    # Check all buffers on GPU
    for name, buf in model.named_buffers():
        if buf.device.type != "cuda":
            print(f"CPU buffer: {name}")
            break
    else:
        print(f"All buffers on GPU")

    # Check forward/backward
    x = torch.randint(0, config.vocab_size, (2, 512), device=device)
    with torch.amp.autocast("cuda"):
        logits, loss, aux = model(x, targets=x, use_grad_ckpt=True)
    loss.backward()
    model.zero_grad(set_to_none=True)

    # Check GPU util
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5
    )
    print(f"nvidia-smi: {out.stdout.strip()}")
    print(f"Forward+backward OK: loss={loss.item():.4f}")
