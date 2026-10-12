"""Round-83: adapter merging (task arithmetic) — ONE model with both skills.

The measured interference (v11/v14: one adapter can't hold the tool-call and
the direct-answer behaviors) is addressed by merging instead of routing:
  merged = base + λ_tool·(tool − base) + λ_spatial·(spatial − base)

Task arithmetic is the "surprisingly strong baseline" (2607.18026: small-ratio
interpolation transfers complementary capabilities; near-balanced collapses —
so the λ grid matters). Optionally runs the two evals right after the merge.

Usage:
  python3 src/merge_models.py --base out/ambatron_430m_v2/ckpt_best.pt \
      --tool out/ambatron_tool_v13/ckpt.pt --spatial out/ambatron_tool_v12/ckpt.pt \
      --lt 0.5 --ls 0.5 --out out/ambatron_merged/ckpt_0505.pt --eval
"""
import argparse, sys, pathlib, subprocess, re
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch
from storage import safe_save  # B-211: atomic save — merge often targets the SSD directly
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tool", required=True, help="tool adapter (v13)")
    ap.add_argument("--spatial", required=True, help="spatial/chat adapter (v12)")
    ap.add_argument("--lt", type=float, required=True, help="λ for the tool delta")
    ap.add_argument("--ls", type=float, required=True, help="λ for the spatial delta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", action="store_true", help="run the two evals after")
    ap.add_argument("--dare", type=float, default=0.0,
                    help="DARE-style: drop this fraction of each delta's entries "
                         "(random mask) + rescale by 1/(1-p) — reduces the "
                         "interference between merged deltas")
    ap.add_argument("--layer_lambdas", type=str, default=None,
                    help="B-105: comma-separated per-layer coefficients "
                         "(Hyper-ES 2608.05541 style), e.g. 1.0,0.8,0.5,0.9,1.0")
    args = ap.parse_args()

    from moe_model import MoEConfig
    import torch.serialization as ser
    ser.add_safe_globals([MoEConfig])

    base = torch.load(args.base, map_location="cpu", weights_only=True)
    tool = torch.load(args.tool, map_location="cpu", weights_only=True)
    spat = torch.load(args.spatial, map_location="cpu", weights_only=True)
    bs, ts, ss = base["model_state"], tool["model_state"], spat["model_state"]
    if set(bs) != set(ts) or set(bs) != set(ss):
        print(f"[merge] state-dict mismatch: base {len(bs)} tool {len(ts)} "
              f"spatial {len(ss)} keys")
        return
    # Sep-5 (audit): key-set equality is NOT enough — a shape mismatch (e.g.
    # changed n_embd) raised an opaque broadcast RuntimeError mid-loop. Check
    # every key's shape up-front with a clear message.
    for k in bs:
        if bs[k].shape != ts[k].shape or bs[k].shape != ss[k].shape:
            print(f"[merge] SHAPE mismatch at {k}: base {tuple(bs[k].shape)} "
                  f"tool {tuple(ts[k].shape)} spatial {tuple(ss[k].shape)} — "
                  f"cannot merge (different architectures)")
            return
    merged = {}
    rng = torch.Generator().manual_seed(7)
    # B-105: layer-wise coefficients (Hyper-ES 2608.05541 style) — when
    # --layer_lambdas is passed (e.g. "1.0,0.8,0.5,0.9,1.0"), each layer's
    # delta is scaled by its own lambda; --lt/--ls remain the global scale.
    layer_ls = None
    if args.layer_lambdas:
        try:
            layer_ls = [float(x) for x in args.layer_lambdas.split(",")]
        except ValueError:
            print(f"[merge] bad --layer_lambdas: {args.layer_lambdas}")
            return
    for k in bs:
        b = bs[k].float()
        lam = 1.0
        if layer_ls is not None:
            m = re.search(r"(?:layers|blocks)\.(\d+)", k)
            if m:
                idx = int(m.group(1))
                lam = layer_ls[idx % len(layer_ls)]
        dt = args.lt * lam * (ts[k].float() - b)
        ds = args.ls * lam * (ss[k].float() - b)
        if args.dare > 0.0 and dt.numel() > 0:
            scale = 1.0 / (1.0 - args.dare)
            mt = torch.rand(dt.shape, generator=rng, device=dt.device) >= args.dare
            ms = torch.rand(ds.shape, generator=rng, device=ds.device) >= args.dare
            dt = dt * mt * scale
            ds = ds * ms * scale
        merged[k] = b + dt + ds
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    from storage import safe_save  # B-211: atomic save — merge targets the SSD
    safe_save({"model_state": merged, "config": base["config"],
                "args": {"merge": vars(args)}}, out_path)
    print(f"[merge] λ_tool={args.lt} λ_spatial={args.ls} -> {out_path}")

    if args.eval:
        py = sys.executable
        ev1 = subprocess.run(
            [py, "src/eval_tool_use.py", "--ckpt", str(out_path),
             "--repeat", "3", "--heldout", "--compiled"],
            capture_output=True, text=True)
        print("[merge] tool held-out:", [l for l in ev1.stdout.splitlines()
                                         if "emit-rate" in l or "correct :" in l])
        ev2 = subprocess.run(
            [py, "src/eval_spatial.py", "--ckpt", str(out_path), "--n", "40"],
            capture_output=True, text=True)
        print("[merge] spatial:", [l for l in ev2.stdout.splitlines()
                                   if "spatial-eval" in l])


if __name__ == "__main__":
    main()