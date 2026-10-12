"""B-127: BC-only runner — resumes straight from a saved demo bundle without
loading the physics env or pretrained SigLIP, sized to fit tight RAM windows
(the full gen_demos peaked at ~4.8G; this targets <4.2G).

Usage: systemd-run --user ... python3 src/bc_only.py --demos out/demos_v6.pt \
    --vla_ckpt out/vla_rl/vla_rl_phys17.pt --bc_updates 6000 --bc_batch 16 \
    --out out/vla_rl/vla_bc_v6.pt
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="out/demos_v2yaw_npy")
    ap.add_argument("--allow-untrusted-bundle", action="store_true",
                    help="Explicitly permit loading a .pt demo BUNDLE with "
                         "weights_only=False (RCE surface on a tampered/corrupt "
                         "file). Prefer --demos pointing at a .npy DIR (mmap-safe).")
    ap.add_argument("--vla_ckpt", required=True)
    ap.add_argument("--bc_updates", type=int, default=6000)
    ap.add_argument("--bc_batch", type=int, default=16)
    ap.add_argument("--bc_lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import time, torch
    import torch.serialization as ser
    from moe_model import MoEConfig
    ser.add_safe_globals([MoEConfig])
    from vla_model import VLAPolicy
    from lora_adapter import add_lora

    t0 = time.time()
    import numpy as np
    from pathlib import Path
    # B-138: comma-separated dataset DIRS, each mmap'd; batches sample across
    # all of them proportionally (base demos + DAgger corrections co-train).
    dirs = [Path(x) for x in args.demos.split(",")]
    mms = []
    sizes = []
    for dd in dirs:
        dd = Path(str(dd).strip())
        if dd.is_dir():
            m = {k: np.load(dd / f"{k}.npy", mmap_mode="r")
                 for k in ("imgs", "states", "actions", "tasks")}
            # B-141b: DAgger dirs carry expert.npy -- the EXPERT action is the
            # regression target for policy-visited states (that is the entire
            # point of DAgger; the v4-v11 runs trained on self-generated
            # labels because this file was ignored).
            if (dd / "expert.npy").exists():
                m["target"] = np.load(dd / "expert.npy", mmap_mode="r")
                m["_dagger"] = True
            elif (dd / "expert_actions.npy").exists():
                # Sep-5 (audit): split_demos wrote expert_actions.npy pre-fix —
                # accept it as a fallback so old converted bundles stay DAgger.
                m["target"] = np.load(dd / "expert_actions.npy", mmap_mode="r")
                m["_dagger"] = True
            else:
                m["target"] = m["actions"]
                m["_dagger"] = False
            mms.append(m)
            sizes.append(len(m["tasks"]))
        else:
            if not args.allow_untrusted_bundle:
                sys.exit(f"[bconly] REFUSING to load .pt demo bundle {dd} with "
                         f"weights_only=False. This is an untrusted-pickle RCE "
                         f"surface. Prefer a .npy DIR (mmap-safe), or convert "
                         f"with split_demos.py. Re-run with --allow-untrusted-bundle "
                         f"to override.")
            d = torch.load(dd.as_posix(), map_location="cpu",
                           weights_only=False)
            m = {k: np.asarray(d[k]) for k in
                 ("imgs", "states", "actions", "tasks")}
            # ROUND-5 (H1): the .pt-bundle branch built m without 'target' —
            # get() KeyError'd on the FIRST sample. Non-DAgger bundle: target
            # == the executed actions (same as the npy-dir fallback).
            m["target"] = np.asarray(d["expert_actions"]) if "expert_actions" in d \
                else m["actions"]
            m["_dagger"] = "expert_actions" in d
            mms.append(m)
            sizes.append(len(m["tasks"]))
    total = sum(sizes)
    cum = np.cumsum([0] + sizes)
    def get(i):
        for mi in range(len(mms)):
            if i < cum[mi + 1]:
                r = {k: mms[mi][k][i - cum[mi]] for k in
                     ("imgs", "states", "actions", "tasks", "target")}
                r["is_dagger"] = mms[mi].get("_dagger", False)
                return r
    def sample(idxs):
        rows = [get(int(i)) for i in idxs]
        out = {k: np.stack([r[k] for r in rows])
               for k in ("imgs", "states", "actions", "tasks", "target")}
        out["any_dagger"] = any(r["is_dagger"] for r in rows)
        return out
    print(f"[bconly] demos: {total} transitions across {len(mms)} dirs "
          f"(mmap) | {time.time()-t0:.0f}s", flush=True)

    ck = torch.load(args.vla_ckpt, map_location="cpu", weights_only=True)
    # B-127: use_pretrained_siglip=False — saves ~1GB; this experiment tests
    # whether the direct task-proprioception token alone can drive navigation
    # B-128b: construct BORN-ON-GPU via the device context -- the host never
    # holds a full fp32 copy of the model (the CPU-construct path peaked at
    # construct 1.7G + ckpt 1.7G simultaneously -> cgroup oom-kill).
    cfg = ck["config"]
    del ck
    import gc; gc.collect()
    with torch.device("cuda"):
        vla = VLAPolicy(backbone_config=cfg, action_dim=4, chunk_len=16,
                        freeze_backbone=True, use_pretrained_siglip=False)
    add_lora(vla.backbone, r=16)
    ck2 = torch.load(args.vla_ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = vla.load_state_dict(ck2["policy"], strict=False)
    del ck2; gc.collect()
    real_missing = [k for k in missing
                    if not ("task_mlp" in k or "task_head" in k
                            or k.startswith("task_proj"))]
    assert not real_missing and not unexpected, \
        f"ckpt mismatch: {real_missing[:3]} / {unexpected[:3]}"
    opt = torch.optim.Adam([p for p in vla.parameters() if p.requires_grad],
                           lr=args.bc_lr)
    task_params = [p for n, p in vla.named_parameters()
                   if "task_mlp" in n or "task_head" in n]
    task_ids = {id(p) for p in task_params}
    opt_task = torch.optim.Adam(task_params, lr=args.bc_lr * 5)
    WARMUP = max(1, args.bc_updates // 4)

    rng = np.random.default_rng(7)
    instr = ["Navigate and pick up the red block."] * args.bc_batch
    loss_log = None
    for step in range(args.bc_updates):
        idx = rng.integers(0, total, size=args.bc_batch)
        bt = sample(idx)
        # B-141c: regress toward EXPERT labels on dagger rows (fall back to
        # executed actions elsewhere) -- true DAgger relabeling at last
        tgt = bt["target"] if bt.get("any_dagger") else bt["actions"]
        batch_img = torch.from_numpy(bt["imgs"]).to("cuda").float() / 255.0
        batch_st = torch.from_numpy(np.asarray(bt["states"], dtype=np.float32)).to("cuda")
        batch_a = torch.from_numpy(np.asarray(tgt, dtype=np.float32)).to("cuda")
        batch_t = torch.from_numpy(np.asarray(bt["tasks"], dtype=np.float32)).to("cuda")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            cond = vla._cond_features(batch_img, instr, batch_st)
        B = cond.shape[0]
        x0 = torch.zeros(B, vla.chunk_len, vla.action_dim,
                         device="cuda", dtype=torch.float32)
        tt = torch.zeros(B, device="cuda", dtype=torch.float32)
        v = vla.expert(x0, tt, cond.float(), task_vec=batch_t)
        if vla.expert.tanh_output:
            v = torch.tanh(v)
        a_det = v[:, 0].float()
        loss = (a_det - batch_a).pow(2).mean()
        if step < WARMUP:
            vla.zero_grad(set_to_none=True)  # ROUND-172 (N8): mirror gen_demos — zero ALL params, not just opt_task's
            opt_task.zero_grad(set_to_none=True)
            loss.backward()
            opt_task.step()
        else:
            if step == WARMUP:
                print("[bconly] phase 2: joint fine-tune", flush=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        loss_log = loss.item()
        if step % 500 == 0:
            print(f"[bconly] step {step}/{args.bc_updates} | loss "
                  f"{loss.item():.4f} | {time.time() - t0:.0f}s", flush=True)
    vla.eval()
    from storage import safe_save
    safe_save({"policy": vla.state_dict(), "config": cfg, "args": vars(args)},
              args.out, retries=2, delay=3)
    print(f"[bconly] done -> {args.out} (loss {loss_log:.4f}, "
          f"{time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()