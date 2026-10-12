"""B-133: DAgger round — aggregate expert corrections on POLICY-rollout
states. Classic BC compounds errors: the policy visits states the expert
demos never cover, where its (imperfect) mapping steers further off.
DAgger fixes exactly this: roll the CURRENT policy, label every visited
state with the expert's action, add to the dataset, retrain.

Usage: python3 src/dagger_round.py --ckpt out/vla_rl/vla_bc_v9.pt \
           --n_eps 60 --out out/demos_dagger1.pt [--append out/demos_v8.pt]
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_eps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--noise", type=float, default=0.15,
                    help="rollout exploration noise on top of the policy")
    ap.add_argument("--max_trans", type=int, default=8000,
                    help="B-134: HARD cap on accumulated transitions — the "
                         "first DAgger attempt grew an unbounded list (worst "
                         "case ~4.7GB) during the Aug-23 hard freeze window")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import time
    import numpy as np
    import torch
    import torch.serialization as ser
    from moe_model import MoEConfig
    ser.add_safe_globals([MoEConfig])
    from vla_model import VLAPolicy
    from lora_adapter import add_lora
    from sim_environment import RoboticsEnv
    # B-160: reuse the SCALED uint8 converter -- the old inline
    # `img.astype(np.uint8)` on [0,1] floats truncated every pixel to {0,1}:
    # all dagger-collected images stored ~BLACK (vision channel dead in
    # v10/v11 training; their gains came from the task-vector skip alone).
    from train_vla_rl import _to_uint8

    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    with torch.device("cuda"):
        vla = VLAPolicy(backbone_config=ck["config"], action_dim=4,
                        chunk_len=16, freeze_backbone=True,
                        use_pretrained_siglip=False)
    add_lora(vla.backbone, r=16)
    missing, unexpected = vla.load_state_dict(ck["policy"], strict=False)
    real_missing = [k for k in missing if "task" not in k]
    assert not real_missing and not unexpected, \
        f"mismatch {real_missing[:3]} / {unexpected[:3]}"
    del ck
    import gc; gc.collect()
    vla.eval().to("cuda")

    env = RoboticsEnv(use_physics=True)
    # B-137: STREAM transitions to disk instead of accumulating in host RAM.
    # Fixed-record appends are page-cache backed (evictable under pressure),
    # so this job's resident set stays ~model-only regardless of episode
    # count -- the Aug-23 near-crash cause is structurally removed.
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "imgs":   open(out_dir / "imgs.raw", "wb"),
        "states": open(out_dir / "states.raw", "wb"),
        "actions": open(out_dir / "actions.raw", "wb"),
        "tasks":  open(out_dir / "tasks.raw", "wb"),
        "expert": open(out_dir / "expert.raw", "wb"),
    }
    n_trans = 0
    INSTR = "Navigate and pick up the red block."
    rng = np.random.default_rng(args.seed)
    n_reached = 0
    # D11 (18th-audit): close raw file handles on ANY exit (exception/disk-drop
    # mid-episode previously leaked them and skipped the .npy conversion).
    import atexit as _atexit
    _atexit.register(lambda: [f.close() for f in rec.values()])
    for ep in range(args.n_eps):
        obs, _ = env.reset(seed=args.seed * 1000 + ep)
        reached = False
        for _ in range(500):
            tv = env.task_vector()
            img = obs.visual.get_tensor()
            st = obs.get_combined_state()[150528:].astype(np.float32)
            img_t = (torch.from_numpy(np.clip(img, 0, 255)).float().cuda().unsqueeze(0) / 255.0
                     if img.dtype == np.uint8 or img.max() > 1.5
                     else torch.from_numpy(np.clip(img, 0, 1)).float().cuda().unsqueeze(0))
            st_t = torch.from_numpy(st).cuda().unsqueeze(0)
            tv_t = torch.from_numpy(tv).cuda().unsqueeze(0)
            with torch.no_grad():
                pol_a = vla.deterministic_action(img_t, [INSTR], st_t,
                                                 task=tv_t)[0].cpu().numpy()
            act = np.clip(pol_a + rng.standard_normal(4).astype(np.float32)
                          * args.noise, -1, 1)
            # DAgger core: EXPERT labels THIS (policy-visited) state
            exp_a = __import__("gen_demos").expert_action(env, noise=0.0)
            nxt, reward, term, trunc, _ = env.step(act)
            if n_trans < args.max_trans:
                img_u8 = _to_uint8(img)
                rec["imgs"].write(img_u8.tobytes())
                rec["states"].write(st.astype(np.float32).tobytes())
                rec["actions"].write(act.astype(np.float32).tobytes())
                rec["tasks"].write(tv.astype(np.float32).tobytes())
                rec["expert"].write(exp_a.astype(np.float32).tobytes())
                n_trans += 1
            obs = nxt
            if term or trunc:
                reached = reached or term
                break
        n_reached += reached
        if n_trans >= args.max_trans:
            print(f"[dagger] transition cap {args.max_trans} reached at "
                  f"ep {ep + 1}", flush=True)
            break
        if (ep + 1) % 20 == 0:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[dagger] ep {ep + 1}/{args.n_eps} | transitions "
                  f"{n_trans} | reached {n_reached}/{ep + 1} | "
                  f"rss {rss:.0f}MB | {time.time() - t0:.0f}s", flush=True)
    for f in rec.values():
        f.close()
    print(f"[dagger] DONE: {n_trans} corrected transitions, "
          f"{n_reached}/{args.n_eps} policy-reached ({time.time() - t0:.0f}s)",
          flush=True)
    import numpy as np
    dt_map = {"imgs": np.uint8, "states": np.float32, "actions": np.float32,
              "tasks": np.float32, "expert": np.float32}
    shapes = {"states": 1872, "actions": 4, "tasks": 5, "expert": 4}
    # Sep-5 (audit): validate record counts BEFORE converting — collect_demos
    # does this; a truncated/interrupted .raw previously converted to a
    # misaligned bundle with no error (mismatched record counts across keys).
    _rec_bytes = {"states": 1872 * 4, "actions": 4 * 4, "tasks": 5 * 4,
                  "expert": 4 * 4}
    _counts = {}
    for k in _rec_bytes:
        sz = (out_dir / f"{k}.raw").stat().st_size
        if sz % _rec_bytes[k] != 0:
            raise SystemExit(f"[dagger] {k}.raw size {sz} NOT a multiple of "
                             f"record size {_rec_bytes[k]} — interrupted stream")
        _counts[k] = sz // _rec_bytes[k]
    _img_count = 0
    if (out_dir / "imgs.raw").exists():
        _img_count = (out_dir / "imgs.raw").stat().st_size // (224 * 224 * 3)
    _all = set(_counts.values()) | ({_img_count} if (out_dir / "imgs.raw").exists() else set())
    if len(_all) > 1:
        raise SystemExit(f"[dagger] record-count mismatch across .raw files "
                         f"({sorted(_all)}) — refusing to build a misaligned bundle")
    for k, dt in dt_map.items():
        arr = np.fromfile(out_dir / f"{k}.raw", dtype=dt)
        if k == "imgs":
            arr = arr.reshape(-1, 224, 224, 3)
        else:
            arr = arr.reshape(-1, shapes[k])
        np.save(out_dir / f"{k}.npy", arr)
        (out_dir / f"{k}.raw").unlink()
    print(f"[dagger] saved mmap-ready npy bundle -> {out_dir} "
          f"({n_trans} transitions)", flush=True)


if __name__ == "__main__":
    main()