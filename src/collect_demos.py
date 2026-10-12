"""B-130: slim demo collector — environment + expert only, no model loading.
Demo collection never needed the VLA (the expert reads simulator state
directly); separating it from BC drops the collection footprint from ~4.8G
to <1G, so demos can be gathered in any resource window.

Usage: python3 src/collect_demos.py --n_eps 120 --min_spawn 1.2 --seed 61 \
           --out out/demos_v7.pt
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eps", type=int, default=120)
    ap.add_argument("--min_spawn", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=61)
    ap.add_argument("--out", default="out/demos_v7.pt")
    ap.add_argument("--goal_min", type=float, default=-1.0)
    ap.add_argument("--goal_max", type=float, default=1.0)
    args = ap.parse_args()

    import time
    import numpy as np
    import torch
    from sim_environment import RoboticsEnv
    from gen_demos import expert_action, _to_uint8

    t0 = time.time()
    env = RoboticsEnv(use_physics=True, goal_range=(args.goal_min, args.goal_max))
    env.min_spawn = args.min_spawn  # B-183: was parsed but never assigned
    rng = np.random.default_rng(args.seed)
    # B-163: STREAM transitions to fixed-record .raw files (B-137 pattern) --
    # page-cache backed appends keep the resident set flat regardless of
    # episode count; the old list-of-tuples + b"".join double-buffer oom-
    # killed the save twice (Aug-26, at MemoryMax=4000M AND 6000M).
    out_p = pathlib.Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "imgs":   open(out_p.with_suffix(".imgs.raw"), "wb"),
        "states": open(out_p.with_suffix(".states.raw"), "wb"),
        "actions": open(out_p.with_suffix(".actions.raw"), "wb"),
        "tasks":  open(out_p.with_suffix(".tasks.raw"), "wb"),
    }
    # ROUND-169: register atexit BEFORE the loop (the old placement was after —
    # a Python-level exception mid-loop still leaked the fds until OS cleanup).
    import atexit as _atexit
    _atexit.register(lambda: [f.close() for f in rec.values()])
    n_trans = 0
    n_reached = 0
    for ep in range(args.n_eps):
        obs, _ = env.reset(seed=args.seed * 1000 + ep)
        # B-65-style curriculum knob stays available via min_spawn
        reached = False
        for _ in range(500):
            a = expert_action(env, noise=0.05)
            task = env.task_vector()  # B-132 body-frame convention
            nxt, reward, term, trunc, _ = env.step(a)
            im_bytes = _to_uint8(obs.visual.get_tensor()).tobytes()
            state = obs.get_combined_state()[150528:].astype(np.float32)
            rec["imgs"].write(im_bytes)
            rec["states"].write(state.tobytes())
            rec["actions"].write(a.astype(np.float32).tobytes())
            rec["tasks"].write(task.astype(np.float32).tobytes())
            n_trans += 1
            obs = nxt
            if term or trunc:
                reached = reached or term
                break
        n_reached += reached
        if (ep + 1) % 20 == 0:
            import resource
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"[collect] ep {ep + 1}/{args.n_eps} | transitions "
                  f"{n_trans} | reached {n_reached}/{ep + 1} | "
                  f"rss {rss:.0f}MB | {time.time() - t0:.0f}s", flush=True)
        if ep + 1 == args.n_eps:
            break
    # close the .raw handles in the normal path; on a hard kill (PyBullet
    # hang -> SIGKILL) the OS closes the fds itself and the conversion below
    # size-validates every .raw and fails loud on a partial/interrupted stream
    # (it never silently salvages). Partial .raw files are left behind on a
    # crash; disk_hygiene / the next run's .raw regeneration handle them.
    # (atexit was registered before the loop — covers mid-loop exceptions too.)
    if n_trans == 0:
        raise SystemExit("[collect] ZERO transitions — refusing to build an "
                         "empty bundle")
    for f in rec.values():
        f.close()
    print(f"[collect] DONE: {n_trans} transitions, {n_reached}/{args.n_eps} "
          f"reached ({time.time() - t0:.0f}s)", flush=True)

    # B-175: save as individual .npy files, ONE array at a time.
    # No bundle dict, no torch.save serialization double-buffer — peak
    # RSS is a single array (~0.75GB for imgs), not the full bundle.
    # Also: each np.save writes directly to disk, no in-memory copy.
    n_rec = out_p.with_suffix(".states.raw").stat().st_size // (1872 * 4)
    specs = [("imgs",   np.uint8,  (n_rec, 224, 224, 3)),
             ("states", np.float32, (n_rec, 1872)),
             ("actions", np.float32, (n_rec, 4)),
             ("tasks",  np.float32, (n_rec, 5))]
    out_dir = out_p.parent / (out_p.stem + "_npy")
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, dt, shape in specs:
        rp = out_p.with_suffix(f".{k}.raw")
        expected = abs(int(np.prod(shape)))
        arr = np.fromfile(rp, dtype=dt)
        if arr.size != expected:
            raise RuntimeError(
                f"[collect] {k}.raw has {arr.size} elems, expected "
                f"{expected} ({n_rec} records) -- collection was interrupted")
        arr = arr.reshape(shape)
        np.save(str(out_dir / f"{k}.npy"), arr)
        del arr; rp.unlink()  # free array + raw before next
        import gc; gc.collect()
    print(f"[collect] saved {out_dir}/ ({n_rec} transitions)", flush=True)


if __name__ == "__main__":
    main()