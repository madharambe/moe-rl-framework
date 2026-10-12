"""B-135: split a demo .pt bundle into mmap-able .npy files.
File-backed arrays load via np.load(mmap_mode='r') — pages stream from disk,
are evictable under pressure (no anonymous-RAM bombs), and training jobs'
host RSS drops to model+batch only.

Usage: python3 src/split_demos.py --in out/demos_v8.pt --dir out/demos_v8_npy
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--dir", dest="out", required=True)
    ap.add_argument("--allow-untrusted-bundle", action="store_true",
                    help="Explicitly permit loading a .pt demo BUNDLE with "
                         "weights_only=False (RCE surface on a tampered file).")
    args = ap.parse_args()
    import torch
    import numpy as np
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # ROUND-172 (N5): demo bundles are plain tensors (numpy arrays) — they load
    # under weights_only=True safely, so the refusal is NOT the default any
    # more. The explicit flag keeps the escape hatch for genuinely-exotic
    # bundles with a loud warning.
    if args.allow_untrusted_bundle:
        print("[split] WARN: --allow-untrusted-bundle set — loading with "
              "weights_only=False (RCE surface on a tampered file)")
        d = torch.load(args.inp, map_location="cpu", weights_only=False)
    else:
        try:
            d = torch.load(args.inp, map_location="cpu", weights_only=True)
        except Exception as e:
            sys.exit(f"[split] bundle {args.inp} not loadable with "
                     f"weights_only=True ({type(e).__name__}: {e}). If this is a "
                     f"trusted legacy bundle, re-run with --allow-untrusted-bundle.")
    for k in ("imgs", "states", "actions", "tasks"):
        if k not in d:
            continue
        arr = np.asarray(d[k])
        np.save(out / f"{k}.npy", arr)
        print(f"[split] {k}: {arr.shape} {arr.dtype}", flush=True)
        del d[k]
    if "expert_actions" in d:
        arr = np.asarray(d["expert_actions"])
        # Sep-5 (audit): bc_only checks expert.npy — writing expert_actions.npy
        # made converted DAgger bundles silently NON-DAgger (self-generated
        # labels regression). Write the name bc_only reads.
        np.save(out / "expert.npy", arr)
        print(f"[split] expert.npy: {arr.shape}", flush=True)
    print(f"[split] done -> {out}", flush=True)


if __name__ == "__main__":
    main()