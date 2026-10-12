"""HellaSwag external capability signal for the MoE model.

Zero-dependency (no pip installs — golden rule): stdlib urllib for the data,
torch + tokenizers from .venv. Downloads the balanced validation split and
scores up to --samples of "gen"-activity rows by next-token loss over each
ending (predict the ending with the LOWER loss). This is the honest EXTERNAL
anchor the /15 honest-eval deliberately isn't: a 430M seeing a few hundred
million tokens lands ~25-35%; the number only moves with real pretraining
(agents.md objective #1: curious + self-verifying; this is the measurable
trend line for capability objectivism).

Usage:
  .venv/bin/python3 src/hellaswag_eval.py --ckpt out/ambatron_430m_v2/ckpt_best.pt --bf16 --samples 1000
"""
import argparse, json, sys, urllib.request
from pathlib import Path
sys.path.insert(0, __file__.rsplit("/", 1)[0])

URL = ("https://raw.githubusercontent.com/rowanz/hellaswag/"
       "master/data/hellaswag_val.jsonl")
CACHE = Path("/tmp/hellaswag_val.jsonl")  # /tmp on the main SSD, not the ext drive


def _download():
    if not CACHE.exists():
        print(f"[hellaswag] downloading {URL}")
        urllib.request.urlretrieve(URL, CACHE)
        print("[hellaswag] cached to /tmp/hellaswag_val.jsonl")
    with open(CACHE) as f:
        rows = [json.loads(l) for l in f]
    # the val split ships pre-balanced; every row has 4 endings + a label idx
    return [r for r in rows if len(r.get("endings", [])) >= 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--device", default="cuda",
                    help="'cuda' (fast) or 'cpu' (to measure mid-training, when "
                         "the GPU is fully held by the base-trainer)")
    args = ap.parse_args()

    import torch
    import torch.serialization as ser
    from moe_model import MoEGPT, MoEConfig
    from tokenizers import Tokenizer
    ser.add_safe_globals([MoEConfig])
    tok = Tokenizer.from_file(str(Path(__file__).resolve().parent.parent
                                       / "data/processed/bpe_tokenizer/tokenizer.json"))
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    m = MoEGPT(ck["config"]).eval()
    m.load_state_dict(ck["model_state"], strict=False)
    if args.bf16 and args.device == "cuda":
        m.to(torch.bfloat16)
    m.to(args.device)

    rows = _download()
    rows = rows[:args.samples]
    correct = 0
    import re as _re
    with torch.no_grad():
        for r in rows:
            endings = r["endings"]
            best_i, best_loss = 0, float("inf")
            for i, end in enumerate(endings):
                # B-151d (audit): canonical lm-eval style — normalize the ending
                # loss by ITS token count (acc_norm, what HF cards report) and
                # apply the bracket/double-space preprocessing that HellaSwag's
                # own loader uses; whole-seq unnormalized loss isn't comparable.
                ctx = _re.sub(r"\s+", " ", r["ctx"]).strip()
                # HIGH (Sep-5 audit): canonical HellaSwag joins ctx+end with a
                # SPACE ("roof. he" + "is using" -> "roof. he is"); the old
                # `ctx + end` merged tokens ("heis") and the +1 end_len offset
                # misaligned the loss slice — every reported score was invalid.
                text = ctx + " " + end
                ids = tok.encode(text).ids
                t = torch.tensor([ids], dtype=torch.long, device=args.device)
                logits, _, _ = m(t[:, :-1])
                # conditioned loss over the ENDING positions only: boundary at
                # len(encode(ctx + " ")) — exact, no +1 fudge.
                bound = len(tok.encode(ctx + " ").ids)
                end_len = len(t) - bound
                seg = logits[0, bound - 1:].reshape(-1, logits.size(-1))
                tgt = t[0, bound:].reshape(-1)
                # F5 (Sep-5): cross_entropy MEAN over end_len rows + extra
                # /end_len double-divided (biased AGAINST longer endings).
                # acc_norm = sum / (end_len-1): use reduction="sum" then divide
                # exactly once by the ACTUAL token count (len(tgt), not end_len-
                # 1 which can be 0/negative for degenerate boundaries).
                loss = torch.nn.functional.cross_entropy(
                    seg.float(), tgt, ignore_index=0, reduction="sum") / max(tgt.numel(), 1)
                if float(loss) < best_loss:
                    best_loss, best_i = float(loss), i
            correct += (best_i == int(r["label"]))
    print(f"[hellaswag] {args.ckpt} {args.samples} samples | "
          f"accuracy {correct}/{len(rows)} = {correct/len(rows)*100:.1f}%")
    return correct / len(rows)


if __name__ == "__main__":
    main()