"""Round-98b: honesty-set augmentation — the 12 hand-authored honest
conversations x topic-swaps -> hundreds. The measured failure: the v17d's
12 records drowned in the code-prior; the honesty signal needs the
surface-breadth. The topic-swaps operate on the topic-NOUNS in the
user turns and their echoes in the answers (the honesty patterns are the
transferable part — the boundary-acknowledgment, the constructive help,
the refusal-to-guess).

Usage: python3 src/augment_chat_honesty.py [--variants 20]
"""
import argparse, json, random, re
from pathlib import Path

# (topic-noun, replacements) — matched case-insensitively as whole words
TOPICS = [
    ("car", ["car", "scooter", "motorcycle", "boat", "van"]),
    ("engine", ["engine", "motor", "battery", "transmission"]),
    ("plants", ["plants", "lawn", "garden", "aquarium", "herbs"]),
    ("keyboard", ["keyboard", "mouse", "headset", "monitor", "laptop"]),
    ("sleep", ["sleep", "focus", "energy", "appetite"]),
    ("weather", ["weather", "forecast", "traffic", "tide"]),
    ("cake", ["cake", "bread", "pastry", "soup"]),
    ("dog", ["dog", "cat", "hamster", "bird"]),
    ("price", ["price", "rate", "deal", "quote"]),
    ("road trip", ["road trip", "weekend trip", "move", "commute"]),
    ("headphones", ["headphones", "speakers", "microphone", "camera",
                    "earbuds", "headset"]),
    ("wireless", ["wireless", "bluetooth", "wired", "usb-c"]),
    ("kitchen", ["kitchen", "bathroom", "garage", "office", "pantry"]),
    ("compost", ["compost", "recycling", "freezing", "meal planning"]),
    ("back", ["back", "neck", "shoulders", "wrists"]),
    ("melatonin", ["melatonin", "magnesium", "caffeine", "vitamins"]),
    ("inspection", ["inspection", "test drive", "service history", "warranty"]),
]
OPENERS = ["Any ideas?", "What should I do?", "What's going on?", "How do I fix it?",
           "Any advice?", "Is that a problem?", "What do you think?"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=20)
    ap.add_argument("--in", dest="inp", default="data/processed/chat_honesty.jsonl")
    ap.add_argument("--out", default="data/processed/chat_honesty_aug.jsonl")
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    convos = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    out = []
    for r in convos:
        segs = r["segments"]
        u1 = segs[1][1][6:].strip()
        out.append(r)
        used = set()
        for _ in range(args.variants):
            u1v = u1
            sub = {}
            for topic, repls in TOPICS:
                pat = re.compile(r"\b" + topic + r"\b", re.IGNORECASE)
                if pat.search(u1v):
                    chosen = rng.choice(repls)
                    if chosen.lower() == topic and topic not in ("car-2", "dog-2"):
                        continue
                    u1v = pat.sub(chosen, u1v, count=1)
                    sub[topic] = chosen
            # rephrase the opening
            q = u1v[:u1v.rfind("?") + 1] if "?" in u1v else u1v
            pre = u1v[:u1v.rfind("?")].rstrip(" ,.") if "?" in u1v else u1v.rstrip(" .")
            u1v = pre + ", " + rng.choice(OPENERS)
            # the second turn: the topic-echo or a follow-up variant
            u2v = segs[3][1][6:].strip()
            for topic, chosen in sub.items():
                u2v = re.sub(r"\b" + topic + r"\b", chosen, u2v,
                             count=1, flags=re.IGNORECASE)
            if u1v in used:
                continue
            used.add(u1v)
            rec2 = json.loads(json.dumps(r))
            rec2["segments"][1][1] = f"User: {u1v}\n"
            rec2["segments"][3][1] = f"User: {u2v}\n"
            out.append(rec2)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[honesty-aug] {len(out)} records ({len(convos)} base + "
          f"{len(out) - len(convos)} variants) -> {args.out}")


if __name__ == "__main__":
    main()