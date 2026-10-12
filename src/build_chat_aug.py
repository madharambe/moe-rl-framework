"""Round-92: chat-data augmentation — systematic diversity from the 70
hand-authored conversations. The objective's chat skill is data-starved
(70 conversations vs the spatial's 2,000); this generates topic + phrasing +
follow-up variants of each base conversation so the model sees many surface
forms of the same underlying conversational patterns. The base set was
hand-authored for QUALITY; the variants preserve the answer content while
varying the question wording, the object, and the follow-up.

Usage: python3 src/build_chat_aug.py [--variants 10] [--out data/processed/chat_aug.jsonl]
"""
import argparse, json, random
from pathlib import Path
import importlib.util

BASE = Path(__file__).resolve().parent.parent
SYS = "You are Ambatron, a friendly, helpful assistant."

# topic substitutions: (match-substring, replacements) applied to the USER turns
TOPIC_VARIANTS = [
    ("blender", ["blender", "kettle", "microwave", "vacuum", "air fryer"]),
    ("phone", ["phone", "tablet", "laptop", "watch"]),
    ("wifi", ["wifi", "bluetooth", "ethernet", "internet"]),
    ("git", ["git", "svn", "a version-control tool"]),
    ("printer", ["printer", "scanner", "monitor", "router"]),
    ("robot", ["robot", "rover", "drone", "gripper"]),
    ("servo", ["servo", "motor", "actuator", "stepper"]),
    ("battery", ["battery", "power supply", "charger"]),
    ("database", ["database", "data store", "file format"]),
    ("code", ["code", "script", "project", "program"]),
]
# question-phrase variants for the FIRST user turn
OPENERS = [
    "Any ideas?", "What should I do?", "How do I fix it?", "What's going on?",
    "Any advice?", "Why?", "What do you think?", "Can you help me?",
    "I'm stuck, what now?", "Is there a fix?", "What are my options?",
    "Any suggestions?", "Where do I start?",
]
# follow-up variants for the SECOND user turn
FOLLOWUPS = [
    "I'll try that. Anything else?",
    "That makes sense. What about a long-term fix?",
    "Good to know. How long should it take?",
    "OK, what's the most important part?",
    "I see. Is that the usual cause?",
    "Thanks. Any common mistakes to avoid?",
    "Interesting. What if that doesn't work?",
    "Got it. Should I do this regularly?",
    "Makes sense. Is it expensive?",
    "Alright. What else could be wrong?",
]


def load_base():
    spec = importlib.util.spec_from_file_location(
        "build_chat_sft", str(BASE / "src" / "build_chat_sft.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load build_chat_sft.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONVOS, mod.SYS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10,
                    help="variants per base conversation")
    ap.add_argument("--out", type=str, default="data/processed/chat_aug.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    convos, sys_txt = load_base()
    rng = random.Random(args.seed)
    records = []
    for ci, (u1, a1, u2, a2) in enumerate(convos):
        records.append({
            "segments": [("system", sys_txt), ("user", f"User: {u1}\n"),
                         ("assistant", f"Assistant: {a1}"),
                         ("user", f"User: {u2}\n"),
                         ("assistant", f"Assistant: {a2}")],
            "domain": "chat"})
        for v in range(args.variants):
            u1v, u2v = u1, u2
            def _sub_first(txt, sub, repls):
                # B-150: detection was case-insensitive but str.replace was
                # case-sensitive, so capitalized variants silently skipped.
                i = txt.lower().find(sub)
                if i < 0:
                    return txt
                return txt[:i] + rng.choice(repls) + txt[i + len(sub):]
            for sub, repls in TOPIC_VARIANTS:
                u1v = _sub_first(u1v, sub, repls)
                u2v = _sub_first(u2v, sub, repls)
            # rephrase the opening question
            if "?" in u1v:
                q = u1v[:u1v.rfind("?") + 1]
                pre = u1v[:u1v.rfind("?")].rstrip(" ,.")
                u1v = pre + ", " + rng.choice(OPENERS)
            # vary the follow-up (a subset of variants)
            if v % 2 == 0:
                u2v = rng.choice(FOLLOWUPS)
            if u1v != u1 or u2v != u2:
                records.append({
                    "segments": [("system", sys_txt),
                                 ("user", f"User: {u1v}\n"),
                                 ("assistant", f"Assistant: {a1}"),
                                 ("user", f"User: {u2v}\n"),
                                 ("assistant", f"Assistant: {a2}")],
                    "domain": "chat"})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[chat-aug] {len(records)} records ({len(convos)} base + "
          f"{len(records) - len(convos)} variants) -> {out}")


if __name__ == "__main__":
    main()