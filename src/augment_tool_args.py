"""Round-93: tool-args augmentation — the wrong-args failure bucket (38%)
is the tool's biggest error. This generates value-shifted variants of the
compiled tool conversations: the arithmetic patterns (modulo / divisibility
/ product / sum / percentage / average) are fully RECOMPUTED so the
query-code-result-answer chain stays consistent; the extractive queries
(repo / path / topic tokens) get aligned token swaps across all segments.
The model then learns the VALUE-EXTRACTION, not the memorized values.

Usage: python3 src/augment_tool_args.py [--variants 2] [--in data/processed/tool_sft_compiled.jsonl]
"""
import argparse, json, random, re
from pathlib import Path

NUM = re.compile(r"-?\d+(?:\.\d+)?")

def recompute(u):
    """Return (code, result, answer_template) for known arithmetic queries."""
    m = re.search(r"Compute\s+(-?\d+)\s+modulo\s+(-?\d+)\.?", u)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"{a}%{b}", str(a % b), f"{a} modulo {b} is {a % b}."
    m = re.search(r"Is\s+(-?\d+)\s+divisible by\s+(-?\d+)\?", u)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return f"{a}%{b}==0", str(a % b == 0), f"{a} divided by {b}: {a % b == 0}."
    m = re.search(r"Compute the product of \[([^\]]+)\]\.?", u)
    if m:
        xs = [int(x) for x in NUM.findall(m.group(1))]
        p = 1
        for x in xs:
            p *= x
        code = "*".join(str(x) for x in xs)
        return code, str(p), f"The product of {m.group(1)} is {p}."
    m = re.search(r"Compute the sum of \[([^\]]+)\]\.?", u)
    if m:
        xs = [int(x) for x in NUM.findall(m.group(1))]
        s = sum(xs)
        return "+".join(str(x) for x in xs), str(s), f"The sum of {m.group(1)} is {s}."
    m = re.search(r"average of \[([^\]]+)\]", u)
    if m:
        xs = [int(x) for x in NUM.findall(m.group(1))]
        return f"({' + '.join(str(x) for x in xs)})/{len(xs)}", str(sum(xs) / len(xs)), \
            f"The average of {m.group(1)} is {sum(xs) / len(xs)}."
    m = re.search(r"What is\s+(-?\d+)\s*%\s*of\s+(-?\d+)\?", u)
    if m:
        p, b = int(m.group(1)), int(m.group(2))
        return f"{b}*0.01*{p}", str(b * p / 100), f"{p}% of {b} is {b * p / 100}."
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--in", dest="inp", default="data/processed/tool_sft_compiled.jsonl")
    ap.add_argument("--out", default="data/processed/tool_sft_args.jsonl")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    records = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    out = []
    for r in records:
        # B-201: a tool record has TWO "assistant" segments (the tool-call JSON,
        # then the final answer). A dict keyed by role collapses them to the LAST
        # one, so `code`/`ans` read the WRONG segment. Key by position instead.
        roles = [s[0] for s in r["segments"]]
        texts = [s[1] for s in r["segments"]]
        def _role_text(role, nth=1):
            idx = [i for i, rl in enumerate(roles) if rl == role]
            k = min(nth, len(idx)) - 1 if idx else -1
            return texts[idx[k]] if 0 <= k < len(texts) else ""
        u = _role_text("user", 1).removeprefix("User: ").strip()
        code = _role_text("assistant", 1)   # first assistant = the tool-call
        res = _role_text("tool", 1)
        ans = _role_text("assistant", 2)    # last assistant = the final answer
        rc = recompute(u)
        variants = []
        if rc:
            c0, r0, a0 = rc
            old_nums = re.findall(r"-?\d+(?:\.\d+)?", u)
            for _ in range(args.variants):
                u2 = NUM.sub(lambda m: str(int(m.group()) + rng.randint(-20, 20)),
                             u, count=len(old_nums))
                rc2 = recompute(u2)
                if not rc2:
                    continue
                c2, r2, a2 = rc2
                new_nums = re.findall(r"-?\d+(?:\.\d+)?", u2)
                if len(old_nums) != len(new_nums):
                    continue
                num_map = dict(zip(old_nums, new_nums))
                num_map[r0] = r2
                rec2 = json.loads(json.dumps(r))
                for seg in rec2["segments"]:
                    if seg[0] == "user":
                        seg[1] = f"User: {u2}\n"
                        continue
                    for t, s in num_map.items():
                        seg[1] = seg[1].replace(t, s)
                    if seg[0] == "tool":
                        seg[1] = f"result: {r2}\n"
                variants.append(rec2)
        else:
            # answer-style records: the arg tokens are echoed in the answer;
            # swap only the repo/path/doc tokens (safe — the knowledge
            # topic swaps would break the answer content)
            ans_txt = ans if ans else code
            if "use tool:" in code:
                code_txt = code
                tokens = {x for x in re.findall(r"[\w./-]{3,}", u)
                          if "/" in x or x in ("README", "CHANGELOG", "LICENSE")}
                args_blob = code_txt[code_txt.find("{") + 1: code_txt.rfind("}")]
            else:
                code_txt = ans_txt
                tokens = {x for x in re.findall(r"[\w./-]{3,}", u)
                          if "/" in x or x in ("README", "CHANGELOG", "LICENSE")}
                args_blob = ans_txt
            subs = {}
            for t in tokens:
                if t not in subs:
                    if t in ("README", "CHANGELOG", "LICENSE", "docs/readme"):
                        subs[t] = rng.choice(["README", "CHANGELOG", "LICENSE",
                                              "docs/readme"])
                    else:
                        subs[t] = rng.choice([
                            "example/robot-arm", "myorg/servo-gym",
                            "example/sensor-gym", "acme/rover",
                            "open-source-lab/vision", "crafty/actuator",
                            "fleet/tracker", "spark/controller",
                            "example/legacy", "green/embedded"])
            if subs:
                for _ in range(args.variants):
                    rec2 = json.loads(json.dumps(r))
                    for seg in rec2["segments"]:
                        for t, s in subs.items():
                            seg[1] = seg[1].replace(t, s)
                    variants.append(rec2)
        if not variants:
            out.append(r)
            continue
        out.extend(variants)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"[tool-args] {len(out)} augmented records ({len(records)} base + "
          f"{len(out) - len(records)} variants) -> {args.out}")


if __name__ == "__main__":
    main()