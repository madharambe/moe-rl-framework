"""B-145: arithmetic-family SFT generator — the Round-118 flywheel target.
Live usage exposed that v13 never learned squares/cubes/times/plus/minus
("What is 14 squared?" -> garbage). This generates compiled-schema records
(identical format to tool_sft_compiled.jsonl) covering the newly-routed
arithmetic patterns with computed answers (no label guessing -- every
result is calculated, B-30 style).

Usage: python3 src/gen_arith_sft.py [--out data/processed/arith_sft.jsonl]
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/arith_sft.jsonl")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # B-146: use the FROZEN system prompt from v13's own training bundle.
    # The runtime registry now emits execute_python(code, timeout_s) and
    # lacks the 4 github tools -- records built from it were
    # out-of-distribution versus what v13 adapted to, so the standalone
    # arith adapter could not anchor the tool-call format at all.
    import json as _json
    with open("data/processed/tool_sft_compiled.jsonl") as _f:
        system = _json.loads(_f.readline())["segments"][0][1]

    def rec(q, code, result, answer):
        return {"segments": [
            ["system", system],
            ["user", f"User: {q}\n"],
            ["assistant", "Assistant: use tool:" + json.dumps(
                {"name": "execute_python", "arguments": {"code": code}})],
            ["tool", f"result: {result}\n"],
            ["assistant", f"Assistant: {answer}"],
        ], "domain": "math"}

    def verbs():
        return rng.choice(["Compute", "Calculate", "What is", ""])

    records = []
    # squares 2..40 (three phrasings each)
    for n in range(2, 100):
        v = rng.choice(["What is", "Compute", ""])
        q = f"{v + ' ' if v else ''}{n} squared?"
        records.append(rec(q, f"{n}**2", str(n * n), f"{n} squared is {n * n}."))
    # cubes 2..16
    for n in range(2, 31):
        q = f"What is {n} cubed?"
        records.append(rec(q, f"{n}**3", str(n ** 3), f"{n} cubed is {n ** 3}."))
    # times: a in 2..30, b in 2..20
    for _ in range(3000):
        a, b = rng.randint(2, 99), rng.randint(2, 60)
        v = rng.choice(["What is", "Compute", "How much is"])
        q = f"{v + ' ' if v else ''}{a} times {b}?"
        records.append(rec(q, f"{a}*{b}", str(a * b), f"{a} times {b} is {a * b}."))
    # plus / minus
    for _ in range(2400):
        a, b = rng.randint(10, 900), rng.randint(10, 800)
        if rng.random() < 0.5:
            q = f"What is {a} plus {b}?"
            records.append(rec(q, f"{a}+{b}", str(a + b), f"{a} plus {b} is {a + b}."))
        else:
            hi, lo = max(a, b), min(a, b)
            q = f"What is {hi} minus {lo}?"
            records.append(rec(q, f"{hi}-{lo}", str(hi - lo),
                               f"{hi} minus {lo} is {hi - lo}."))
    # divided by (integer results only)
    for _ in range(1800):
        b = rng.randint(2, 24)
        c = rng.randint(2, 90)
        a = b * c
        q = f"What is {a} divided by {b}?"
        records.append(rec(q, f"{a}//{b}", str(a // b),
                           f"{a} divided by {b} is {a // b}."))
    # percent of -- B-145: keep code/result types consistent (int when exact,
    # else one-decimal float; the old version stored ints for float-executing
    # code, teaching the model wrong result formats)
    for _ in range(1600):
        p = rng.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90])
        y = rng.randint(20, 990)
        val = y * p / 100
        if float(val).is_integer():
            code = f"{y}*{p}//100"   # // so the tool's own output is an int
            res_s = str(int(val))
            ans = f"{p}% of {y} is {int(val)}."
        else:
            code = f"round({y}*{p}/100, 2)"
            res_s = str(round(val, 2))
            ans = f"{p}% of {y} is about {res_s}."
        records.append(rec(q := f"What is {p}% of {y}?", code, res_s, ans))
    rng.shuffle(records)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"[arith-sft] {len(records)} records -> {args.out}")


if __name__ == "__main__":
    main()