"""Round-77: 3D spatial-reasoning SFT data generator (Objective 4c — "good
understanding of 3D space" for the LLM itself). Synthetic, computed-correct
(never guessed) spatial QAs: relative positions, cardinal moves + turns,
above/below with explicit heights, mental 90-deg rotation. Deterministic RNG
(SEED) so the set is reproducible; write to data/processed/spatial_sft.jsonl
and train with train_tool_sft.py --sft_data ... (direct-answer records).

Usage:
  python3 src/build_spatial_sft.py [--n 3000] [--out data/processed/spatial_sft.jsonl]
"""
import argparse, json, random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SEED = 7

OBJECTS = ["box", "cube", "ball", "cone", "cylinder", "vase", "bottle",
           "lamp", "chair", "book", "key", "cup", "mug", "plant", "plate"]
ROWS = ["north", "south", "east", "west", "north-east", "north-west",
        "south-east", "south-west"]
TURNS = {"north": "east", "east": "south", "south": "west", "west": "north",
         "north-east": "south-east", "south-east": "south-west",
         "south-west": "north-west", "north-west": "north-east"}
# 90-degree turns only move between the 4 cardinal directions
CYCLE = ["north", "east", "south", "west"]
CYCLE8 = ["north", "north-east", "east", "south-east", "south", "south-west",
          "west", "north-west"]


def turn(dirn, direction):
    if dirn in CYCLE:
        i = CYCLE.index(dirn)
        return CYCLE[(i + 1) % 4] if direction == "right" else CYCLE[(i - 1) % 4]
    i = CYCLE8.index(dirn)
    return CYCLE8[(i + 1) % 8] if direction == "right" else CYCLE8[(i - 1) % 8]


def rel_dir(a, b):
    """a is (x,y), b is (x,y) -> compass direction from a to b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    if dy < 0 and dx == 0:
        return "north"
    if dy > 0 and dx == 0:
        return "south"
    if dx > 0 and dy == 0:
        return "east"
    if dx < 0 and dy == 0:
        return "west"
    if dx > 0 and dy < 0:
        return "north-east"
    if dx < 0 and dy < 0:
        return "north-west"
    if dx > 0 and dy > 0:
        return "south-east"
    return "south-west"


def gen(rng, i):
    kind = i % 4
    if kind == 0:  # relative position on a 2D grid (5x5, unique cells)
        cells = rng.sample([(x, y) for x in range(5) for y in range(5)], 2)
        a, b = cells
        o1, o2 = rng.sample(OBJECTS, 2)
        q = (f"A {o1} and a {o2} are on a grid. The {o1} is at row {a[1]}, "
             f"column {a[0]}; the {o2} is at row {b[1]}, column {b[0]}. "
             f"Which direction is the {o2} from the {o1}?")
        return q, rel_dir(a, b)
    if kind == 1:  # face dirn, turn left/right, re-face
        d = rng.choice(["north", "east", "south", "west"])
        t = rng.choice(["left", "right"])
        q = (f"You are facing {d}. If you turn {t} 90 degrees, which "
             f"direction are you now facing?")
        return q, turn(d, t)
    if kind == 2:  # cardinal move on a grid: start -> moves -> end relative
        start = rng.choice([(x, y) for x in range(5) for y in range(5)])
        moves = []
        pos = list(start)
        for _ in range(rng.randint(1, 3)):
            m = rng.choice([(0, -1), (0, 1), (1, 0), (-1, 0)])
            pos = [max(0, min(4, pos[0] + m[0])), max(0, min(4, pos[1] + m[1]))]
            moves.append({(0, -1): "one step north", (0, 1): "one step south",
                          (1, 0): "one step east", (-1, 0): "one step west"}[m])
        q = (f"You start at row {start[1]}, column {start[0]}. You move "
             + ", then ".join(moves) + ". What is your final position?")
        return q, f"row {pos[1]}, column {pos[0]}"
    # kind == 3: above/below with heights
    o = rng.choice(OBJECTS)
    h1, h2 = rng.sample(range(1, 6), 2)
    q = (f"A {o} is {h1} meters above the floor. It is moved down "
         f"{h1 - h2} meters. How far above the floor is it now?")
    return q, f"{h2} meters"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", type=str, default="data/processed/spatial_sft.jsonl")
    args = ap.parse_args()
    rng = random.Random(SEED)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for i in range(args.n):
            q, a = gen(rng, i)
            rec = {
                "segments": [
                    ("system", "You are a helpful assistant with good spatial reasoning."),
                    ("user", f"User: {q}\n"),
                    ("assistant", f"Assistant: {a}"),
                ],
                "domain": "spatial",
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[spatial] {args.n} spatial-QA records -> {out}")


if __name__ == "__main__":
    main()
