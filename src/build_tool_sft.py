"""
Generate tool-use SFT data for Ambatron (function calling must be TAUGHT —
a pretrained model cannot reliably emit tool-call syntax; see LLM Stack
tool-use chapter, 2026).

Produces multi-turn conversations in the ToolUseMoE prompt format:
    <tools section>
    User: <query>
    Assistant: <tool call OR direct answer>
    result: <tool result>
    Assistant: <final answer>

Each record: {"segments": [(role, text), ...]} where role in
{"system", "user", "assistant", "tool"}. The trainer masks the loss to
assistant segments only.

Includes "no tool" examples (research: teaching WHEN NOT to call tools is
half of reliable function calling).
"""
import json, random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "processed" / "tool_sft.jsonl"

SEED = 0
N_PER_TOOL = 120      # tool-call examples per tool
N_DIRECT = 60        # direct-answer examples (no tool)

_TOOL_TEMPLATES = {
    "execute_python": [
        # All results EXACTLY match the B-30 harness output (exec/eval fallback):
        # expression-style code -> repr(value). Verified against tool_use._execute_python.
        ("Compute the first 10 Fibonacci numbers and report the last one.",
         {"code": "__import__('functools').reduce(lambda a,b:a+[a[-1]+a[-2]], range(8), [0,1])[-1]"},
         "34", "The 10th Fibonacci number is 34."),
        ("What is the sum of squares of 1..10?",
         {"code": "sum(i*i for i in range(1,11))"},
         "385", "The sum of squares from 1 to 10 is 385."),
        ("Check whether 1729 is divisible by 7.",
         {"code": "1729 % 7"},
         "0", "1729 % 7 == 0, so 1729 is divisible by 7."),
        ("Sort [3,1,4,1,5] and report the second element.",
         {"code": "sorted([3,1,4,1,5])[1]"},
         "1", "Sorted: [1,1,3,4,5]; the second element is 1."),
        ("What is 2 + 2?",
         {"code": "2+2"},
         "4", "2 + 2 is 4."),
        ("Compute 12 multiplied by 12.",
         {"code": "12*12"},
         "144", "12 x 12 = 144."),
        ("How many characters are in the string 'python'?",
         {"code": "len('python')"},
         "6", "The string 'python' has 6 characters."),
        ("Find the largest value in [7, 2, 9, 4].",
         {"code": "max([7,2,9,4])"},
         "9", "The maximum of [7, 2, 9, 4] is 9."),
        ("Compute 2 to the power of 10.",
         {"code": "2**10"},
         "1024", "2 to the power of 10 is 1024."),
        ("How many elements are in [1, 2, 3]?",
         {"code": "len([1,2,3])"},
         "3", "The list has 3 elements."),
        ("Is 1001 divisible by 7?",
         {"code": "1001 % 7"},
         "0", "1001 % 7 == 0, so 1001 is divisible by 7."),
        ("Compute the average of 4, 6, and 8.",
         {"code": "(4+6+8)/3"},
         "6.0", "The average of 4, 6 and 8 is 6.0."),
        ("Sum the first five positive integers.",
         {"code": "1+2+3+4+5"},
         "15", "The sum is 15."),
        ("What is 7 times 7?",
         {"code": "7*7"},
         "49", "7 times 7 is 49."),
        ("Report the odd numbers in [1, 1, 2, 3, 5].",
         {"code": "[f for f in [1,1,2,3,5] if f%2==1]"},
         "[1, 1, 3, 5]", "The odd numbers are 1, 1, 3 and 5."),
        ("Compute 3 to the power of 4.",
         {"code": "3**4"}, "81", "3 to the power of 4 is 81."),
        ("What is 15 modulo 6?",
         {"code": "15 % 6"}, "3", "15 modulo 6 is 3."),
        ("Find the minimum of [9, 3, 7, 1].",
         {"code": "min([9,3,7,1])"}, "1", "The minimum is 1."),
        ("Sum the list [2, 4, 6].",
         {"code": "sum([2,4,6])"}, "12", "The sum is 12."),
        ("What is the absolute value of -17?",
         {"code": "abs(-17)"}, "17", "The absolute value is 17."),
        ("Compute 6 divided by 4.",
         {"code": "6/4"}, "1.5", "6 divided by 4 is 1.5."),
        ("Count how many 'a' characters are in 'banana'.",
         {"code": "'banana'.count('a')"}, "3", "banana has 3 a's."),
        ("What is 2 to the power of 0?",
         {"code": "2**0"}, "1", "Anything to the power of 0 is 1."),
        ("Round 3.14159 to 2 decimal places.",
         {"code": "round(3.14159, 2)"}, "3.14", "3.14159 rounds to 3.14."),
        ("Check if 15 is even.",
         {"code": "15 % 2"}, "1", "15 % 2 = 1, so 15 is odd."),
        ("Compute the product of [3, 4, 5].",
         {"code": "3*4*5"}, "60", "3 times 4 times 5 is 60."),
        ("What is 7 factorial?",
         {"code": "5040"}, "5040", "7! = 5040."),
        ("Find the second-largest value in [4, 1, 9, 6].",
         {"code": "sorted([4,1,9,6])[-2]"}, "6", "The second-largest is 6."),
        ("Compute 10 modulo 3.",
         {"code": "10 % 3"}, "1", "10 modulo 3 is 1."),
        ("What is the length of [10, 20, 30, 40, 50]?",
         {"code": "len([10,20,30,40,50])"}, "5", "The list has 5 elements."),
        ("Compute 9 times 9.",
         {"code": "9*9"}, "81", "9 times 9 is 81."),
        ("Is 81 a perfect square (sqrt is integer)?",
         {"code": "81**0.5"}, "9.0", "The square root of 81 is 9.0."),
        ("What is the result of 1 + 2 + 3 + 4?",
         {"code": "1+2+3+4"}, "10", "The sum is 10."),
        ("Compute the remainder of 22 divided by 7.",
         {"code": "22 % 7"}, "1", "22 divided by 7 leaves remainder 1."),
        ("What is the maximum of [2, 8, 5, 11, 3]?",
         {"code": "max([2,8,5,11,3])"}, "11", "The maximum is 11."),
        ("Sum the squares of [1, 2, 3].",
         {"code": "sum(x*x for x in [1,2,3])"}, "14", "1+4+9 = 14."),
        ("Compute 0.5 plus 0.25.",
         {"code": "0.5+0.25"}, "0.75", "0.5 + 0.25 = 0.75."),
        ("What is 100 divided by 8 (floor)?",
         {"code": "100//8"}, "12", "100 floor-divided by 8 is 12."),
        ("Check whether 'python' starts with 'py'.",
         {"code": "'python'.startswith('py')"}, "True", "Yes, python starts with py."),
        ("How many vowels are in 'hello'?",
         {"code": "sum(c in 'aeiou' for c in 'hello')"}, "2", "hello has 2 vowels."),
        ("What is -5 squared?",
         {"code": "(-5)**2"}, "25", "(-5)^2 = 25."),
        ("Compute 2 to the power of 8.",
         {"code": "2**8"}, "256", "2^8 = 256."),
        ("What is the average of [10, 20, 30]?",
         {"code": "sum([10,20,30])/3"}, "20.0", "The average is 20.0."),
        ("Find the smallest value in [6, 2, 9].",
         {"code": "min([6,2,9])"}, "2", "The smallest is 2."),
    ],
    "search_knowledge": [
        ("Explain PPO in one sentence.",
         {"query": "ppo"},
         "Proximal Policy Optimization: on-policy RL with clipped surrogate objective.",
         "PPO is an on-policy RL algorithm with a clipped surrogate objective."),
        ("What is SAC?",
         {"query": "sac"},
         "Soft Actor-Critic: off-policy RL with maximum entropy.",
         "SAC is an off-policy maximum-entropy RL algorithm."),
        ("What is a mixture of experts?",
         {"query": "moe"},
         "Mixture of Experts: sparse activation with multiple expert networks.",
         "MoE is sparse activation with multiple expert networks."),
        ("What does GQA mean in transformer models?",
         {"query": "gqa"},
         "Grouped Query Attention: fewer KV heads than query heads.",
         "GQA shares key/value heads across query groups to reduce KV cache."),
    ],
    "clone_repo_to_corpus": [
        ("Add the fastapi repository to the training corpus.",
         {"repo": "fastapi/fastapi"},
         "{'cloned': 'fastapi/fastapi', 'py_files': 412, 'unique_texts': 398, 'added_seqs': 245, 'corpus_seqs': 171221}",
         "Cloned fastapi/fastapi and appended 245 new sequences to the corpus."),
        ("Grow the corpus with the numpy repo.",
         {"repo": "numpy/numpy"},
         "{'cloned': 'numpy/numpy', 'py_files': 2914, 'unique_texts': 2801, 'added_seqs': 1204, 'corpus_seqs': 172425}",
         "Added numpy/numpy — 1,204 new sequences absorbed into the training corpus."),
    ],
    "corpus_stats": [
        ("How big is the training corpus right now?",
         {},
         "{'seqs': 170976, 'tokens': 154752165, 'dtype': 'torch.int16', 'py_files_in_raw': 43509}",
         "The corpus has 170,976 sequences and about 154.7 million tokens."),
        ("Report the corpus size.",
         {},
         "{'seqs': 170976, 'tokens': 154752165, 'dtype': 'torch.int16', 'py_files_in_raw': 43509}",
         "Currently 170,976 sequences / ~154.7M tokens."),
    ],
    "search_repos": [
        ("Find a gymnasium robot environment on GitHub.",
         {"query": "gymnasium robot"},
         "[{'full_name': 'example/robot-gym', 'description': 'Gymnasium robot envs', 'stars': 42, 'language': 'Python'}]",
         "Found example/robot-gym, a Python gymnasium robot environment."),
    ],
    "fetch_raw_file": [
        ("Fetch the README of example/robot-gym.",
         {"repo": "example/robot-gym", "path": "README.md"},
         "# robot-gym\\nGymnasium environments for robots.",
         "Fetched the README of example/robot-gym."),
    ],
}

_DIRECT_EXAMPLES = [
    ("Explain the difference between L1 and L2 regularization.",
     "L1 adds the absolute weights (sparse, feature selection); L2 adds the squared weights (smooth shrinkage)."),
    ("What does the tanh activation do?",
     "tanh squashes inputs to (-1, 1), centering the output and avoiding unbounded growth."),
    ("Why use grouped query attention?",
     "GQA shares key/value heads across query groups, cutting KV-cache memory with little quality loss."),
    ("What is gradient clipping?",
     "Gradient clipping caps the gradient norm before the optimizer step to prevent divergence."),
    ("Give a one-line summary of the RSI project.",
     "RSI trains a custom MoE LLM and a vision-language-action policy for robotics on a single laptop GPU."),
    ("Explain the difference between supervised and unsupervised learning.",
     "Supervised learning maps labeled inputs to outputs; unsupervised learning finds structure in unlabeled data."),
    ("What is overfitting?",
     "Overfitting is when a model memorizes training noise and generalizes poorly to new data."),
    ("What does backpropagation do?",
     "Backpropagation computes parameter gradients by the chain rule, propagating error from the loss back through the network."),
    ("Name two common activation functions.",
     "ReLU and tanh are common activations; ReLU zeroes negatives, tanh maps to (-1, 1)."),
    ("What is a learning rate?",
     "The learning rate scales the parameter updates each optimizer step; too high diverges, too low is slow."),
    ("What is a loss function?",
     "A loss function measures how wrong the model's prediction is and is the training objective."),
    ("Why do we normalize inputs?",
     "Normalization keeps feature scales similar, which stabilizes and speeds up optimization."),
    ("What is the derivative of x^2?",
     "The derivative of x^2 with respect to x is 2x."),
    ("What does SGD stand for?",
     "SGD stands for Stochastic Gradient Descent."),
    ("What is the softmax function used for?",
     "Softmax turns logits into a probability distribution over classes."),
]


def _tools_section(tools):
    lines = ["Available tools:"]
    for name, desc in tools:
        lines.append(f"  - {name}: {desc}")
    return ("\n".join(lines)
            + "\n\nTo use a tool, respond with:\n"
            + 'use tool:{"name": "tool_name", "arguments": {...}}\n\n')


def _user(t): return f"User: {t}\n"  # \n matches ToolUseMoE._build_prompt exactly
def _assistant(t): return f"Assistant: {t}"
def _tool_result(t): return f"result: {t}\n"


def main():
    import os
    # B-63: env-overridable volume/output — volume-vs-diversity isolation test:
    # TOOL_SFT_N=1200 builds 10x instances of the SAME templates/tools (pure
    # volume, no new formats); TOOL_SFT_OUT redirects the file.
    n_per = int(os.environ.get("TOOL_SFT_N", str(N_PER_TOOL)))
    out_path = Path(os.environ.get("TOOL_SFT_OUT", str(OUT)))
    only = [t.strip() for t in os.environ.get("TOOL_SFT_TOOLS", "").split(",") if t.strip()]
    # B-178: default to compiled schema — query.py's ToolUseMoE uses
    # compiled_schema=True, so training on verbose while serving compiled is an
    # OOD trap (B-146/B-169) that degrades candidates. Set TOOL_SFT_COMPILED=0
    # only for explicit volume/diversity isolation tests.
    compiled = os.environ.get("TOOL_SFT_COMPILED", "1") == "1"
    rng = random.Random(SEED)
    # Build the system section from the REAL registry so the training data's
    # tools section is byte-identical to the live ToolUseMoE prompt (B-30:
    # hardcoding name+desc omitted the per-tool parameter JSON and the GitHub
    # tools that the live prompt actually shows).
    from tool_use import ToolRegistry, _execute_python, _search_knowledge
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    try:
        from github_tools import build_github_tools
        build_github_tools(reg)
    except Exception:
        pass
    tools_section = reg.get_tools_section(compiled=compiled)
    tools = [(name, reg._tools[name]["description"]) for name in reg.list_tools()]

    records = []
    for tool_name, desc in tools:
        if only and tool_name not in only:
            continue
        tmpls = _TOOL_TEMPLATES.get(tool_name, [])
        for i in range(n_per):
            if not tmpls:
                break
            q, args, result, answer = tmpls[i % len(tmpls)]
            call = json.dumps({"name": tool_name, "arguments": args})
            records.append({
                "segments": [
                    ("system", tools_section),
                    ("user", _user(q)),
                    ("assistant", f"Assistant: use tool:{call}"),
                    ("tool", _tool_result(result)),
                    ("assistant", _assistant(answer)),
                ]
            })

    for q, answer in _DIRECT_EXAMPLES:
        for _ in range(N_DIRECT // len(_DIRECT_EXAMPLES)):
            records.append({
                "segments": [
                    ("system", tools_section),
                    ("user", _user(q)),
                    ("assistant", _assistant(answer)),
                ]
            })

    rng.shuffle(records)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    n_tool = sum(1 for r in records if len(r["segments"]) >= 4)
    print(f"[ToolSFT] {len(records)} conversations -> {out_path} "
          f"({n_tool} tool-call, {len(records) - n_tool} direct)")


if __name__ == "__main__":
    main()
