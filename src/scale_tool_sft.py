"""
Scale tool-use SFT data by template-based generation with REAL execution
verification. Every record's expected result is produced by ACTUALLY executing
the template's code in the same subprocess sandbox as build_tool_sft.py /
rest_round.py (tool_use._execute_python), so the labels are exact, not guessed.

Categories: math, string processing, file operations, data analysis,
robotics-adjacent (angles/coordinates/matrices/quaternions). Includes a
~10% no-call "pure reasoning" negative class and 2-3 turn multi-turn chains.

Output JSONL (same segment format as build_tool_sft.py / auto_tool_train):
    {"segments": [[role, text], ...]} with user/assistant/tool/result framing.
"""
import sys, json, random
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data" / "processed" / "auto_tool_train_v2.jsonl"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tool_use import _execute_python


def _user(t): return f"User: {t}\n"
def _assistant(t): return f"Assistant: {t}"
def _tool_result(t): return f"result: {t}\n"
def _tool_call(name, args): return f"Assistant: use tool:{json.dumps({'name': name, 'arguments': args})}"


def _exec_code(code):
    """Execute raw Python code string in the sandbox and return the result."""
    return _execute_python(code)


def _verify(code):
    """Run <code> through the execution sandbox and return (result_str, ok).
    ok=False means the sandbox errored (any Python *Error, traceback, or
    'no output' from a multi-statement template) — we skip this record so no
    broken template ever writes a garbage label."""
    r = _exec_code(code)
    stripped = r.strip()
    lowered = stripped.lower()
    bad_prefix = ("error", "traceback", "exception", "sandbox error",
                  "code executed successfully (no output)")
    import re
    if any(lowered.startswith(b) for b in bad_prefix):
        return r, False
    if re.search(r"^[A-Za-z_]+Error\b", stripped) or re.search(r"Error: ", stripped):
        return r, False
    if "[timed out" in lowered or "[truncated" in lowered:
        return r, False
    return r, True


# Each entry: (query, code, answer_hint). The actual result is captured by
# executing <code> and inserting the returned value into the final answer.
_MATH = [
    ("Compute the sum of all even numbers from 1 to 40.", "sum(i for i in range(1,41) if i%2==0)"),
    ("Sum the first 25 odd positive integers.", "sum(2*i+1 for i in range(25))"),
    ("What is the product of the first 6 prime numbers?", "__import__('math').prod([2,3,5,7,11,13])"),
    ("Compute the mean of [12, 15, 18, 21, 24].", "sum([12,15,18,21,24])/5"),
    ("Find the median of [7, 1, 9, 3, 5, 11].", "__import__('statistics').median([7,1,9,3,5,11])"),
    ("Compute the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9].", "round(__import__('statistics').pstdev([2,4,4,4,5,5,7,9]), 4)"),
    ("What is 2^12?", "2**12"),
    ("Compute the factorial of 12.", "__import__('math').factorial(12)"),
    ("What is the GCD of 48 and 36?", "__import__('math').gcd(48,36)"),
    ("Compute the LCM of 12 and 18.", "__import__('math').lcm(12,18)"),
    ("Evaluate the linear equation 3x + 5 = 20 for x.", "(20-5)/3"),
    ("What is sqrt(169)?", "__import__('math').isqrt(169)"),
    ("Compute 7!/3!.", "__import__('math').factorial(7)//__import__('math').factorial(3)"),
    ("Find the number of ways to choose 3 items from 10.", "__import__('math').comb(10,3)"),
    ("Compute the sum 1 + 1/2 + 1/3 + ... + 1/10 (harmonic).", "round(sum(1/i for i in range(1,11)), 4)"),
    ("What is the determinant of [[2,3],[1,4]]?", "2*4-3*1"),
    ("Solve the quadratic 2x^2 + 5x - 3 = 0 (positive root).", "(-5 + (25+4*2*3)**0.5)/(4)"),
    ("Compute the sum of digits of 12345.", "sum(int(c) for c in str(12345))"),
    ("What is log base 2 of 1024?", "__import__('math').log2(1024)"),
    ("Compute 15% of 240.", "0.15*240"),
    ("What is the integral of x^2 from 0 to 3 (exact)?", "3**3/3"),
    ("What is the derivative coefficient: slope of y=4x+2?", "4"),
    ("Compute the 20th triangular number.", "20*21//2"),
    ("Find the geometric mean of 4 and 16.", "(4*16)**0.5"),
    ("Compute |−3.7| rounded.", "round(abs(-3.7),1)"),
    ("What is 5 choose 2?", "__import__('math').comb(5,2)"),
    ("Compute the sum of cubes of 1..6.", "sum(i**3 for i in range(1,7))"),
    ("What is the remainder of 2^10 divided by 7?", "1024 % 7"),
    ("Compute the average of the first 10 squares.", "sum(i*i for i in range(1,11))/10"),
    ("What is the 7th Fibonacci number?",      # nth -> F(n-1): index [6] = 8 (the
     "([0,1,1,2,3,5,8,13][6])"),            # old [7] returned 13 = F(7): the ONLY
                                            # 1-indexed instance -> generalization
                                            # mismatch with every other fib probe)
]

_STRING = [
    ("Join the list ['alpha','beta','gamma'] with a hyphen.", "'-'.join(['alpha','beta','gamma'])"),
    ("Split 'a,b,c,d' on commas and report the count.", "len('a,b,c,d'.split(','))"),
    ("How many uppercase letters are in 'Hello World Python'?", "sum(c.isupper() for c in 'Hello World Python')"),
    ("Reverse the string 'racecar' and check if it is a palindrome.",
     "('racecar'[::-1], 'racecar' == 'racecar'[::-1])"),
    ("Strip all whitespace from '  spaced  out  '.", "'  spaced  out  '.replace(' ','')"),
    ("Find all digits in 'a1b2c34' using regex and sum them.", "__import__('re').findall(r'\\d', 'a1b2c34')"),
    ("Replace all vowels in 'hello world' with '*'.",
     "'hello world'.translate(str.maketrans('aeiou','*****'))"),
    ("Count the words in 'the quick brown fox jumps over the lazy dog'.", "len('the quick brown fox jumps over the lazy dog'.split())"),
    ("Report the last 4 characters of 'abcdefgh'.", "'abcdefgh'[-4:]"),
    ("Encode 'data' as UTF-8 and report its length in bytes.", "len('data'.encode('utf-8'))"),
    ("What is the ASCII value of the first char of 'Ambatron'?", "ord('Ambatron'[0])"),
    ("Repeat the string 'ab' 5 times.", "'ab'*5"),
    ("Replace spaces with underscores in 'hello world'.", "'hello world'.replace(' ','_')"),
    ("Convert 'hello world' to title case.", "'hello world'.title()"),
    ("Sort the characters of 'openai' alphabetically and join.", "''.join(sorted('openai'))"),
    ("Find the first occurrence index of 'lo' in 'hello world'.", "'hello world'.find('lo')"),
    ("How many times does 'ana' appear in 'banana'?", "'banana'.count('ana')"),
    ("Check if '12345' consists only of digits.", "'12345'.isdigit()"),
    ("Capitalize the first letter of 'ambatron'.", "'ambatron'.capitalize()"),
    ("Extract the substring from index 2 to 6 of 'abcdefghi'.", "'abcdefghi'[2:6]"),
]

_FILEOPS = [
    ("Write 'hello' to a file named tmp_read.txt and read it back.",
     "(__import__('pathlib').Path('sft_tmp_read2.txt').write_text('hello'), __import__('pathlib').Path('sft_tmp_read2.txt').read_text())[1]"),
    ("Create a directory 'tmpdir_a' and check it exists (in the sandbox).",
     "(__import__('os').makedirs('sft_tmpdir_b', exist_ok=True), __import__('os').path.isdir('sft_tmpdir_b'))[1]"),
    ("List the files in the current (sandbox) directory.",
     "sorted(__import__('os').listdir('.'))[:5]"),
    ("Get the basename of the path '/home/user/docs/report.pdf'.",
     "__import__('posixpath').basename('/home/user/docs/report.pdf')"),
    ("Join two path components '/tmp/notes' and 'report.txt'.",
     "__import__('posixpath').join('/tmp/notes','report.txt')"),
    ("Check whether 'report.py' ends with '.py'.",
     "'report.py'.endswith('.py')"),
    ("What is the size in bytes of a file you write to sft_tmp_sz.txt?",
     "(__import__('pathlib').Path('sft_tmp_sz2.txt').write_text('12345'), __import__('pathlib').Path('sft_tmp_sz2.txt').stat().st_size)[1]"),
    ("Split '/a/b/c/d.log' into directory and filename.",
     "__import__('posixpath').split('/a/b/c/d.log')"),
    ("Get the file extension of 'archive.tar.gz'.",
     "__import__('posixpath').splitext('archive.tar.gz')[1]"),
    ("Create a file sft_tmp_abs.txt in the sandbox and report its absolute path.",
     "(__import__('pathlib').Path('sft_tmp_abs2.txt').write_text('x'), str(__import__('pathlib').Path('sft_tmp_abs2.txt').resolve()))[1]"),
]

_DATA = [
    ("From [5,3,9,1,7], find the second largest value.", "sorted([5,3,9,1,7])[-2]"),
    ("Given the dict {'a':1,'b':2,'c':3}, report the sum of values.", "sum({'a':1,'b':2,'c':3}.values())"),
    ("How many unique elements are in [1,2,2,3,3,3,4]?", "len(set([1,2,2,3,3,3,4]))"),
    ("Concatenate [1,2,3] and [4,5] and report the sum.", "sum([1,2,3]+[4,5])"),
    ("Using a list comprehension, square [1,2,3,4] and sum the squares.", "sum(x*x for x in [1,2,3,4])"),
    ("Zip two lists [1,2,3] and ['a','b','c'] into pairs.", "list(zip([1,2,3],['a','b','c']))"),
    ("From the list of words, find the longest: ['cat','elephant','dog','giraffe'].", "max(['cat','elephant','dog','giraffe'], key=len)"),
    ("Report the running total of [1,2,3,4] as a list.",
     "[sum([1,2,3,4][:i+1]) for i in range(4)]"),
    ("Flip the dict {'x':1,'y':2} keys and values.", "{v:k for k,v in {'x':1,'y':2}.items()}"),
    ("Get the frequency of each character in 'aabbc'.", "dict(__import__('collections').Counter('aabbc'))"),
    ("Find all numbers divisible by 3 in [1..15].",
     "[x for x in range(1,16) if x%3==0]"),
    ("What is the dot product of [1,2,3] and [4,5,6]?", "sum(a*b for a,b in zip([1,2,3],[4,5,6]))"),
    ("Sort the dict {'b':2,'a':3,'c':1} by values ascending.",
     "[k for k,v in sorted({'b':2,'a':3,'c':1}.items(), key=lambda kv: kv[1])]"),
    ("From [[1,2],[3,4],[5,6]], flatten to one list.", "[x for row in [[1,2],[3,4],[5,6]] for x in row]"),
    ("Report the index of the maximum in [4, 9, 2, 7].", "[4,9,2,7].index(max([4,9,2,7]))"),
    ("Compute the pairwise sum of [1,2,3] and [10,20,30].",
     "[a+b for a,b in zip([1,2,3],[10,20,30])]"),
    ("Group numbers 1..10 into even and odd lists.",
     "[x for x in range(1,11) if x%2==0], [x for x in range(1,11) if x%2==1]"),
    ("Deliver the transpose of [[1,2,3],[4,5,6]].", "list(map(list, zip(*[[1,2,3],[4,5,6]])))"),
    ("What is 2 plus 2, expressed via reduce?",
     "__import__('functools').reduce(lambda a,b:a+b, [2,2])"),
    ("Use a comprehension to extract all keys from {'red':1,'green':2,'blue':3}.",
     "[k for k in {'red':1,'green':2,'blue':3}]"),
]

_ROBO = [
    ("Convert 90 degrees to radians.", "round(__import__('math').radians(90), 4)"),
    ("Convert 1.5708 radians to degrees.", "round(__import__('math').degrees(1.5708), 2)"),
    ("Compute the roll,pitch given a rotation of 30 deg about z (use cos/sin).", "(round(__import__('math').cos(__import__('math').radians(30)),3), round(__import__('math').sin(__import__('math').radians(30)),3))"),
    ("Rotate the 2D point (1, 0) by 90 degrees counterclockwise.", "(round(1*__import__('math').cos(__import__('math').radians(90)) - 0*__import__('math').sin(__import__('math').radians(90)),3), round(1*__import__('math').sin(__import__('math').radians(90)) + 0*__import__('math').cos(__import__('math').radians(90)),3))"),
    ("Compute the distance between points (0,0) and (3,4).", "__import__('math').hypot(3,4)"),
    ("Convert Cartesian (0, 1) to polar (r, theta in degrees).", "(round(__import__('math').hypot(0,1),3), round(__import__('math').degrees(__import__('math').atan2(1,0)),1))"),
    ("Compute the 3x3 identity matrix as a nested list.",
     "[[1 if i==j else 0 for j in range(3)] for i in range(3)]"),
    ("Multiply the matrices [[1,2],[3,4]] and [[5,6],[7,8]].",
     "[[1*5+2*7, 1*6+2*8],[3*5+4*7,3*6+4*8]]"),
    ("Add two quaternions (1+i) and (2+3j) component-wise (as tuples).", "(1+2, 1+3)"),
    ("Normalize the 2D vector (3,4) to unit length.",
     "(lambda x,y,m:(x/m, y/m))(3, 4, __import__('math').hypot(3,4))"),
    ("Compute the angle (degrees) of the vector (1,1) in the plane.",
     "round(__import__('math').degrees(__import__('math').atan2(1,1)),1)"),
    ("Euler angle 90 deg about z -> quaternion (w,x,y,z) for pure Z rotation.",
     "(round(__import__('math').cos(__import__('math').radians(45)),3), 0, 0, round(__import__('math').sin(__import__('math').radians(45)),3))"),
    ("Compute the wheel speeds for a differential drive with v=1, w=0, L=0.5.",
     "(1 + 0*0.5/2, 1 - 0*0.5/2)"),
    ("Find the determinant of the rotation matrix [[0,-1],[1,0]].", "0*0 - (-1)*1"),
    ("Compute the trace of a 2x2 matrix [[1,2],[3,4]].", "1+4"),
    ("Convert polar (r=2, theta=60 deg) to Cartesian.",
     "(round(2*__import__('math').cos(__import__('math').radians(60)),3), round(2*__import__('math').sin(__import__('math').radians(60)),3))"),
    ("Compute the cross product of 2D vectors (1,0) and (0,1) (scalar z).", "1*1 - 0*0"),
    ("Reflect the point (1,0) across the y-axis.", "(-1, 0)"),
    ("Compute the rotation of point (2,0) by 45 degrees.",
     "(round(2*__import__('math').cos(__import__('math').radians(45)) - 0*__import__('math').sin(__import__('math').radians(45)),3), round(2*__import__('math').sin(__import__('math').radians(45)) + 0*__import__('math').cos(__import__('math').radians(45)),3))"),
    ("Compute the average angular velocity of 3 samples [1.0, 1.5, 2.0] rad/s.",
     "sum([1.0,1.5,2.0])/3"),
]

_NO_CALL = [
    ("Explain the difference between an encoder and a decoder in transformers.",
     "An encoder processes the full input bidirectionally; a decoder generates output left-to-right using attention over prior tokens."),
    ("Why is momentum important in gradient descent?",
     "Momentum accumulates past gradients to smooth updates, damping oscillations and accelerating convergence in ravines."),
    ("What does a covariance matrix represent?",
     "It captures pairwise variances (diagonal) and covariances (off-diagonal) of multiple random variables."),
    ("Explain what a quaternion is in one sentence.",
     "A quaternion is a four-component number (w,x,y,z) that represents a 3D rotation without gimbal lock."),
    ("What is the purpose of a loss function in robotics policy learning?",
     "The loss function quantifies how far the policy's predicted actions are from the desired target, guiding gradient updates."),
    ("Describe the robot kinematics concept of forward versus inverse kinematics.",
     "Forward kinematics maps joint angles to end-effector pose; inverse kinematics computes the joint angles to reach a desired pose."),
    ("What is the difference between L1 and L2 regularization?",
     "L1 drives weights toward zero (sparse); L2 shrinks large weights smoothly (rounded), both reducing overfitting."),
    ("Explain what receptive field means in a neural network.",
     "The receptive field is the region of the input that influences a given neuron's activation."),
    ("Why use bfloat16 over float16 for training?",
     "bfloat16 keeps the same exponent range as float32, avoiding the underflow/overflow issues of fp16 while halving memory."),
    ("What is the role of a reward function in reinforcement learning?",
     "It provides the scalar signal that tells the agent how good a state/action is, guiding the policy toward the goal."),
]


def _build_tool_record(tools_section, query, call_args, result, cat):
    segments = [
        ["system", tools_section],
        ["user", _user(query)],
        ["assistant", _tool_call("execute_python", call_args)],
        ["tool", _tool_result(result)],
        ["assistant", _assistant(f"The answer is {result}.")],
    ]
    return {"segments": segments, "category": cat, "verified": True}


def _build_multi_turn(tools_section, query1, call1, result1, query2, call2, result2, cat):
    segments = [
        ["system", tools_section],
        ["user", _user(query1)],
        ["assistant", _tool_call("execute_python", call1)],
        ["tool", _tool_result(result1)],
        ["assistant", _assistant(f"Intermediate result: {result1}.")],
        ["user", _user(query2)],
        ["assistant", _tool_call("execute_python", call2)],
        ["tool", _tool_result(result2)],
        ["assistant", _assistant(f"The final answer is {result2}.")],
    ]
    return {"segments": segments, "category": cat, "verified": True}


def _build_no_call(tools_section, query, answer):
    return {"segments": [["system", tools_section],
                          ["user", _user(query)],
                          ["assistant", _assistant(answer)]],
            "category": "no_call", "verified": True}


def _tools_section():
    from tool_use import ToolRegistry, _execute_python, _search_knowledge
    reg = ToolRegistry()
    reg.register(_execute_python, name="execute_python",
                 description="Execute Python code and return output")
    reg.register(_search_knowledge, name="search_knowledge",
                 description="Search internal knowledge base for information")
    return reg.get_tools_section(compiled=True)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Scale tool-SFT data with execution-verified templates")
    ap.add_argument("--count", type=int, default=0,
                    help="Number of records to generate (0 = all templates)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--multi_turn", type=int, default=0,
                    help="Number of 2-turn chain records to generate")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Repeat each verified record N times (volume scaling)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tools_section = _tools_section()

    records = []
    failed = 0

    categories = [("math", _MATH), ("string", _STRING), ("fileops", _FILEOPS),
                  ("data", _DATA), ("robotics", _ROBO)]

    tool_pool = []
    for cat, items in categories:
        for (query, code) in items:
            tool_pool.append((cat, query, code))

    for _ in range(args.multi_turn):
        if len(tool_pool) < 2:
            break
        i1, i2 = rng.sample(range(len(tool_pool)), 2)
        cat1, q1, c1 = tool_pool[i1]
        cat2, q2, c2 = tool_pool[i2]
        r1, ok1 = _verify(c1)
        r2, ok2 = _verify(c2)
        if not (ok1 and ok2):
            failed += (0 if ok1 else 1) + (0 if ok2 else 1)
            continue
        cat = f"multi_{cat1}_{cat2}"
        records.append(_build_multi_turn(tools_section, q1, {"code": c1}, r1,
                                         q2, {"code": c2}, r2, cat))

    tool_pool_shuffled = list(tool_pool)
    rng.shuffle(tool_pool_shuffled)
    verified_templates = []
    for cat, query, code in tool_pool_shuffled:
        result, ok = _verify(code)
        if not ok:
            failed += 1
            continue
        call_args = {"code": code}
        rec = _build_tool_record(tools_section, query, call_args, result, cat)
        verified_templates.append(rec)

    for rec in verified_templates:
        for rep in range(args.repeat):
            records.append(rec)

    # 15th-audit: no-call negatives must be ~10% of the FINAL record count, not
    # 10% of the ~100-template pool — the old `max(5, 10% of tools)` yielded only
    # 10 no-call vs ~600 tool records = 1.6%, so the model never learned when NOT
    # to call a tool (the root of the tool axis's wrong-tool failures). Now target
    # ~10% of whatever was actually emitted (single + multi-turn chains + repeats).
    n_no_call = max(5, int(len(records) * 0.10))
    for i in range(n_no_call):
        q, a = _NO_CALL[i % len(_NO_CALL)]
        records.append(_build_no_call(tools_section, q, a))

    rng.shuffle(records)

    if args.count > 0:
        records = records[:args.count]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    n_tool = sum(1 for r in records if r.get("category") != "no_call")
    print(f"[ScaleToolSFT] {len(records)} records -> {out_path} "
          f"({n_tool} tool-call, {len(records) - n_tool} no-call, {failed} exec-failed skipped)")


if __name__ == "__main__":
    main()
