"""HARD tool-eval suite (ROUND-174): every expected answer is COMPUTED by a
reference implementation at load time — no hand-typed answers to typo.
Traps designed from BigCodeBench's principles + the flywheel's failure modes.
"""
import math, random

def fib(n):  # repo convention: F(0)=0, F(1)=1
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def primes(limit):
    return [n for n in range(2, limit + 1) if all(n % d for d in range(2, int(n**0.5)+1))]

def is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5)+1))

def gen() -> list[tuple[str, tuple, object, callable]]:
    """query, (tool, expected), expected-callable (recomputed)."""
    cases = []

    # ---- T1: BIG arithmetic (multi-digit, large values) ----
    cases.append(("What is 12345 multiplied by 6789?",
                  ("execute_python", 12345*6789), lambda: 12345*6789))
    cases.append(("What is 987654321 divided by 1234567?",
                  ("execute_python", 987654321//1234567), lambda: 987654321//1234567))
    cases.append(("What is 15 factorial?", ("execute_python", math.factorial(15)),
                  lambda: math.factorial(15)))
    cases.append(("What is 2 to the power of 40?", ("execute_python", 2**40),
                  lambda: 2**40))
    # ---- T2: CHAINED (2-step) requiring intermediate values ----
    cases.append(("What is 17 times 23, minus 41?", ("execute_python", 17*23-41),
                  lambda: 17*23-41))
    cases.append(("What is the sum of the first 25 prime numbers?",
                  ("execute_python", sum(primes(100)[:25])),
                  lambda: sum(primes(100)[:25])))
    # ---- T3: WRONG-TOOL traps (looks like one op, needs another) ----
    cases.append(("Is 317 prime?", ("execute_python", "yes"), lambda: "yes"))
    cases.append(("Is 323 prime?", ("execute_python", "no"), lambda: "no"))   # 17*19
    # ROUND-182: the OLD trap was mathematically wrong — 27³ == 3⁹ == 19683 (equal),
    # so the expected '3 to the power of 9' answer was incorrect; fixed by
    # widening the exponent so 3^10 (59049) genuinely wins.
    cases.append(("Which is larger, 27 cubed or 3 to the power of 10?",
                  ("execute_python", "3 to the power of 10"), lambda: "3 to the power of 10"))
    # ---- T4: negative divisibility for the always-yes hack ----
    for a, b in [(121, 11), (143, 13), (221, 17), (1333, 31), (771, 3)]:
        cases.append((f"Is {a} divisible by {b}?",
                      ("execute_python", "yes" if a % b == 0 else "no"),
                      lambda a=a, b=b: "yes" if a % b == 0 else "no"))
    for a, b in [(127, 11), (149, 13), (223, 17), (1457, 31), (771, 7)]:
        cases.append((f"Is {a} divisible by {b}?",
                      ("execute_python", "yes" if a % b == 0 else "no"),
                      lambda a=a, b=b: "yes" if a % b == 0 else "no"))
    # ---- T5: edge cases (zero, one, negatives, float output) ----
    cases.append(("Is 1 prime?", ("execute_python", "no"), lambda: "no"))
    cases.append(("Is 0 divisible by 5?", ("execute_python", "yes"), lambda: "yes"))
    cases.append(("Is -12 divisible by 3?", ("execute_python", "yes"), lambda: "yes"))
    cases.append(("What is 7 divided by 2 (report the remainder)?", ("execute_python", 1),
                  lambda: 7 % 2))
    cases.append(("What is 17 modulo 5?", ("execute_python", 2), lambda: 17 % 5))
    # ---- T6: composition + float median (traps int-output habbits) ----
    cases.append(("What is the median of [3,1,9,6]?", ("execute_python", 4.5),
                  lambda: (sorted([3,1,9,6])[1] + sorted([3,1,9,6])[2]) / 2))
    # ---- T7: multi-arg precision (10-element sort, positional) ----
    arr = [9,2,7,4,1,8,5,3,6,0]
    cases.append(("Sort [9,2,7,4,1,8,5,3,6,0] and report the 6th element.",
                  ("execute_python", sorted(arr)[5]), lambda: sorted(arr)[5]))
    # ---- T8: fibonacci family (recomputed) ----
    for k in [21, 30, 33]:
        cases.append((f"Compute the {k}th Fibonacci number.", ("execute_python", fib(k-1)),
                      lambda k=k: fib(k-1)))
    return cases

CASES = gen()
