"""Generated alien rule worlds (a template for hand-made scenarios) -> scenarios.

Each world has its own numeral system (base b, made-up digit words) and --ops made-up binary operators,
each with a definition and a precedence level; expressions are written without parentheses. The document
is short, but applying it (base conversion, precedence, chaining) is a multi-step procedure, so the knowledge
volume and the application difficulty can be turned up independently (--ops, --length).
Graded by exact match of the final numeral.

Usage:
    python -m teachlab.adapters.rule_world --worlds 10 --ops 6 --length 3
"""

import argparse
import random

from ..schema import write_scenarios

SYLLABLES = ["ka", "mo", "ri", "zu", "te", "lo", "vi", "sha", "pe", "nu", "go", "xi", "da", "fe", "yo", "bu"]
OP_WORDS = ["glorp", "snev", "tikka", "voom", "blen", "quarr", "zib", "frell", "mox", "drun", "plim", "skee"]

# (description with a and b, python implementation); results stay non-negative.
OP_TEMPLATES = [
    ("twice a plus b", lambda a, b: 2 * a + b),
    ("a plus b plus 1", lambda a, b: a + b + 1),
    ("a times b, plus a", lambda a, b: a * b + a),
    ("the absolute difference of a and b", lambda a, b: abs(a - b)),
    ("the larger of a and b, doubled", lambda a, b: 2 * max(a, b)),
    ("the smaller of a and b, plus 3", lambda a, b: min(a, b) + 3),
    ("the remainder of a times b divided by 7, plus a", lambda a, b: (a * b) % 7 + a),
    ("a plus three times b", lambda a, b: a + 3 * b),
    ("a times a, minus b if that is not negative, otherwise b minus a times a", lambda a, b: abs(a * a - b)),
    ("the integer part of (a plus b) divided by 2", lambda a, b: (a + b) // 2),
    ("a if a is even, otherwise a plus b", lambda a, b: a if a % 2 == 0 else a + b),
    ("b times 2 if a is greater than b, otherwise a times 2", lambda a, b: 2 * b if a > b else 2 * a),
]


class World:
    def __init__(self, rng, n_ops):
        self.base = rng.randint(5, 9)
        self.digits = rng.sample(SYLLABLES, self.base)
        words = rng.sample(OP_WORDS, n_ops)
        templates = rng.sample(OP_TEMPLATES, n_ops)
        self.ops = {w: {"desc": d, "fn": f, "prec": rng.randint(1, 3)} for w, (d, f) in zip(words, templates)}

    def numeral(self, n):
        out = []
        while True:
            out.append(self.digits[n % self.base])
            n //= self.base
            if n == 0:
                return "-".join(reversed(out))

    def evaluate(self, nums, ops):
        """Precedence climbing: higher level binds tighter, equal levels go left to right."""
        nums, ops = list(nums), list(ops)
        for level in (3, 2, 1):
            i = 0
            while i < len(ops):
                if self.ops[ops[i]]["prec"] == level:
                    nums[i:i + 2] = [self.ops[ops[i]]["fn"](nums[i], nums[i + 1])]
                    ops.pop(i)
                else:
                    i += 1
        return nums[0]

    def document(self, name):
        digits = ", ".join(f"{d} = {i}" for i, d in enumerate(self.digits))
        ops = "\n".join(f"- {w} (binding level {o['prec']}): a {w} b is {o['desc']}." for w, o in self.ops.items())
        w1, w2 = list(self.ops)[:2]
        return f"""# The arithmetic of {name}

## Numerals
{name} writes numbers in base {self.base}. The digits are: {digits}.
A numeral lists its digits from the most significant to the least significant, joined by hyphens.
For example, {self.numeral(self.base + 2)} means 1 x {self.base} + 2 = {self.base + 2}. Zero is written {self.digits[0]}.

## Operators
Every operator takes a left value a and a right value b (both whole numbers):
{ops}

## Order of operations
Expressions are written without parentheses. Operators with a higher binding level are applied first.
Operators with the same binding level are applied from left to right.
For example, in "x {w1} y {w2} z", {"the " + w1 + " is applied first" if self.ops[w1]["prec"] >= self.ops[w2]["prec"] else "the " + w2 + " is applied first, because its binding level is higher"}.

## Answers
Results are always written as {name} numerals."""


def make_items(world, rng, n, length, split, sid, max_val):
    items = []
    for i in range(n):
        nums = [rng.randint(0, max_val) for _ in range(length + 1)]
        ops = [rng.choice(list(world.ops)) for _ in range(length)]
        expr = " ".join(f"{world.numeral(x)} {op}" for x, op in zip(nums, ops)) + f" {world.numeral(nums[-1])}"
        items.append({
            "id": f"{sid}/{split}{i}", "split": split,
            "messages": [{"role": "user", "content": f"Evaluate the expression:\n\n{expr}\n\n"
                                                     "Write the result as a numeral of this world, inside \\boxed{}."}],
            "grader": {"type": "exact", "answer": world.numeral(world.evaluate(nums, ops))},
        })
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=10)
    p.add_argument("--ops", type=int, default=6, help="Operators per world (knowledge volume)")
    p.add_argument("--length", type=int, default=3, help="Operators per expression (application difficulty)")
    p.add_argument("--max-val", type=int, default=20)
    p.add_argument("--stream", type=int, default=20)
    p.add_argument("--probe", type=int, default=10)
    p.add_argument("--test", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rng = random.Random(args.seed)

    variant = f"o{args.ops}l{args.length}{f'_s{args.seed}' if args.seed else ''}"
    scenarios = []
    for w in range(args.worlds):
        world, name = World(rng, args.ops), rng.choice(["Velm", "Oruth", "Kesh", "Tamboa", "Ilvar", "Prenn"])
        sid = f"rule_world/{variant}/w{w:02d}"
        items = [it for split, n in (("stream", args.stream), ("probe", args.probe), ("test", args.test))
                 for it in make_items(world, rng, n, args.length, split, sid, args.max_val)]
        scenarios.append({
            "id": sid, "source": "rule_world",
            "meta": {"base": world.base, "ops": args.ops, "length": args.length, "seed": args.seed},
            "knowledge": world.document(name),
            "student_system": f"You are a careful assistant who does arithmetic in the world of {name}.",
            "items": items,
        })
    write_scenarios(scenarios, args.out or f"teachlab/scenarios/rule_world_{variant}.jsonl")


if __name__ == "__main__":
    main()
