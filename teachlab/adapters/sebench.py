"""SE-Bench (NumPy renamed to the unseen library `zwc`) -> scenarios.

Each course is a group of ~--funcs functions, grown greedily from the function sets of multiple_test items so that some multi-function test items fall inside the course. knowledge = zwc's quick start, the array constructor and the rewritten docs of the course's functions. stream / probe = train items over those functions (one function each), test = single_test and multiple_test items over those functions.

Graded by execution against the test cases (see graders.grade_exec).

Data: data/sebench/{train,single_test,multiple_test}.jsonl from huggingface.co/datasets/jintailin/SE-Bench,
data/sebench/api_doc.jsonl and the zwc package in data/sebench/zwc from github.com/thunlp/SE-Bench
(fetch all of it with `bash teachlab/download_sebench.sh`).

Usage:
    python -m teachlab.adapters.sebench --funcs 8 --courses 20
    python -m teachlab.adapters.sebench --doc-scope library   # knowledge = docs of every zwc function
"""

import argparse
import json
import os
import random

from ..schema import write_scenarios

DATA = "data/sebench"

STUDENT_SYSTEM = ("You are a Python programmer. Your runtime has an unfamiliar numerical library called zwc and "
                  "NumPy is NOT installed, so every solution must be built on zwc functions.")

TASK_PROMPT = """### Problem
{query}

### Example test case (the data structures are valid; the output may not be exact)
{example}

### Function to complete
{signature}

### Requirements
- Solve the problem with the zwc library (`import zwc`). NumPy is not available, and re-implementing the logic without zwc is not allowed.
- Keep the function name and parameters unchanged. You may import Python built-in modules.
- Return a value with the same data structure as the example output. zwc arrays have no `.tolist()`; returning a zwc array is accepted, since the grader compares printed values.
- End your answer with the complete implementation in a single ```python``` block."""


def load(name):
    with open(os.path.join(DATA, f"{name}.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def to_item(row, split, uid):
    funcs = row["selected_functions"]
    return {
        "id": uid, "split": split, "functions": funcs,
        "messages": [{"role": "user", "content": TASK_PROMPT.format(
            query=row["query"].strip(), example=row["example"], signature=row["function_name"].strip())}],
        "grader": {"type": "exec", "entry": row["function_name"].split("def ")[1].split("(")[0].strip(),
                   "test_cases": row["test_cases"], "expected": row["right_exe_result"].split("#"), "lib": "zwc"},
    }


def grow_courses(multi, n_funcs, n_courses, rng):
    """Greedy: seed with a multiple_test function set, keep adding the set that adds the fewest new functions."""
    sets = [frozenset(r["selected_functions"]) for r in multi]
    unused, courses = list(range(len(sets))), []
    rng.shuffle(unused)
    while unused and len(courses) < n_courses:
        funcs = set(sets[unused.pop()])
        while True:
            fits = [i for i in unused if len(funcs | sets[i]) <= n_funcs]
            if not fits:
                break
            best = min(fits, key=lambda i: (len(sets[i] - funcs), rng.random()))
            funcs |= sets[best]
            unused.remove(best)
        courses.append(sorted(funcs))
    return courses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--funcs", type=int, default=8, help="Functions per course")
    p.add_argument("--courses", type=int, default=20)
    p.add_argument("--probe-frac", type=float, default=0.2, help="Fraction of a course's train items held out as probe")
    p.add_argument("--min-stream", type=int, default=10)
    p.add_argument("--doc-scope", choices=["course", "library"], default="course",
                   help="course: docs of the course's functions only; library: all ~270 zwc functions (~97k words)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rng = random.Random(args.seed)

    docs = {}
    for line in open(os.path.join(DATA, "api_doc.jsonl")):
        d = json.loads(line)
        docs[d["original_name"]] = d
    quick = open(os.path.join(DATA, "zwc", "zwc", "README.md")).read().split("## Features")[0].strip()
    quick = quick.replace("pip install zwc", "# zwc is already installed")

    def knowledge(funcs):
        names = sorted(docs) if args.doc_scope == "library" else sorted(set(funcs) | {"array"})
        body = "\n\n".join(f"### zwc.{docs[f]['curr_name']}  (function {i + 1})\n{docs[f]['rewritten_doc'].strip()}"
                           for i, f in enumerate(names) if f in docs)
        return f"{quick}\n\n## Function reference\n\n{body}"

    train = [r for r in load("train") if r.get("is_valid", True)]
    single = [r for r in load("single_test") if r.get("is_valid", True)]
    multi = [r for r in load("multiple_test") if r.get("is_valid", True)]

    # The variant tag goes into every id, so scenarios built with different settings never collide in one run.
    variant = f"f{args.funcs}{'_lib' if args.doc_scope == 'library' else ''}{f'_s{args.seed}' if args.seed else ''}"
    scenarios = []
    for ci, funcs in enumerate(grow_courses(multi, args.funcs, args.courses * 3, rng)):
        fs = set(funcs)
        inside = lambda r: set(r["selected_functions"]) <= fs
        tr = [r for r in train if inside(r)]
        rng.shuffle(tr)
        n_probe = max(1, round(len(tr) * args.probe_frac))
        if len(tr) - n_probe < args.min_stream:
            continue
        sid = f"sebench/{variant}/c{len(scenarios):02d}"
        items = ([to_item(r, "probe", f"{sid}/probe{i}") for i, r in enumerate(tr[:n_probe])]
                 + [to_item(r, "stream", f"{sid}/stream{i}") for i, r in enumerate(tr[n_probe:])]
                 + [to_item(r, "test", f"{sid}/single{i}") for i, r in enumerate(r for r in single if inside(r))]
                 + [to_item(r, "test", f"{sid}/multi{i}") for i, r in enumerate(r for r in multi if inside(r))])
        kn = knowledge(funcs)
        scenarios.append({
            "id": sid, "source": "sebench",
            "meta": {"functions": funcs, "zwc_names": [docs[f]["curr_name"] for f in funcs if f in docs],
                     "doc_scope": args.doc_scope, "knowledge_words": len(kn.split())},
            "knowledge": kn, "student_system": STUDENT_SYSTEM, "items": items,
        })
        if len(scenarios) == args.courses:
            break
    out = args.out or f"teachlab/scenarios/sebench_{variant}.jsonl"
    write_scenarios(scenarios, out)
    for s in scenarios:
        n = {sp: sum(i["split"] == sp for i in s["items"]) for sp in ("stream", "probe", "test")}
        print(f"  {s['id']}: {len(s['meta']['functions'])} funcs, {s['meta']['knowledge_words']} words, {n}")


if __name__ == "__main__":
    main()
