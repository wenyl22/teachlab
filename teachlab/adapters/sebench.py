"""SE-Bench (NumPy renamed to the unseen library `zwc`) -> scenarios.

Each course is a group of ~--funcs functions, grown greedily from the function sets of multiple_test items so that some multi-function test items fall inside the course. knowledge = zwc's quick start, the array constructor and the rewritten docs of the course's functions. stream / probe = train items over those functions (one function each), test = single_test and multiple_test items over those functions.

Graded by execution against the test cases (see graders.grade_exec).

Only verified rows are used by default: rows whose own NumPy reference solution (`example_output`), run through
the same harness, reproduces the stored ground truth. The others are ambiguous or mislabeled (about 12% of
train, 12% of single_test and 19% of multiple_test) and would also give the student wrong feedback in study
sessions. The check runs once and is cached in data/sebench/verified.json; --all-rows skips it.

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
import re
from concurrent.futures import ThreadPoolExecutor

from .. import graders
from ..schema import write_scenarios

DATA = "data/sebench"
VERIFIED = os.path.join(DATA, "verified.json")
FILES = ("train", "single_test", "multiple_test")

# Runtime facts the API docs do not state. Without them a solver with the docs open still fails about half of
# its wrong answers on these (NumPy habits such as abs(arr), m1 & m2, arr.sum(), zwc.pi), which only execution
# feedback can reveal. Verified against data/sebench/zwc (ZWCArray in zwc/__init__.py).
STUDENT_SYSTEM = """You are a Python programmer. Your runtime has an unfamiliar numerical library called zwc and NumPy is NOT installed, so every solution must be built on zwc functions.

Runtime notes on zwc arrays (they hold for every zwc function):
- zwc functions accept Python numbers, lists and nested lists, and return a ZWCArray (or a plain Python number for a single value). A solution may return a ZWCArray directly.
- A ZWCArray supports only: indexing and slicing, including boolean masks (a[a > 0]); item assignment; len() and iteration; the operators + - * / ** with numbers or arrays; the comparisons == != < <= > >=; and the attributes .shape, .dtype, .size, .ndim and .T. Indexing one element gives a plain Python number.
- Nothing else exists on a ZWCArray: no methods such as .sum(), .mean(), .reshape(), .astype() or .tolist() (use the zwc function for the operation, or plain Python, e.g. [float(x) for x in a]); no unary minus or abs() on a whole array (write 0 - a, or use the zwc function); no & | ~ or @ (for masks, m1 * m2 means "and", (m1 + m2) > 0 means "or", m == False means "not").
- The zwc module has no constants (no zwc.pi, zwc.inf, zwc.nan, zwc.newaxis): use math.pi, float("inf"), float("nan"). The Python standard library is available."""

TASK_PROMPT = """### Problem
{query}

### Example test case (the data structures are valid; the output may not be exact)
{example}

### Function to complete
{signature}

### Requirements
- Solve the problem with the zwc library (`import zwc`). NumPy is not available, and re-implementing the logic without zwc is not allowed.
- Keep the function name and parameters unchanged. You may import Python built-in modules.
- Return a value with the same data structure as the example output. zwc arrays have no `.tolist()`; returning a zwc array is accepted (the grader converts it to plain Python values).
- End your answer with the complete implementation in a single ```python``` block."""


def load(name):
    """Rows of data/sebench/<name>.jsonl, each tagged with `_row` = "<name>/<0-based line index>"."""
    with open(os.path.join(DATA, f"{name}.jsonl")) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    for i, r in enumerate(rows):
        r["_row"] = f"{name}/{i}"
    return rows


def entry_name(row):
    return row["function_name"].split("def ")[1].split("(")[0].strip()


def to_item(row, split, uid):
    funcs = row["selected_functions"]
    return {
        "id": uid, "split": split, "functions": funcs, "source_row": row["_row"],
        "messages": [{"role": "user", "content": TASK_PROMPT.format(
            query=row["query"].strip(), example=row["example"], signature=row["function_name"].strip())}],
        "grader": {"type": "exec", "entry": entry_name(row),
                   "test_cases": row["test_cases"], "expected": row["right_exe_result"].split("#"), "lib": "zwc"},
    }


def zwc_path(doc):
    """Full call path of a doc's function, e.g. zwc.falekef or zwc.rfx.gicopuf (the 31 linear-algebra functions
    live in the rfx submodule; `zwc.<curr_name>` does not exist for them)."""
    for path in json.loads(doc["func_mapping"]).values():
        if path.split(".")[-1] == doc["curr_name"]:
            return path
    return f"zwc.{doc['curr_name']}"


def reference_ok(row):
    """Does the row's NumPy reference solution reproduce its stored ground truth through our harness?"""
    code = row["example_output"][0]
    entry = entry_name(row)
    if not re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", code):  # reference names its function differently
        m = re.search(r"\bdef\s+([A-Za-z_]\w*)\s*\(", code)
        if not m:
            return False
        entry = m.group(1)
    results, _, _ = graders.run_and_compare(code, entry, row["test_cases"], row["right_exe_result"].split("#"))
    return results is not None and all(r["ok"] for r in results)


def verified_rows(rows_by_file):
    """Set of verified `_row` ids, computed once and cached in VERIFIED (invalidated if the data changes)."""
    sizes = {name: len(rows) for name, rows in rows_by_file.items()}
    if os.path.exists(VERIFIED):
        cached = json.load(open(VERIFIED))
        if cached.get("sizes") == sizes:
            return set(cached["ok"])
    rows = [r for name in FILES for r in rows_by_file[name]]
    print(f"verifying the NumPy reference solutions of {len(rows)} rows (once; cached in {VERIFIED}) ...")
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(reference_ok, rows))
    good = [r["_row"] for r, o in zip(rows, ok) if o]
    try:
        import numpy
        np_version = numpy.__version__
    except ImportError:
        np_version = None
    json.dump({"numpy": np_version, "sizes": sizes, "ok": good,
               "bad": [r["_row"] for r, o in zip(rows, ok) if not o]}, open(VERIFIED, "w"), indent=0)
    for name in FILES:
        n_ok = sum(g.startswith(name + "/") for g in good)
        print(f"  {name}: {n_ok}/{sizes[name]} verified")
    return set(good)


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
    p.add_argument("--courses", type=int, default=10)
    p.add_argument("--probe-frac", type=float, default=0.2, help="Fraction of a course's train items held out as probe")
    p.add_argument("--min-stream", type=int, default=10)
    p.add_argument("--doc-scope", choices=["course", "library"], default="course",
                   help="course: docs of the course's functions only; library: all ~270 zwc functions (~97k words)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-rows", action="store_true",
                   help="Also use rows whose NumPy reference does not reproduce the ground truth (see module doc)")
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
        body = "\n\n".join(f"### {zwc_path(docs[f])}  (function {i + 1})\n{docs[f]['rewritten_doc'].strip()}"
                           for i, f in enumerate(names) if f in docs)
        return f"{quick}\n\n## Function reference\n\n{body}"

    raw = {name: load(name) for name in FILES}
    verified = None if args.all_rows else verified_rows(raw)
    # Rows naming a function without documentation (one multiple_test row lists ['np', 'np', 'np']) are dropped.
    usable = lambda r: (r.get("is_valid", True) and (verified is None or r["_row"] in verified)
                        and all(f in docs for f in r["selected_functions"]))
    train, single, multi = ([r for r in raw[name] if usable(r)] for name in FILES)
    if not args.all_rows:
        print(f"using verified rows: train {len(train)}, single_test {len(single)}, multiple_test {len(multi)}")

    # The variant tag goes into every id, so scenarios built with different settings never collide in one run.
    # "v" marks verified-only courses, so they never share ids with courses built from all rows (--all-rows).
    variant = (f"f{args.funcs}{'' if args.all_rows else 'v'}{'_lib' if args.doc_scope == 'library' else ''}"
               f"{f'_s{args.seed}' if args.seed else ''}")
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
            "meta": {"functions": funcs, "zwc_names": [zwc_path(docs[f]) for f in funcs if f in docs],
                     "doc_scope": args.doc_scope, "verified_only": not args.all_rows,
                     "knowledge_words": len(kn.split())},
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
