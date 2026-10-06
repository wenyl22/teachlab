"""Learning-curve tables for a run: mean solo ability by source x condition x k.

Usage:
    python -m teachlab.summarize teachlab/outputs/pilot [--split test] [--metric score|passed|doc]
      --metric doc: CL-bench pass rate on rubrics that need document knowledge (by_type["doc"])
"""

import argparse
import glob
import os
from collections import defaultdict

from .schema import group_of, load_jsonl


def value(ev, metric):
    if metric in ("score", "passed"):
        return ev.get(metric)
    return (ev.get("by_type") or {}).get(metric)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--split", default="test")
    p.add_argument("--metric", default="score", help="score | passed | a rubric type such as doc / instruction")
    args = p.parse_args()

    cell = defaultdict(list)        # (source, label, k) -> values over scenarios x orders
    side = defaultdict(lambda: defaultdict(list))
    for path in sorted(glob.glob(os.path.join(args.run_dir, "*.jsonl"))):
        for r in load_jsonl(path):
            if "label" not in r:  # other files in the run dir, e.g. diagnose_memory's output
                continue
            ev = (r.get("eval") or {}).get(args.split)
            key = (group_of(r["scenario"]), r["label"])
            side[key]["memory_words"].append(r.get("memory_words", 0))
            if r.get("log"):
                side[key]["teacher_words"].append(r["log"].get("teacher_words", 0))
            if ev is None:
                continue
            v = value(ev, args.metric)
            if v is not None:
                cell[(group_of(r["scenario"]), r["label"], r["k"])].append(v)

    for source in sorted({s for s, _, _ in cell}):
        labels = sorted({l for s, l, _ in cell if s == source})
        ks = sorted({k for s, _, k in cell if s == source})
        print(f"\n## {source} - {args.split} {args.metric} (mean over scenarios x orders; n in brackets)\n")
        print("| condition | " + " | ".join(f"k={k}" for k in ks) + " | mem words | teacher words/session |")
        print("|---|" + "---|" * len(ks) + "---|---|")
        for label in labels:
            row = []
            for k in ks:
                v = cell.get((source, label, k))
                row.append(f"{sum(v) / len(v):.3f} ({len(v)})" if v else "")
            sd = side[(source, label)]
            mem = sum(sd["memory_words"]) / len(sd["memory_words"]) if sd["memory_words"] else 0
            tw = sum(sd["teacher_words"]) / len(sd["teacher_words"]) if sd["teacher_words"] else 0
            print(f"| {label} | " + " | ".join(row) + f" | {mem:.0f} | {tw:.0f} |")


if __name__ == "__main__":
    main()
