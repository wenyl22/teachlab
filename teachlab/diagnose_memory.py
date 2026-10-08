"""Same teaching, different memory: is the student's own notebook the bottleneck?

For each scenario of a finished run, take all the sessions of the run (dialogues, final attempts, feedback)
and re-test the student with the memory rebuilt in other ways:
    notebook     the run's own final state (student-written notebook, rewritten after each session)
    transcript   the full record of every session, unbounded (the pilot's memory)
    teacher_nb   the teacher compresses the same records into a notebook of at most --budget words, WITHOUT
                 the document, so it can only keep what was actually taught
Only the memory differs; the sessions are identical.

Usage:
    python -m teachlab.diagnose_memory teachlab/outputs/clbench_core/tutor.jsonl --scenarios teachlab/scenarios/clbench.jsonl
    (add --ks 1,2,4,6,8,10 to rebuild the memory after each of those sessions, for a learning curve)
"""

import argparse
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from .llm import add_model_args, model_from_args
from .schema import append_jsonl, load_jsonl, load_scenarios
from .session import Ctx, enforce_budget, evaluate, memory_text, render_session, words

COMPRESS = """Below is the full record of a student's study sessions with a tutor in an unfamiliar domain. The student will later have to solve new problems of this kind with nothing but a notebook - no tutor, no document, no record of these sessions.

Write that notebook from what was taught in these sessions: the rules, procedures, facts, conventions and pitfalls the student will need. At most {budget} words. Output only the notebook.

=== STUDY SESSIONS ===
{sessions}
=== END ==="""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_file", help="A finished run's condition file, e.g. teachlab/outputs/<run>/tutor.jsonl")
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--memories", default="transcript,teacher_nb")
    p.add_argument("--ks", default=None,
                   help="Comma-separated k: rebuild the memory from the first k sessions only (default: all sessions)")
    p.add_argument("--out", default=None, help="Default: <run dir>/diagnose_<condition>.jsonl")
    add_model_args(p, "student", "gpt-5-mini")
    add_model_args(p, "teacher", "gpt-5.5")
    add_model_args(p, "judge", "gpt-5.1")
    p.add_argument("--budget", type=int, default=1500)
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--eval-splits", default="test")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--eval-workers", type=int, default=16)
    args = p.parse_args()

    scenarios = {s["id"]: s for s in load_scenarios(args.scenarios)}
    runs = defaultdict(list)
    for r in load_jsonl(args.run_file):
        if r["order"] == 0:
            runs[r["scenario"]].append(r)
    out = args.out or os.path.join(os.path.dirname(args.run_file),
                                   f"diagnose_{os.path.splitext(os.path.basename(args.run_file))[0]}.jsonl")
    done = {(r["scenario"], r["memory"], r["k"]) for r in load_jsonl(out)}
    ctx = Ctx(model_from_args(args, "student"), model_from_args(args, "teacher"), model_from_args(args, "judge"), args)

    def job(sid, kind, k):
        scn, recs = scenarios[sid], sorted(runs[sid], key=lambda r: r["k"])
        sessions = [r["session"] for r in recs if 0 < r["k"] <= k]
        if kind == "transcript":
            state = {"transcript": sessions}
        else:
            text = "\n\n".join(f"--- Session {i + 1} ---\n{render_session(s, 'Student', 'Tutor')}"
                               for i, s in enumerate(sessions))
            msgs = [{"role": "user", "content": COMPRESS.format(budget=args.budget, sessions=text)}]
            note, _ = enforce_budget(ctx.teacher, msgs, ctx.teacher.chat(msgs), args.budget)
            state = {"notebook": note}
        ev = {sp: evaluate(ctx, scn, state, sp) for sp in args.eval_splits.split(",")}
        append_jsonl({"scenario": sid, "memory": kind, "k": k, "memory_words": words(memory_text(state)),
                      "state": state, "eval": {k: v for k, v in ev.items() if v}}, out)

    last_k = {sid: max(r["k"] for r in recs) for sid, recs in runs.items()}
    jobs = [(sid, kind, k) for sid in runs for kind in args.memories.split(",")
            for k in ([int(x) for x in args.ks.split(",")] if args.ks else [last_k[sid]])
            if k <= last_k[sid] and (sid, kind, k) not in done]
    print(f"{len(jobs)} jobs -> {out}")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for f in [ex.submit(job, *j) for j in jobs]:
            try:
                f.result()
            except Exception as e:
                print(f"   ❌ failed: {str(e)[:300]}")

    # Table: the run's own final notebook vs the rebuilt memories, per scenario.
    rows = defaultdict(dict)
    for sid, recs in runs.items():
        last = max(recs, key=lambda r: r["k"])
        if last.get("eval"):
            rows[sid]["notebook"] = (last["eval"]["test"], last["memory_words"])
    for r in load_jsonl(out):
        if r["k"] == last_k.get(r["scenario"]):  # the table compares final memories; per-k results are in the file
            rows[r["scenario"]][r["memory"]] = (r["eval"]["test"], r["memory_words"])
    kinds = ["notebook"] + args.memories.split(",")
    for metric in ("doc", "instruction", "score"):
        print(f"\n## test {metric}\n\n| scenario | " + " | ".join(kinds) + " |\n|---|" + "---|" * len(kinds))
        means = defaultdict(list)
        for sid in sorted(rows):
            cells = []
            for k in kinds:
                if k not in rows[sid]:
                    cells.append("")
                    continue
                ev, w = rows[sid][k]
                v = ev["score"] if metric == "score" else (ev.get("by_type") or {}).get(metric)
                if v is not None:
                    means[k].append(v)
                cells.append(f"{v:.2f} ({w}w)" if v is not None else "")
            print(f"| {sid} | " + " | ".join(cells) + " |")
        print("| **mean** | " + " | ".join(f"**{sum(means[k]) / len(means[k]):.3f}**" if means[k] else "" for k in kinds) + " |")


if __name__ == "__main__":
    main()
