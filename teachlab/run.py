"""Learning curves of solo ability across sessions, for any mix of scenarios and conditions.

For every scenario x condition x session order: test the student at k = 0 (empty memory), then run one
session per stream item and, at each k in --eval-at, test the student with only its memory on the probe
and/or test items. Each k is appended to the output as soon as it is done, with the full memory state, so
an interrupted run resumes from the last recorded k.

Usage:
    python -m teachlab.run --scenarios teachlab/scenarios/sebench_f8.jsonl teachlab/scenarios/clbench.jsonl \
        --conditions none,doc,summary,self,tutor --sessions 10 --run-name pilot
"""

import argparse
import os
import time
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from . import graders
from .llm import add_model_args, model_from_args
from .schema import append_jsonl, group_of, items, load_jsonl, load_scenarios
from .session import (CONDITIONS, Ctx, empty_state, evaluate, memory_text, ngram_overlap, oneshot_state, run_session,
                      words)


def stream_for(scn, order, sessions, cycle):
    stream = items(scn, "stream")
    if order:
        stream = stream[:]
        random.Random(f"{scn['id']}/{order}").shuffle(stream)
    if cycle and stream:
        stream = [stream[i % len(stream)] for i in range(sessions)]
    return stream[:sessions]


def eval_record(ctx, scn, state, open_book=False):
    out = {}
    for split in ctx.args.eval_splits.split(","):
        r = evaluate(ctx, scn, state, split, open_book)
        if r is not None:
            out[split] = r
    return out


def state_stats(state, knowledge):
    text = memory_text(state)
    return {"memory_words": words(text), "memory_copy_rate": ngram_overlap(text, knowledge)}


def run_job(ctx, scn, cond, label, order, out_path, existing):
    a = ctx.args
    base = {"scenario": scn["id"], "source": scn["source"], "condition": cond, "label": label, "order": order,
            "student": ctx.student.name, "teacher": ctx.teacher.name, "judge": ctx.judge.name,
            "memory_policy": a.memory, "budget": a.budget, "teacher_until": a.teacher_until}
    spec = CONDITIONS[cond]
    if "oneshot" in spec:
        if existing:
            return
        state, open_book, log = oneshot_state(ctx, scn, spec["oneshot"])
        append_jsonl({**base, "k": 0, "state": state, "log": log, **state_stats(state, scn["knowledge"]),
                      "eval": eval_record(ctx, scn, state, open_book)}, out_path)
        return

    stream = stream_for(scn, order, a.sessions, a.cycle)
    eval_at = {int(x) for x in a.eval_at.split(",")} | {len(stream)}
    last = max(existing, key=lambda r: r["k"]) if existing else None
    if last is None:
        state = empty_state(a)
        append_jsonl({**base, "k": 0, "state": state, **state_stats(state, scn["knowledge"]),
                      "eval": eval_record(ctx, scn, state)}, out_path)
        start = 1
    else:
        state, start = last["state"], last["k"] + 1
    for k in range(start, len(stream) + 1):
        state, rec, log = run_session(ctx, scn, cond, stream[k - 1], state, k)
        append_jsonl({**base, "k": k, "item_id": stream[k - 1]["id"], "session": rec, "log": log, "state": state,
                      **state_stats(state, scn["knowledge"]),
                      "eval": eval_record(ctx, scn, state) if k in eval_at else {}}, out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", required=True, help="Scenario jsonl files (from teachlab.adapters.*)")
    p.add_argument("--ids", default=None, help="Comma-separated scenario ids (or id prefixes) to keep")
    p.add_argument("--limit", type=int, default=None, help="Keep at most this many scenarios per scenario set (id prefix)")
    p.add_argument("--conditions", default="none,doc,summary,self,tutor")
    p.add_argument("--output-dir", default="teachlab/outputs")
    p.add_argument("--run-name", default="dev")
    add_model_args(p, "student", "gpt-5-mini")
    add_model_args(p, "teacher", "gpt-5.5")
    add_model_args(p, "judge", "gpt-5.1")
    p.add_argument("--memory", choices=["notebook", "transcript"], default="notebook")
    p.add_argument("--budget", type=int, default=1500, help="Notebook / summary limit B, in words")
    p.add_argument("--sessions", type=int, default=10, help="Horizon H (capped by the stream length unless --cycle)")
    p.add_argument("--cycle", action="store_true", help="Repeat the stream to reach --sessions")
    p.add_argument("--orders", type=int, default=1, help="Session orders per scenario (0 = file order, then shuffles)")
    p.add_argument("--teacher-until", type=int, default=None, help="Withdraw the teacher after this session")
    p.add_argument("--retries", type=int, default=2, help="Extra attempts per session in self_retry")
    p.add_argument("--max-rounds", type=int, default=8, help="Max teacher messages per tutor session")
    p.add_argument("--max-words", type=int, default=300, help="Max words per teacher message")
    p.add_argument("--max-copy", type=float, default=0.25,
                   help="Max 8-gram overlap of teacher text with the document (1.0 disables the check)")
    p.add_argument("--eval-at", default="0,1,2,4,6,8,10,15,20")
    p.add_argument("--eval-splits", default="probe,test")
    p.add_argument("--samples", type=int, default=3, help="Student samples per evaluated item")
    p.add_argument("--workers", type=int, default=8, help="Parallel scenario x condition x order jobs")
    p.add_argument("--eval-workers", type=int, default=16, help="Parallel calls within one evaluation")
    p.add_argument("--exec-url", default=None,
                   help="Run generated code in SE-Bench's Docker sandbox at this URL (e.g. http://localhost:8111/run) "
                        "instead of a local subprocess")
    args = p.parse_args()
    if args.exec_url:
        graders.EXEC_URL = args.exec_url

    scenarios = load_scenarios(args.scenarios)
    if args.ids:
        wanted = args.ids.split(",")
        scenarios = [s for s in scenarios if any(s["id"] == w or s["id"].startswith(w) for w in wanted)]
    if args.limit:
        per, kept = defaultdict(int), []
        for s in scenarios:
            per[group_of(s["id"])] += 1
            if per[group_of(s["id"])] <= args.limit:
                kept.append(s)
        scenarios = kept
    conds = args.conditions.split(",")
    for c in conds:
        assert c in CONDITIONS, f"unknown condition {c}; choose from {', '.join(CONDITIONS)}"

    ctx = Ctx(model_from_args(args, "student"), model_from_args(args, "teacher"), model_from_args(args, "judge"), args)
    run_dir = os.path.join(args.output_dir, args.run_name)
    jobs = []
    for cond in conds:
        oneshot = "oneshot" in CONDITIONS[cond]
        label = cond + (f"@w{args.teacher_until}" if args.teacher_until is not None and not oneshot else "")
        out_path = os.path.join(run_dir, f"{label}.jsonl")
        existing = defaultdict(list)
        for r in load_jsonl(out_path):
            existing[(r["scenario"], r["order"])].append(r)
        for s in scenarios:
            for order in range(1 if oneshot else args.orders):
                have = existing.get((s["id"], order), [])
                target = 0 if oneshot else len(stream_for(s, order, args.sessions, args.cycle))
                if not have or max(r["k"] for r in have) < target:
                    jobs.append((s, cond, label, order, out_path, have))
    print(f"{len(scenarios)} scenarios, {len(jobs)} scenario x condition x order jobs to run or resume -> {run_dir}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_job, ctx, *job): (job[0]["id"], job[2], job[3]) for job in jobs}
            for f in tqdm(as_completed(futures), total=len(futures)):
                try:
                    f.result()
                except Exception as e:
                    print(f"   ❌ {futures[f]} failed: {str(e)[:300]}")
    finally:
        # Token usage of this invocation, one line per role; a resumed run appends another line.
        usage = {role: {"model": m.name, **m.usage} for role, m in
                 (("student", ctx.student), ("teacher", ctx.teacher), ("judge", ctx.judge))}
        append_jsonl({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "conditions": conds, "usage": usage},
                     os.path.join(run_dir, "usage.jsonl"))
        for role, u in usage.items():
            if u["calls"]:
                print(f"   {role} {u['model']}: {u['calls']} calls, {u['prompt_tokens']} prompt "
                      f"({u['cached_tokens']} cached), {u['completion_tokens']} completion "
                      f"({u['reasoning_tokens']} reasoning) tokens")


if __name__ == "__main__":
    main()
