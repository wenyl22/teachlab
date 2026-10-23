"""GEPA over the student's notebook: one optimisation per course (scenario).

The text being optimised is the notebook the student answers with - the same memory as in every notebook condition.
Per course:
    trainset   stream items (teaching problems; only the first N with --train-items N). GEPA minibatches: the
               student attempts them with the current notebook, the teacher rewrites it from the graded attempts, and
               the rewrite is kept only if it scores higher on the same minibatch.
    valset     probe items (default) - GEPA selects candidates on these scores (only scores; the teacher never sees
               probe problems). With --val-from-stream N, the last N stream items are held back as the valset instead,
               so probe and test stay untouched, as in the other conditions.
    test       evaluated once at the end with --samples, like run.py's evaluations.
The teacher's rewrite prompt is the teacher_note prompt, with a minibatch of graded attempts instead of one session.
The copy check (--max-copy) and the word budget (--budget) apply as in teacher-written notes.

Writes <out>/<label>.jsonl in run.py's record format (k = 0, state.notebook, eval: probe + test) plus a "gepa" field
(every candidate, parents, validation scores, budget used), <out>/proposals_<label>.jsonl (every teacher call), and
<out>/usage.jsonl (token usage per role). Finished courses are skipped and unfinished ones resume from their GEPA state.

Requires `pip install gepa`. Example (Qwen3 served by vLLM):
    python -m teachlab.gepa_notes --scenarios teachlab/scenarios/rule_world_o6l3.jsonl --out teachlab/outputs/gepa_rw \
        --student qwen3-4b --student-url http://127.0.0.1:8101/v1 --teacher qwen3.8-27b \
        --teacher-url http://127.0.0.1:8102/v1 --student-extra '{"chat_template_kwargs": {"enable_thinking": false}}'
"""

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import gepa
from gepa.core.adapter import EvaluationBatch

from .graders import feedback
from .llm import add_model_args, model_from_args
from .schema import append_jsonl, items, load_jsonl, load_scenarios, problem_text
from .session import (Ctx, checked_chat, enforce_budget, evaluate, ngram_overlap, role_block, safe_grade,
                      student_answer, student_messages, words)

PROPOSE = """You are an expert tutor. Your student keeps a study notebook that you write for them. The notebook is the ONLY thing the student keeps: later they solve new problems of this kind with nothing but the notebook - no tutor and no document.

=== STUDENT'S CURRENT NOTEBOOK ===
{notebook}
=== END ===

The student just answered these practice problems using ONLY the notebook above. Each attempt comes with the grader's verdict.
=== THE STUDENT'S ATTEMPTS ===
{attempts}
=== END ===

Rewrite the notebook so that it best prepares this student for future problems of this kind, given what these attempts show about how the student uses the notebook and where it fails them. At most {budget} words. Output only the new notebook.
{role}
=== PRIVATE REFERENCE DOCUMENT ===
{knowledge}
=== END OF DOCUMENT ==="""

MAX_RESPONSE_CHARS = 6000  # per attempt shown to the teacher


class CourseLogger:
    """GEPA's default Logger (used when run_dir is set) swaps the process-wide sys.stdout for a tee and closes it when
    optimize() returns, which breaks every other course optimised in a parallel thread ("I/O operation on closed
    file"). This one only writes to its own file."""

    def __init__(self, path):
        self.f, self.lock = open(path, "a"), threading.Lock()

    def log(self, message, *args, **kwargs):
        with self.lock:
            self.f.write(f"{message}\n")
            self.f.flush()

    def close(self):
        self.f.close()


class NotebookAdapter:
    """GEPA adapter: candidate {"notebook": text} -> the student's scores on a batch of teachlab items."""

    def __init__(self, ctx, scn, args, log_path, label):
        self.ctx, self.scn, self.args = ctx, scn, args
        self.log_path, self.label, self.lock = log_path, label, threading.Lock()
        self.n_proposals = 0

    def cap(self, text):
        w = (text or "").split()
        return " ".join(w[:self.args.budget]) if len(w) > self.args.budget else (text or "")

    def evaluate(self, batch, candidate, capture_traces=False):
        state = {"notebook": self.cap(candidate["notebook"])}

        def one(item):
            response, _ = student_answer(self.ctx, student_messages(self.scn, item, state))
            g = safe_grade(self.ctx, item, response)
            return response, (g["score"] if g else 0.0), (feedback(item, g) if g else "(The grader is unavailable.)")

        with ThreadPoolExecutor(max_workers=max(1, min(self.args.eval_workers, len(batch)))) as ex:
            res = list(ex.map(one, batch))
        traj = [{"item": it, "response": r, "score": s, "feedback": f} for it, (r, s, f) in zip(batch, res)] \
            if capture_traces else None
        return EvaluationBatch(outputs=[r for r, _, _ in res], scores=[s for _, s, _ in res], trajectories=traj)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        recs = [{"problem": problem_text(t["item"]), "answer": (t["response"] or "")[-MAX_RESPONSE_CHARS:],
                 "feedback": t["feedback"], "score": t["score"]} for t in eval_batch.trajectories]
        return {c: recs for c in components_to_update}

    def propose_new_texts(self, candidate, reflective_dataset, components_to_update):
        recs = reflective_dataset["notebook"]
        attempts = "\n\n".join(
            f"--- Problem {i + 1} (score {r['score']:.2f}) ---\n{r['problem']}\n\n[Student's answer]\n{r['answer']}\n\n"
            f"[Grader]\n{r['feedback']}" for i, r in enumerate(recs))
        msgs = [{"role": "user", "content": PROPOSE.format(
            notebook=candidate["notebook"] or "(empty)", attempts=attempts, budget=self.args.budget,
            role=role_block(self.scn), knowledge=self.scn["knowledge"])}]
        t0 = time.time()
        text, violations = checked_chat(self.ctx, self.ctx.teacher, msgs, self.scn["knowledge"])
        text, truncated = enforce_budget(self.ctx.teacher, msgs, text, self.args.budget)
        with self.lock:
            self.n_proposals += 1
            append_jsonl({"scenario": self.scn["id"], "condition": self.label, "proposal": self.n_proposals,
                          "parent": candidate["notebook"], "attempts": recs, "new": text, "words": words(text),
                          "violations": violations, "truncated": truncated, "seconds": round(time.time() - t0, 1)},
                         self.log_path)
        return {"notebook": text}


def splits(scn, args):
    train = items(scn, "stream")
    if args.val_from_stream:
        n = min(args.val_from_stream, len(train) - 1)
        train, val = train[:-n], train[-n:]
    else:
        val = items(scn, "probe")
    if args.train_items:  # e.g. 10 = the problems a 10-session teaching run sees
        train = train[:args.train_items]
    return train, val


def run_course(ctx, scn, seed_text, args, label, out_path, log_path):
    train, val = splits(scn, args)
    adapter = NotebookAdapter(ctx, scn, args, log_path, label)
    run_dir = os.path.join(args.out, "gepa_runs", label, scn["id"].replace("/", "_"))
    os.makedirs(run_dir, exist_ok=True)
    logger = CourseLogger(os.path.join(run_dir, "run_log.txt"))
    # --max-rewrites caps the number of teacher proposals (e.g. to match teacher_note's sessions).
    stoppers = [lambda state: adapter.n_proposals >= args.max_rewrites] if args.max_rewrites else None
    t0 = time.time()
    try:
        res = gepa.optimize(
            seed_candidate={"notebook": seed_text}, trainset=train, valset=val, adapter=adapter,
            reflection_minibatch_size=args.minibatch, max_metric_calls=args.max_metric_calls, stop_callbacks=stoppers,
            candidate_selection_strategy="pareto", skip_perfect_score=True, perfect_score=1.0, use_merge=False,
            run_dir=run_dir, logger=logger, seed=args.seed, display_progress_bar=False, raise_on_exception=True)
    finally:
        logger.close()
    best = adapter.cap(res.best_candidate["notebook"])
    state = {"notebook": best}
    ev = {sp: r for sp in ("probe", "test") if (r := evaluate(ctx, scn, state, sp)) is not None}
    append_jsonl({
        "scenario": scn["id"], "source": scn["source"], "condition": label, "label": label, "order": 0,
        "student": ctx.student.name, "teacher": ctx.teacher.name, "judge": ctx.judge.name, "memory_policy": "notebook",
        "budget": args.budget, "teacher_until": None, "k": 0, "state": state, "memory_words": words(best),
        "memory_copy_rate": ngram_overlap(best, scn["knowledge"]), "eval": ev,
        "gepa": {"seed": args.seed_from, "val": "stream" if args.val_from_stream else "probe",
                 "n_train": len(train), "n_val": len(val), "best_idx": res.best_idx,
                 "n_candidates": len(res.candidates), "val_scores": res.val_aggregate_scores, "parents": res.parents,
                 "candidates": [c["notebook"] for c in res.candidates], "total_metric_calls": res.total_metric_calls,
                 "n_proposals": adapter.n_proposals, "minutes": round((time.time() - t0) / 60, 1),
                 "max_metric_calls": args.max_metric_calls, "max_rewrites": args.max_rewrites,
                 "minibatch": args.minibatch}}, out_path)
    return scn["id"], len(res.candidates), res.val_aggregate_scores[res.best_idx], (ev.get("test") or {}).get("score")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenarios", nargs="+", required=True)
    p.add_argument("--ids", default=None, help="Comma-separated scenario ids or id prefixes to keep")
    p.add_argument("--seed-from", default="none",
                   help="'none' (empty notebook) or a run dir whose summary.jsonl gives each course's seed notes")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--max-metric-calls", type=int, default=300, help="GEPA budget per course, in student rollouts")
    p.add_argument("--max-rewrites", type=int, default=None, help="Also stop after this many teacher rewrites")
    p.add_argument("--minibatch", type=int, default=4, help="Teaching problems per GEPA minibatch")
    p.add_argument("--val-from-stream", type=int, default=0,
                   help="Hold back the last N stream items as GEPA's valset instead of the probe split")
    p.add_argument("--train-items", type=int, default=0,
                   help="Use only the first N (remaining) stream items as GEPA's trainset; the label gets _t<N>")
    p.add_argument("--budget", type=int, default=1500, help="Notebook limit B, in words")
    p.add_argument("--max-copy", type=float, default=0.25,
                   help="Max fraction of the notebook copied verbatim from the document (rewritten once if over)")
    p.add_argument("--samples", type=int, default=3, help="Student samples per item in the final probe/test eval")
    p.add_argument("--eval-workers", type=int, default=8, help="Parallel student calls within one evaluation")
    p.add_argument("--workers", type=int, default=10, help="Courses optimised in parallel")
    p.add_argument("--seed", type=int, default=0)
    add_model_args(p, "student", "gpt-5-mini")
    add_model_args(p, "teacher", "gpt-5.5")
    add_model_args(p, "judge", "gpt-5.1")
    args = p.parse_args()
    ctx = Ctx(model_from_args(args, "student"), model_from_args(args, "teacher"), model_from_args(args, "judge"), args)

    scns = load_scenarios(args.scenarios)
    if args.ids:
        want = args.ids.split(",")
        scns = [s for s in scns if any(s["id"] == w or s["id"].startswith(w) for w in want)]
    seeds = {}
    if args.seed_from != "none":
        seeds = {r["scenario"]: r["state"]["notebook"] for r in load_jsonl(os.path.join(args.seed_from, "summary.jsonl"))}
    label = "gepa_" + ("empty" if args.seed_from == "none" else "summary") + \
        (f"_r{args.max_rewrites}" if args.max_rewrites else "") + ("_sval" if args.val_from_stream else "") + \
        (f"_t{args.train_items}" if args.train_items else "")
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{label}.jsonl")
    log_path = os.path.join(args.out, f"proposals_{label}.jsonl")
    done = {r["scenario"] for r in load_jsonl(out_path)}
    todo = [s for s in scns if s["id"] not in done]
    print(f"{label}: {len(todo)} courses to optimise -> {out_path}", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_course, ctx, s, seeds.get(s["id"], ""), args, label, out_path, log_path) for s in todo]
            for f in futs:
                try:
                    sid, n, val, test = f.result()
                    print(f"   {sid}: {n} candidates, best val {val:.2f}, test {test if test is None else round(test, 2)}",
                          flush=True)
                except Exception as e:
                    print(f"   ❌ failed: {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        usage = {role: {"model": m.name, **m.usage} for role, m in
                 (("student", ctx.student), ("teacher", ctx.teacher), ("judge", ctx.judge))}
        append_jsonl({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "conditions": [label], "usage": usage},
                     os.path.join(args.out, "usage.jsonl"))


if __name__ == "__main__":
    main()
