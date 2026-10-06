"""One study session under each condition, the student's memory, and the solo-ability test.

The student's state is what it carries between sessions:
    {"notebook": str}      - a notebook it rewrites itself after every session, at most --budget words (v2 setup)
    {"transcript": [...]}  - the full record of every session, unbounded (the CL-bench pilot setup)
Solo ability A(m) is measured by answering held-out items with only that state: no teacher, no document,
no record of the dialogue (unless the state is a transcript).

Conditions (CONDITIONS below):
    none          no study, empty memory                                   
    doc           the document itself as memory, open book                 
    summary       a teacher-written note of at most --budget words, once
    self          attempt -> grader feedback -> student updates its memory  
    self_retry    the same with --retries extra attempts per session        
    self_doc      attempt with the document open -> feedback -> update     
    critique      attempt -> feedback -> one teacher critique -> update
    tutor         teacher-led dialogue -> final attempt -> feedback -> update
    placebo       tutor without the document
    teacher_note  attempt -> feedback -> the teacher rewrites the student's notebook
With --teacher-until W, sessions after W run as `self` (the teacher is withdrawn).
"""

from concurrent.futures import ThreadPoolExecutor

from . import prompts as P
from .graders import feedback, grade
from .llm import NON_RETRYABLE
from .schema import items, problem_text

CONDITIONS = {
    "none": {"oneshot": "none"},
    "doc": {"oneshot": "doc"},
    "summary": {"oneshot": "summary"},
    "self": {"teacher": None},
    "self_retry": {"teacher": None, "retry": True},
    "self_doc": {"teacher": None, "open_book": True},
    "critique": {"teacher": "critique"},
    "tutor": {"teacher": "tutor"},
    "placebo": {"teacher": "placebo"},
    "teacher_note": {"teacher": "note"},
}


class Ctx:
    """Models and settings shared by every session of a run."""

    def __init__(self, student, teacher, judge, args):
        self.student, self.teacher, self.judge, self.args = student, teacher, judge, args


def words(text):
    return len((text or "").split())


def ngram_overlap(text, reference, n=8):
    """Fraction of text's word n-grams that appear verbatim in reference (copying detector)."""
    tw, rw = (text or "").split(), (reference or "").split()
    grams = {tuple(tw[i:i + n]) for i in range(len(tw) - n + 1)}
    if not grams:
        return 0.0
    ref = {tuple(rw[i:i + n]) for i in range(len(rw) - n + 1)}
    return len(grams & ref) / len(grams)


# --- memory -----------------------------------------------------------------------------------

def empty_state(args):
    return {"transcript": []} if args.memory == "transcript" else {"notebook": ""}


def render_session(rec, you="You", other="Tutor"):
    parts = [f"Example problem:\n{rec['problem']}"]
    for s, t in rec.get("dialogue", []):
        parts.append(f"[{other if s == 'teacher' else you}]\n{t}")
    for i, a in enumerate(rec.get("attempts", [])):
        label = "final attempt" if rec.get("dialogue") else f"attempt {i + 1}"
        parts.append(f"[{you}: {label}]\n{a['response']}\n\n[Grader]\n{a['feedback']}")
    if rec.get("critique"):
        parts.append(f"[{other}: comments on {'your' if you == 'You' else 'the'} attempt]\n{rec['critique']}")
    return "\n\n".join(parts)


def memory_text(state, you="You", other="Tutor"):
    if "notebook" in state:
        return state["notebook"]
    return "\n\n".join(f"--- Session {i + 1} ---\n{render_session(r, you, other)}"
                       for i, r in enumerate(state["transcript"]))


def memory_block(state):
    text = memory_text(state)
    if not text.strip():
        return ""
    kind, title = ("notebook", "NOTEBOOK") if "notebook" in state else ("record", "STUDY RECORD")
    return P.MEMORY_BLOCK.format(kind=kind, title=title, memory=text)


def student_messages(scn, item, state, open_book=False):
    system = scn["student_system"] + memory_block(state)
    if open_book:
        system += P.DOC_BLOCK.format(knowledge=scn["knowledge"])
    return [{"role": "system", "content": system}] + [dict(m) for m in item["messages"]]


def enforce_budget(model, messages, text, budget):
    """One request to shorten an over-long notebook, then a hard cut at `budget` words."""
    if words(text) > budget * 1.1:
        text = model.chat(messages + [{"role": "assistant", "content": text},
                                      {"role": "user", "content": P.SHORTEN.format(words=words(text), budget=budget)}])
    w = text.split()
    return (" ".join(w[:budget]), True) if len(w) > budget else (text, False)


def update_memory(ctx, scn, state, rec, writer):
    if "transcript" in state:
        return {"transcript": state["transcript"] + [rec]}, {}
    budget = ctx.args.budget
    if writer == "teacher":
        msgs = [{"role": "user", "content": P.TEACHER_NOTE_UPDATE.format(
            notebook=state["notebook"] or "(empty)", session=render_session(rec, "Student", "You"),
            budget=budget, role=role_block(scn), knowledge=scn["knowledge"])}]
        text, violations = checked_chat(ctx, ctx.teacher, msgs, scn["knowledge"])
        model = ctx.teacher
    else:
        msgs = [{"role": "system", "content": scn["student_system"]},
                {"role": "user", "content": P.NOTEBOOK_UPDATE.format(
                    notebook=state["notebook"] or "(empty)", session=render_session(rec), budget=budget)}]
        text, violations = ctx.student.chat(msgs), []
        model = ctx.student
    text, truncated = enforce_budget(model, msgs, text, budget)
    return {"notebook": text}, {"notebook_truncated": truncated, "note_violations": violations}


# --- teacher helpers --------------------------------------------------------------------------

def role_block(scn):
    return P.STUDENT_ROLE.format(student_system=scn["student_system"])


def rule_problems(text, knowledge, args, max_words=None):
    problems = []
    copy = ngram_overlap(text, knowledge)
    if copy > args.max_copy:
        problems.append(f"{copy:.0%} of it is copied verbatim from the document (limit {args.max_copy:.0%})")
    if max_words and words(text) > max_words * 1.2:
        problems.append(f"it is {words(text)} words long (limit {max_words})")
    return problems


def checked_chat(ctx, model, messages, knowledge, max_words=None):
    """Generate; if the text breaks the copying/length rules, ask for one rewrite. Returns (text, violations)."""
    draft = model.chat(messages)
    problems = rule_problems(draft.replace(P.END_TOKEN, ""), knowledge, ctx.args, max_words)
    if not problems:
        return draft, []
    rewritten = model.chat(messages + [{"role": "assistant", "content": draft},
                                       {"role": "user", "content": P.REWRITE.format(problems="; ".join(problems))}])
    if P.END_TOKEN in draft and P.END_TOKEN not in rewritten:
        rewritten = rewritten.rstrip() + "\n" + P.END_TOKEN
    still = rule_problems(rewritten.replace(P.END_TOKEN, ""), knowledge, ctx.args, max_words)
    return rewritten, [{"draft_problems": problems, "after_rewrite": still}]


def flip(dialogue, me):
    return [{"role": "assistant" if s == me else "user", "content": t} for s, t in dialogue]


def teacher_view_of_memory(state):
    text = memory_text(state, you="Student", other="You (tutor)")
    if not text.strip():
        return ""
    title = "STUDENT'S CURRENT NOTEBOOK" if "notebook" in state else "PREVIOUS SESSIONS"
    return f"\n=== {title} ===\n{text}\n=== END ===\n"


# --- session steps ----------------------------------------------------------------------------

def safe_grade(ctx, item, response):
    """grade(), but a judge that keeps failing yields None instead of killing the whole run."""
    try:
        return grade(item, response, ctx.judge)
    except Exception as e:
        print(f"   ⚠️ grading {item['id']} failed: {str(e)[:150]}")
        return None


def student_answer(ctx, messages):
    """The student's reply, or ("", error) if the request can never succeed - e.g. a transcript memory that no
    longer fits the context window. That answer is graded as empty and the error recorded, rather than killing
    the whole scenario x condition job. Transient failures (server down after all retries) still raise, so an
    outage stops the job (which then resumes) instead of being scored as wrong answers."""
    try:
        return ctx.student.chat(messages), None
    except NON_RETRYABLE as e:
        return "", f"{type(e).__name__}: {str(e)[:300]}"


def attempt_and_grade(ctx, scn, item, messages):
    response, error = student_answer(ctx, messages)
    result = safe_grade(ctx, item, response)
    out = {"response": response}
    if error:
        out["generation_error"] = error
    if result is None:
        return {**out, "score": None, "passed": False, "feedback": "(The grader is unavailable.)"}
    return {**out, "score": result["score"], "passed": result["passed"], "feedback": feedback(item, result)}


def attempts_phase(ctx, scn, item, state, open_book, retry):
    msgs = student_messages(scn, item, state, open_book)
    out = [attempt_and_grade(ctx, scn, item, msgs)]
    for _ in range(ctx.args.retries if retry else 0):
        if out[-1]["passed"]:
            break
        msgs = msgs + [{"role": "assistant", "content": out[-1]["response"]},
                       {"role": "user", "content": P.RETRY.format(feedback=out[-1]["feedback"])}]
        out.append(attempt_and_grade(ctx, scn, item, msgs))
    return out


def critique_phase(ctx, scn, item, state, rec):
    system = P.CRITIQUE_SYSTEM.format(max_words=ctx.args.max_words, role=role_block(scn),
                                      student_memory=teacher_view_of_memory(state), knowledge=scn["knowledge"])
    user = render_session({"problem": rec["problem"], "attempts": rec["attempts"]}, "Student", "You")
    return checked_chat(ctx, ctx.teacher, [{"role": "system", "content": system}, {"role": "user", "content": user}],
                        scn["knowledge"], ctx.args.max_words)


def tutor_phase(ctx, scn, item, state, placebo):
    a = ctx.args
    problem = problem_text(item)
    fmt = dict(max_words=a.max_words, end=P.END_TOKEN, max_rounds=a.max_rounds, role=role_block(scn),
               student_memory=teacher_view_of_memory(state), problem=problem)
    teacher_sys = {"role": "system", "content": P.PLACEBO_SYSTEM.format(**fmt) if placebo
                   else P.TUTOR_SYSTEM.format(**fmt, knowledge=scn["knowledge"])}
    student_sys = {"role": "system", "content": P.STUDENT_IN_SESSION.format(
        student_system=scn["student_system"], memory=memory_block(state), problem=problem)}
    start = {"role": "user", "content": "(The student has joined the session. Begin.)"}
    dialogue, violations, ended_by = [], [], "max_rounds"
    for r in range(a.max_rounds):
        text, v = checked_chat(ctx, ctx.teacher, [teacher_sys, start] + flip(dialogue, "teacher"),
                               "" if placebo else scn["knowledge"], a.max_words)
        violations += [{"round": r + 1, **x} for x in v]
        if P.END_TOKEN in text:
            text = text.replace(P.END_TOKEN, "").strip()
            if text:
                dialogue.append(("teacher", text))
            ended_by = "teacher"
            break
        dialogue.append(("teacher", text))
        reply, error = student_answer(ctx, [student_sys] + flip(dialogue, "student"))
        if error:  # the student cannot answer (e.g. context overflow): end the dialogue, keep the record
            ended_by = "student_error"
            violations.append({"round": r + 1, "student_error": error})
            break
        dialogue.append(("student", reply))
    # The final attempt is in the session's own context (the dialogue), then graded like any attempt.
    final = attempt_and_grade(ctx, scn, item, [student_sys] + flip(dialogue, "student")
                              + [{"role": "user", "content": P.FINAL_ATTEMPT}])
    return dialogue, final, {"ended_by": ended_by, "violations": violations}


def run_session(ctx, scn, cond, item, state, t):
    """Session t on stream item `item`. Returns (new state, session record, log)."""
    spec = dict(CONDITIONS[cond])
    withdrawn = ctx.args.teacher_until is not None and t > ctx.args.teacher_until
    teacher = None if withdrawn else spec.get("teacher")
    rec = {"item_id": item["id"], "problem": problem_text(item)}
    log = {"teacher_present": teacher is not None}

    if teacher in ("tutor", "placebo"):
        dialogue, final, info = tutor_phase(ctx, scn, item, state, teacher == "placebo")
        rec.update(dialogue=dialogue, attempts=[final])
        log.update(info)
    else:
        rec["attempts"] = attempts_phase(ctx, scn, item, state, spec.get("open_book") and not withdrawn,
                                         spec.get("retry"))
        if teacher == "critique":
            rec["critique"], log["violations"] = critique_phase(ctx, scn, item, state, rec)

    teacher_text = "\n".join([t for s, t in rec.get("dialogue", []) if s == "teacher"] + [rec.get("critique", "")])
    new_state, info = update_memory(ctx, scn, state, rec, "teacher" if teacher == "note" else "student")
    log.update(info)
    log.update({
        "first_attempt_score": rec["attempts"][0]["score"], "last_attempt_score": rec["attempts"][-1]["score"],
        "n_attempts": len(rec["attempts"]), "teacher_words": words(teacher_text),
        "generation_errors": sum(1 for a in rec["attempts"] if a.get("generation_error")),
        "teacher_msgs": sum(1 for s, _ in rec.get("dialogue", []) if s == "teacher") + bool(rec.get("critique")),
        "teacher_copy_rate": ngram_overlap(teacher_text, scn["knowledge"]),
    })
    return new_state, rec, log


def oneshot_state(ctx, scn, kind):
    """State for the conditions with no sessions. Returns (state, open_book, log)."""
    if kind == "none":
        return {"notebook": ""}, False, {}
    if kind == "doc":
        return {"notebook": ""}, True, {}
    msgs = [{"role": "user", "content": P.SUMMARY.format(budget=ctx.args.budget, role=role_block(scn),
                                                         knowledge=scn["knowledge"])}]
    text, violations = checked_chat(ctx, ctx.teacher, msgs, scn["knowledge"])
    text, truncated = enforce_budget(ctx.teacher, msgs, text, ctx.args.budget)
    return {"notebook": text}, False, {"note_violations": violations, "notebook_truncated": truncated}


# --- solo ability -----------------------------------------------------------------------------

def evaluate(ctx, scn, state, split, open_book=False):
    """Every item of `split` x --samples, answered with only `state`. Returns a summary and per-item results."""
    todo = items(scn, split)
    if not todo:
        return None
    n = ctx.args.samples

    def one(item):
        response, error = student_answer(ctx, student_messages(scn, item, state, open_book))
        return response, safe_grade(ctx, item, response), error

    jobs = [it for it in todo for _ in range(n)]
    with ThreadPoolExecutor(max_workers=max(1, min(ctx.args.eval_workers, len(jobs)))) as ex:
        results = list(ex.map(one, jobs))
    # Failed generations count as empty (wrong) answers; their number is reported as generation_errors.
    gen_errors = [e for _, _, e in results if e]
    outs = [(r, g) for r, g, _ in results]
    per_item, by_type, errors = [], {}, 0
    for i, it in enumerate(todo):
        graded = [(r, g) for r, g in outs[i * n:(i + 1) * n] if g is not None]
        errors += n - len(graded)
        if not graded:
            continue
        samples = graded
        entry = {"id": it["id"], "score": sum(g["score"] for _, g in samples) / len(samples),
                 "passed": sum(g["passed"] for _, g in samples) / len(samples),
                 "scores": [g["score"] for _, g in samples], "output": samples[0][0]}
        if "status" in samples[0][1]:
            entry["statuses"] = [g["status"] for _, g in samples]  # per-rubric pass/fail, for later analysis
        types = {}
        for _, g in samples:
            for k, v in (g.get("by_type") or {}).items():
                types.setdefault(k, []).append(v)
        if types:
            entry["by_type"] = {k: sum(v) / len(v) for k, v in types.items()}
            for k, v in entry["by_type"].items():
                by_type.setdefault(k, []).append(v)
        per_item.append(entry)
    gen = {"generation_errors": len(gen_errors)}
    if gen_errors:
        gen["generation_error_example"] = gen_errors[0]
    if not per_item:
        return {"score": None, "passed": None, "n_items": 0, "grading_errors": errors, **gen, "items": []}
    summary = {"score": sum(e["score"] for e in per_item) / len(per_item),
               "passed": sum(e["passed"] for e in per_item) / len(per_item), "n_items": len(per_item),
               "grading_errors": errors, **gen}
    if by_type:
        summary["by_type"] = {k: sum(v) / len(v) for k, v in by_type.items()}
    return {**summary, "items": per_item}
