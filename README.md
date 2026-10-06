# teachlab: multi-session teaching experiments across benchmarks

One protocol for every scenario source: after the student has studied over several sessions, how well can it do **alone**, with no teacher and no document (solo ability)?

```

adapters/*  
-> scenarios/*.jsonl (unified format)
-> run.py (condition × session × evaluation) 
-> outputs/<run>/*.jsonl
-> summarize.py

```

## Unified scenario format

One scenario is one course. The format is defined in [schema.py](teachlab/schema.py). The main fields:

- `knowledge`: the private document. The teacher always sees it; the student sees it only under the `doc` and `self_doc` conditions.

- `student_system`: the system prompt the student answers under.

- `items`: each item has a `split`.
  - `stream`: one item per session;
  - `probe`: tested after each session and is disjoint from `test`;
  - `test`: the sealed final evaluation.

- Each item carries its own `grader`, of one of three types:
  - `rubric`: an LLM judge using CL-bench's judge prompt, with optional breakdown by doc / instruction / general rubrics;
  - `exec`: runs the code against test cases (SE-Bench);
  - `exact`: exact match on the final answer.

## Sources

| adapter | command | size | grading |
|---|---|---|---|
| CL-bench (selected courses) | `python -m teachlab.adapters.clbench --contexts-from data/clbench/selected.jsonl` | 7–10 stream and ~3 test items per course; documents ~4–6k words | rubric |
| SE-Bench (the zwc library) | `bash teachlab/download_sebench.sh && python -m teachlab.adapters.sebench --funcs 8` | 7–8 functions, 14–19 stream, 4–5 probe and 10–13 test items per course; documents ~2–4k words | exec |
| SE-Bench, full-library docs | the command above with `--doc-scope library` | document ~97k words, too large for the notebook | exec |
| Generated rule worlds | `python -m teachlab.adapters.rule_world --ops 6 --length 3` | any number of items; `--ops` sets the knowledge volume, `--length` the difficulty of applying it | exact |

**Adding a source**: write an adapter that outputs jsonl in the format above. Hand-written scenarios can also be written directly as jsonl.

## Conditions

Full definitions are in [session.py](teachlab/session.py).

### Core: five conditions, two questions

| condition | what it is | role |
|---|---|---|
| `none` | no study, empty memory (S0) | floor |
| `doc` | the document in context, open book (S1) | ceiling |
| `summary` | the teacher writes one note of at most B words, once | "just tell the rules" baseline |
| `self` | each session: attempt → grader feedback → the student updates its notebook | learning without a teacher |
| `tutor` | each session: teacher-led dialogue → final attempt → feedback → the student updates its notebook | learning with a teacher |

1. **Does a teacher help a persistent learner?** Compare `tutor` against `self` across k.

2. **Is multi-session teaching worth more than telling the rules once?** Compare `tutor` against `summary`. `none` and `doc` set the range that the other three fall in.


- **Memory**:
  - `--memory notebook`: the notebook is rewritten after every session, capped at `--budget` words (B);
  - `--memory transcript`: the full record of every session, unbounded.

## Running

- Default models: student gpt-5-mini, teacher gpt-5.5, judge gpt-5.1.
- Any OpenAI-compatible endpoint works, e.g. Qwen3 served locally by vLLM:

```bash
--student Qwen/Qwen3-8B --student-url http://localhost:8000/v1 \
  --student-extra '{"chat_template_kwargs": {"enable_thinking": false}}'
```

- An HTML report (learning curves, per-scenario curves, table view, memory diagnostic, and a session browser with dialogues, attempts, feedback, notebooks and test outputs):

```bash
python -m teachlab.visualize teachlab/outputs/clbench_core teachlab/outputs/sebench_small -o teachlab/outputs/report.html
open teachlab/outputs/report.html
```
