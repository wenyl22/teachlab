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
| SE-Bench (the zwc library) | `bash teachlab/download_sebench.sh && python -m teachlab.adapters.sebench --funcs 8` | 7–8 functions, 10–19 stream, 3–5 probe and 9–12 test items per course (verified rows); documents ~2.6–3.6k words | exec |
| SE-Bench, full-library docs | the command above with `--doc-scope library` | document ~97k words, too large for the notebook | exec |
| Generated rule worlds | `python -m teachlab.adapters.rule_world --ops 6 --length 3` | any number of items; `--ops` sets the knowledge volume, `--length` the difficulty of applying it | exact |

**Adding a source**: write an adapter that outputs jsonl in the format above. Hand-written scenarios can also be written directly as jsonl.

### SE-Bench details

- **Verified rows only (default).** The adapter keeps rows whose own NumPy reference solution reproduces the stored ground truth. The rest are ambiguous or mislabeled: about 12% of train, 12% of single_test and 19% of multiple_test. The check runs once and is cached in `data/sebench/verified.json`.
  - Verified courses have ids `sebench/f8v/...` and are written to `scenarios/sebench_f8v.jsonl`.
  - `--all-rows` keeps every row, with the old `sebench/f8/...` ids.
- **Function names.** The docs give each function's full call path, e.g. `zwc.rfx.gicopuf` for the 31 linear-algebra functions in the `rfx` submodule.
- **Harness.** Its own `print()` calls are captured, so they cannot leak into case outputs. A returned zwc array is printed as plain Python values: zwc arrays print in NumPy's rounded display form, which otherwise fails correct answers.
- **Execution.** Local execution uses pre-warmed fork servers on POSIX, so importing NumPy from a slow filesystem does not eat into the time limit. On other systems it uses one interpreter per program. Settings, as environment variables:
  - `TEACHLAB_EXEC_TIMEOUT`: seconds per program; default 3, as in SE-Bench.
  - `TEACHLAB_EXEC_SERVERS`: number of fork servers; default 4.
  - `TEACHLAB_EXEC_MODE`: `fork` or `subprocess`.
- **NumPy version.** Install `numpy==2.3.0`, the version zwc pins and SE-Bench's sandbox uses.

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
  --student-extra '{"chat_template_kwargs": {"enable_thinking": false}}' --student-max-tokens 8192
```

- `--{role}-max-tokens` caps completion tokens (sent as `max_completion_tokens`).
- Requests that cannot succeed are not retried. That includes a prompt longer than the context window, which a growing `--memory transcript` will eventually produce on a small student.
- A student request that can never succeed counts as an empty answer and is reported as `generation_errors` in the eval summary and the session log. It no longer stops the job.
- Transient failures, such as a server that is down after all retries, still stop the job. The job then resumes.

- An HTML report (learning curves, per-scenario curves, table view, memory diagnostic, and a session browser with dialogues, attempts, feedback, notebooks and test outputs):

```bash
python -m teachlab.visualize teachlab/outputs/clbench_core teachlab/outputs/sebench_small -o teachlab/outputs/report.html
open teachlab/outputs/report.html
```
