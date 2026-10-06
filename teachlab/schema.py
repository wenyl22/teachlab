"""The unified scenario format that every benchmark adapter writes and the runner reads.

A scenario is one course: a private knowledge document, a stream of items to study one per session,
and held-out items to measure the student's solo ability. One JSON object per line:

{
  "id":             "sebench/f8/c03",          # unique across files: source/variant/course
  "source":         "sebench",                 # clbench | sebench | rule_world | <your own>
  "meta":           {...},                     # free-form (category, functions, generator params...)
  "knowledge":      "...",                     # privileged document: teacher sees it, student never does
                                               # (except in the open-book `doc` / `self_doc` conditions)
  "student_system": "...",                     # system prompt the student always acts under
  "items": [
    {
      "id":       "...",
      "split":    "stream" | "probe" | "test", # stream: one per session; probe: per-session reward
                                               # (disjoint from test); test: sealed final evaluation
      "messages": [{"role": "user", "content": "..."}, ...],   # the task; earlier turns are allowed
      "grader":   {"type": "rubric", "rubrics": [...], "rubric_types": [...]}       # LLM judge
               | {"type": "exec", "entry": "solve", "test_cases": [...], "expected": [...], "lib": "zwc"}
               | {"type": "exact", "answer": "..."}                                  # \\boxed{} or last line
    }
  ]
}
"""

import json
import os

SPLITS = ("stream", "probe", "test")
GRADER_TYPES = ("rubric", "exec", "exact")


def validate(scenario):
    for key in ("id", "source", "knowledge", "student_system", "items"):
        assert key in scenario, f"scenario missing '{key}'"
    ids = set()
    for item in scenario["items"]:
        assert item["id"] not in ids, f"{scenario['id']}: duplicate item id {item['id']}"
        ids.add(item["id"])
        assert item["split"] in SPLITS, f"{item['id']}: bad split {item['split']}"
        assert item["messages"] and item["messages"][-1]["role"] == "user", f"{item['id']}: must end with a user turn"
        assert item["grader"]["type"] in GRADER_TYPES, f"{item['id']}: bad grader {item['grader']['type']}"
    return scenario


def group_of(scenario_id):
    """The scenario set an id belongs to: "sebench/f8/c03" -> "sebench/f8", "clbench/1d7a1bf8" -> "clbench".
    Results are averaged and --limit is applied per group, never across groups built with different settings."""
    return scenario_id.rsplit("/", 1)[0]


def items(scenario, split):
    return [i for i in scenario["items"] if i["split"] == split]


def problem_text(item):
    """The task as plain text, including the conversation it is part of, for prompts that embed it."""
    msgs = item["messages"]
    if len(msgs) == 1:
        return msgs[0]["content"]
    turns = "\n\n".join(f"[{m['role'].capitalize()}]\n{m['content']}" for m in msgs[:-1])
    return f"Earlier in the conversation:\n\n{turns}\n\nNow the user asks:\n{msgs[-1]['content']}"


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(item, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_scenarios(scenarios, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in scenarios:
            f.write(json.dumps(validate(s), ensure_ascii=False) + "\n")
    print(f"wrote {len(scenarios)} scenarios to {path}")


def load_scenarios(paths):
    out = []
    for p in paths:
        out += [validate(s) for s in load_jsonl(p)]
    return out
