"""CL-bench courses (built by teaching/build_doc_courses.py) -> scenarios.

stream = the teaching tasks, test = the original CL-bench tasks (rubric-graded by the LLM judge). Test tasks that need long verbatim passages of the document are dropped.

Usage:
    python -m teachlab.adapters.clbench --courses data/clbench/doc_courses_alltest.jsonl \
        --contexts-from data/clbench/selected_5mini.jsonl
"""

import argparse
import json
import os

from ..schema import load_jsonl, write_scenarios

GENERIC_SYSTEM = "You are a helpful assistant."


def to_scenario(course, rubric_types, verbatim):
    rulebook = course["type"] == "rulebook"
    items = []
    for t in course["tasks"]:
        if t["split"] == "test" and verbatim.get(t["task_id"], {}).get("requires_verbatim"):
            continue
        spec = {"type": "rubric", "rubrics": t["rubrics"]}
        if t["task_id"] in rubric_types and len(rubric_types[t["task_id"]]) == len(t["rubrics"]):
            spec["rubric_types"] = rubric_types[t["task_id"]]
        history = [] if rulebook else [dict(m) for m in t.get("history") or []]
        items.append({"id": t["task_id"], "split": "stream" if t["split"] == "teach" else "test",
                      "messages": history + [{"role": "user", "content": t["question"]}], "grader": spec,
                      "synthetic": bool(t.get("synthetic"))})
    return {
        "id": f"clbench/{course['context_id'][:8]}", "source": "clbench",
        "meta": {"context_id": course["context_id"], "category": course.get("category"),
                 "sub_category": course.get("sub_category"), "type": course["type"]},
        "knowledge": course["knowledge"],
        # Rulebook courses keep their knowledge in the system prompt, so the student acts under a generic one.
        "student_system": GENERIC_SYSTEM if rulebook else course["system"],
        "items": items,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--courses", default="data/clbench/doc_courses_alltest.jsonl")
    p.add_argument("--contexts-from", default=None, help="Keep only context_ids present in this jsonl")
    p.add_argument("--rubric-types", default="data/clbench/rubric_types.json")
    p.add_argument("--verbatim", default="data/clbench/verbatim_flags.json")
    p.add_argument("--out", default="teachlab/scenarios/clbench.jsonl")
    args = p.parse_args()

    courses = load_jsonl(args.courses)
    if args.contexts_from:
        keep = {r["context_id"] for r in load_jsonl(args.contexts_from)}
        courses = [c for c in courses if c["context_id"] in keep]
    load = lambda path: json.load(open(path)) if os.path.exists(path) else {}
    rubric_types, verbatim = load(args.rubric_types), load(args.verbatim)
    write_scenarios([to_scenario(c, rubric_types, verbatim) for c in courses], args.out)


if __name__ == "__main__":
    main()
