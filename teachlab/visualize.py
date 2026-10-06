"""One self-contained HTML report for one or more runs: learning curves, per-scenario curves, a table view,
the memory diagnostic (if present), and a session browser (dialogues, attempts, feedback, notebooks, test outputs).

Usage:
    python -m teachlab.visualize teachlab/outputs/clbench_core teachlab/outputs/sebench_small -o teachlab/outputs/report.html
    open teachlab/outputs/report.html
"""

import argparse
import glob
import json
import os

from .schema import group_of, load_jsonl

MAX_TEXT = 6000  # characters kept per long text field (model outputs, dialogue turns)


def cut(text, n=MAX_TEXT):
    text = text or ""
    return text if len(text) <= n else text[:n] + f"\n… [{len(text) - n} more characters]"


def compact_eval(ev, with_items):
    out = {}
    for split, e in (ev or {}).items():
        if not e or e.get("score") is None:
            continue
        out[split] = {"score": e["score"], "passed": e.get("passed"), "bt": e.get("by_type") or {}}
        if with_items:
            out[split]["items"] = [{"id": i["id"], "score": i["score"], "scores": i.get("scores"),
                                    "bt": i.get("by_type") or {}, "output": cut(i.get("output"))}
                                   for i in e.get("items", [])]
    return out


def compact_session(rec):
    if not rec:
        return None
    return {"problem": cut(rec.get("problem")),
            "dialogue": [[s, cut(t)] for s, t in rec.get("dialogue", [])],
            "attempts": [{"response": cut(a["response"]), "score": a["score"], "feedback": cut(a["feedback"])}
                         for a in rec.get("attempts", [])],
            "critique": cut(rec.get("critique")) if rec.get("critique") else None}


def load_run(run_dir):
    curves, sessions, diagnose = [], {}, []
    for path in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        is_diag = os.path.basename(path).startswith("diagnose_")  # teachlab.diagnose_memory output
        for r in load_jsonl(path):
            if "label" not in r:  # not a run record (diagnose output, or other analyses in the run dir)
                if is_diag:
                    diagnose.append({"s": r["scenario"], "memory": r["memory"], "words": r.get("memory_words"),
                                     "ev": compact_eval(r.get("eval"), False)})
                continue
            ev = compact_eval(r.get("eval"), False)
            curves.append({"s": r["scenario"], "src": group_of(r["scenario"]), "label": r["label"], "cond": r["condition"],
                           "k": r["k"], "o": r.get("order", 0), "ev": ev})
            state = r.get("state") or {}
            log = r.get("log") or {}
            sessions.setdefault(f'{r["scenario"]}|{r["label"]}|{r.get("order", 0)}', []).append({
                "k": r["k"], "item": r.get("item_id"), "session": compact_session(r.get("session")),
                "notebook": state.get("notebook"), "memWords": r.get("memory_words"),
                "log": {k: log.get(k) for k in ("teacher_present", "ended_by", "teacher_words", "teacher_msgs",
                                                "teacher_copy_rate", "notebook_truncated", "first_attempt_score",
                                                "last_attempt_score") if k in log},
                "violations": len(log.get("violations") or []),
                "ev": compact_eval(r.get("eval"), True)})
    for v in sessions.values():
        v.sort(key=lambda e: e["k"])
    meta = {}
    first = next((r for p in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))) for r in load_jsonl(p)
                  if "label" in r), None)
    if first:
        meta = {k: first.get(k) for k in ("student", "teacher", "judge", "memory_policy", "budget")}
    return {"name": os.path.basename(os.path.normpath(run_dir)), "meta": meta, "curves": curves,
            "sessions": sessions, "diagnose": diagnose}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("-o", "--out", default=None, help="Default: <run_dir>/report.html (one run) or teachlab/outputs/report.html")
    args = p.parse_args()
    runs = [load_run(d) for d in args.run_dirs]
    out = args.out or (os.path.join(args.run_dirs[0], "report.html") if len(runs) == 1 else "teachlab/outputs/report.html")
    data = json.dumps(runs, ensure_ascii=False).replace("</", "<\\/")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html"), encoding="utf-8") as f:
        html = f.read().replace("__DATA__", data)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB; {', '.join(r['name'] for r in runs)})")


if __name__ == "__main__":
    main()
