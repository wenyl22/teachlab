"""Graders, one per `item["grader"]["type"]`, and the feedback the student gets after a session.

Every grader returns a dict with at least
    score   - in [0, 1]: rubric pass rate / fraction of test cases passed / 0-1 exact match
    passed  - bool: the strict, all-or-nothing verdict (all rubrics / all cases / exact)
plus details used to build the feedback message.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

from .vendor.sebench.EqualityChecker import EqulityChecker
from .vendor.sebench.ast_zwc_checker import ASTSourceValidator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZWC_PATH = os.path.join(ROOT, "data", "sebench", "zwc")

from .vendor.clbench.judge import judge_prompt  # CL-bench's own judge prompt, with the opener fix


# --- rubric (LLM judge) -----------------------------------------------------------------------

def grade_rubric(spec, response, judge, max_attempts=3):
    rubrics = spec["rubrics"]
    if not response or not response.strip():
        status = ["no"] * len(rubrics)
    else:
        prompt, status, error = judge_prompt(rubrics, response), None, None
        for attempt in range(max_attempts):
            note = (f"\nNote: there are exactly {len(rubrics)} rubrics, so the status list must have exactly "
                    f"{len(rubrics)} entries, one per numbered rubric.") if attempt else ""
            try:
                text = judge.chat([{"role": "user", "content": prompt + note}]).strip()
                text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                status = json.loads(text)["List of Requirement Satisfaction Status"]
                if len(status) != len(rubrics):
                    raise ValueError(f"{len(status)} statuses for {len(rubrics)} rubrics")
                break
            except Exception as e:
                status, error = None, e
        if status is None:
            raise RuntimeError(f"judge failed: {str(error)[:200]}")
    ok = [str(s).strip().lower() == "yes" for s in status]
    result = {"score": sum(ok) / len(ok) if ok else 0.0, "passed": all(ok), "status": ok}
    types = spec.get("rubric_types")
    if types:
        by = {}
        for t, o in zip(types, ok):
            by.setdefault(t, []).append(o)
        result["by_type"] = {t: sum(v) / len(v) for t, v in by.items()}
    return result


def feedback_rubric(spec, result):
    missed = [r for r, o in zip(spec["rubrics"], result["status"]) if not o]
    if not missed:
        return f"Grader: your answer met all {len(spec['rubrics'])} requirements."
    lines = "\n".join(f"- {r}" for r in missed)
    return (f"Grader: your answer met {len(spec['rubrics']) - len(missed)} of {len(spec['rubrics'])} "
            f"requirements. Unmet requirements:\n{lines}")


# --- exec (SE-Bench style: run code against test cases) ----------------------------------------

def extract_code(response):
    blocks = re.findall(r"```python\n([\s\S]*?)```", response or "")
    return blocks[-1] if blocks else ""


def _uses_numpy(code):
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "numpy" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "numpy":
            return True
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__" and node.args \
                and isinstance(node.args[0], ast.Constant) and str(node.args[0].value).startswith("numpy"):
            return True
    return False


def _harness(code, entry, test_cases):
    """SE-Bench's harness: call the function on every case and print results separated by '#'."""
    args = [re.sub(r"\bnp\.array\b", "zwc.yitaf", tc["input"]) for tc in test_cases]
    calls = "\n".join(f"    try:\n        print({entry}({a}))\n"
                      f"    except Exception as e:\n        print('__ERROR__', type(e).__name__, e)\n    print('#')"
                      for a in args)
    return f"{code}\nimport zwc\n\nif __name__ == '__main__':\n    print('====================')\n{calls}\n"


# Where generated code runs: None = a local subprocess; a URL = SE-Bench's Docker sandbox
# (teachlab/sandbox/build_and_run.sh), e.g. http://localhost:8111/run. Set by run.py's --exec-url.
EXEC_URL = os.getenv("TEACHLAB_EXEC_URL")


def run_python(code, timeout):
    if EXEC_URL:
        # The sandbox enforces its own 3 s limit per program, as in SE-Bench (returns stderr "TLE").
        req = urllib.request.Request(EXEC_URL, data=json.dumps({"code": code}).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read())
                return out["stdout"], out["stderr"]
            except OSError as e:
                error = e
        raise RuntimeError(f"sandbox at {EXEC_URL} unreachable: {str(error)[:150]}")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w") as f:
            f.write(code)
        # Generated code runs with no API keys in its environment, in a scratch directory.
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ZWC_PATH, "HOME": tmp}
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env)
            return r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return "", "TIMEOUT"


def grade_exec(spec, response, timeout=10):
    cases, expected = spec["test_cases"], spec["expected"]
    fail = lambda reason, **kw: {"score": 0.0, "passed": False, "error": reason, "cases": [], **kw}
    code = extract_code(response)
    if not code.strip():
        return fail("no ```python``` code block in the answer")
    try:
        if _uses_numpy(code):
            return fail("the code imports numpy, which is not available; use the zwc library")
    except SyntaxError as e:
        return fail(f"syntax error: {e}")
    lib = spec.get("lib")
    if lib:
        check = ASTSourceValidator(lib).check_source(code + f"\nimport {lib}\n", spec["entry"])
        if not check["passed"]:
            return fail(f"the solution must call {lib} functions ({check['error']})")
    stdout, stderr = run_python(_harness(code, spec["entry"], cases), timeout)
    if "====================" not in stdout:
        return fail("the code crashed before running the test cases", stderr=stderr[-1500:])
    outs = [s.strip() for s in stdout.split("====================")[-1].split("#")[:-1]]
    checker = EqulityChecker(tolerance=1e-2)
    results = []
    for i, (tc, want) in enumerate(zip(cases, expected)):
        got = outs[i] if i < len(outs) else ""
        try:
            ok = got == want.strip() or checker.compare_stdout_outputs(got, want.strip(), 1e-2)
        except Exception:
            ok = False
        results.append({"input": tc["input"], "expected": want.strip(), "got": got, "ok": bool(ok)})
    n_ok = sum(r["ok"] for r in results)
    return {"score": n_ok / len(cases), "passed": n_ok == len(cases), "cases": results,
            "stderr": stderr[-1500:] if stderr else ""}


def feedback_exec(spec, result, max_cases=3):
    if result.get("error"):
        msg = f"Grader: {result['error']}."
        if result.get("stderr"):
            msg += f"\nError output:\n{result['stderr'][-800:]}"
        return msg
    cases = result["cases"]
    n_ok = sum(c["ok"] for c in cases)
    if n_ok == len(cases):
        return f"Grader: all {len(cases)} test cases passed."
    shown = [c for c in cases if not c["ok"]][:max_cases]
    lines = "\n".join(f"- input: {c['input'][:300]}\n  expected: {c['expected'][:300]}\n  got: {c['got'][:300]}"
                      for c in shown)
    return f"Grader: {n_ok} of {len(cases)} test cases passed. Some failing cases:\n{lines}"


# --- exact (final answer match) ---------------------------------------------------------------

def extract_answer(response):
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", response or "")
    if boxed:
        return boxed[-1].strip()
    lines = [l for l in (response or "").strip().splitlines() if l.strip()]
    if not lines:
        return ""
    return re.sub(r"^(final\s+)?answer\s*[:：]\s*", "", lines[-1].strip(), flags=re.I).strip()


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().strip(".").lower()


def grade_exact(spec, response):
    got = extract_answer(response)
    answers = spec["answer"] if isinstance(spec["answer"], list) else [spec["answer"]]
    ok = any(_norm(got) == _norm(a) for a in answers)
    return {"score": float(ok), "passed": ok, "got": got}


def feedback_exact(spec, result, reveal=True):
    if result["passed"]:
        return "Grader: correct."
    answer = spec["answer"] if isinstance(spec["answer"], str) else spec["answer"][0]
    return f"Grader: incorrect (your final answer: {result['got'] or '(none)'})." + \
        (f" The correct answer is {answer}." if reveal else "")


# --- dispatch ---------------------------------------------------------------------------------

def grade(item, response, judge):
    spec = item["grader"]
    if spec["type"] == "rubric":
        return grade_rubric(spec, response, judge)
    if spec["type"] == "exec":
        return grade_exec(spec, response)
    if spec["type"] == "exact":
        return grade_exact(spec, response)
    raise ValueError(spec["type"])


def feedback(item, result):
    spec = item["grader"]
    return {"rubric": feedback_rubric, "exec": feedback_exec, "exact": feedback_exact}[spec["type"]](spec, result)
