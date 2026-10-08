"""Graders, one per `item["grader"]["type"]`, and the feedback the student gets after a session.

Every grader returns a dict with at least
    score   - in [0, 1]: rubric pass rate / fraction of test cases passed / 0-1 exact match
    passed  - bool: the strict, all-or-nothing verdict (all rubrics / all cases / exact)
plus details used to build the feedback message.
"""

import ast
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
import warnings

from .vendor.sebench.EqualityChecker import EqulityChecker

# EqulityChecker eval()s and ast.literal_eval()s printed outputs; strings like "[1 2](3)" or "1if" make Python emit
# SyntaxWarnings from "<string>" (eval) or "<unknown>" (literal_eval).
warnings.filterwarnings("ignore", category=SyntaxWarning, module="<(string|unknown)>")
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


SEP = "===================="

# Harness helpers, appended after the solution. Two deliberate differences from SE-Bench's harness:
#  - stdout is captured while the solution runs, so its own print() calls cannot leak into the case outputs
#    (SE-Bench computes every result before printing the separator, which has the same effect);
#  - a returned ZWCArray (also nested in lists / tuples / dicts) is printed as plain Python values. Its own
#    str() is NumPy's display form (8 significant digits, scientific notation, line wrapping), which fails the
#    1e-2 absolute tolerance on large or long outputs: 2.7% of single_test and 5.0% of multiple_test correct
#    zwc solutions were marked wrong by it.
_HARNESS_HELPERS = '''
import contextlib as _tl_contextlib
import io as _tl_io
import zwc

def _tl_plain(x):
    if isinstance(x, zwc.ZWCArray):
        return x._data.tolist()
    if isinstance(x, (list, tuple)):
        return type(x)(_tl_plain(v) for v in x)
    if isinstance(x, dict):
        return {k: _tl_plain(v) for k, v in x.items()}
    return x

def _tl_case(call):
    try:
        with _tl_contextlib.redirect_stdout(_tl_io.StringIO()):
            result = call()
        print(_tl_plain(result))
    except Exception as e:
        print('__ERROR__', type(e).__name__, e)
    print('#')
'''


def _harness(code, entry, test_cases):
    """SE-Bench's harness: call the function on every case and print results separated by '#'."""
    args = [re.sub(r"\bnp\.array\b", "zwc.yitaf", tc["input"]) for tc in test_cases]
    calls = "\n".join(f"    _tl_case(lambda: {entry}({a}))" for a in args)
    return f"{code}\n{_HARNESS_HELPERS}\nif __name__ == '__main__':\n    print('{SEP}')\n{calls}\n"


# Where generated code runs: None = locally; a URL = SE-Bench's Docker sandbox
# (teachlab/sandbox/build_and_run.sh), e.g. http://localhost:8111/run. Set by run.py's --exec-url.
EXEC_URL = os.getenv("TEACHLAB_EXEC_URL")
# Wall-clock limit per program, excluding interpreter start-up in the local fork servers (SE-Bench: 3 s).
EXEC_TIMEOUT = float(os.getenv("TEACHLAB_EXEC_TIMEOUT", 3))
# local: "fork" = pre-warmed fork servers (POSIX); "subprocess" = a fresh interpreter per program.
EXEC_MODE = os.getenv("TEACHLAB_EXEC_MODE", "fork" if os.name == "posix" else "subprocess")
# Generated code runs with no API keys in its environment, and single-threaded BLAS.
_EXEC_ENV = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ZWC_PATH, "PYTHONBREAKPOINT": "0",
             "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
             **({"SYSTEMROOT": os.environ["SYSTEMROOT"]} if "SYSTEMROOT" in os.environ else {})}  # Windows needs it

# One long-lived, single-threaded server per slot imports numpy/zwc once, then forks a child per program:
# fresh temp dir, stdout/stderr to files, stdin closed, 8 GB address space and 64 MB file limits, SIGKILL at
# the deadline. Importing numpy from a network-filesystem env can take 30 s+, so a fresh interpreter per
# program would time out on start-up alone.
_EXEC_SERVER = r'''
import json, os, resource, runpy, shutil, signal, sys, tempfile, time, traceback
import numpy, zwc
pipe_in, pipe_out = sys.stdin.buffer, sys.stdout.buffer
while True:
    line = pipe_in.readline()
    if not line:
        break
    req = json.loads(line)
    td = tempfile.mkdtemp(prefix="teachlab-exec-")
    path, so, se = (os.path.join(td, n) for n in ("main.py", "out", "err"))
    with open(path, "w", encoding="utf-8") as f:
        f.write(req["code"])
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        code = 1
        try:
            os.chdir(td)
            os.environ["HOME"] = td
            os.dup2(os.open(so, os.O_WRONLY | os.O_CREAT, 0o600), 1)
            os.dup2(os.open(se, os.O_WRONLY | os.O_CREAT, 0o600), 2)
            os.close(0)
            for lim, val in ((getattr(resource, "RLIMIT_AS", None), 8 << 30), (resource.RLIMIT_FSIZE, 64 << 20)):
                if lim is not None:
                    try:
                        resource.setrlimit(lim, (val, val))
                    except (ValueError, OSError):
                        pass
            sys.argv = [path]
            runpy.run_path(path, run_name="__main__")
            code = 0
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except BaseException:
            traceback.print_exc()
        finally:
            try:
                sys.stdout.flush(); sys.stderr.flush()
            finally:
                os._exit(code)
    deadline, timed_out, nap = time.time() + req["timeout"], False, 0.001
    while True:
        done, _ = os.waitpid(pid, os.WNOHANG)
        if done:
            break
        if time.time() > deadline:
            os.kill(pid, signal.SIGKILL); os.waitpid(pid, 0); timed_out = True
            break
        time.sleep(nap); nap = min(nap * 2, 0.05)
    def tail(p, n):
        try:
            with open(p, "rb") as f:
                f.seek(0, 2); f.seek(max(0, f.tell() - n)); return f.read().decode("utf-8", "replace")
        except OSError:
            return ""
    res = {"stdout": tail(so, 200000), "stderr": tail(se, 4000), "timed_out": timed_out}
    shutil.rmtree(td, ignore_errors=True)
    pipe_out.write((json.dumps(res) + "\n").encode()); pipe_out.flush()
'''


class _ExecPool:
    """TEACHLAB_EXEC_SERVERS fork servers shared by all threads; started on first use, respawned if one dies."""

    def __init__(self, n):
        self.n, self.free, self.started, self.lock = n, queue.Queue(), False, threading.Lock()

    def _spawn(self):
        # cwd is a private empty dir: `python -c` puts the cwd first on sys.path, and a stray file there
        # (e.g. a struct.py in a shared /tmp) would shadow the standard library.
        log = tempfile.TemporaryFile()
        p = subprocess.Popen([sys.executable, "-u", "-c", _EXEC_SERVER], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, env=_EXEC_ENV,
                             cwd=tempfile.mkdtemp(prefix="teachlab-server-"))
        # Self-test: a server that cannot run zwc would otherwise turn every answer into a silent failure.
        reply = self._ask(p, "import zwc\nprint(zwc.yitaf([1, 2]))", 120)
        if not reply or reply["stdout"].strip() != "[1 2]":
            log.seek(0)
            err = log.read().decode("utf-8", "replace")[-2000:]
            p.kill()
            raise RuntimeError(f"exec server failed its self-test (is {ZWC_PATH} present?): {reply} {err}")
        return p

    @staticmethod
    def _ask(p, code, timeout):
        p.stdin.write((json.dumps({"code": code, "timeout": timeout}) + "\n").encode())
        p.stdin.flush()
        line = p.stdout.readline()
        return json.loads(line) if line else None

    def run(self, code, timeout):
        with self.lock:
            if not self.started:
                for _ in range(self.n):
                    self.free.put(self._spawn())
                self.started = True
        p = self.free.get()
        try:
            reply = self._ask(p, code, timeout)
            if reply is None:
                raise BrokenPipeError("exec server exited")
            return reply["stdout"], reply["stderr"], reply["timed_out"]
        except (BrokenPipeError, OSError, ValueError) as e:
            p.kill()
            p = self._spawn()
            return "", f"sandbox error: {e!r}", False
        finally:
            self.free.put(p)


_POOL = _ExecPool(int(os.getenv("TEACHLAB_EXEC_SERVERS", 4)))
_STARTUP_ALLOWANCE = 120  # subprocess mode only: interpreter + numpy import time on top of EXEC_TIMEOUT


def run_python(code, timeout=None):
    """Run a program; returns (stdout, stderr, timed_out). stdout printed before a timeout is kept."""
    timeout = EXEC_TIMEOUT if timeout is None else timeout
    if EXEC_URL:
        # The sandbox enforces its own 3 s limit per program, as in SE-Bench (returns stderr "TLE").
        req = urllib.request.Request(EXEC_URL, data=json.dumps({"code": code}).encode(),
                                     headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    out = json.loads(r.read())
                return out["stdout"], out["stderr"], out["stderr"].strip() == "TLE"
            except OSError as e:
                error = e
        raise RuntimeError(f"sandbox at {EXEC_URL} unreachable: {str(error)[:150]}")
    if EXEC_MODE == "fork":
        return _POOL.run(code, timeout)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "main.py")
        with open(path, "w") as f:
            f.write(code)
        try:
            r = subprocess.run([sys.executable, path], capture_output=True, text=True, cwd=tmp,
                               timeout=timeout + _STARTUP_ALLOWANCE, env={**_EXEC_ENV, "HOME": tmp})
            return r.stdout, r.stderr, False
        except subprocess.TimeoutExpired as e:
            out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return out, "", True


def run_and_compare(code, entry, test_cases, expected, timeout=None):
    """Run `code` on the test cases through the harness and compare with `expected`, case by case.
    Returns (per-case results, stderr, timed_out), or (None, stderr, timed_out) if no case ran."""
    stdout, stderr, timed_out = run_python(_harness(code, entry, test_cases), timeout)
    if SEP not in stdout:
        return None, stderr, timed_out
    outs = [s.strip() for s in stdout.split(SEP)[-1].split("#")[:-1]]
    checker = EqulityChecker(tolerance=1e-2)
    results = []
    for i, (tc, want) in enumerate(zip(test_cases, expected)):
        got = outs[i] if i < len(outs) else ""
        try:
            ok = got == want.strip() or checker.compare_stdout_outputs(got, want.strip(), 1e-2)
        except Exception:
            ok = False
        results.append({"input": tc["input"], "expected": want.strip(), "got": got, "ok": bool(ok)})
    return results, stderr, timed_out


def grade_exec(spec, response, timeout=None):
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
    timeout = EXEC_TIMEOUT if timeout is None else timeout
    results, stderr, timed_out = run_and_compare(code, spec["entry"], cases, expected, timeout)
    if results is None:
        if timed_out:
            return fail(f"time limit exceeded ({timeout:g} s) before any test case finished")
        return fail("the code crashed before running the test cases", stderr=stderr[-1500:])
    n_ok = sum(r["ok"] for r in results)
    out = {"score": n_ok / len(cases), "passed": n_ok == len(cases), "cases": results,
           "stderr": stderr[-1500:] if stderr else ""}
    if timed_out:
        out["timeout"] = f"time limit exceeded ({timeout:g} s); cases after the last printed one count as failed"
    return out


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
    note = f"\nNote: {result['timeout']}." if result.get("timeout") else ""
    return f"Grader: {n_ok} of {len(cases)} test cases passed. Some failing cases:\n{lines}{note}"


# --- exact (final answer match) ---------------------------------------------------------------

def _last_boxed(text):
    """Content of the last \\boxed{...}, with nested braces (e.g. \\boxed{\\text{ka-te}})."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i = start + len("\\boxed{")
    depth = 1
    for j in range(i, len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[i:j]
    return text[i:].split("\n")[0].rstrip("}")  # unclosed box: the rest of its line


def extract_answer(response):
    boxed = _last_boxed(response or "")
    if boxed is not None:
        # Unwrap LaTeX text commands models put inside the box: \text{ka-te}, ka\text{-}te, \mathrm{...}
        text_cmd = re.compile(r"\\(?:text|mathrm|textrm|texttt|mathtt|textbf|mathbf)\{([^{}]*)\}")
        while text_cmd.search(boxed):
            boxed = text_cmd.sub(r"\1", boxed)
        boxed = re.sub(r"\\(?:text|mathrm|textrm|texttt|mathtt|textbf|mathbf)\{", "", boxed)  # left unclosed
        boxed = re.sub(r"(\\[)\]]|[.\s$])+$", "", boxed)  # a math delimiter or period caught inside the box
        return boxed.replace("\\-", "-").strip().strip("$").strip()
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
