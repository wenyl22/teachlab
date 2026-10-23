"""TeachMix grader types, dispatched from teachlab.graders.grade() / feedback() (see GRADER_TYPES in schema.py).

    numeric  final number (by `pattern`, or RuleArena tax's owed/overpaid sentence, or the last \\boxed{} / number),
             correct if within max(atol, rtol * |answer|)
    nba      RuleArena NBA: "Answer: False." or "Answer: True. Illegal Operation: X. Problematic Team: Y."
    choice   one of `options`, by `pattern` (last match)
    kor      KOR-Bench's own checker (eval/eval_utils.evaluate_response_vs_answer), else [[...]] string match
    pyexec   CodeUpdateArena: problem imports + update implementation (setattr) + student code + unit tests
    chrf     MTOB: chrF2 (character 6-grams, beta 2) of the \\boxed{} translation against the reference, in [0, 1]
    rg       Reasoning Gym's score_answer on the <answer></answer> content

The KOR checker is imported from the KOR-Bench repository under $TEACHMIX_DATA (default data/teachmix).
"""
import collections, os, re, subprocess, sys, tempfile

DATA = os.path.abspath(os.path.expanduser(os.getenv("TEACHMIX_DATA", "data/teachmix")))  # absolute: builders chdir
KOR_ROOT = os.path.join(DATA, "KOR-Bench")
NUM = r"-?\$?\s*-?[\d,]*\d(?:\.\d+)?"


def _num(s):
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _last_boxed(text):
    from ..graders import extract_answer  # lazy: teachlab.graders imports this module
    return extract_answer(text)


# --- numeric -----------------------------------------------------------------------------------

def grade_numeric(spec, response):
    text = response or ""
    got = None
    if spec.get("tax_sign"):
        m = re.findall(r"total tax (owed|overpaid) is \$?\s*(-?[\d,]+(?:\.\d+)?)", text, re.I)
        if m:
            status, v = m[-1]
            got = _num(v)
            got = -got if (got is not None and status.lower() == "overpaid") else got
    if got is None and spec.get("pattern"):
        m = re.findall(spec["pattern"], text, re.I)
        if m:
            got = _num(m[-1] if isinstance(m[-1], str) else m[-1][-1])
    if got is None:
        b = _last_boxed(text) if "\\boxed{" in text else None
        cand = re.findall(NUM, b if b else text)
        got = _num(cand[-1]) if cand else None
    ans = float(spec["answer"])
    ok = got is not None and abs(got - ans) <= max(spec.get("atol", 0.0), spec.get("rtol", 0.0) * abs(ans))
    return {"score": float(ok), "passed": ok, "got": got}


def feedback_numeric(spec, result):
    if result["passed"]:
        return "Grader: correct."
    a = spec["answer"]
    a = int(a) if float(a).is_integer() else a
    return f"Grader: incorrect (your final answer: {result['got'] if result['got'] is not None else '(none)'}). " \
           f"The correct answer is {a}."


# --- nba / choice --------------------------------------------------------------------------------

def grade_nba(spec, response):
    t = response or ""
    m = re.findall(r"Answer:\s*\**\s*(True|False)", t, re.I)
    got = m[-1].lower() == "true" if m else None
    op = re.findall(r"Illegal Operation:\s*\**\s*([A-Z])", t)
    team = re.findall(r"Problematic Team:\s*\**\s*([A-Z])", t)
    if got is None:
        ok = False
    elif not spec["answer"]:
        ok = got is False
    else:
        ok = got is True and bool(op) and bool(team) and op[-1] == spec["op"] and team[-1] == spec["team"]
    return {"score": float(ok), "passed": ok, "got": (got, op[-1] if op else None, team[-1] if team else None)}


def feedback_nba(spec, result):
    if result["passed"]:
        return "Grader: correct."
    gold = "Answer: False." if not spec["answer"] else \
        f"Answer: True. Illegal Operation: {spec['op']}. Problematic Team: {spec['team']}."
    return f"Grader: incorrect. The correct conclusion is \"{gold}\""


def grade_choice(spec, response):
    m = re.findall(spec["pattern"], response or "", re.I)
    got = m[-1].lower() if m else None
    ok = got == spec["answer"]
    return {"score": float(ok), "passed": ok, "got": got}


def feedback_choice(spec, result):
    return "Grader: correct." if result["passed"] else \
        f"Grader: incorrect (your answer: {result['got'] or '(none)'}). The correct answer is {spec['answer']}."


# --- kor --------------------------------------------------------------------------------------------

_KOR = None


def _kor_eval():
    global _KOR
    if _KOR is None:
        try:
            sys.path.insert(0, os.path.abspath(KOR_ROOT))
            from eval.eval_utils import evaluate_response_vs_answer
            _KOR = evaluate_response_vs_answer
        except Exception as e:  # missing optional deps (e.g. antlr for latex parsing)
            print(f"   ⚠️ KOR-Bench checker unavailable ({type(e).__name__}: {e}); using [[...]] string match")
            _KOR = False
    return _KOR


def _brackets(text):
    m = re.findall(r"\[\[(.*?)\]\]", text or "", re.S)
    return re.sub(r"\s+", " ", m[-1]).strip().lower() if m else None


def grade_kor(spec, response):
    f = _kor_eval()
    ok = None
    if f:
        try:
            ok = bool(f(response or "", spec["answer"], spec["category"], spec["rule_id"], spec["idx"]))
        except Exception:
            ok = None
    if ok is None:
        ok = _brackets(response) is not None and _brackets(response) == _brackets(spec["answer"])
    return {"score": float(ok), "passed": ok, "got": _brackets(response)}


def feedback_kor(spec, result):
    return "Grader: correct." if result["passed"] else \
        f"Grader: incorrect (your final answer: [[{result['got'] or ''}]]). The correct answer is {spec['answer']}."


# --- pyexec (CodeUpdateArena) --------------------------------------------------------------------------

RUNNER = """
import traceback as _tb
_names = [n for n in list(globals()) if n.startswith("test_") and callable(globals()[n])]
_ok = 0
for _n in _names:
    try:
        globals()[_n](); _ok += 1; print("PASS", _n)
    except Exception as _e:
        print("FAIL", _n, type(_e).__name__, str(_e)[:200].replace(chr(10), " "))
print("RESULT", _ok, len(_names))
"""


def _code(response):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", response or "", re.S)
    return blocks[-1] if blocks else (response or "")


def grade_pyexec(spec, response):
    prog = "\n\n".join([spec["imports"], spec["setup"], _code(response), spec["tests"], RUNNER])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prog.py")
        open(path, "w").write(prog)
        try:
            p = subprocess.run([sys.executable, path], cwd=d, capture_output=True, text=True, timeout=spec.get("timeout", 20))
            out, err, to = p.stdout, p.stderr, False
        except subprocess.TimeoutExpired:
            out, err, to = "", "", True
    m = re.findall(r"RESULT (\d+) (\d+)", out)
    if to or not m:
        return {"score": 0.0, "passed": False, "error": "timed out" if to else "the program crashed before the tests ran",
                "stderr": err[-800:], "cases": []}
    ok, n = map(int, m[-1])
    cases = re.findall(r"^(PASS|FAIL) (\S+)(.*)$", out, re.M)
    return {"score": ok / n if n else 0.0, "passed": n > 0 and ok == n, "cases": cases}


def feedback_pyexec(spec, result):
    if result.get("error"):
        return f"Grader: {result['error']}.\n{result.get('stderr', '')}".strip()
    fails = [f"- {name}: {rest.strip()}" for st, name, rest in result["cases"] if st == "FAIL"]
    if not fails:
        return f"Grader: all {len(result['cases'])} tests passed."
    return f"Grader: {len(result['cases']) - len(fails)} of {len(result['cases'])} tests passed. Failing:\n" + "\n".join(fails[:4])


# --- chrf (MTOB) ---------------------------------------------------------------------------------------

def _ngrams(s, n):
    s = re.sub(r"\s+", "", s)
    return collections.Counter(s[i:i + n] for i in range(len(s) - n + 1))


def chrf(hyp, ref, max_n=6, beta=2.0):
    precs, recs = [], []
    for n in range(1, max_n + 1):
        h, r = _ngrams(hyp, n), _ngrams(ref, n)
        if not h or not r:
            continue
        overlap = sum((h & r).values())
        precs.append(overlap / sum(h.values())); recs.append(overlap / sum(r.values()))
    if not precs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    return 0.0 if p + r == 0 else (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def grade_chrf(spec, response):
    got = _last_boxed(response or "") if "\\boxed{" in (response or "") else (response or "").strip().split("\n")[-1]
    s = chrf(got or "", spec["answer"])
    return {"score": s, "passed": s >= 0.5, "got": got}


def feedback_chrf(spec, result):
    return f"Grader: chrF {result['score']:.2f} (your translation: {result['got'] or '(none)'}). " \
           f"Reference translation: {spec['answer']}"


# --- rg (Reasoning Gym) -----------------------------------------------------------------------------------

def grade_rg(spec, response):
    import reasoning_gym as rg
    m = re.findall(r"<answer>(.*?)</answer>", response or "", re.S)
    got = m[-1].strip() if m else None
    try:
        s = float(rg.get_score_answer_fn(spec["task"])(got, spec["entry"])) if got is not None else 0.0
    except Exception:
        s = 0.0
    return {"score": s, "passed": s >= 1.0, "got": got}


def feedback_rg(spec, result):
    return "Grader: correct." if result["passed"] else \
        f"Grader: incorrect (your answer: {result['got'] or '(none)'}). The correct answer is {spec['entry'].get('answer')}."


GRADERS = {"numeric": (grade_numeric, feedback_numeric), "nba": (grade_nba, feedback_nba),
           "choice": (grade_choice, feedback_choice), "kor": (grade_kor, feedback_kor),
           "pyexec": (grade_pyexec, feedback_pyexec), "chrf": (grade_chrf, feedback_chrf), "rg": (grade_rg, feedback_rg)}

