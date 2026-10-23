"""Course builders for the TeachMix source benchmarks (also usable on their own, e.g. for screening).

    RuleArena  airline, nba, tax         doc = the real rulebook (AA baggage policy / NBA CBA rules / IRS form texts)
    DeonticBench sara_numeric, sara_binary, housing (one course per state)   doc = statutes
    KOR-Bench  5 categories x 5 groups of 5 rules (50 items each)            doc = the 5 rules
    CodeUpdateArena  one course per package, one update per API function     doc = the updates' docs (not the code)
    MTOB       Kalamang->English and English->Kalamang                       doc = the "medium" grammar excerpt
    Reasoning Gym  a few procedurally generated tasks (if reasoning_gym is installed)  doc = the task's docstring
Each course: stream / probe / test splits (default 20 / 10 / up to 40, smaller courses split 40/20/40).
Grader types: teachlab/teachmix/graders.py (numeric, nba, choice, kor, pyexec, chrf, rg).
Each item keeps its dataset-native id as "source_id" (its "id" is rewritten to <course>/<split><k>).

    python -m teachlab.teachmix.sources --only rulearena,deontic,kor,cua,mtob,rg    # -> scenarios/new_<source>.jsonl
Data under --data (default $TEACHMIX_DATA or data/teachmix); see teachlab/teachmix/README.md for the downloads.
"""
import argparse, glob, json, os, random, re, sys, collections

DATA = os.path.abspath(os.path.expanduser(os.getenv("TEACHMIX_DATA", "data/teachmix")))  # absolute: builders chdir
FMT_NUM = 'End your response with "The answer is $xxx." (xxx is a number).'


def split(items, seed, n_stream=20, n_probe=10, n_test=40):
    items = items[:]
    random.Random(seed).shuffle(items)
    n = len(items)
    if n < n_stream + n_probe + n_test:
        n_stream, n_probe = round(0.4 * n), round(0.2 * n)
        n_test = n - n_stream - n_probe
    out = []
    for i, it in enumerate(items[:n_stream + n_probe + n_test]):
        sp = "stream" if i < n_stream else "probe" if i < n_stream + n_probe else "test"
        out.append({**it, "split": sp, "id": f"{it['id']}"})
    return out


def scenario(sid, source, knowledge, student_system, items, meta=None, seed=0):
    its = split(items, seed)
    for k, it in enumerate(its):
        it.setdefault("source_id", it["id"])
        it["id"] = f"{sid}/{it['split']}{k}"
    return {"id": sid, "source": source, "meta": meta or {}, "knowledge": knowledge, "student_system": student_system,
            "items": its}


def user(text):
    return [{"role": "user", "content": text}]


# --- RuleArena ----------------------------------------------------------------------------------

def rulearena(root):
    out = []
    # airline
    sys.path.insert(0, os.path.join(root, "airline")); cwd = os.getcwd(); os.chdir(os.path.join(root, "airline"))
    from compute_answer import compute_answer, load_checking_fee
    tables = load_checking_fee()
    rules = open("reference_rules.txt").read()
    items = []
    for lvl in range(3):
        for i, l in enumerate(open(f"synthesized_problems/comp_{lvl}.jsonl")):
            d = json.loads(l)
            fee, _ = compute_answer(**d["info"], check_base_tables=tables)
            items.append({"id": f"a{lvl}_{i}", "level": lvl, "messages": user(
                d["prompt"] + " Compute the total cost for him step by step (don't omit any bag) and end your response "
                "with \"The total cost is $xxx.\" (xxx is a number)"),
                "grader": {"type": "numeric", "answer": float(fee), "pattern": r"total cost is \$?\s*(-?[\d,]+(?:\.\d+)?)",
                           "atol": 1.0}})
    os.chdir(cwd)
    sys_air = ("You are a helpful assistant at American Airlines. You are given the information of a passenger, his / her "
               "items, his / her special needs. You should compute the total cost (including the flight ticket fee, "
               "checked bag fees, cost of special needs) according to American Airlines' policies for the passenger.")
    out.append(scenario("rulearena/airline", "rulearena", "# American Airlines baggage policies\n\n" + rules, sys_air, items))

    # nba
    rules = open(os.path.join(root, "nba", "reference_rules.txt")).read()
    assume = ("Assume:\n* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;\n"
              "* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;\n"
              "* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;\n"
              "* the Luxury Tax is $170,814,000;\n* the First Apron Level is $178,132,000;\n"
              "* the Second Apron Level is $188,931,000;\n* the Team Salary of each team listed under \"Team Situations:\" "
              "do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.\n")
    items = []
    for lvl in range(3):
        for i, d in enumerate(json.load(open(os.path.join(root, "nba", "annotated_problems", f"comp_{lvl}.json")))):
            ops = d["operations"] if isinstance(d["operations"], list) else eval(d["operations"])
            q = (assume + "\nTeam Situations:\n" + "\n".join(d["team_situations"]) + "\n\nPlayer Situations:\n" +
                 "\n".join(d["player_situations"]) + "\n\nOperations:\n" + "\n".join(ops) +
                 "\n\nDecide whether any operation by any team violates the rules. Conclude your response with:\n"
                 "* \"Answer: False.\" if there is no violation to the rules;\n* \"Answer: True. Illegal Operation: X. "
                 "Problematic Team: Y.\" if Team Y in Operation X violates the rules. Both X and Y should be a single "
                 "capital letter as A/B/C/...")
            ans = str(d["answer"]).strip().lower() == "true"
            items.append({"id": f"n{lvl}_{i}", "level": lvl, "messages": user(q),
                          "grader": {"type": "nba", "answer": ans, "op": d.get("illegal_operation"),
                                     "team": d.get("problematic_team")}})
    sys_nba = ("You are a helpful NBA team consultant. You are given the situations of some teams and players and the "
               "operations they conduct. You should determine whether each operation complies with the NBA Collective "
               "Bargaining Agreement.")
    out.append(scenario("rulearena/nba", "rulearena", "# NBA Collective Bargaining Agreement (excerpts)\n\n" + rules,
                        sys_nba, items))

    # tax
    sys.path.insert(0, os.path.join(root, "tax")); os.chdir(os.path.join(root, "tax"))
    import prompt as TP
    from structured_forms import TaxPayer
    from micro_evaluation import compute_answer as tax_answer
    forms = "".join(getattr(TP, n) for n in ("basic_forms_textual", "itemized_forms", "self_employ_forms", "edu_forms",
                                              "f8863_part_iii", "schedule_8812") if hasattr(TP, n))
    forms = forms.replace("$TBD", "[__]")
    items = []
    for lvl in range(3):
        for i, p in enumerate(json.load(open(f"synthesized_problems/comp_{lvl}.json"))):
            d, tp = p["dict"], TaxPayer(**p["pydantic"])
            ans, _ = tax_answer(tp)
            facts = {k: d[k] for k in ("name", "age", "spouse_age", "blind", "spouse_blind", "filing_status", "itemized",
                                       "self_employed", "num_qualifying_children", "num_other_dependents") if k in d}
            data = {k: v for k, v in d["data"].items() if v != "$TBD"}
            q = ("Tax payer information:\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items()) +
                 "\n\nAmounts reported on the tax payer's forms (field: value; amounts in $):\n" +
                 "\n".join(f"- {k}: {v:,}" if isinstance(v, (int, float)) else f"- {k}: {v}" for k, v in data.items()) +
                 "\n\nCalculate the tax owed by this payer for 2024 step by step. DO NOT round numbers without explicit "
                 "instructions. End your response with:\n1. \"The total tax owed is $xxx.\" (xxx is a number) if there "
                 "is tax owed.\n2. \"The total tax overpaid is $xxx.\" (xxx is a number) if there is tax overpaid (and "
                 "should be refunded).")
            items.append({"id": f"t{lvl}_{i}", "level": lvl, "messages": user(q),
                          "grader": {"type": "numeric", "answer": float(ans), "tax_sign": True, "atol": 1.0}})
    os.chdir(cwd)
    sys_tax = ("You are a helpful US tax assistant. Given a tax payer's information you calculate the 2024 federal income "
               "tax they owe or overpaid.")
    out.append(scenario("rulearena/tax", "rulearena",
                        "# IRS Form 1040 (2024) and schedules: lines and instructions\n\nFields written as $name refer "
                        "to the amounts given for the tax payer; [__] marks a line you must compute.\n\n" + forms,
                        sys_tax, items))
    return out


# --- DeonticBench ---------------------------------------------------------------------------------

def deontic(root, hf):
    import pandas as pd
    out = []
    sara = "\n\n".join(f"## {f}\n" + open(os.path.join(root, "statutes", "sara", f)).read()
                       for f in sorted(os.listdir(os.path.join(root, "statutes", "sara"))))
    sys_tax = "You are a careful tax lawyer. You apply the given statutes of the US Internal Revenue Code to the case."
    w = pd.read_parquet(os.path.join(hf, "sara_numeric", "whole-00000-of-00001.parquet"))
    items = [{"id": r["id"], "messages": user(f"Case: {r['text']}\n\nQuestion: {r['question']}\n\nWork it out step by "
                                               f"step under the statutes. " + FMT_NUM),
              "grader": {"type": "numeric", "answer": float(r["label"]), "pattern": r"answer is \$?\s*(-?[\d,]+(?:\.\d+)?)",
                         "atol": 1.0, "rtol": 1e-4}} for _, r in w.iterrows()]
    out.append(scenario("deontic/sara_numeric", "deontic", "# US Internal Revenue Code, selected sections\n\n" + sara,
                        sys_tax, items))
    w = pd.read_parquet(os.path.join(hf, "sara_binary", "whole-00000-of-00001.parquet"))
    items = [{"id": r["id"], "messages": user(
        f"Case: {r['text']}\n\nClaim: {r['question']}\n\nDoes the claim follow from the statutes and the case "
        "(Entailment) or not (Contradiction)? End your response with \"Answer: Entailment\" or \"Answer: Contradiction\"."),
        "grader": {"type": "choice", "answer": "entailment" if str(r["label"]) == "1" else "contradiction",
                   "options": ["entailment", "contradiction"], "pattern": r"answer:\s*\**\s*(entailment|contradiction)"}}
        for _, r in w.iterrows()]
    out.append(scenario("deontic/sara_binary", "deontic", "# US Internal Revenue Code, selected sections\n\n" + sara,
                        sys_tax, items))
    w = pd.read_parquet(os.path.join(hf, "housing", "whole-00000-of-00001.parquet"))
    for state, g in w.groupby("state"):
        if state not in HOUSING_STATES:  # 33 states have >= 40 cases; a few suffice for the screening
            continue
        statutes = []
        for s in g["statutes"]:
            for part in re.split(r"\n(?=\[\d+\])", str(s)):
                p = re.sub(r"^\[\d+\]\s*", "", part).strip()
                if p and p not in statutes:
                    statutes.append(p)
        doc = f"# {state} housing statutes (excerpts)\n\n" + "\n\n".join(statutes)
        if len(doc.split()) > 12000:
            continue
        items = [{"id": r["id"], "messages": user(f"Question about {state} housing law: {r['question']}\n\nEnd your "
                                                   "response with \"Answer: Yes\" or \"Answer: No\"."),
                  "grader": {"type": "choice", "answer": str(r["label"]).strip().lower(), "options": ["yes", "no"],
                             "pattern": r"answer:\s*\**\s*(yes|no)"}} for _, r in g.iterrows()]
        out.append(scenario(f"deontic/housing_{state.lower().replace(' ', '_')}", "deontic", doc,
                            f"You are a careful lawyer who answers questions about {state} housing law.", items))
    return out


HOUSING_STATES = ("Texas", "Florida", "New York")


# --- KOR-Bench ------------------------------------------------------------------------------------

def kor(root):
    out = []
    for cat in ("operation", "logic", "cipher", "puzzle", "counterfactual"):
        S = [json.loads(l) for l in open(os.path.join(root, "data", cat, "sample.jsonl"))]
        R = {str(r["idx"]): r for r in map(json.loads, open(os.path.join(root, "data", cat, "rule.jsonl")))}
        ids = sorted(R, key=int)
        for g in range(5):
            group = ids[5 * g:5 * g + 5]
            doc = f"# Rules ({cat})\n\n" + "\n\n".join(f"## Rule {i}: {R[i].get('title', '')}\n{R[i]['rule_content']}"
                                                     for i in group)
            items = [{"id": f"{cat}_{s['rule_id']}_{s['idx']}", "messages": user(
                f"(This question uses Rule {s['rule_id']}.)\n\n{s['question']}"),
                "grader": {"type": "kor", "answer": s["answer"], "category": cat, "rule_id": s["rule_id"],
                           "idx": s["idx"]}} for s in S if str(s["rule_id"]) in group]
            out.append(scenario(f"kor/{cat}_g{g}", "kor", doc,
                                "You are an intelligent assistant that solves problems by strictly following the given rules.",
                                items))
    return out


SKIP_PKGS = ("torch",)  # torch is not installed in the grading environment


def keep_solvable(items):
    """Keep items whose reference solution passes its tests in this environment (library versions differ)."""
    from concurrent.futures import ThreadPoolExecutor
    from . import graders as G
    fence = "```"
    ok = list(ThreadPoolExecutor(16).map(
        lambda it: G.grade_pyexec(it["grader"], f"{fence}python\n{it['ref']}\n{fence}")["passed"], items))
    kept = [{k: v for k, v in it.items() if k != "ref"} for it, o in zip(items, ok) if o]
    print(f"   CodeUpdateArena: {len(kept)}/{len(items)} items solvable by their reference solution")
    return kept


# --- CodeUpdateArena -----------------------------------------------------------------------------

def cua(path):
    R = [json.loads(l) for l in open(path)]
    by_upd = collections.defaultdict(list)
    for r in R:
        by_upd[r["update_id"]].append(r)
    by_pkg = collections.defaultdict(dict)  # package -> api_path -> (update_id, problems)
    for uid, ps in by_upd.items():
        u = ps[0]["update"]
        best = by_pkg[u["package"]].get(u["api_path"])
        if best is None or len(ps) > len(best[1]):
            by_pkg[u["package"]][u["api_path"]] = (uid, ps)
    out = []
    for pkg, apis in by_pkg.items():
        docs, items = [], []
        for api, (uid, ps) in sorted(apis.items()):
            u = ps[0]["update"]
            docs.append(f"## {api}\n**Change:** {u['description']}\n**New signature:** `{u['signature']}`\n\n{u['docstring']}")
            parent, name = api.rsplit(".", 1)
            setup = (u["imports"] + "\n" + u["implementation"] + f"\nsetattr({parent}, '{name}', {name})\n")
            for k, p in enumerate(ps):
                if pkg in SKIP_PKGS:
                    continue
                q = (f"{p['scenario']}\n\n{p['problem']}\n\nWrite the Python function `{p['solution_signature']}` "
                     f"(you may import modules inside your code). Use the current version of `{api}` as documented. "
                     "Give the complete code in one ```python block.")
                items.append({"id": f"{uid}#{k}", "messages": user(q), "grader": {
                    "type": "pyexec", "setup": setup, "imports": p["imports"], "tests": p["unit_tests"],
                    "timeout": 20}, "ref": p["ref_solution"]})
        items = keep_solvable(items)
        if len(items) < 25:
            continue
        doc = (f"# Recent changes to the `{pkg}` API\n\nThe following functions changed in the version installed in "
               "your environment. Code must use the new behaviour.\n\n" + "\n\n".join(docs))
        out.append(scenario(f"cua/{pkg}", "cua", doc, "You are an expert Python programmer.", items))
    return out


# --- MTOB ----------------------------------------------------------------------------------------

def mtob(root):
    d = os.path.join(root, "data")
    grammar = open(os.path.join(d, "resources", "grammar_book_for_claude_medium.txt")).read()
    load = lambda f: [x for x in json.load(open(os.path.join(d, "splits", f))) if "original" in x]
    train = load("train_examples.json")
    out = []
    for direction, test_f, src, tgt in (("ke", "test_examples_ke.json", "Kalamang", "English"),
                                        ("ek", "test_examples_ek.json", "English", "Kalamang")):
        test = load(test_f)
        pool = [{"src": x["translation"] if direction == "ke" else x["original"],
                 "ref": x["original"] if direction == "ke" else x["translation"], "id": f"tr{i}"}
                for i, x in enumerate(train)]
        random.Random(1).shuffle(pool)
        mk = lambda s, r, i, sp: {"id": i, "split": sp, "messages": user(
            f"Translate the following {src} sentence into {tgt}.\n\n{s}\n\nWrite only the translation, inside \\boxed{{}}."),
            "grader": {"type": "chrf", "answer": r}}
        items = ([mk(x["src"], x["ref"], x["id"], "stream") for x in pool[:30]] +
                 [mk(x["src"], x["ref"], x["id"], "probe") for x in pool[30:45]] +
                 [mk(x["original"], x["ground_truth"], f"te{i}", "test") for i, x in enumerate(test)])
        for k, it in enumerate(items):
            it["source_id"], it["id"] = it["id"], f"mtob/{direction}/{it['split']}{k}"
        out.append({"id": f"mtob/{direction}", "source": "mtob", "meta": {"direction": direction},
                    "knowledge": "# A grammar of Kalamang (Visser 2022), excerpts\n\n" + grammar,
                    "student_system": f"You are a careful translator between {src} and {tgt}.", "items": items})
    return out


# --- Reasoning Gym --------------------------------------------------------------------------------

RG_TASKS = ["knights_knaves", "zebra_puzzles", "cryptarithm", "propositional_logic", "circuit_logic"]


def reasoning_gym():
    try:
        import reasoning_gym as rg
    except ImportError:
        print("reasoning_gym not installed; skipping")
        return []
    out = []
    for task in RG_TASKS:
        try:
            ds = rg.create_dataset(task, size=70, seed=7)
        except Exception as e:
            print("rg skip", task, e)
            continue
        doc_src = type(ds).__doc__ or task
        items = [{"id": f"{task}_{i}", "messages": user(e["question"] + "\n\nGive your final answer inside "
                                                         "<answer></answer>."),
                  "grader": {"type": "rg", "task": task, "entry": json.loads(json.dumps(e, default=str))}}
                 for i, e in enumerate(ds)]
        out.append(scenario(f"rg/{task}", "rg", f"# Task: {task}\n\n{doc_src.strip()}",
                            "You are a careful problem solver.", items))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=DATA)
    p.add_argument("--out", default="teachlab/scenarios")
    p.add_argument("--only", default="rulearena,deontic,kor,cua,mtob,rg")
    a = p.parse_args()
    a.data = os.path.abspath(a.data)
    sources = a.only.split(",")
    builders = {"rulearena": lambda: rulearena(os.path.join(a.data, "RuleArena")),
                "deontic": lambda: deontic(os.path.join(a.data, "DeonticBench"), os.path.join(a.data, "hf_DeonticBench")),
                "kor": lambda: kor(os.path.join(a.data, "KOR-Bench")),
                "cua": lambda: cua(os.path.join(a.data, "hf_CodeUpdateArena", "arena-ungrouped.jsonl")),
                "mtob": lambda: mtob(os.path.join(a.data, "mtob")),
                "rg": reasoning_gym}
    out_dir = os.path.abspath(a.out)
    for s in sources:
        scns = builders[s]()
        path = os.path.join(out_dir, f"new_{s}.jsonl")
        with open(path, "w") as f:
            for sc in scns:
                f.write(json.dumps(sc) + "\n")
        for sc in scns:
            c = collections.Counter(it["split"] for it in sc["items"])
            print(f"{sc['id']:34s} doc {len(sc['knowledge'].split()):6d} words  " + " ".join(f"{k}={v}" for k, v in c.items()))
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
