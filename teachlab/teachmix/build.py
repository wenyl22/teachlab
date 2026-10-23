"""TeachMix: one teaching benchmark from several datasets, calibrated so that the student (Qwen3-4B, best of without /
with the doc) solves ~15% and the teacher (Qwen3.8-27B with the doc) ~60% of the items. See README.md here.

    pool    write teachlab/scenarios/mixpool.jsonl: every candidate item of every viable course, all in split "test",
            so that one none/doc evaluation scores each item with a few samples (the calibration runs)
    select  read the calibration runs (outputs/mixcal_s4b: none + doc, outputs/mixcal_s27b: doc), pick 45 items per
            course whose calibration means come closest to 15 / 60 (see select()), split each course 15 / 7 / 23 into
            stream / probe / test (= train / valid / test) and write teachlab/scenarios/teachmix_v1.jsonl
    v1      rebuild teachlab/scenarios/teachmix_v1.jsonl exactly from v1_items.json (no calibration runs needed)

Viable = a correct/incorrect metric, a 27B-with-doc vs 4B gap, and not solved by the doc alone (2026-10-14 screening):
    deontic/sara_numeric, rulearena/airline, kor/{cipher,logic,puzzle}_g*, cua/*, rule_world/o6l3/*, sebench/f8v/*
Left out: sara_binary and housing (yes/no, 50% floor), MTOB (chrF, and the grammar overflows the 4B), Reasoning Gym
(the 27B does not need the doc), NBA and tax (the 27B fails too), KOR counterfactual / operation (the doc suffices).

Same logic everywhere: one session per train item (attempt -> grader feedback with the correct answer -> memory
update), closed-book evaluation on test; every grader returns a 0/1 pass. Final answers go in \\boxed{} (code: one
```python block; KOR keeps its [[...]] convention, which its questions spell out).

Rule-world and SE-Bench courses are read from scenarios/rule_world_o6l3.jsonl and scenarios/sebench_f8v.jsonl (build
them first with teachlab.adapters.rule_world --ops 6 --length 3 and teachlab.adapters.sebench --funcs 8).

    python -m teachlab.teachmix.build pool | select | v1
"""
import argparse, collections, json, os, random

from . import sources as B

DATA = B.DATA
SCN = "teachlab/scenarios"
OUT = "teachlab/outputs"
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = {"rulearena/airline": 150, "cua": 80}  # items per course in the pool (calibration cost)
BOX_NUM = "End your response with the final answer, a single number, in \\boxed{}."


def domain(cid):
    return "/".join(cid.split("/")[:2]).split("_g")[0] if cid.startswith("kor") else cid.split("/")[0]


def boxed_numeric(scn):
    """RuleArena airline / SARA numeric: ask for \\boxed{number} instead of the dataset's own end sentence."""
    for it in scn["items"]:
        m = it["messages"][-1]
        c = m["content"]
        for tail in (' and end your response with "The total cost is $xxx." (xxx is a number)', " " + B.FMT_NUM):
            c = c.replace(tail, "")
        m["content"] = c.rstrip(". ") + ". " + BOX_NUM
        it["grader"].pop("pattern", None)
    return scn


def pool_courses(sources=("deontic", "rulearena", "kor", "cua", "rule_world", "sebench")):
    """Every candidate course, all items in split "test", in a fixed order (pool ids are <course>/test<k>)."""
    B.split = lambda items, seed, *x: [{**it, "split": "test"} for it in items]  # keep everything, one split
    out = []
    if "deontic" in sources:
        out += [boxed_numeric(s) for s in B.deontic(os.path.join(DATA, "DeonticBench"),
                                                    os.path.join(DATA, "hf_DeonticBench")) if s["id"] == "deontic/sara_numeric"]
    if "rulearena" in sources:
        out += [boxed_numeric(s) for s in B.rulearena(os.path.join(DATA, "RuleArena")) if s["id"] == "rulearena/airline"]
    if "kor" in sources:
        out += [s for s in B.kor(os.path.join(DATA, "KOR-Bench")) if s["id"].split("/")[1].split("_g")[0] in
                ("cipher", "logic", "puzzle")]
    if "cua" in sources:
        out += B.cua(os.path.join(DATA, "hf_CodeUpdateArena", "arena-ungrouped.jsonl"))
    for src, f in (("rule_world", "rule_world_o6l3.jsonl"), ("sebench", "sebench_f8v.jsonl")):
        if src in sources:
            for l in open(os.path.join(SCN, f)):
                s = json.loads(l)
                s["items"] = [{**it, "split": "test", "source_id": it["id"]} for it in s["items"]]
                out.append(s)
    for s in out:
        cap = CAP.get(s["id"], CAP.get(domain(s["id"])))
        if cap and len(s["items"]) > cap:
            s["items"] = random.Random(0).sample(s["items"], cap)
    return out


def pool(a):
    with open(os.path.join(SCN, "mixpool.jsonl"), "w") as f:
        for s in pool_courses():
            f.write(json.dumps(s) + "\n")
            print(f"{s['id']:28s} {len(s['items']):4d} items  doc {len(s['knowledge'].split()):6d} words")
    print("wrote", os.path.join(SCN, "mixpool.jsonl"))


def item_scores(run, cond):
    """item id -> list of sample scores, from the calibration run."""
    res = {}
    for l in open(os.path.join(OUT, run, f"{cond}.jsonl")):
        r = json.loads(l)
        for it in (r["eval"].get("test") or {}).get("items", []):
            res[it["id"]] = it["scores"]
    return res


def select(a):
    """Per course, pick N items whose calibration means come closest to (target_s, target_t) (swap local search);
    keep courses that get within reach (student <= max_course_s, teacher >= min_course_t); then, while the pool's
    teacher mean is above target, drop the most teacher-solved course of any domain that keeps >= keep_per_domain.
    The calibration items split almost bimodally (solved by both / by neither / by the teacher only), so a 15 / 60
    mix means roughly 15% items the student knows, 45% teacher-only items and 40% items neither solves."""
    mean = lambda v: sum(v) / len(v) if v else 0.0
    s_none, s_doc = item_scores("mixcal_s4b", "none"), item_scores("mixcal_s4b", "doc")
    t_doc = item_scores("mixcal_s27b", "doc")
    scns = [json.loads(l) for l in open(os.path.join(SCN, "mixpool.jsonl"))]
    rng = random.Random(a.seed)
    f = lambda s, t: (s - a.target_s) ** 2 + (t - a.target_t) ** 2

    def solve(rows, n):
        rows = rows[:]
        rng.shuffle(rows)
        sel, rest = rows[:n], rows[n:]
        while True:
            S, T = sum(r[1] for r in sel), sum(r[2] for r in sel)
            cur, best = f(S / n, T / n), None
            for i, x in enumerate(sel):
                for j, y in enumerate(rest):
                    g = f((S - x[1] + y[1]) / n, (T - x[2] + y[2]) / n)
                    if g < cur - 1e-12 and (best is None or g < best[0]):
                        best = (g, i, j)
            if best is None:
                return sel
            _, i, j = best
            sel[i], rest[j] = rest[j], sel[i]

    cand, report = {}, []
    for s in scns:
        rows = [(it, max(mean(s_none.get(it["id"], [])), mean(s_doc[it["id"]])), mean(t_doc[it["id"]]))
                for it in s["items"] if it["id"] in s_doc and it["id"] in t_doc]
        if len(rows) < a.min_items:
            report.append(f"{s['id']:28s} dropped (only {len(rows)} items)")
            continue
        sel = solve(rows, min(a.per_course, len(rows)))
        ms, mt = mean([r[1] for r in sel]), mean([r[2] for r in sel])
        if ms > a.max_course_s or mt < a.min_course_t:
            report.append(f"{s['id']:28s} dropped (best mix student {ms:.2f} teacher {mt:.2f})")
            continue
        cand[s["id"]] = sel
    while mean([r[2] for v in cand.values() for r in v]) > a.target_t + a.tol:
        cnt = collections.Counter(domain(c) for c in cand)
        cs = [c for c in cand if cnt[domain(c)] > a.keep_per_domain]
        if not cs:
            break
        c = max(cs, key=lambda c: mean([r[2] for r in cand[c]]) - 0.5 * mean([r[1] for r in cand[c]]))
        report.append(f"{c:28s} dropped (teacher {mean([r[2] for r in cand.pop(c)]):.2f}; pool teacher mean above target)")

    out = []
    by_id = {s["id"]: s for s in scns}
    for cid, rows in cand.items():
        rng.shuffle(rows)
        n = len(rows)
        n_tr, n_va = round(a.train_frac * n), round(a.valid_frac * n)
        sp = lambda k: "stream" if k < n_tr else "probe" if k < n_tr + n_va else "test"
        s = finalize(by_id[cid], [(it, sp(k), round(ps, 3), round(pt, 3)) for k, (it, ps, pt) in enumerate(rows)],
                     domain(cid))
        out.append(s)
        t = [x for x in s["items"] if x["split"] == "test"]
        report.append(f"{cid:28s} train {n_tr:3d} valid {n_va:3d} test {len(t):3d}   calib student "
                      f"{mean([x['calib']['p_student'] for x in t]):.2f} teacher {mean([x['calib']['p_teacher'] for x in t]):.2f}")
    path = os.path.join(SCN, "teachmix_v1.jsonl")
    with open(path, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    write_spec(out, os.path.join(SCN, "teachmix_v1_items.json"))
    print("\n".join(report))
    dom = collections.defaultdict(list)
    for s in out:
        dom[s["meta"]["domain"]] += [x["calib"] for x in s["items"] if x["split"] == "test"]
    allt = [c for v in dom.values() for c in v]
    print("\n== test items by domain (calibration samples; fresh samples will differ)")
    for d, v in sorted(dom.items()):
        print(f"{d:16s} n={len(v):4d}  student {mean([c['p_student'] for c in v]):.2f}  teacher {mean([c['p_teacher'] for c in v]):.2f}")
    print(f"{'ALL':16s} n={len(allt):4d}  student {mean([c['p_student'] for c in allt]):.2f}  "
          f"teacher {mean([c['p_teacher'] for c in allt]):.2f}   courses={len(out)}")
    print("wrote", path)


def finalize(scn, rows, domain_name, benchmark="teachmix_v1"):
    """rows: [(item, split, p_student, p_teacher)] in course order -> the course as written by select()."""
    s = dict(scn)
    s["items"] = [{**it, "split": sp, "id": f"{scn['id']}/{sp}{k}", "orig_id": it["id"],
                   "calib": {"p_student": ps, "p_teacher": pt}} for k, (it, sp, ps, pt) in enumerate(rows)]
    s["meta"] = {**s.get("meta", {}), "domain": domain_name, "benchmark": benchmark}
    return s


def write_spec(courses, path):
    """The selection (pool id, source id, split, calibration scores per item), in v1_items.json's format."""
    json.dump({"benchmark": "teachmix_v1", "courses": {
        s["id"]: [{"pool_id": it["orig_id"], "source_id": it.get("source_id"), "split": it["split"],
                   "p_student": it["calib"]["p_student"], "p_teacher": it["calib"]["p_teacher"]} for it in s["items"]]
        for s in courses}}, open(path, "w"), indent=1)


def v1(a):
    """Rebuild teachmix_v1.jsonl from v1_items.json: per course, the pool items (by pool id, checked against their
    dataset-native source_id) in the stored order and splits, with their calibration scores."""
    spec = json.load(open(os.path.join(HERE, "v1_items.json")))
    courses = {s["id"]: s for s in pool_courses(tuple({c.split("/")[0] for c in spec["courses"]}))}
    out = []
    for cid, rows in spec["courses"].items():
        by_id = {it["id"]: it for it in courses[cid]["items"]}
        picked = []
        for r in rows:
            it = by_id[r["pool_id"]]
            assert it.get("source_id") == r["source_id"], \
                f"{cid}: pool item {r['pool_id']} is {it.get('source_id')}, expected {r['source_id']} (source data changed?)"
            picked.append((it, r["split"], r["p_student"], r["p_teacher"]))
        out.append(finalize(courses[cid], picked, domain(cid)))
    path = os.path.join(SCN, "teachmix_v1.jsonl")
    with open(path, "w") as f:
        for s in out:
            f.write(json.dumps(s) + "\n")
    n = collections.Counter(it["split"] for s in out for it in s["items"])
    print(f"wrote {path}: {len(out)} courses, " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["pool", "select", "v1"])
    p.add_argument("--target-s", type=float, default=0.15, help="student (4B, best of none / doc) mean")
    p.add_argument("--target-t", type=float, default=0.60, help="teacher (27B with the doc) mean")
    p.add_argument("--tol", type=float, default=0.06)
    p.add_argument("--per-course", type=int, default=45, help="items per course (train + valid + test)")
    p.add_argument("--train-frac", type=float, default=0.34)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--min-items", type=int, default=25)
    p.add_argument("--max-course-s", type=float, default=0.25, help="drop courses the student mostly knows")
    p.add_argument("--min-course-t", type=float, default=0.40, help="drop courses the teacher cannot do")
    p.add_argument("--keep-per-domain", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    {"pool": pool, "select": select, "v1": v1}[a.cmd](a)
