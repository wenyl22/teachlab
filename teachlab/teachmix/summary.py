"""Test scores on TeachMix (or any scenario set) per domain and over all test items, for one or more run directories.

Every (condition, k) with a test evaluation becomes a column; scores are averaged over courses weighted by their number
of test items. Domains: kor/<category>, otherwise the first path component of the course id.

    python -m teachlab.teachmix.summary teachlab/outputs/mix_s4b teachlab/outputs/mix_gepa
"""
import argparse, collections, glob, json, os


def domain(cid):
    return cid.split("_g")[0] if cid.startswith("kor") else cid.split("/")[0]


def load(run_dir):
    """(course, column) -> (score, n_items) from every <run_dir>/*.jsonl record with a test evaluation."""
    res = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            t = (r.get("eval") or {}).get("test")
            if t:
                col = r["condition"] + (f" k{r['k']}" if r.get("k") else "")
                res[(r["scenario"], col)] = (t["score"], t["n_items"])
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="Run directories (teachlab/outputs/<run-name>)")
    p.add_argument("--courses", action="store_true", help="Also print one row per course")
    a = p.parse_args()
    for run in a.runs:
        res = load(run)
        if not res:
            print(f"{run}: no test evaluations")
            continue
        cols = sorted({c for _, c in res}, key=lambda c: (c.split(" k")[0], int(c.split(" k")[1]) if " k" in c else 0))
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for (cid, col), x in res.items():
            agg[domain(cid)][col].append(x)
            agg["ALL"][col].append(x)
        width = max(14, *(len(c) + 2 for c in cols))
        print(f"\n== {run}\n{'':24s}" + "".join(f"{c:>{width}s}" for c in cols))
        if a.courses:
            for cid in sorted({c for c, _ in res}):
                print(f"{cid:24s}" + "".join(f"{res[(cid, c)][0]:{width}.2f}" if (cid, c) in res else f"{'-':>{width}s}"
                                            for c in cols))
        for g in sorted(agg, key=lambda g: (g == "ALL", g)):
            row = []
            for c in cols:
                v = agg[g][c]
                row.append(f"{sum(s * n for s, n in v) / sum(n for _, n in v):{width}.2f}" if v else f"{'-':>{width}s}")
            print(f"{g:24s}" + "".join(row))


if __name__ == "__main__":
    main()
