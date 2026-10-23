# TeachMix

TeachMix is a teaching benchmark built from several datasets. In every course the teacher holds a private reference document (statutes, a rulebook, invented rules or a synthetic arithmetic) that the student never sees. The student must learn to *apply* it. Without the document the student fails, and with it a strong model mostly succeeds.

TeachMix v1 is calibrated for student Qwen3-4B and teacher Qwen3.8-27B (reasoning off). The student (best of without and with the document) solves about 15% of the items, and the teacher with the document about 60%.

## v1 at a glance

There are 9 courses with 45 items each, split 15 train (`stream`), 7 valid (`probe`) and 23 test. That gives 207 test items in all.

| Domain | Courses | What the student must learn | Source |
|---|---|---|---|
| SARA tax amount | `deontic/sara_numeric` | Apply selected US tax statutes to a case and compute the tax | DeonticBench (SARA) |
| RuleArena airline | `rulearena/airline` | Compute baggage fees from American Airlines' policy: tables plus exceptions | RuleArena |
| KOR cipher | `kor/cipher_g3`, `kor/cipher_g4` | Run 5 invented encryption/decryption rules | KOR-Bench |
| KOR logic | `kor/logic_g0` | Apply 5 invented logic/argument rules | KOR-Bench |
| KOR puzzle | `kor/puzzle_g0`, `kor/puzzle_g1` | Solve puzzles under 5 invented rule sets | KOR-Bench |
| Rule-world | `rule_world/o6l3/w00`, `w08` | Evaluate expressions under a synthetic arithmetic | `teachlab.adapters.rule_world --ops 6 --length 3` |

**Same logic everywhere:**
- One session per train item: an attempt, grader feedback with the correct answer, then a memory update.
- Closed-book evaluation on test.
- Every grader returns a 0/1 pass.
- Final answers go in `\boxed{}`. KOR keeps the `[[...]]` format its questions spell out.

**Left out after screening** (4B and 27B, without and with the document):
- SARA yes/no and housing law: yes/no answers give a 50% floor.
- MTOB Kalamang: chrF instead of pass/fail, and the grammar overflows the 4B's context.
- Reasoning Gym: the 27B does not need the document.
- RuleArena NBA and tax: the 27B fails too.
- KOR counterfactual and operation: the document alone is enough.
- CodeUpdateArena and SE-Bench: the 4B already solves most items.
- The other KOR groups: the 4B knows too much, or the 27B too little.

**Selection:**
- Every candidate item (the pool, 2,215 items) got 3 calibration answers each from the 4B (without and with the document) and from the 27B with the document.
- Per course, a swap search picked the 45 items whose means came closest to student 0.15 / teacher 0.60.
- Courses that could not get close were dropped. Rule-world (the 27B solves nearly everything) was capped at 2 worlds.
- Teaching results were never used.
- On fresh answers, v1 scores student 0.09 (with the document) and teacher 0.69.

## Results (test, 2 answers per item)

Settings for all runs:
- 4,000-word notebook, 10 sessions unless noted.
- Student answers capped at 6,144 tokens.
- At most 6 tutor messages per session.
- GEPA trains on the first 10 train items, with 300 student attempts per course and no cap on rewrites.

Student Qwen3-4B:

| Domain | no doc | + doc | 27B summary | `teacher_note` k10 | `tutor` k10 | `tutor_note` k20 | GEPA | 27B + doc |
|---|---|---|---|---|---|---|---|---|
| SARA tax amount | .09 | .09 | .09 | .35 | .09 | .24 | .35 | .61 |
| RuleArena airline | .00 | .00 | .04 | .13 | .00 | .24 | .09 | .74 |
| KOR cipher | .00 | .03 | .10 | .13 | .00 | .08 | .20 | .65 |
| KOR logic | .13 | .13 | .17 | .33 | .15 | .33 | .39 | .41 |
| KOR puzzle | .10 | .10 | .13 | .21 | .08 | .12 | .14 | .59 |
| Rule-world | .00 | .15 | .52 | .64 | .05 | .71 | .80 | .97 |
| **All** | **.05** | **.09** | **.20** | **.31** | **.06** | **.29** | **.35** | **.69** |

Student Qwen3-1.7B, overall:

| none | doc | summary | `teacher_note` k10 | `tutor` k10 | `tutor_note` k10 | GEPA |
|---|---|---|---|---|---|---|
| .05 | .04 | .06 | .10 | .03 | .09 | .07 |

**Findings:**
- **Notes the 27B writes are what transfers.** The 4B goes from 0.09 to 0.29–0.35.
- **Notes the student writes itself don't work** (`tutor`). The tutored 4B solves 30–55% of training problems while the tutor is present, but its own notebook loses it.
- **More sessions help `tutor_note` but not `teacher_note`.** `teacher_note` is flat from session 10 to 20 (0.24–0.26 in the 20-session run); `tutor_note` is still rising at session 20.
- **GEPA's gain comes from selecting notebooks by the student's valid-split score.** With the 1.7B it overfits the 7 valid problems: up to 0.43 on valid, 0.04–0.17 on test.
- **Run-to-run variation is about ±0.07 overall.** `teacher_note` at session 10 scored 0.31 and 0.24 in two identical runs.

## Reproducing

Data goes under `$TEACHMIX_DATA` (default `data/teachmix`). The KOR grader imports KOR-Bench's own checker from there.

| Directory | Source (version used for v1) |
|---|---|
| `RuleArena/` | `git clone https://github.com/skyriver-2000/RuleArena` (59f60f6) |
| `KOR-Bench/` | `git clone https://github.com/KOR-Bench/KOR-Bench` (bb8194d) |
| `DeonticBench/` | `git clone https://github.com/guangyaodou/DeonticBench` (3c168f9) |
| `hf_DeonticBench/` | `huggingface_hub.snapshot_download("gydou/DeonticBench", repo_type="dataset", local_dir=...)` |
| `hf_CodeUpdateArena/` | `snapshot_download("leo-liuzy/CodeUpdateArena", repo_type="dataset", ...)` (pool only) |
| `mtob/` | `git clone https://github.com/lukemelas/mtob` (0f5c939), then unzip its dataset archive (screening only) |
| Reasoning Gym | `pip install reasoning-gym --no-build-isolation` (screening only) |

The pool also reads `scenarios/rule_world_o6l3.jsonl` and `scenarios/sebench_f8v.jsonl`. Build those with the adapters in the main README.

```bash
python -m teachlab.teachmix.build v1     # rebuild scenarios/teachmix_v1.jsonl from v1_items.json
```

`v1_items.json` lists every item by pool id and dataset-native `source_id`, with its split and calibration scores.

The v1 runs used Rule-world prompts from before commit 06bf145, which reworded the task descriptions. Rebuilding v1 from the current adapter gives the same items with the newer wording.

To recalibrate for another student/teacher pair:

```bash
python -m teachlab.teachmix.build pool   # scenarios/mixpool.jsonl, all candidates in split "test"
python -m teachlab.run --run-name mixcal_s4b --scenarios teachlab/scenarios/mixpool.jsonl --conditions none,doc \
    --eval-splits test --samples 3 --student-max-tokens 6144 ...        # the student
python -m teachlab.run --run-name mixcal_s27b --scenarios teachlab/scenarios/mixpool.jsonl --conditions doc \
    --eval-splits test --samples 3 ...                                  # the teacher as the student
python -m teachlab.teachmix.build select  # scenarios/teachmix_v1.jsonl + teachmix_v1_items.json
```

Teaching runs and the results table:

```bash
python -m teachlab.run --run-name mix_s4b --scenarios teachlab/scenarios/teachmix_v1.jsonl \
    --conditions none,doc,summary,teacher_note,tutor,tutor_note --budget 4000 --sessions 10 --eval-at 5,10 \
    --eval-splits test --samples 2 --max-rounds 6 --student-max-tokens 6144 --student-timeout 1800 ...
python -m teachlab.gepa_notes --scenarios teachlab/scenarios/teachmix_v1.jsonl --out teachlab/outputs/mix_gepa \
    --train-items 10 --max-metric-calls 300 --budget 4000 --samples 2 ...
python -m teachlab.teachmix.summary teachlab/outputs/mix_s4b teachlab/outputs/mix_gepa
```

The screening sets for every source course (`scenarios/new_<source>.jsonl`) come from `python -m teachlab.teachmix.sources`.
