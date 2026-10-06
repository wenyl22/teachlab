"""CL-bench's grading prompt, rebuilt from eval.py's source with only the opener replaced.

eval.py's opening line ("Starting now, you are a rigorous instruction-following grading teacher...") makes gpt-5.x
reject every request with `invalid_prompt`, deterministically. Replacing only that line fixes it.
"""

import inspect
import textwrap

from . import eval as clbench_eval

_ORIG_JUDGE_OPENER = ("Starting now, you are a rigorous instruction-following grading teacher. Your task is to "
                      "accurately grade and score student answers based on the 【Rubrics】.\n\n")
JUDGE_OPENER = "You are a strict grader. Grade the student response below against the 【Rubrics】.\n\n"


def judge_prompt(rubrics, output):
    src = inspect.getsource(clbench_eval.call_judge_api)
    body = textwrap.dedent(src[src.index("grading_prompt = ("):src.index("messages = [")])
    ns = {"rubrics_text": clbench_eval.build_rubrics_text(rubrics), "model_output": output}
    exec(body, ns)
    prompt = ns["grading_prompt"]
    assert prompt.startswith(_ORIG_JUDGE_OPENER), "eval.py's judge prompt changed; update JUDGE_OPENER"
    return JUDGE_OPENER + prompt[len(_ORIG_JUDGE_OPENER):]
