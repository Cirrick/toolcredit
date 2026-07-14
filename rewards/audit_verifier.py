"""M2: verifier audit on 200 real (model output, gold) pairs from the M1 probe.

Draws a fixed-seed random sample from data/probe/trajectories_{cot,tir}.jsonl,
judges each pair with rewards.verifier (both regimes), and writes an audit sheet
grouped for human review:
  A. verdict=True  via math_verify/sympy (strings differ)  -> false-positive hunt
  B. verdict=False with an extracted answer                -> false-negative hunt
  C. verdict=False without extraction                      -> quick scan
  D. verdict=True  via exact string                        -> certain, sample-checked

Output: data/verifier_audit.jsonl (full) + printed group summaries.
"""

import json
import os
import random
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from rewards.verifier import verify_answer  # noqa: E402

PROBE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "probe")
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "verifier_audit.jsonl")
N_SAMPLE = 200


def main() -> None:
    records: list[dict[str, Any]] = []
    for arm in ("cot", "tir"):
        with open(os.path.join(PROBE_DIR, f"trajectories_{arm}.jsonl"), encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    records = [r for r in records if r["messages"] and not r["truncated"]]
    rng = random.Random(7)
    sample = rng.sample(records, N_SAMPLE)

    audited = []
    for r in sample:
        final = r["messages"][-1]["content"]
        strict = verify_answer(final, r["gold"], strict_boxed=True)
        lenient = verify_answer(final, r["gold"], strict_boxed=False)
        audited.append(
            {
                "id": r["id"], "arm": r["arm"], "sample_idx": r["sample_idx"], "gold": r["gold"],
                "extracted": strict["extracted"], "strict_correct": strict["correct"],
                "lenient_correct": lenient["correct"], "method": strict["method"],
                "probe_correct": r.get("correct"),  # what M1 probe scoring said
                "final_tail": final[-300:],
            }
        )

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for a in audited:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")

    groups = {"A_true_nonexact": [], "B_false_extracted": [], "C_false_noextract": [], "D_true_exact": []}
    for a in audited:
        if a["strict_correct"] and a["method"] != "string":
            groups["A_true_nonexact"].append(a)
        elif not a["strict_correct"] and a["extracted"] is not None:
            groups["B_false_extracted"].append(a)
        elif not a["strict_correct"]:
            groups["C_false_noextract"].append(a)
        else:
            groups["D_true_exact"].append(a)
    print({k: len(v) for k, v in groups.items()}, "->", OUT_PATH)
    for key in ("A_true_nonexact", "B_false_extracted"):
        print(f"\n===== {key} =====")
        for a in groups[key]:
            print(f"[{a['id']}/{a['arm']}/{a['sample_idx']}] method={a['method']} "
                  f"gold={a['gold']!r} extracted={a['extracted']!r}")


if __name__ == "__main__":
    main()
