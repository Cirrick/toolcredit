"""M1: materialize the RL training subset from the cleaned pool.

Criterion (reports/01_tool_gain.md, user-approved 2026-07-13): all of levels 3-5.
Potential tool gain (P(correct|tool ok) - P(correct|no tool)) and GRPO group
variance (mixed-outcome fraction 0.44-0.51) both concentrate in these levels.
"""

import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DATA_DIR, "processed", "math_train_clean.jsonl")
DST = os.path.join(DATA_DIR, "processed", "train_subset.jsonl")
LEVELS = {3, 4, 5}


def main() -> None:
    rows = [json.loads(line) for line in open(SRC, encoding="utf-8")]
    subset = [r for r in rows if r["level"] in LEVELS]
    with open(DST, "w", encoding="utf-8") as f:
        for r in subset:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_level = {lv: sum(r["level"] == lv for r in subset) for lv in sorted(LEVELS)}
    print(f"wrote {DST}: {len(subset)} rows, per level {per_level}")


if __name__ == "__main__":
    main()
