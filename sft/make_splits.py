"""M3: carve a 200-question held-out set from the training subset (PLAN §7.2).

Held-out questions are excluded from distillation-trace generation and used for
SFT acceptance (tool-format success rate, pass@1 vs zero-shot).
Outputs: sft/data/heldout_200.jsonl, sft/data/sft_pool.jsonl (stratified, seed 42).
"""

import json
import os
import random
from collections import defaultdict

SFT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_SUBSET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "train_subset.jsonl"
)
N_HELDOUT = 200


def main() -> None:
    rows = [json.loads(line) for line in open(TRAIN_SUBSET, encoding="utf-8")]
    by_level: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)

    rng = random.Random(42)
    heldout: list[dict] = []
    for level in sorted(by_level):
        pool = sorted(by_level[level], key=lambda r: r["id"])
        n = round(N_HELDOUT * len(pool) / len(rows))
        heldout.extend(rng.sample(pool, n))
    heldout_ids = {r["id"] for r in heldout}
    sft_pool = [r for r in rows if r["id"] not in heldout_ids]

    os.makedirs(SFT_DATA_DIR, exist_ok=True)
    for name, data in (("heldout_200.jsonl", heldout), ("sft_pool.jsonl", sft_pool)):
        path = os.path.join(SFT_DATA_DIR, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        levels = defaultdict(int)
        for r in data:
            levels[r["level"]] += 1
        print(f"{path}: {len(data)} rows, per level {dict(sorted(levels.items()))}")


if __name__ == "__main__":
    main()
