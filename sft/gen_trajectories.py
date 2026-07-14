"""M3: distillation-trace generation with rejection sampling (PLAN §7.1).

Teacher: local Qwen3-8B behind sglang (scripts/m1/serve_sglang.sh with MODEL=...),
prompted in the exact training-time format (bare TIR, hermes tool calls, sandbox
really executes). Reuses the M1 probe conversation loop.

Rejection-sampling filters (plans/M3.md, all failures counted, never silent):
  keep_correct     strict-boxed verifier says the final answer is right
  keep_tool        >=1 successful tool call (error-then-recovery traces allowed)
  keep_not_trunc   not truncated by the max_turns budget
  keep_length      student-tokenized response fits the 3072-token training budget

Usage:
  python sft/gen_trajectories.py --limit 12          # dry-run into sft/data/gen_dryrun
  python sft/gen_trajectories.py                     # full run into sft/data/gen
  python sft/gen_trajectories.py --filter-only      # re-filter existing raw traces
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any

import openai

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
from data.tool_gain_probe import run_conversation  # noqa: E402
from rewards.verifier import verify_answer  # noqa: E402

SFT_DATA_DIR = os.path.join(PROJECT_DIR, "sft", "data")
POOL_PATH = os.path.join(SFT_DATA_DIR, "sft_pool.jsonl")
STUDENT_TOKENIZER_PATH = os.path.expanduser("~/Qwen/Qwen3-1.7B")  # same vocab family as teacher
N_SAMPLES = 2
TEMPERATURE = 0.7
MAX_RESPONSE_TOKENS = 3072  # student training budget (PLAN §3.2)

logger = logging.getLogger("gen_traj")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


async def generate(args: argparse.Namespace, questions: list[dict[str, Any]]) -> None:
    client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY", max_retries=3)
    model = (await client.models.list()).data[0].id
    logger.info("teacher model: %s", model)

    raw_path = os.path.join(args.out_dir, "raw_traces.jsonl")
    done: set[tuple[str, int]] = set()
    if os.path.exists(raw_path):
        done = {(r["id"], r["sample_idx"]) for r in read_jsonl(raw_path)}
        logger.info("resume: %d conversations already done", len(done))

    conv_args = SimpleNamespace(tir_style="bare", temperature=TEMPERATURE)
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    n_done, n_total = 0, len(questions) * N_SAMPLES - len(done)

    async def one(row: dict[str, Any], sample_idx: int) -> None:
        nonlocal n_done
        async with sem:
            try:
                result = await run_conversation(client, model, row["question"], "tir", conv_args)
            except Exception as e:
                logger.error("conversation failed id=%s k=%d: %r", row["id"], sample_idx, e)
                result = {"messages": [], "n_tool_calls": 0, "n_tool_errors": 0, "truncated": True,
                          "finish_reason": f"api_error: {e!r}"}
        record = {"id": row["id"], "level": row["level"], "gold": row["answer"],
                  "sample_idx": sample_idx, **result}
        async with write_lock:
            with open(raw_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 200 == 0 or n_done == n_total:
                logger.info("progress: %d/%d", n_done, n_total)

    tasks = [one(row, k) for row in questions for k in range(N_SAMPLES) if (row["id"], k) not in done]
    logger.info("generating %d conversations (concurrency=%d)", len(tasks), args.concurrency)
    await asyncio.gather(*tasks)


def response_token_len(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    """Student-tokenizer length of everything after the initial user turn."""
    full = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, enable_thinking=False
    )
    prompt = tokenizer.apply_chat_template(
        messages[:1], tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    return len(full) - len(prompt)


def subsample(kept: list[dict[str, Any]], max_traces: int, seed: int = 42) -> list[dict[str, Any]]:
    """Cap the dataset at max_traces: 1 trace/question first (prefer error-recovery,
    then shorter), stratified by level; fill remaining quota with second traces."""
    import random

    rng = random.Random(seed)
    by_q: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in kept:
        by_q[r["id"]].append(r)
    primary, extras = [], []
    for traces in by_q.values():
        ranked = sorted(traces, key=lambda r: (-(r["n_tool_errors"] > 0), r["response_tokens"]))
        primary.append(ranked[0])
        extras.extend(ranked[1:])
    if len(primary) >= max_traces:
        by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in primary:
            by_level[r["level"]].append(r)
        out: list[dict[str, Any]] = []
        for level, rows in sorted(by_level.items()):
            quota = round(max_traces * len(rows) / len(primary))
            out.extend(rng.sample(rows, min(quota, len(rows))))
        return out
    rng.shuffle(extras)
    return primary + extras[: max_traces - len(primary)]


def filter_traces(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER_PATH)
    raw = read_jsonl(os.path.join(args.out_dir, "raw_traces.jsonl"))
    kept: list[dict[str, Any]] = []
    reasons: Counter = Counter()
    per_level_kept: Counter = Counter()
    per_level_total: Counter = Counter()

    for r in raw:
        per_level_total[r["level"]] += 1
        final = r["messages"][-1]["content"] if r["messages"] else ""
        if r["truncated"]:
            reasons["truncated"] += 1
            continue
        if r["n_tool_calls"] - r["n_tool_errors"] < 1:
            reasons["no_successful_tool_call"] += 1
            continue
        if any(
            set(json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str)
                else tc["function"]["arguments"]) != {"code"}
            for m in r["messages"] for tc in (m.get("tool_calls") or [])
        ):  # malformed arg keys would teach the wrong call format (and break replay)
            reasons["bad_tool_args"] += 1
            continue
        verdict = verify_answer(final, r["gold"], strict_boxed=True)
        if verdict["invalid"]:
            reasons["verify_invalid"] += 1
            continue
        if not verdict["correct"]:
            reasons["wrong_answer"] += 1
            continue
        n_tokens = response_token_len(tokenizer, r["messages"])
        if n_tokens > MAX_RESPONSE_TOKENS:
            reasons["too_long"] += 1
            continue
        kept.append({**r, "response_tokens": n_tokens})
    reasons["kept_before_subsample"] = len(kept)
    if args.max_traces and len(kept) > args.max_traces:
        kept = subsample(kept, args.max_traces)
    for r in kept:
        per_level_kept[r["level"]] += 1
    reasons["kept"] = len(kept)

    out_path = os.path.join(SFT_DATA_DIR, "sft_traces.jsonl" if not args.limit else "sft_traces_dryrun.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_err_recovery = sum(1 for r in kept if r["n_tool_errors"] > 0)
    stats = {
        "teacher": "Qwen3-8B", "temperature": TEMPERATURE, "n_samples": N_SAMPLES,
        "raw": len(raw), "kept": len(kept), "yield": round(len(kept) / max(1, len(raw)), 4),
        "filter_reasons": dict(reasons),
        "kept_per_level": {str(k): v for k, v in sorted(per_level_kept.items())},
        "raw_per_level": {str(k): v for k, v in sorted(per_level_total.items())},
        "kept_with_error_recovery": n_err_recovery,
        "questions_covered": len({r["id"] for r in kept}),
        "avg_response_tokens": round(sum(r["response_tokens"] for r in kept) / max(1, len(kept)), 1),
        "avg_tool_calls": round(sum(r["n_tool_calls"] for r in kept) / max(1, len(kept)), 2),
    }
    with open(os.path.join(args.out_dir, "gen_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("traces -> %s", out_path)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--limit", type=int, default=0, help="dry-run on N questions")
    parser.add_argument("--concurrency", type=int, default=48)
    parser.add_argument("--filter-only", action="store_true")
    parser.add_argument("--max-traces", type=int, default=2500, help="PLAN §7.1 target ceiling; 0 = no cap")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(SFT_DATA_DIR, "gen_dryrun" if args.limit else "gen")
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out_dir, "gen.log"))],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # keep the log readable

    questions = read_jsonl(POOL_PATH)
    if args.limit:
        import random

        questions = random.Random(0).sample(questions, args.limit)
    logger.info("questions: %d", len(questions))

    if not args.filter_only:
        asyncio.run(generate(args, questions))
    filter_traces(args)


if __name__ == "__main__":
    main()
