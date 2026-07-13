"""M1: tool-gain probe (PLAN §5.2) — CoT vs TIR pass@1 per MATH level.

Stratified sample (100/level, seed 42) from the cleaned train pool; two arms:
  cot: plain prompting, no tools;
  tir: native tool calling (hermes format, same `code_interpreter` schema as the
       M0 official verl SandboxTool) + local sandbox execution, max 4 tool turns.
Both arms: temperature 0.6, n=4 samples/question, pass@1 = mean correctness.

Requires a running sglang server (scripts/m1/serve_sglang.sh, tmux).
Generation is resumable (trajectories are appended as conversations finish);
scoring (math-verify) runs after generation and rewrites the trajectory files.

Usage:
  python data/tool_gain_probe.py --limit 10        # dry-run into data/probe_dryrun
  python data/tool_gain_probe.py                   # full run into data/probe
  python data/tool_gain_probe.py --score-only      # re-score existing trajectories
"""

import argparse
import ast
import asyncio
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any

import openai

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from env.sandbox import run_python  # noqa: E402

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CLEAN = os.path.join(DATA_DIR, "processed", "math_train_clean.jsonl")
# Same instruction suffix as the training data / M0 official example.
PROMPT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."
TIR_SYSTEM_PROMPT = """\
You have access to a code_interpreter tool that executes Python code. Use it for any \
non-trivial computation (arithmetic, algebra, enumeration) instead of computing by hand, \
then continue reasoning from its output.

Rules for code_interpreter:
- The "code" argument must be raw Python source; never call code_interpreter from inside the code.
- Each call runs in a fresh, stateless Python process: repeat all imports and variable \
definitions in every call.
- Write plain top-level Python with no leading indentation; print() every value you need to see.
- You have at most 4 tool calls per problem; after that, give your final answer.

Always finish with the final answer within \\boxed{}."""
# One worked example prepended as conversation history for --tir-style fewshot
# (PLAN §5.2 少样本示例引导工具格式): reason → clean printed code → read output → boxed answer.
TIR_FEWSHOT_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "What is the remainder when $2^{100}$ is divided by $1000$?" + " Let's think"
     " step by step and output the final answer within \\boxed{}."},
    {
        "role": "assistant",
        "content": "Computing $2^{100} \\bmod 1000$ by hand is error-prone, so I will use Python.",
        "tool_calls": [{
            "id": "call_fewshot_1",
            "type": "function",
            "function": {"name": "code_interpreter", "arguments": json.dumps({"code": "print(pow(2, 100, 1000))"})},
        }],
    },
    {"role": "tool", "content": "376\n", "tool_call_id": "call_fewshot_1"},
    {"role": "assistant", "content": "The code shows $2^{100} \\equiv 376 \\pmod{1000}$. "
     "The final answer is $\\boxed{376}$."},
]

# Schema identical to M0 SandboxTool.get_openai_tool_schema() output.
CODE_INTERPRETER_TOOL = {
    "type": "function",
    "function": {
        "name": "code_interpreter",
        "description": "Execute the code in the sandbox.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The code to be executed."}},
            "required": ["code"],
        },
    },
}
MAX_TURNS = 4  # tool-call rounds per trajectory (PLAN §3.2)
TOP_P = 0.95
MAX_TOKENS = 2048
CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)

logger = logging.getLogger("probe")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_questions(pool_path: str, per_level: int, seed: int = 42) -> list[dict[str, Any]]:
    by_level: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(pool_path):
        by_level[row["level"]].append(row)
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []
    for level in sorted(by_level):
        rows = sorted(by_level[level], key=lambda r: r["id"])
        picked.extend(rng.sample(rows, per_level))
    return picked


def preprocess_code(raw: str) -> str:
    """Fence strip + REPL-style auto-print.

    Robust version of the M0 SandboxTool heuristic: only wraps the last statement
    in print() when it is a bare expression (the official line-based version breaks
    on indented/assignment last lines, producing SyntaxErrors).
    """
    matches = CODE_FENCE_RE.findall(raw)
    code = matches[0].strip() if matches else raw
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # let the sandbox report the error verbatim
    last = tree.body[-1] if tree.body else None
    already_print = (
        isinstance(last, ast.Expr)
        and isinstance(last.value, ast.Call)
        and getattr(last.value.func, "id", "") == "print"
    )
    if isinstance(last, ast.Expr) and not already_print:
        tree.body[-1] = ast.Expr(
            ast.Call(func=ast.Name(id="print", ctx=ast.Load()), args=[last.value], keywords=[])
        )
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    return code


async def run_conversation(
    client: openai.AsyncOpenAI, model: str, question: str, arm: str, args: argparse.Namespace
) -> dict[str, Any]:
    """Run one (possibly multi-turn) conversation; returns messages + tool stats."""
    tir_style = args.tir_style
    messages: list[dict[str, Any]] = []
    if arm == "tir" and tir_style == "system":
        messages.append({"role": "system", "content": TIR_SYSTEM_PROMPT})
    elif arm == "tir" and tir_style == "fewshot":
        messages.extend(TIR_FEWSHOT_MESSAGES)
    # tir_style == "bare": no system prompt, exactly the M4 training-time setup
    # (verl tool_agent attaches tools via chat template only).
    messages.append({"role": "user", "content": question + PROMPT_SUFFIX})
    n_history = len(messages) - 1  # strip few-shot prefix from saved trajectories

    n_tool_calls = 0
    n_tool_errors = 0
    truncated = False
    finish_reason = ""
    for _ in range(MAX_TURNS + 1):  # MAX_TURNS tool rounds + 1 final answer round
        completion = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[CODE_INTERPRETER_TOOL] if arm == "tir" else openai.NOT_GIVEN,
            temperature=args.temperature,
            top_p=TOP_P,
            max_tokens=MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = completion.choices[0]
        finish_reason = choice.finish_reason or ""
        msg = choice.message
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_msg)
        if arm != "tir" or not msg.tool_calls:
            break
        if n_tool_calls >= MAX_TURNS:
            truncated = True  # still asking for tools after budget exhausted
            break
        for tc in msg.tool_calls:
            n_tool_calls += 1
            try:
                code = json.loads(tc.function.arguments)["code"]
                result = await asyncio.to_thread(run_python, preprocess_code(code))
                output = result["stdout"] + result["stderr"]
                if result["status"] != "ok":
                    n_tool_errors += 1
            except (json.JSONDecodeError, KeyError) as e:
                output = f"Error: invalid tool arguments ({e})"
                n_tool_errors += 1
            messages.append({"role": "tool", "content": output, "tool_call_id": tc.id})
    else:
        truncated = True

    return {
        "messages": messages[n_history:],
        "n_tool_calls": n_tool_calls,
        "n_tool_errors": n_tool_errors,
        "truncated": truncated,
        "finish_reason": finish_reason,
    }


async def generate(args: argparse.Namespace, questions: list[dict[str, Any]]) -> None:
    client = openai.AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY", max_retries=3)
    models = await client.models.list()
    model = models.data[0].id
    logger.info("server model: %s", model)

    done: dict[str, set[tuple[str, int]]] = {"cot": set(), "tir": set()}
    for arm in args.arms:
        path = traj_path(args.out_dir, arm)
        if os.path.exists(path):
            done[arm] = {(r["id"], r["sample_idx"]) for r in read_jsonl(path)}
            logger.info("resume %s: %d conversations already done", arm, len(done[arm]))

    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    n_total = sum(len(questions) * args.n_samples - len(done[arm]) for arm in args.arms)
    n_done = 0

    async def one(row: dict[str, Any], arm: str, sample_idx: int) -> None:
        nonlocal n_done
        async with sem:
            try:
                result = await run_conversation(client, model, row["question"], arm, args)
            except Exception as e:  # persistent API failure after retries: record, don't crash the batch
                logger.error("conversation failed id=%s arm=%s k=%d: %r", row["id"], arm, sample_idx, e)
                result = {"messages": [], "n_tool_calls": 0, "n_tool_errors": 0, "truncated": False,
                          "finish_reason": f"api_error: {e!r}"}
        record = {"id": row["id"], "level": row["level"], "subject": row["subject"], "gold": row["answer"],
                  "arm": arm, "sample_idx": sample_idx, **result}
        async with write_lock:
            with open(traj_path(args.out_dir, arm), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 100 == 0 or n_done == n_total:
                logger.info("progress: %d/%d", n_done, n_total)

    tasks = [
        one(row, arm, k)
        for arm in args.arms
        for row in questions
        for k in range(args.n_samples)
        if (row["id"], k) not in done[arm]
    ]
    logger.info("generating %d conversations (concurrency=%d)", len(tasks), args.concurrency)
    await asyncio.gather(*tasks)


def traj_path(out_dir: str, arm: str) -> str:
    return os.path.join(out_dir, f"trajectories_{arm}.jsonl")


def score_record(record: dict[str, Any], parse: Any, verify: Any) -> dict[str, Any]:
    """Score one trajectory with math-verify; failures are logged, never silent (禁止事项 #2)."""
    if record["truncated"]:  # PLAN §3.2: truncation forces reward 0; not an extraction failure
        return {**record, "correct": False, "invalid": False}
    final_text = record["messages"][-1]["content"] if record["messages"] else ""
    correct, invalid = False, False
    try:
        gold = parse("\\boxed{" + record["gold"] + "}")
        pred = parse(final_text)
        if not pred:
            invalid = True
            logger.warning("no answer extracted: id=%s arm=%s k=%d", record["id"], record["arm"], record["sample_idx"])
        else:
            correct = bool(verify(gold, pred))
    except Exception as e:
        invalid = True
        logger.warning("verify error id=%s arm=%s k=%d: %r", record["id"], record["arm"], record["sample_idx"], e)
    return {**record, "correct": correct, "invalid": invalid}


def score_and_report(args: argparse.Namespace) -> None:
    from math_verify import parse, verify

    metrics: dict[str, Any] = {
        "pool": TRAIN_CLEAN, "per_level": args.per_level, "n_samples": args.n_samples,
        "temperature": args.temperature, "top_p": TOP_P, "max_turns": MAX_TURNS,
        "tir_style": args.tir_style, "levels": {}, "overall": {},
    }
    scored_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in args.arms:
        records = read_jsonl(traj_path(args.out_dir, arm))
        scored = [score_record(r, parse, verify) for r in records]
        with open(traj_path(args.out_dir, arm), "w", encoding="utf-8") as f:
            for r in scored:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        scored_by_arm[arm] = scored

    def arm_stats(rows: list[dict[str, Any]], arm: str) -> dict[str, float]:
        n = len(rows)
        stats = {"n": n, "pass1": round(sum(r["correct"] for r in rows) / n, 4),
                 "invalid_rate": round(sum(r["invalid"] for r in rows) / n, 4)}
        if arm == "tir":
            stats.update(
                avg_tool_calls=round(sum(r["n_tool_calls"] for r in rows) / n, 3),
                tool_error_rate=round(
                    sum(r["n_tool_errors"] for r in rows) / max(1, sum(r["n_tool_calls"] for r in rows)), 4),
                truncated_rate=round(sum(r["truncated"] for r in rows) / n, 4),
                no_tool_use_rate=round(sum(r["n_tool_calls"] == 0 for r in rows) / n, 4),
            )
        return stats

    levels = sorted({r["level"] for rows in scored_by_arm.values() for r in rows})
    for level in levels:
        entry = {}
        for arm in args.arms:
            rows = [r for r in scored_by_arm[arm] if r["level"] == level]
            entry[arm] = arm_stats(rows, arm)
        if "cot" in entry and "tir" in entry:
            entry["gain"] = round(entry["tir"]["pass1"] - entry["cot"]["pass1"], 4)
        metrics["levels"][str(level)] = entry
    for arm in args.arms:
        metrics["overall"][arm] = arm_stats(scored_by_arm[arm], arm)
    if {"cot", "tir"} <= set(args.arms):
        metrics["overall"]["gain"] = round(
            metrics["overall"]["tir"]["pass1"] - metrics["overall"]["cot"]["pass1"], 4)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("metrics -> %s", metrics_path)
    print(json.dumps(metrics["levels"], indent=2))
    print("overall:", json.dumps(metrics["overall"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--per-level", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="dry-run: use only N questions total")
    parser.add_argument("--arms", nargs="+", default=["cot", "tir"], choices=["cot", "tir"])
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n-samples", type=int, default=4, help="samples per question (1 with temperature 0)")
    parser.add_argument("--out-dir", default=None, help="default: data/probe (data/probe_dryrun with --limit)")
    parser.add_argument("--tir-style", default="bare", choices=["bare", "system", "fewshot"],
                        help="bare = match M4 training setup (no system prompt); "
                             "system = instruction prompt; fewshot = 1 worked example as history")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(DATA_DIR, "probe_dryrun" if args.limit else "probe")
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(os.path.join(args.out_dir, "probe.log"))],
    )

    questions = sample_questions(TRAIN_CLEAN, args.per_level)
    if args.limit:
        rng = random.Random(0)
        questions = rng.sample(questions, args.limit)
    logger.info("questions: %d, arms: %s, out_dir: %s", len(questions), args.arms, args.out_dir)

    if not args.score_only:
        asyncio.run(generate(args, questions))
    score_and_report(args)


if __name__ == "__main__":
    main()
