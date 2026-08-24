"""Build and verify the preregistered ToolCredit diagnostic freeze bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "eval/diagnostic_protocol_v1.json"
PANEL_PATH = ROOT / "eval/diagnostic_panel_v1.jsonl"
TAXONOMY_PATH = ROOT / "eval/diagnostic_taxonomy_v1.json"
LEDGER_PATH = ROOT / "plans/M7_DIAGNOSTIC_FREEZE.md"
MANIFEST_PATH = ROOT / "eval/diagnostic_freeze_v1.sha256"

SOURCE_FILES = (
    ("data/processed/math500.jsonl", 500),
    ("data/processed/aime24.jsonl", 30),
    ("data/processed/aime25.jsonl", 30),
    ("data/processed/gsm8k_test200.jsonl", 200),
)
PRIMARY_VAL = "rl/data/e3_val_math500_100.parquet"
TRAIN_DATA = "rl/data/e3_train.parquet"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob(path: Path) -> str:
    result = subprocess.run(
        ["git", "hash-object", str(path)], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def normalized_question_hash(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", question).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def parquet_ids(relative: str) -> set[str]:
    rows = pq.read_table(ROOT / relative, columns=["extra_info"]).to_pylist()
    return {str(row["extra_info"]["id"]) for row in rows}


def build_panel() -> tuple[list[dict[str, Any]], dict[str, int]]:
    primary_ids = parquet_ids(PRIMARY_VAL)
    train_ids = parquet_ids(TRAIN_DATA)
    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for relative, expected_count in SOURCE_FILES:
        source_rows = [
            json.loads(line)
            for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(source_rows) != expected_count:
            raise ValueError(f"{relative}: expected {expected_count}, found {len(source_rows)}")
        for item in source_rows:
            question_id = str(item["id"])
            roles = ["m7_full"]
            if question_id in primary_ids:
                roles.insert(0, "m6_primary_greedy")
            rows.append(
                {
                    "level": item.get("level"),
                    "panel_roles": roles,
                    "question_id": question_id,
                    "question_sha256_nfkc_whitespace": normalized_question_hash(item["question"]),
                    "source": item["source"],
                    "source_file": relative,
                    "split": item["split"],
                    "stratum": (
                        f"MATH500_level_{item['level']}"
                        if item["source"] == "MATH500"
                        else item["source"]
                    ),
                }
            )
        counts[source_rows[0]["source"]] = len(source_rows)

    ids = [row["question_id"] for row in rows]
    if len(ids) != 760 or len(ids) != len(set(ids)):
        raise ValueError("diagnostic universe must contain exactly 760 unique IDs")
    actual_primary = {row["question_id"] for row in rows if "m6_primary_greedy" in row["panel_roles"]}
    if actual_primary != primary_ids or len(actual_primary) != 100:
        raise ValueError("M6 primary IDs do not exactly match the fixed E3 validation parquet")
    if set(ids) & train_ids:
        raise ValueError("diagnostic panel contains E3 train IDs")
    PANEL_PATH.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
    return rows, counts


def taxonomy() -> dict[str, Any]:
    labels = [
        "tool_json_or_parser_failure",
        "no_tool_correct",
        "no_tool_wrong_tool_likely_helpful",
        "stateless_variable_reuse",
        "python_or_sandbox_execution_error",
        "no_recovery_after_tool_error",
        "tool_result_not_adopted_or_misread",
        "tool_correct_semantic_or_constraint_miss",
        "exact_to_low_precision_approximation",
        "boxed_or_verifier_format_error",
        "tool_overuse",
        "budget_exhaustion_or_truncation",
    ]
    return {
        "adjudication": {
            "automatic_rules": "coarse proposals only",
            "blind_to_checkpoint_identity": True,
            "correct_trajectory_primary": None,
            "rationale_required": True,
            "secondary_labels": "zero or more labels from labels; behavior labels allowed when correct",
            "wrong_trajectory_primary": "exactly one label from labels",
        },
        "labels": labels,
        "primary_decision_tree": [
            {
                "order": 1,
                "rule": "If budget exhaustion or truncation directly prevents completion, use budget_exhaustion_or_truncation.",
            },
            {
                "order": 2,
                "rule": "If the mathematical answer is otherwise correct but strict boxing/verifier formatting fails, use boxed_or_verifier_format_error.",
            },
            {
                "order": 3,
                "rule": "If no tool was called: correct is primary null with no_tool_correct secondary; wrong and a tool was plausibly useful is no_tool_wrong_tool_likely_helpful.",
            },
            {
                "order": 4,
                "rule": "For malformed tool JSON or parser rejection, use tool_json_or_parser_failure.",
            },
            {
                "order": 5,
                "rule": "For stateless reuse of a prior variable or Python/sandbox execution failure, use the more specific stateless_variable_reuse or python_or_sandbox_execution_error.",
            },
            {
                "order": 6,
                "rule": "After an execution/parser error, if no effective recovery occurs, use no_recovery_after_tool_error.",
            },
            {
                "order": 7,
                "rule": "If a visible tool result is unused or misread, use tool_result_not_adopted_or_misread.",
            },
            {
                "order": 8,
                "rule": "If tool computation is correct but semantic reasoning or a constraint is missed, use tool_correct_semantic_or_constraint_miss.",
            },
            {
                "order": 9,
                "rule": "If an exact value becomes a harmful low-precision approximation, use exact_to_low_precision_approximation.",
            },
            {
                "order": 10,
                "rule": "Use tool_overuse as primary only when unnecessary calls are the direct failure cause; when they cause exhaustion, exhaustion is primary and tool_overuse secondary.",
            },
        ],
        "taxonomy_version": "toolcredit_failure_taxonomy_v1",
        "version_change_rule": "Any label or decision-rule change requires a version bump and rerunning every compared checkpoint.",
    }


def protocol(frozen_at: str, git_head: str, source_counts: dict[str, int]) -> dict[str, Any]:
    hashes = {
        relative: sha256(ROOT / relative)
        for relative in [
            *(item[0] for item in SOURCE_FILES),
            TRAIN_DATA,
            PRIMARY_VAL,
            "rl/data/manifest.json",
            "data/contamination_report_m4.md",
            "sft/checkpoints/qwen3-1.7b-sft/model.safetensors",
            "sft/checkpoints/qwen3-1.7b-sft/chat_template.jinja",
            "sft/checkpoints/qwen3-1.7b-sft/tokenizer_config.json",
            "env/sandbox.py",
            "rl/custom/sandbox_tool.py",
            "rl/custom/tool_config.yaml",
            "rewards/verifier.py",
            "rewards/composite_reward.py",
            "rewards/format_reward.py",
            "rl/custom/reward.py",
        ]
    }
    return {
        "bootstrap": {
            "confidence_interval": "two-sided percentile 95%",
            "m6_resampling_unit": "paired question_id",
            "m7_resampling": "within frozen source strata while retaining original source weights",
            "recompute_full_statistic_each_replicate": True,
            "replicates": 10000,
            "report": ["point_estimate", "lower", "upper", "effective_replicates"],
            "seed": 42,
            "zero_denominator": "NA (0 denominator); no pseudocount",
        },
        "checkpoints": {
            "e3": "rl/runs/e3_grpo_baseline_20260819_093123/global_step_200",
            "e5": "the approved canonical E5 step-200 checkpoint; unresolved before training",
            "raw": "Qwen/Qwen3-1.7B at the project-pinned revision",
            "sft": "sft/checkpoints/qwen3-1.7b-sft",
        },
        "data_isolation": {
            "contamination_method": "normalized exact match plus word-level 13-gram overlap",
            "contamination_report": "data/contamination_report_m4.md",
            "id_overlap_with_e3_train": 0,
            "train_file": TRAIN_DATA,
        },
        "file_sha256": hashes,
        "frozen_at": frozen_at,
        "generation": {
            "base_seed": 42,
            "greedy": {"do_sample": False, "n": 1, "temperature": 0.0, "top_k": -1, "top_p": 1.0},
            "max_assistant_turns": 5,
            "max_parallel_tool_calls": 1,
            "max_response_tokens_total": 3072,
            "max_user_tool_turns": 4,
            "sample_pairing_key": ["question_id", "sample_index"],
            "sampling": {"do_sample": True, "n": 4, "temperature": 0.6, "top_k": -1, "top_p": 1.0},
            "seed_derivation": {
                "expression": "(int.from_bytes(SHA256(utf8('toolcredit-diagnostic-v1\\0' + question_id + '\\0' + str(sample_index)))[:8], 'big') & 0x7fffffff) XOR 42",
                "sample_index_origin": 0,
            },
        },
        "git_head_at_freeze": git_head,
        "pairing_and_transitions": {
            "greedy_key": ["question_id"],
            "mandatory_counts": [
                "denominator",
                "fixed_failures",
                "new_failures",
                "unchanged_correct",
                "unchanged_failed",
                "failure_category_transition_matrix",
            ],
            "sampling_key": ["question_id", "sample_index"],
            "transition_denominators": {
                "conditional_fix_rate": "E3 failures in the named category",
                "failure_category_transition": "E3 failures in the source category",
                "new_failure_rate": "E3 correct questions",
            },
        },
        "panel": {
            "m6_primary": {
                "count": 100,
                "file": PRIMARY_VAL,
                "generation": "greedy",
                "selection": "all fixed E3 validation IDs; 20 per MATH500 level 1-5",
            },
            "m7_full": {
                "count": 760,
                "source_counts": source_counts,
                "selection": "all IDs, in source-file order, from the four frozen processed test files",
            },
            "question_hash_normalization": "Unicode NFKC, collapse all whitespace runs to one ASCII space, SHA256 UTF-8",
        },
        "persistence": {
            "all_trajectories": True,
            "required_fields": [
                "checkpoint_role",
                "question_id",
                "source",
                "sample_index",
                "sample_seed",
                "prompt",
                "response",
                "reward",
                "verifier_outcome",
                "tool_metadata",
                "truncated",
                "primary_failure",
                "secondary_labels",
                "adjudication_rationale",
            ],
            "selection_policy": "save correct, failed, no-tool, error, and truncated trajectories without filtering",
        },
        "prompt": {
            "chat_template_file": "sft/checkpoints/qwen3-1.7b-sft/chat_template.jinja",
            "suffix": " Let's think step by step and output the final answer within \\boxed{}.",
        },
        "protocol_change_rule": "Any change requires a protocol version bump and complete rerun for raw, SFT, E3, and E5.",
        "protocol_version": "toolcredit_diagnostic_v1",
        "schema_version": "toolcredit_diagnostic_schema_v1",
        "taxonomy": {
            "file": "eval/diagnostic_taxonomy_v1.json",
            "version": "toolcredit_failure_taxonomy_v1",
        },
        "tool": {
            "format": "hermes",
            "max_tool_response_length": {
                "limit": 512,
                "marker": "...(truncated)...",
                "meaning": "Python str length / Unicode code points before tokenization; not a token budget",
                "operation": "if len(text) > 512: text[:256] + marker + text[-256:]",
                "side": "middle",
                "visible_only_adoption": "Only this exact post-processed/post-truncation text encoded in the rollout may contribute to s_t; raw sandbox stdout is audit-only.",
            },
            "name": "code_interpreter",
            "parallel_calls": 1,
            "sandbox_timeout_seconds": 5.0,
            "schema_file": "rl/custom/tool_config.yaml",
            "tool_class": "rl.custom.sandbox_tool.ToolCreditSandboxTool",
        },
        "verifier": {
            "invalid_policy": "answer extraction, format parsing, verifier, or reward errors are visible and counted in invalid_rate",
            "mode": "strict boxed answer with project verifier and composite reward",
            "version": "toolcredit_project_verifier_at_freeze_hash",
        },
    }


def closure_files() -> list[str]:
    return [
        "plans/M7_DIAGNOSTIC_FREEZE.md",
        "eval/diagnostic_protocol_v1.json",
        "eval/diagnostic_panel_v1.jsonl",
        "eval/diagnostic_taxonomy_v1.json",
        "eval/build_diagnostic_freeze.py",
        "eval/__init__.py",
        "rl/configs/e3_grpo_baseline.yaml",
        "rl/configs/e5_turn_credit.yaml",
        "rl/custom/turn_advantage.py",
        "rl/custom/turn_credit_agent_loop.py",
        "rl/custom/turn_credit_agent_loop_config.yaml",
        "rl/launch/e5_turn_credit.py",
        "rl/analyze_e5.py",
        "rewards/verifier.py",
        "rewards/composite_reward.py",
        "rewards/format_reward.py",
        "rl/custom/reward.py",
        "env/sandbox.py",
        "rl/custom/sandbox_tool.py",
        "rl/custom/tool_config.yaml",
        "data/contamination_report_m4.md",
        "rl/data/manifest.json",
        "rl/data/e3_train.parquet",
        "rl/data/e3_val_math500_100.parquet",
        "data/processed/math500.jsonl",
        "data/processed/aime24.jsonl",
        "data/processed/aime25.jsonl",
        "data/processed/gsm8k_test200.jsonl",
        "sft/checkpoints/qwen3-1.7b-sft/model.safetensors",
        "sft/checkpoints/qwen3-1.7b-sft/chat_template.jinja",
        "sft/checkpoints/qwen3-1.7b-sft/tokenizer_config.json",
    ]


def build(frozen_at: str) -> None:
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    rows, source_counts = build_panel()
    write_json(TAXONOMY_PATH, taxonomy())
    write_json(PROTOCOL_PATH, protocol(frozen_at, git_head, source_counts))

    component_lines = []
    for path in (PROTOCOL_PATH, PANEL_PATH, TAXONOMY_PATH):
        component_lines.append(
            f"| `{path.relative_to(ROOT)}` | `{sha256(path)}` | `{git_blob(path)}` |"
        )
    LEDGER_PATH.write_text(
        "# M7 diagnostic protocol v1 freeze ledger\n\n"
        f"- 冻结时间（UTC）：`{frozen_at}`\n"
        f"- Git HEAD：`{git_head}`\n"
        "- 决定：冻结并批准用于 M6 E5 smoke 前门禁及后续 E3→E5/M7 配对诊断。\n"
        "- 批准依据：用户于 2026-08-22 明确授权创建、验证、hash 并冻结完整 M7 bundle；所有门禁通过后可继续 5-step E5 smoke。\n"
        "- 状态：`FROZEN_V1`。任何协议、panel、taxonomy、generation、tool、verifier、pairing 或 bootstrap 变更都必须 version bump，并对 raw/SFT/E3/E5 全部重跑。\n\n"
        "## 冻结内容\n\n"
        f"M6 primary 是 fixed MATH500-100 的全部 100 题；M7 universe 是四个冻结 test source 的全部 {len(rows)} 题。"
        "panel 与 E3 train ID 交集为 0；既有 normalized exact + 13-gram contamination report 命中为 0。\n\n"
        "`max_tool_response_length=512` 冻结为 pin veRL 0.8.0 的 Python 字符串长度截断：tokenization 前检查 `len(text)`，"
        "超过时使用 `text[:256] + \"...(truncated)...\" + text[-256:]`。它不是 512-token budget。"
        "adoption 只能读取实际编码进 rollout、模型可见的这个 post-processed/post-truncation 文本；raw sandbox stdout 仅供 audit。\n\n"
        "## Canonical component ledger\n\n"
        "JSON 使用 UTF-8、sorted keys、compact separators、单个尾随换行；JSONL 每行按同一规则 canonicalize。\n\n"
        "| artifact | SHA256 | Git blob ID |\n|---|---|---|\n"
        + "\n".join(component_lines)
        + "\n\n完整闭包由 `eval/diagnostic_freeze_v1.sha256` 锁定；manifest 不自哈希。"
        "其 SHA256 必须由每次 smoke/formal preflight 单独记录。\n",
        encoding="utf-8",
    )

    manifest_lines = [f"{sha256(ROOT / relative)}  {relative}" for relative in closure_files()]
    MANIFEST_PATH.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")


def verify() -> None:
    entries: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split(maxsplit=1)
        if relative in entries or len(digest) != 64:
            raise ValueError(f"invalid manifest entry: {line}")
        entries[relative] = digest
    expected = set(closure_files())
    if set(entries) != expected:
        raise ValueError(f"manifest closure mismatch: missing={expected-set(entries)}, extra={set(entries)-expected}")
    for relative, expected_hash in entries.items():
        actual = sha256(ROOT / relative)
        if actual != expected_hash:
            raise ValueError(f"hash mismatch for {relative}: {actual} != {expected_hash}")

    protocol_value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    taxonomy_value = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    panel_rows = [json.loads(line) for line in PANEL_PATH.read_text(encoding="utf-8").splitlines()]
    if protocol_value["tool"]["max_tool_response_length"]["meaning"] != (
        "Python str length / Unicode code points before tokenization; not a token budget"
    ):
        raise ValueError("tool response truncation semantics drifted")
    if protocol_value["tool"]["max_tool_response_length"]["visible_only_adoption"].startswith("Only this exact") is False:
        raise ValueError("visible-only adoption requirement missing")
    if len(panel_rows) != 760 or sum("m6_primary_greedy" in r["panel_roles"] for r in panel_rows) != 100:
        raise ValueError("diagnostic panel counts drifted")
    if len(taxonomy_value["labels"]) != 12:
        raise ValueError("taxonomy count drifted")
    if "FROZEN_V1" not in LEDGER_PATH.read_text(encoding="utf-8"):
        raise ValueError("freeze ledger lacks approval state")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-at", help="UTC ISO-8601 timestamp; required when building")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        if not args.frozen_at:
            parser.error("--frozen-at is required when building")
        build(args.frozen_at)
        verify()
        print(f"manifest_sha256={sha256(MANIFEST_PATH)}")


if __name__ == "__main__":
    main()
