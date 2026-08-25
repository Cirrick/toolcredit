"""Frozen-protocol M7 generation wiring, key universe, and durable shard helpers.

Importing this module never loads a model or starts generation. The ``generate-shard``
command additionally requires an explicit environment approval guard; the approved
pre-smoke work uses only ``preflight`` and CPU fixtures.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import unicodedata
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from rl.custom.turn_advantage import LEDGER_KEY
from rl.custom.turn_credit_agent_loop import TurnCreditAgentLoop

from eval.checkpoints import COMMON_TOKENIZER, ROOT, resolved_checkpoint_roles, sha256

CONFIG_PATH = ROOT / "eval/m7_eval_config.json"
PANEL_PATH = ROOT / "eval/diagnostic_panel_v1.jsonl"
PROTOCOL_V2_PATH = ROOT / "eval/diagnostic_protocol_v2.json"
ROLE_NAMES = (
    "raw_base_model",
    "sft_start",
    "e3_grpo_baseline_step_200",
    "e5_turn_credit_step_200",
)
SOURCE_COUNTS = {"MATH500": 500, "AIME2024": 30, "AIME2025": 30, "GSM8K": 200}
EVALUATOR_VERSION = "toolcredit_m7_evaluator_v1"
SEED_PREFIX = b"toolcredit-diagnostic-v1\0"


class M7MetadataAgentLoop(TurnCreditAgentLoop):
    """M7 audit-only name for the frozen loop; generation methods are not overridden."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON mapping at {path}")
    return value


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def question_hash(text: str) -> str:
    return hashlib.sha256(normalize_question(text).encode("utf-8")).hexdigest()


def derive_sample_seed(question_id: str, sample_index: int, base_seed: int = 42) -> int:
    if not question_id or "\0" in question_id:
        raise ValueError("question_id must be non-empty and cannot contain NUL")
    if sample_index < 0:
        raise ValueError("sample_index must be nonnegative")
    payload = SEED_PREFIX + question_id.encode("utf-8") + b"\0" + str(sample_index).encode("ascii")
    return (int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF) ^ base_seed


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"partial JSONL line at {path}:{line_number}")
            if not line.strip():
                raise ValueError(f"blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"non-object JSONL row at {path}:{line_number}")
            rows.append(value)
    return rows


def load_panel() -> list[dict[str, Any]]:
    panel = _read_jsonl(PANEL_PATH)
    if len(panel) != 760 or len({row.get("question_id") for row in panel}) != 760:
        raise ValueError("frozen diagnostic panel must contain 760 unique question IDs")
    observed: dict[str, int] = {}
    for row in panel:
        source = str(row.get("source"))
        observed[source] = observed.get(source, 0) + 1
    if observed != SOURCE_COUNTS:
        raise ValueError(f"frozen panel source counts drifted: {observed}")
    return panel


def load_panel_questions() -> list[dict[str, Any]]:
    panel = load_panel()
    sources: dict[str, dict[str, Any]] = {}
    config = load_json(CONFIG_PATH)
    for relative in config["source_files"]:
        for row in _read_jsonl(ROOT / relative):
            question_id = str(row.get("id"))
            if question_id in sources:
                raise ValueError(f"duplicate question ID across source files: {question_id}")
            sources[question_id] = row
    joined: list[dict[str, Any]] = []
    for panel_row in panel:
        question_id = panel_row["question_id"]
        if question_id not in sources:
            raise KeyError(f"panel question is absent from frozen source file: {question_id}")
        source_row = sources[question_id]
        if source_row.get("source") != panel_row.get("source") or source_row.get("split") != "test":
            raise ValueError(f"source/split drift for {question_id}")
        actual_hash = question_hash(str(source_row["question"]))
        if actual_hash != panel_row.get("question_sha256_nfkc_whitespace"):
            raise ValueError(f"question hash drift for {question_id}")
        joined.append({**panel_row, "question": source_row["question"], "answer": source_row["answer"]})
    return joined


def setting_sample_indices(setting: str) -> range:
    if setting == "greedy":
        return range(1)
    if setting == "sampled":
        return range(4)
    raise ValueError(f"unknown evaluation setting: {setting}")


def expected_keys(
    roles: Iterable[str] = ROLE_NAMES,
    settings: Iterable[str] = ("greedy", "sampled"),
) -> list[tuple[str, str, str, int, int]]:
    keys: list[tuple[str, str, str, int, int]] = []
    panel = load_panel()
    for role in roles:
        if role not in ROLE_NAMES:
            raise ValueError(f"unknown role in expected key universe: {role}")
        for setting in settings:
            for row in panel:
                for sample_index in setting_sample_indices(setting):
                    keys.append(
                        (
                            role,
                            setting,
                            str(row["question_id"]),
                            sample_index,
                            derive_sample_seed(str(row["question_id"]), sample_index),
                        )
                    )
    if len(keys) != len(set(keys)):
        raise AssertionError("M7 expected key universe is not unique")
    return keys


def canonical_key(row: dict[str, Any]) -> tuple[str, str, str, int, int]:
    return (
        str(row["checkpoint_role"]),
        str(row["evaluation_setting"]),
        str(row["question_id"]),
        int(row["sample_index"]),
        int(row["sample_seed"]),
    )


def validate_frozen_runtime_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = load_json(CONFIG_PATH) if config is None else config
    v2 = load_json(PROTOCOL_V2_PATH)
    generation = config.get("generation", {})
    expected = v2["generation"]
    exact_pairs = {
        "greedy": (generation.get("greedy"), expected["greedy"]),
        "sampling": (generation.get("sampling"), expected["sampling"]),
        "max_response_tokens_total": (generation.get("max_response_tokens_total"), expected["max_response_tokens_total"]),
        "max_user_tool_turns": (generation.get("max_user_tool_turns"), expected["max_user_tool_turns"]),
        "max_assistant_turns": (generation.get("max_assistant_turns"), expected["max_assistant_turns"]),
        "max_parallel_tool_calls": (generation.get("max_parallel_tool_calls"), expected["max_parallel_tool_calls"]),
        "max_tool_response_length": (generation.get("max_tool_response_length"), v2["tool"]["max_tool_response_length"]["limit"]),
        "tool_response_truncate_side": (generation.get("tool_response_truncate_side"), v2["tool"]["max_tool_response_length"]["side"]),
        "sandbox_timeout_seconds": (generation.get("sandbox_timeout_seconds"), v2["tool"]["sandbox_timeout_seconds"]),
        "prompt_suffix": (generation.get("prompt_suffix"), v2["prompt"]["suffix"]),
    }
    drift = {name: pair for name, pair in exact_pairs.items() if pair[0] != pair[1]}
    if drift:
        raise ValueError(f"M7 runtime config drifted from frozen v2 generation semantics: {drift}")
    if generation.get("enable_thinking") is not False:
        raise ValueError("M7 must use enable_thinking=false")
    if (ROOT / config["common_tokenizer_path"]).resolve() != COMMON_TOKENIZER.resolve():
        raise ValueError("M7 evaluator tokenizer is not the frozen SFT tokenizer")
    evidence = config.get("diagnostic_evidence", {})
    if evidence.get("greedy_paired_transitions") != "primary_diagnostic":
        raise ValueError("greedy paired transitions must remain primary diagnostic evidence")
    if evidence.get("sampled_matched_index_transitions") != "secondary_exploratory":
        raise ValueError("sampled transitions must remain secondary/exploratory")
    if "causal" not in str(evidence.get("sampled_guardrail", "")):
        raise ValueError("sampled transition guardrail must prohibit strong causal claims")
    return config


def validate_raw_generation_row(row: dict[str, Any], question: dict[str, Any]) -> None:
    required = {
        "protocol_version", "schema_version", "evaluator_version", "freeze_manifest_sha256",
        "checkpoint_role", "checkpoint_identity", "evaluation_setting", "question_id", "source",
        "split", "stratum", "question_hash", "sample_index", "sample_seed", "prompt", "prompt_hash",
        "messages", "response", "response_token_count", "termination_reason", "tool_metadata",
        "infrastructure_status", "raw_output_sha256",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"raw generation row is missing fields: {missing}")
    if row["infrastructure_status"] != "ok":
        raise ValueError("infrastructure failures cannot be accepted as model trajectories")
    if row["question_id"] != question["question_id"] or row["source"] != question["source"]:
        raise ValueError("raw generation row question identity mismatch")
    if row["question_hash"] != question["question_sha256_nfkc_whitespace"]:
        raise ValueError("raw generation row question hash mismatch")
    setting = str(row["evaluation_setting"])
    if int(row["sample_index"]) not in setting_sample_indices(setting):
        raise ValueError("sample index is outside its frozen setting")
    expected_seed = derive_sample_seed(str(row["question_id"]), int(row["sample_index"]))
    if int(row["sample_seed"]) != expected_seed:
        raise ValueError("raw generation row sample seed drifted")
    if row["checkpoint_role"] not in ROLE_NAMES:
        raise ValueError("raw generation row has unknown checkpoint role")
    response = str(row["response"])
    if row["raw_output_sha256"] != hashlib.sha256(response.encode("utf-8")).hexdigest():
        raise ValueError("raw generation response hash mismatch")


def validated_resume_rows(path: Path, questions: dict[str, dict[str, Any]]) -> dict[tuple[str, str, str, int, int], dict[str, Any]]:
    if not path.exists():
        return {}
    accepted: dict[tuple[str, str, str, int, int], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        question_id = str(row.get("question_id"))
        if question_id not in questions:
            raise ValueError(f"resume shard contains unknown question ID: {question_id}")
        validate_raw_generation_row(row, questions[question_id])
        key = canonical_key(row)
        if key in accepted:
            if accepted[key] != row:
                raise ValueError(f"conflicting duplicate resume key: {key}")
            raise ValueError(f"duplicate resume key: {key}")
        accepted[key] = row
    return accepted


def append_jsonl_fsync(path: Path, row: dict[str, Any]) -> None:
    """Append one complete JSON object with O_APPEND and fsync durability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short JSONL append: {written} != {len(payload)}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def archive_invalid_shard(path: Path, recovery_root: Path, stamp: str) -> Path:
    """Move an invalid partial shard within its run so canonical output remains clean."""
    target = recovery_root / stamp / path.name
    if target.exists():
        raise FileExistsError(f"recovery target already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, target)
    return target


class SGLangHTTPServer:
    """Minimal TokenOutput adapter for SGLang's native ``/generate`` endpoint."""

    def __init__(self, endpoint: str):
        self.base_url = endpoint.rstrip("/")
        self.endpoint = self.base_url + "/generate"

    async def validate_model_path(self, expected: Path) -> dict[str, Any]:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url + "/get_model_info") as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"SGLang model identity endpoint failed HTTP {response.status}: {body[:2000]}")
        value = json.loads(body)
        advertised = value.get("model_path") or value.get("model-path")
        if not isinstance(advertised, str) or Path(advertised).resolve() != expected.resolve():
            raise RuntimeError(f"SGLang serves the wrong checkpoint: {advertised!r} != {expected}")
        return value

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], **_: Any) -> Any:
        import aiohttp
        from verl.workers.rollout.replica import TokenOutput

        params = to_sglang_sampling_params(sampling_params)
        payload = {"input_ids": prompt_ids, "sampling_params": params}
        async with aiohttp.ClientSession() as session:
            async with session.post(self.endpoint, json=payload) as response:
                body = await response.text()
                if response.status != 200:
                    raise RuntimeError(f"SGLang infrastructure error HTTP {response.status}: {body[:2000]}")
        value = json.loads(body)
        token_ids = value.get("output_ids")
        if not isinstance(token_ids, list) or not all(isinstance(item, int) for item in token_ids):
            raise RuntimeError("SGLang response did not contain integer output_ids")
        finish_reason = value.get("meta_info", {}).get("finish_reason")
        return TokenOutput(token_ids=token_ids, stop_reason=str(finish_reason), num_preempted=0)


def to_sglang_sampling_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Translate frozen evaluator controls to pinned SGLang 0.5.8's native schema."""
    params = dict(sampling_params)
    do_sample = params.pop("do_sample", None)
    params.pop("n", None)
    seed = params.pop("seed", None)
    if not isinstance(seed, int):
        raise ValueError("every M7 SGLang request must carry its frozen integer seed")
    params["sampling_seed"] = seed
    params["max_new_tokens"] = int(params.pop("max_tokens", params.get("max_new_tokens", 3072)))
    if do_sample is False and float(params.get("temperature", 0.0)) != 0.0:
        raise ValueError("greedy do_sample=false requires frozen temperature=0")
    if do_sample is True and float(params.get("temperature", 0.0)) <= 0.0:
        raise ValueError("sampled do_sample=true requires positive frozen temperature")
    allowed = {
        "max_new_tokens", "stop", "stop_token_ids", "temperature", "top_p", "top_k",
        "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty", "min_new_tokens",
        "ignore_eos", "skip_special_tokens", "spaces_between_special_tokens", "sampling_seed",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError(f"M7 sampling parameters are unsupported by pinned SGLang: {unknown}")
    return params


def build_m7_loop(tokenizer: Any, server: Any, config: dict[str, Any]) -> M7MetadataAgentLoop:
    """Assemble the frozen real veRL state machine with the single project sandbox."""
    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.utils.chat_template import initialize_system_prompt

    from rl.custom.sandbox_tool import ToolCreditSandboxTool

    generation = config["generation"]
    tool_config = {"timeout": generation["sandbox_timeout_seconds"]}
    schema = ToolCreditSandboxTool(tool_config, None).get_openai_tool_schema()
    tool = ToolCreditSandboxTool(tool_config, schema)
    loop = M7MetadataAgentLoop.__new__(M7MetadataAgentLoop)
    loop.tokenizer = tokenizer
    loop.processor = None
    loop.apply_chat_template_kwargs = {"enable_thinking": False}
    loop.mm_processor_kwargs = {}
    loop.system_prompt = initialize_system_prompt(tokenizer, enable_thinking=False)
    loop.loop = asyncio.get_running_loop()
    loop.server_manager = server
    loop.tools = {"code_interpreter": tool}
    loop.tool_schemas = [schema.model_dump(exclude_unset=True, exclude_none=True)]
    loop.tool_parser = ToolParser.get_tool_parser("hermes", tokenizer)
    loop.tool_parser_name = "hermes"
    loop.max_user_turns = generation["max_user_tool_turns"]
    loop.max_assistant_turns = generation["max_assistant_turns"]
    loop.max_parallel_calls = generation["max_parallel_tool_calls"]
    loop.max_tool_response_length = generation["max_tool_response_length"]
    loop.tool_response_truncate_side = generation["tool_response_truncate_side"]
    loop.prompt_length = 4096
    loop.response_length = generation["max_response_tokens_total"]
    loop.rollout_config = SimpleNamespace(prompt_length=loop.prompt_length, response_length=loop.response_length)
    return loop


def _plain_metrics(metrics: Any) -> dict[str, Any]:
    if is_dataclass(metrics):
        return asdict(metrics)
    if isinstance(metrics, dict):
        return dict(metrics)
    return dict(vars(metrics))


def _sampling_params(config: dict[str, Any], setting: str, seed: int) -> dict[str, Any]:
    frozen = config["generation"]["greedy" if setting == "greedy" else "sampling"]
    value = dict(frozen)
    value.pop("n", None)
    value["seed"] = seed
    value["max_new_tokens"] = config["generation"]["max_response_tokens_total"]
    return value


async def generate_one(
    loop: M7MetadataAgentLoop,
    tokenizer: Any,
    question: dict[str, Any],
    role: str,
    setting: str,
    sample_index: int,
    checkpoint_identity: dict[str, Any],
    protocol_version: str,
    manifest_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    seed = derive_sample_seed(question["question_id"], sample_index)
    prompt = str(question["question"]) + config["generation"]["prompt_suffix"]
    raw_prompt = [{"role": "user", "content": prompt}]
    try:
        output = await loop.run(
            sampling_params=_sampling_params(config, setting, seed),
            raw_prompt=raw_prompt,
        )
    except Exception as exc:
        raise RuntimeError(
            f"infrastructure/model-serving failure for {(role, setting, question['question_id'], sample_index)}"
        ) from exc
    ledger = output.extra_fields.get(LEDGER_KEY)
    if not isinstance(ledger, dict):
        raise RuntimeError("M7 metadata loop returned no structured audit ledger")
    response = tokenizer.decode(output.response_ids, skip_special_tokens=True)
    prompt_ids_hash = hashlib.sha256(
        b"".join(int(item).to_bytes(4, "big", signed=False) for item in output.prompt_ids)
    ).hexdigest()
    messages: list[dict[str, Any]] = []
    for turn in ledger["assistant_turns"]:
        messages.append({"role": "assistant", "content": turn.get("assistant_text", ""), "turn_audit": turn})
        if turn.get("visible_tool_response_text") is not None:
            messages.append({"role": "tool", "content": turn["visible_tool_response_text"]})
    row = {
        "protocol_version": protocol_version,
        "schema_version": "toolcredit_m7_trajectory_schema_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "freeze_manifest_sha256": manifest_sha256,
        "checkpoint_role": role,
        "checkpoint_identity": checkpoint_identity,
        "evaluation_setting": setting,
        "question_id": question["question_id"],
        "source": question["source"],
        "split": question["split"],
        "stratum": question["stratum"],
        "question_hash": question["question_sha256_nfkc_whitespace"],
        "sample_index": sample_index,
        "sample_seed": seed,
        "prompt": prompt,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_ids_sha256": prompt_ids_hash,
        "messages": messages,
        "response": response,
        "response_token_count": len(output.response_ids),
        "response_mask": output.response_mask,
        "termination_reason": ledger.get("termination_reason"),
        "tool_metadata": {**output.extra_fields, "agent_loop_metrics": _plain_metrics(output.metrics)},
        "infrastructure_status": "ok",
        "raw_output_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
    }
    validate_raw_generation_row(row, question)
    return row


def preflight() -> dict[str, Any]:
    config = validate_frozen_runtime_config()
    panel = load_panel_questions()
    roles = resolved_checkpoint_roles(require_materialized=True)
    keys = expected_keys()
    free_bytes = shutil.disk_usage(ROOT).free
    return {
        "status": "ready_for_gpu_smoke_review",
        "evaluator_version": EVALUATOR_VERSION,
        "panel_count": len(panel),
        "source_counts": SOURCE_COUNTS,
        "expected_key_count": len(keys),
        "greedy_key_count": sum(key[1] == "greedy" for key in keys),
        "sampled_key_count": sum(key[1] == "sampled" for key in keys),
        "checkpoint_roles": roles,
        "common_tokenizer_path": str(COMMON_TOKENIZER.resolve()),
        "config_sha256": sha256(CONFIG_PATH),
        "free_bytes": free_bytes,
        "gpu_execution_started": False,
        "diagnostic_evidence": config["diagnostic_evidence"],
    }


async def generate_shard(
    role: str,
    setting: str,
    source: str,
    endpoint: str,
    output: Path,
    scope: str,
) -> dict[str, Any]:
    """Generate one immutable role/setting/source shard after approval guards."""
    if role not in ROLE_NAMES or source not in SOURCE_COUNTS:
        raise ValueError("unknown M7 role or source")
    if scope not in {"smoke", "canonical"}:
        raise ValueError("scope must be smoke or canonical")
    if os.environ.get("M7_GPU_SMOKE_APPROVED") != "YES":
        raise PermissionError("M7 GPU smoke has not been explicitly approved")
    if scope == "canonical" and os.environ.get("M7_CANONICAL_EVAL_APPROVED") != "YES":
        raise PermissionError("canonical M7 evaluation has not been explicitly approved")
    config = validate_frozen_runtime_config()
    protocol_path = ROOT / config["protocol_path"]
    manifest_path = ROOT / "eval/diagnostic_freeze_v3.sha256"
    if not protocol_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("verified v3 freeze bundle is required before GPU generation")
    from eval.build_diagnostic_freeze_v3 import verify as verify_v3

    verify_v3()
    protocol = load_json(protocol_path)
    roles = resolved_checkpoint_roles(require_materialized=True)
    expected_model = Path(roles[role]["inference_locator"])
    server = SGLangHTTPServer(endpoint)
    await server.validate_model_path(expected_model)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(COMMON_TOKENIZER)
    loop = build_m7_loop(tokenizer, server, config)
    questions = [row for row in load_panel_questions() if row["source"] == source]
    if scope == "smoke":
        questions = questions[:1]
    question_map = {row["question_id"]: row for row in questions}
    existing = validated_resume_rows(output, question_map)
    generated = 0
    for question in questions:
        for sample_index in setting_sample_indices(setting):
            seed = derive_sample_seed(question["question_id"], sample_index)
            key = (role, setting, question["question_id"], sample_index, seed)
            if key in existing:
                continue
            row = await generate_one(
                loop=loop,
                tokenizer=tokenizer,
                question=question,
                role=role,
                setting=setting,
                sample_index=sample_index,
                checkpoint_identity=roles[role],
                protocol_version=protocol["protocol_version"],
                manifest_sha256=sha256(manifest_path),
                config=config,
            )
            append_jsonl_fsync(output, row)
            existing[key] = row
            generated += 1
    expected_count = len(questions) * len(setting_sample_indices(setting))
    if len(existing) != expected_count:
        raise AssertionError(f"shard completion mismatch: {len(existing)} != {expected_count}")
    return {
        "status": "completed",
        "scope": scope,
        "role": role,
        "setting": setting,
        "source": source,
        "expected": expected_count,
        "existing_before": expected_count - generated,
        "generated": generated,
        "output": str(output),
        "output_sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preflight", "expected-keys", "generate-shard"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--role", choices=ROLE_NAMES)
    parser.add_argument("--setting", choices=("greedy", "sampled"))
    parser.add_argument("--source", choices=tuple(SOURCE_COUNTS))
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000")
    parser.add_argument("--scope", choices=("smoke", "canonical"), default="smoke")
    args = parser.parse_args()
    if args.command == "generate-shard":
        if not all((args.role, args.setting, args.source, args.output)):
            parser.error("generate-shard requires --role, --setting, --source, and --output")
        result = asyncio.run(
            generate_shard(args.role, args.setting, args.source, args.endpoint, args.output, args.scope)
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    result: Any = preflight() if args.command == "preflight" else {"count": len(expected_keys()), "keys": expected_keys()}
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
