"""M9 / E3-NoTool launcher gates — plans/M9.md §6 fixtures J and L.

Fixture J (prompt equivalence): the NoTool prompt is exactly what the tokenizer produces
with ``tools=None``, and the TIR prompt still reproduces the canonical E3 rollout input.
Fixture L (config diff gate): E3 -> E3-NoTool differs only on the preregistered fields, and
the launcher refuses any config that could put the tool back.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
import yaml

from rl.launch.e3_grpo_baseline import DEFAULT_CONFIG as E3_CONFIG
from rl.launch.e3_grpo_baseline import PROJECT_ROOT, compose_config
from rl.launch.e3notool_grpo import (
    AGENT_LOOP_NAME,
    DEFAULT_CONFIG,
    E3_DIFF_PATHS,
    M9_PINNED_VERL_HASHES,
    SMOKE_DIFF_PATHS,
    compose_e3notool_config,
    config_diff_gate,
    disk_estimate,
    validate_e3notool_config,
    verl_source_hashes,
)
from rl.launch.e5v2_conserved_credit import diff_paths, resolved

E3_CANONICAL_STEP_1 = PROJECT_ROOT / "rl/runs/e3_grpo_baseline_20260819_224555/predictions/train/1.jsonl"
TOOL_SCHEMA_MARKERS = ("<tools>", "code_interpreter", "<tool_call>", "# Tools")


# --------------------------------------------------------------------------- #
# Fixture J — prompt equivalence
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "sft/checkpoints/qwen3-1.7b-sft"))


@pytest.fixture(scope="module")
def canonical_case() -> tuple[list[dict[str, str]], str]:
    """A real training question plus the decoded TIR prompt E3 actually generated from it."""
    import pyarrow.parquet as pq

    row = json.loads(E3_CANONICAL_STEP_1.read_text(encoding="utf-8").splitlines()[0])
    table = pq.read_table(PROJECT_ROOT / "rl/data/e3_train.parquet", columns=["prompt", "extra_info"])
    record = next(item for item in table.to_pylist() if item["extra_info"]["id"] == row["sample_id"])
    return [dict(message) for message in record["prompt"]], row["input"]


def _tool_schema() -> dict:
    config = yaml.safe_load((PROJECT_ROOT / "rl/custom/tool_config.yaml").read_text(encoding="utf-8"))
    return config["tools"][0]["tool_schema"]


def test_fixture_j_notool_prompt_equals_tools_none_chat_template(tokenizer, canonical_case) -> None:
    """The NoTool prompt is the plain chat template with no tools and no thinking."""
    from verl.utils.chat_template import apply_chat_template

    messages, _ = canonical_case
    kwargs = {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
    through_verl = list(apply_chat_template(tokenizer, messages, tools=None, **kwargs))
    direct = list(tokenizer.apply_chat_template(messages, tools=None, **kwargs))
    assert through_verl == direct
    decoded = tokenizer.decode(through_verl, skip_special_tokens=True)
    assert not [marker for marker in TOOL_SCHEMA_MARKERS if marker in decoded]
    assert decoded.startswith("user\n")


def test_fixture_j_tir_prompt_still_reproduces_the_canonical_e3_input(tokenizer, canonical_case) -> None:
    """Regression: with the tool schema the prompt is byte-identical to E3's logged input."""
    from verl.utils.chat_template import apply_chat_template

    messages, canonical_input = canonical_case
    kwargs = {"add_generation_prompt": True, "tokenize": True, "enable_thinking": False}
    tir = apply_chat_template(tokenizer, messages, tools=[_tool_schema()], **kwargs)
    assert tokenizer.decode(tir, skip_special_tokens=True) == canonical_input
    notool = apply_chat_template(tokenizer, messages, tools=None, **kwargs)
    assert list(tir) != list(notool)
    assert len(tir) > len(notool)


def test_fixture_j_native_single_turn_loop_never_passes_tool_schemas() -> None:
    """veRL's ``single_turn_agent`` calls ``apply_chat_template`` without ``tools``."""
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

    source = inspect.getsource(SingleTurnAgentLoop.run)
    assert "self.apply_chat_template(" in source
    call = source.split("self.apply_chat_template(", 1)[1].split(")", 1)[0]
    assert "tools" not in call
    assert "verl/experimental/agent_loop/single_turn_agent_loop.py" in M9_PINNED_VERL_HASHES
    verl_source_hashes()


# --------------------------------------------------------------------------- #
# Fixture L — config diff gate
# --------------------------------------------------------------------------- #


def test_fixture_l_e3_to_notool_diff_is_exactly_the_preregistered_set() -> None:
    gate = config_diff_gate("same_name")
    assert set(gate["e3_to_notool"]) == E3_DIFF_PATHS
    e3 = resolved(compose_config(E3_CONFIG, "same_name"))
    notool = resolved(compose_e3notool_config("same_name"))
    for section in ("data", "algorithm", "reward"):
        assert e3[section] == notool[section]
    assert e3["actor_rollout_ref"]["actor"] == notool["actor_rollout_ref"]["actor"]
    assert e3["actor_rollout_ref"]["ref"] == notool["actor_rollout_ref"]["ref"]
    rollout_e3 = dict(e3["actor_rollout_ref"]["rollout"])
    rollout_nt = dict(notool["actor_rollout_ref"]["rollout"])
    for key in ("n", "temperature", "top_k", "top_p", "gpu_memory_utilization", "name", "mode"):
        assert rollout_e3[key] == rollout_nt[key]
    assert rollout_nt["multi_turn"]["enable"] is False
    assert rollout_nt["agent"]["default_agent_loop"] == AGENT_LOOP_NAME
    assert rollout_nt["agent"]["agent_loop_config_path"] is None
    assert e3["trainer"]["total_training_steps"] == notool["trainer"]["total_training_steps"] == 200
    assert e3["trainer"]["save_freq"] == notool["trainer"]["save_freq"] == 25


def test_fixture_l_smoke_only_changes_step_count_and_disables_checkpoints() -> None:
    formal = resolved(compose_e3notool_config("same_name"))
    smoke = resolved(compose_e3notool_config("same_name", smoke=True))
    assert diff_paths(formal, smoke) == SMOKE_DIFF_PATHS
    assert smoke["trainer"]["save_freq"] == -1 and smoke["trainer"]["total_training_steps"] == 5
    assert smoke["data"]["train_batch_size"] == 64 and smoke["actor_rollout_ref"]["rollout"]["n"] == 8
    assert Path(smoke["data"]["train_files"]).name == "e3_train.parquet"


def test_fixture_l_validator_rejects_anything_that_restores_the_tool(tmp_path: Path) -> None:
    validate_e3notool_config(compose_e3notool_config("e3notool_validate"), tmp_path / "missing_run")

    config = compose_e3notool_config("e3notool_validate")
    config.actor_rollout_ref.rollout.multi_turn.enable = True
    with pytest.raises(ValueError, match="multi-turn tool loop"):
        validate_e3notool_config(config, tmp_path / "missing_run")

    config = compose_e3notool_config("e3notool_validate")
    config.actor_rollout_ref.rollout.agent.default_agent_loop = "toolcredit_agent"
    with pytest.raises(ValueError, match="agent loop"):
        validate_e3notool_config(config, tmp_path / "missing_run")

    config = compose_e3notool_config("e3notool_validate")
    config.actor_rollout_ref.rollout.agent.agent_loop_config_path = str(
        PROJECT_ROOT / "rl/custom/agent_loop_config.yaml"
    )
    with pytest.raises(ValueError, match="custom agent loop"):
        validate_e3notool_config(config, tmp_path / "missing_run")

    config = compose_e3notool_config("e3notool_validate")
    config.actor_rollout_ref.rollout.n = 4
    with pytest.raises(ValueError, match="rollout n"):
        validate_e3notool_config(config, tmp_path / "missing_run")

    config = compose_e3notool_config("e3notool_validate")
    config.data.train_batch_size = 16
    with pytest.raises(ValueError, match="batch/response"):
        validate_e3notool_config(config, tmp_path / "missing_run")


def test_config_path_and_disk_estimate() -> None:
    assert DEFAULT_CONFIG.name == "e3notool_grpo.yaml"
    estimate = disk_estimate(smoke=False, check=False)
    # Three rotating ~20 GiB checkpoints dominate the peak write.
    assert 55.0 < estimate["remaining_peak_write_gib"] < 80.0
    assert estimate["disk_minimum_gib"] == pytest.approx(estimate["remaining_peak_write_gib"] + 30)


def test_frozen_predecessors_are_untouched_by_m9() -> None:
    """Freeze v2/v3/v4 closures hash the E3/E5 modules; M9 must live in new files."""
    from eval.build_diagnostic_freeze_v2 import verify as verify_v2
    from eval.build_diagnostic_freeze_v4 import verify as verify_v4

    verify_v2()
    verify_v4()
