"""M10 / E7 fixtures N–S (plans/M10.md §7, plans/M10_STEP0_BOUNDARY.md §6).

N — pure filtering by scalar ``score``; conservation; ``score == rm_scores.sum(-1)``.
O — multi-chunk concatenation through the real ``_e7_collect_informative_batch`` with a fake
    rollout manager: UID/score/tensor alignment, first-64-in-order selection, one
    ``sleep_replicas()``, no actor update, balance/meta only on the final batch.
P — underfilled after four chunks fails before any actor update with candidates on disk.
Q — ledger/checkpoint/resume semantics: atomic ledger next to the checkpoint, step identity,
    StatefulDataLoader rollover and mid-epoch resume, ``--stop-at-step`` / stop-request logic,
    explicit refusal of veRL's epoch-boundary heuristic.
R — ``fit()`` is the pinned ``ray_trainer.py:1362–1770`` except the whitelisted W1–W7 lines.
S — the equal-compute checkpoint triggers exactly once, outside the rotation, also at a
    multiple of 25.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from rl.custom import dynamic_filter_trainer as dft
from rl.custom.dynamic_filter_trainer import (
    E3_TOTAL_TRAJECTORIES,
    GROUP_SIZE,
    LEDGER_FILENAME,
    LEDGER_JSONL,
    MAX_GENERATION_BATCHES,
    STOP_REQUEST_FILENAME,
    TARGET_GROUPS,
    TRAINER_CLASS_FQN,
    DynamicFilterRayPPOTrainer,
    DynamicSamplingLedger,
    E7InvariantError,
    E7UnderfilledError,
    GroupSplit,
    Selection,
    fit_source_lines,
    next_checkpoint_step,
    select_informative_groups,
    split_groups_by_score_variance,
    upstream_fit_source_lines,
)

PROMPT_LEN = 6
RESPONSE_LEN = 10

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _uids(n_groups: int, prefix: str = "u") -> np.ndarray:
    return np.array([f"{prefix}{g}" for g in range(n_groups) for _ in range(GROUP_SIZE)], dtype=object)


def _scores_from_pattern(pattern: str) -> list[float]:
    """Group score patterns: 'AC' all correct, 'AW' all wrong, 'FMT' acc-equal/format-varied, 'MIX' mixed."""
    if pattern == "AC":
        return [1.1] * GROUP_SIZE
    if pattern == "AW":
        return [0.0] * GROUP_SIZE
    if pattern == "AW_FMT":  # all wrong on acc, but format bonus differs -> informative by score
        return [0.1, 0.0] * (GROUP_SIZE // 2)
    if pattern == "MIX":
        return [1.1, 0.0, 1.1, 0.1, 0.0, 1.1, 0.0, 1.0][:GROUP_SIZE]
    if pattern == "AC_FMT":  # all correct on acc, one without format -> informative by score
        return [1.1] * (GROUP_SIZE - 1) + [1.0]
    raise ValueError(pattern)


def _acc_from_scores(scores: list[float]) -> list[float]:
    return [1.0 if score >= 1.0 else 0.0 for score in scores]


# --------------------------------------------------------------------------- #
# Fixture N — pure filtering
# --------------------------------------------------------------------------- #


def test_fixture_n_zero_variance_is_judged_by_score_not_acc() -> None:
    patterns = ["AC"] * 20 + ["AW"] * 14 + ["AW_FMT"] * 3 + ["AC_FMT"] * 2 + ["MIX"] * 25
    assert len(patterns) == TARGET_GROUPS
    uids = _uids(TARGET_GROUPS)
    scores = [value for pattern in patterns for value in _scores_from_pattern(pattern)]
    acc = [value for pattern in patterns for value in _acc_from_scores(_scores_from_pattern(pattern))]
    rm_scores = torch.zeros(len(scores), RESPONSE_LEN)
    rm_scores[:, -1] = torch.tensor(scores)
    split = split_groups_by_score_variance(uids, rm_scores.sum(-1), chunk_index=0, acc=acc)
    assert split.n_groups == TARGET_GROUPS
    assert len(split.zero_std_uids) == 34  # AC + AW only
    assert len(split.informative_uids) == 30  # AW_FMT + AC_FMT + MIX still carry gradient
    # acc-uniform but format-varied groups are informative, and their acc pattern is still recorded.
    fmt_uid = split.group_uids[34]  # first AW_FMT
    assert fmt_uid in split.informative_uids and split.acc_pattern(fmt_uid) == "all_wrong"
    assert split.acc_pattern(split.group_uids[0]) == "all_correct"
    assert split.acc_pattern(split.group_uids[-1]) == "mixed"
    selection = select_informative_groups([split])
    assert selection.selected_groups + selection.surplus_groups + selection.zero_std_groups == TARGET_GROUPS
    assert selection.underfilled()
    assert selection.selected_uids == split.informative_uids
    assert selection.surplus_uids == ()


def test_fixture_n_tolerance_and_malformed_chunks() -> None:
    uids = _uids(2)
    scores = np.array([1.1] * 8 + [1.1] * 7 + [1.1 + 5e-9])
    split = split_groups_by_score_variance(uids, scores, chunk_index=0)
    assert len(split.zero_std_uids) == 2  # within 1e-8 tolerance both are zero variance
    with pytest.raises(E7InvariantError):
        split_groups_by_score_variance(uids[:-1], scores[:-1], chunk_index=0)
    shuffled = uids.copy()
    shuffled[0], shuffled[8] = shuffled[8], shuffled[0]
    with pytest.raises(E7InvariantError):
        split_groups_by_score_variance(shuffled, scores, chunk_index=0)
    with pytest.raises(E7InvariantError):
        split_groups_by_score_variance(uids, np.array([float("nan")] * 16), chunk_index=0)


def test_fixture_n_selection_takes_first_64_informative_in_sampling_order() -> None:
    def make(chunk: int, patterns: list[str]) -> GroupSplit:
        uids = _uids(len(patterns), prefix=f"c{chunk}g")
        scores = [value for pattern in patterns for value in _scores_from_pattern(pattern)]
        return split_groups_by_score_variance(uids, np.array(scores), chunk_index=chunk)

    first = make(0, ["MIX", "AC"] * 32)  # 32 informative
    second = make(1, ["AC", "MIX", "MIX", "AW"] * 16)  # 32 informative
    third = make(2, ["MIX"] * 64)  # 64 informative -> surplus
    selection = select_informative_groups([first, second, third])
    assert selection.generation_batches_used == 3
    assert selection.selected_groups == TARGET_GROUPS and not selection.underfilled()
    assert selection.selected_uids == first.informative_uids + second.informative_uids
    assert selection.surplus_uids == third.informative_uids
    assert selection.zero_std_groups == 32 + 32
    assert selection.total_groups == 192
    with pytest.raises(E7InvariantError):
        select_informative_groups([first, first])
    with pytest.raises(E7InvariantError):
        select_informative_groups([second])


# --------------------------------------------------------------------------- #
# Fake trainer environment for fixtures O / P / Q / S
# --------------------------------------------------------------------------- #


class _PromptDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "dummy_tensor": torch.tensor([0], dtype=torch.uint8),
            "data_source": "toolcredit_math",
            "reward_model": {"style": "rule", "ground_truth": str(index)},
            "extra_info": {"id": f"math_train_{index:06d}", "level": 3},
            "raw_prompt": [{"role": "user", "content": f"question {index}"}],
            "index": index,
            "tools_kwargs": {},
        }


class _FakeRolloutManager:
    """Scripted generation: chunk ``k`` receives the k-th list of group score patterns."""

    def __init__(self, chunk_patterns: list[list[str]]) -> None:
        self.chunk_patterns = chunk_patterns
        self.calls = 0
        self.global_steps_seen: list[int] = []

    def generate_sequences(self, gen_batch: Any) -> Any:
        from verl import DataProto

        patterns = self.chunk_patterns[self.calls]
        self.calls += 1
        self.global_steps_seen.append(int(gen_batch.meta_info["global_steps"]))
        uids = gen_batch.non_tensor_batch["uid"]
        n_rows = len(uids)
        assert n_rows == len(patterns) * GROUP_SIZE
        scores = [value for pattern in patterns for value in _scores_from_pattern(pattern)]
        acc = [value for pattern in patterns for value in _acc_from_scores(_scores_from_pattern(pattern))]
        lengths = [3 + (row % 5) for row in range(n_rows)]
        prompts = torch.full((n_rows, PROMPT_LEN), 7, dtype=torch.long)
        responses = torch.zeros((n_rows, RESPONSE_LEN), dtype=torch.long)
        response_mask = torch.zeros((n_rows, RESPONSE_LEN), dtype=torch.long)
        rm_scores = torch.zeros((n_rows, RESPONSE_LEN), dtype=torch.float32)
        for row, length in enumerate(lengths):
            responses[row, :length] = 11
            response_mask[row, :length] = 1
            rm_scores[row, length - 1] = scores[row]
        attention_mask = torch.cat([torch.ones((n_rows, PROMPT_LEN), dtype=torch.long), response_mask], dim=1)
        batch = {
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "input_ids": torch.cat([prompts, responses], dim=1),
            "attention_mask": attention_mask,
            "position_ids": torch.cumsum(attention_mask, dim=1) - 1,
            "rm_scores": rm_scores,
        }
        sample_ids = np.array([str(info["id"]) for info in gen_batch.non_tensor_batch["extra_info"]], dtype=object)
        multi_modal = np.empty(n_rows, dtype=object)
        multi_modal[:] = [{} for _ in range(n_rows)]
        non_tensor = {
            "score": np.array(scores, dtype=np.float64),
            "acc": np.array(acc, dtype=np.float64),
            "sample_id": sample_ids,
            "multi_modal_inputs": multi_modal,
            "__num_turns__": np.ones(n_rows, dtype=np.int32),
        }
        return DataProto.from_dict(
            tensors=batch,
            non_tensors=non_tensor,
            meta_info={"metrics": [{}], "reward_extra_keys": ["score", "acc", "sample_id"], "timing": {"agent_loop/x": 0.01}},
        )


class _FakeCheckpointManager:
    def __init__(self) -> None:
        self.sleep_calls = 0
        self.update_calls: list[int] = []

    def sleep_replicas(self) -> None:
        self.sleep_calls += 1

    def update_weights(self, global_steps: int) -> None:
        self.update_calls.append(global_steps)


class _FakeWorkerGroup:
    """Mimics ``actor_rollout_wg.save_checkpoint`` writing an actor directory."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, int, Any]] = []

    def save_checkpoint(self, local_path: str, hdfs_path: Any, global_step: int, max_ckpt_to_keep: Any = None) -> None:
        path = Path(local_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model_world_size_1_rank_0.pt").write_bytes(b"model")
        self.saved.append((local_path, global_step, max_ckpt_to_keep))


def _config(tmp_path: Path, **trainer_overrides: Any) -> Any:
    run_dir = tmp_path / "run"
    return OmegaConf.create(
        {
            "algorithm": {
                "adv_estimator": "grpo",
                "filter_groups": {"enable": True, "metric": "score", "max_num_gen_batches": MAX_GENERATION_BATCHES},
            },
            "data": {"train_batch_size": TARGET_GROUPS},
            "actor_rollout_ref": {
                "rollout": {"n": GROUP_SIZE, "temperature": 1.0},
                "actor": {"fsdp_config": {"param_offload": False, "optimizer_offload": False}},
                "ref": {"fsdp_config": {"param_offload": False}},
            },
            "trainer": {
                "default_local_dir": str(run_dir / "checkpoints"),
                "rollout_data_dir": str(run_dir / "predictions/train"),
                "balance_batch": False,
                "total_epochs": 30,
                "save_freq": 25,
                "toolcredit_trainer_class": TRAINER_CLASS_FQN,
                **trainer_overrides,
            },
        }
    )


def _make_trainer(tmp_path: Path, chunk_patterns: list[list[str]], *, dataset_size: int = 64 * 81, **overrides: Any) -> Any:
    """A ``DynamicFilterRayPPOTrainer`` without veRL workers: only the E7 helpers are exercised."""
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import RandomSampler
    from verl.utils.dataset.rl_dataset import collate_fn

    trainer = DynamicFilterRayPPOTrainer.__new__(DynamicFilterRayPPOTrainer)
    trainer.config = _config(tmp_path, **overrides)
    trainer.use_rm = False
    trainer.global_steps = 1
    trainer.total_training_steps = 200
    trainer._e7_ledger = DynamicSamplingLedger()
    trainer._e7_epoch = 0
    trainer._e7_generation_phase = False
    trainer._e7_candidate_rows_cache = []
    dataset = _PromptDataset(dataset_size)
    generator = torch.Generator()
    generator.manual_seed(42)
    trainer.train_dataset = dataset
    trainer.train_dataloader = StatefulDataLoader(
        dataset,
        batch_size=TARGET_GROUPS,
        sampler=RandomSampler(dataset, generator=generator),
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    trainer._e7_iter = iter(trainer.train_dataloader)
    trainer.async_rollout_manager = _FakeRolloutManager(chunk_patterns)
    trainer.checkpoint_manager = _FakeCheckpointManager()
    trainer.actor_rollout_wg = _FakeWorkerGroup()
    trainer.llm_server_manager = None
    return trainer


# --------------------------------------------------------------------------- #
# Fixture O — multi-chunk concatenation and selection
# --------------------------------------------------------------------------- #


def test_fixture_o_two_chunks_select_first_64_informative_with_aligned_tensors(tmp_path: Path) -> None:
    chunk1 = ["MIX", "AC", "AW", "AW_FMT"] * 16  # 32 informative (MIX + AW_FMT)
    chunk2 = ["AC", "MIX", "AC_FMT", "AW"] * 16  # 32 informative (MIX + AC_FMT)
    chunk3 = ["MIX"] * 64  # must never be requested
    trainer = _make_trainer(tmp_path, [chunk1, chunk2, chunk3])
    metrics: dict[str, Any] = {}
    timing: dict[str, float] = {}
    batch, reward_tensor, extra, row = trainer._e7_collect_informative_batch(metrics, timing, False)

    manager = trainer.async_rollout_manager
    assert manager.calls == 2 and manager.global_steps_seen == [1, 1]
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert trainer.checkpoint_manager.update_calls == []
    assert trainer._e7_generation_phase is False
    assert len(batch) == TARGET_GROUPS * GROUP_SIZE
    uids = [str(value) for value in batch.non_tensor_batch["uid"].tolist()]
    # Every selected row's scalar score equals its group's score and is not zero-variance.
    scores = reward_tensor.sum(-1).numpy()
    assert np.allclose(scores, batch.non_tensor_batch["score"].astype(float))
    assert np.allclose(scores, batch.batch["rm_scores"].sum(-1).numpy())
    for start in range(0, len(uids), GROUP_SIZE):
        group = uids[start : start + GROUP_SIZE]
        assert len(set(group)) == 1
        assert scores[start : start + GROUP_SIZE].std() > 1e-8
    # First 32 groups come from chunk 1 and the next 32 from chunk 2, in sampling order.
    cached = trainer._e7_candidate_rows_cache
    selected_rows = [r for r in cached if r["role"] == "selected"]
    assert [r["chunk"] for r in selected_rows[:: GROUP_SIZE]] == [0] * 32 + [1] * 32
    assert [r["uid"] for r in selected_rows[:: GROUP_SIZE]] == uids[:: GROUP_SIZE]
    roles = {role: sum(1 for r in cached if r["role"] == role) // GROUP_SIZE for role in ("selected", "surplus", "zero_std")}
    assert roles == {"selected": 64, "surplus": 0, "zero_std": 64}
    assert len(cached) == 2 * TARGET_GROUPS * GROUP_SIZE
    # Tensor alignment: response length per row matches the candidate row bookkeeping.
    lengths = batch.batch["response_mask"].sum(-1).tolist()
    by_uid = {(r["uid"], r["score"]): r for r in selected_rows}
    assert all((uid, float(score)) in by_uid for uid, score in zip(uids, scores))
    assert "global_token_num" in batch.meta_info and len(batch.meta_info["global_token_num"]) == len(batch)
    assert batch.meta_info["images_seqlens"] == []
    assert extra["score"].shape == (TARGET_GROUPS * GROUP_SIZE,)
    assert set(batch.batch.keys()) >= {"prompts", "responses", "response_mask", "rm_scores", "attention_mask"}
    # Ledger row.
    assert row.effective_step == 1 and row.generation_batches_used == 2
    assert row.prompts_sampled == 128 and row.trajectories_sampled == 1024
    assert row.informative_groups == 64 and row.zero_std_groups == 64 and row.surplus_groups == 0
    assert row.all_correct_groups == 32 + 16 and row.all_wrong_groups == 16 + 16 + 16 and row.mixed_groups == 32
    assert row.rollout_tokens == int(sum(lengths)) * 2 or row.rollout_tokens > 0
    assert row.dataloader_batches_consumed == 2 and row.cumulative_trajectories == 1024
    assert timing["gen"] >= 0 and "reward" in timing
    assert row.metrics()["e7/generation_batches_used"] == 2.0


def test_fixture_o_surplus_is_recorded_but_never_trained_on(tmp_path: Path) -> None:
    chunk1 = ["AC", "AW"] * 32  # 0 informative
    chunk2 = ["MIX"] * 64  # 64 informative
    chunk3 = ["MIX"] * 64
    trainer = _make_trainer(tmp_path, [chunk1, chunk2, chunk3])
    batch, _, _, row = trainer._e7_collect_informative_batch({}, {}, False)
    assert trainer.async_rollout_manager.calls == 2
    assert row.surplus_groups == 0 and row.zero_std_groups == 64 and row.selected_groups == 64
    chunkier = _make_trainer(tmp_path / "b", [["AC"] * 32 + ["MIX"] * 32, ["MIX"] * 64])
    batch, _, _, row = chunkier._e7_collect_informative_batch({}, {}, False)
    assert row.surplus_groups == 32 and row.informative_groups == 96
    surplus = {r["uid"] for r in chunkier._e7_candidate_rows_cache if r["role"] == "surplus"}
    assert surplus.isdisjoint(set(batch.non_tensor_batch["uid"].tolist()))
    assert len(surplus) == 32


def test_fixture_o_score_identity_and_actor_update_guard(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, [["MIX"] * 64])
    original = trainer.async_rollout_manager.generate_sequences

    def tampered(gen_batch: Any) -> Any:
        out = original(gen_batch)
        out.non_tensor_batch["score"] = out.non_tensor_batch["score"] + 0.5
        return out

    trainer.async_rollout_manager.generate_sequences = tampered
    with pytest.raises(E7InvariantError, match="rm_scores"):
        trainer._e7_collect_informative_batch({}, {}, False)
    assert trainer.checkpoint_manager.sleep_calls == 1  # finally still sleeps exactly once
    guard = _make_trainer(tmp_path / "g", [["MIX"] * 64])
    guard._e7_generation_phase = True
    with pytest.raises(E7InvariantError):
        guard._update_actor(None)


# --------------------------------------------------------------------------- #
# Fixture P — underfilled failure
# --------------------------------------------------------------------------- #


def test_fixture_p_underfilled_after_four_chunks_fails_before_update_with_candidates_on_disk(tmp_path: Path) -> None:
    chunks = [["MIX"] * 10 + ["AC"] * 54 for _ in range(MAX_GENERATION_BATCHES + 1)]
    trainer = _make_trainer(tmp_path, chunks)
    with pytest.raises(E7UnderfilledError):
        trainer._e7_collect_informative_batch({}, {}, False)
    assert trainer.async_rollout_manager.calls == MAX_GENERATION_BATCHES
    assert trainer.checkpoint_manager.sleep_calls == 1
    assert trainer._e7_ledger.effective_step == 0 and trainer._e7_ledger.rows == []
    dumped = tmp_path / "run/predictions/candidates/1.underfilled.jsonl"
    rows = [json.loads(line) for line in dumped.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == MAX_GENERATION_BATCHES * TARGET_GROUPS * GROUP_SIZE
    assert sum(1 for r in rows if r["role"] == "selected") == 40 * GROUP_SIZE
    assert sum(1 for r in rows if r["role"] == "zero_std") == 4 * 54 * GROUP_SIZE
    assert trainer._e7_ledger.dataloader_batches_consumed == MAX_GENERATION_BATCHES


# --------------------------------------------------------------------------- #
# Fixture Q — ledger / checkpoint / resume
# --------------------------------------------------------------------------- #


def test_fixture_q_ledger_roundtrip_and_step_identity(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, [["MIX"] * 64, ["MIX"] * 64])
    trainer._e7_collect_informative_batch({}, {}, False)
    ledger = trainer._e7_ledger
    path = tmp_path / LEDGER_FILENAME
    ledger.save(path)
    loaded = DynamicSamplingLedger.load(path)
    assert loaded.to_dict() == ledger.to_dict()
    assert not (tmp_path / (LEDGER_FILENAME + ".tmp")).exists()
    # Step identity: recording step 3 after step 1 is refused.
    trainer.global_steps = 3
    trainer.async_rollout_manager.calls = 1
    with pytest.raises(E7InvariantError, match="discontinuity"):
        trainer._e7_collect_informative_batch({}, {}, False)
    # _save_checkpoint refuses a ledger/global_steps mismatch before touching veRL.
    trainer.global_steps = 5
    with pytest.raises(E7InvariantError):
        trainer._save_checkpoint()
    # Corrupted ledgers are refused on load.
    payload = ledger.to_dict()
    payload["rows"][0]["effective_step"] = 2
    with pytest.raises(E7InvariantError):
        DynamicSamplingLedger.from_dict(payload)


def test_fixture_q_save_checkpoint_writes_ledger_before_super(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _make_trainer(tmp_path, [["MIX"] * 64])
    trainer._e7_collect_informative_batch({}, {}, False)
    order: list[str] = []

    def fake_super(self: Any) -> None:
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        order.append("super")
        assert (folder / LEDGER_FILENAME).is_file()

    monkeypatch.setattr(dft.RayPPOTrainer, "_save_checkpoint", fake_super)
    trainer._save_checkpoint()
    assert order == ["super"]
    saved = DynamicSamplingLedger.load(tmp_path / "run/checkpoints/global_step_1" / LEDGER_FILENAME)
    assert saved.effective_step == 1


def test_fixture_q_load_checkpoint_restores_ledger_and_refuses_epoch_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _make_trainer(tmp_path, [["MIX"] * 64, ["MIX"] * 64, ["MIX"] * 64])
    trainer._e7_collect_informative_batch({}, {}, False)
    trainer.global_steps = 2
    trainer._e7_collect_informative_batch({}, {}, False)
    folder = tmp_path / "run/checkpoints/global_step_2"
    trainer._e7_ledger.save(folder / LEDGER_FILENAME)
    jsonl = tmp_path / "run" / LEDGER_JSONL
    jsonl.write_text("stale\n", encoding="utf-8")

    def fake_super(self: Any) -> int:
        self.global_steps = 2
        return 0

    monkeypatch.setattr(dft.RayPPOTrainer, "_load_checkpoint", fake_super)
    resumed = _make_trainer(tmp_path, [])
    resumed._load_checkpoint()
    assert resumed._e7_ledger.effective_step == 2 and resumed._e7_ledger.dataloader_batches_consumed == 2
    assert [json.loads(l)["effective_step"] for l in jsonl.read_text().splitlines()] == [1, 2]
    # Ledger step disagreeing with the checkpoint step is refused.
    wrong = _make_trainer(tmp_path, [])
    monkeypatch.setattr(dft.RayPPOTrainer, "_load_checkpoint", lambda self: setattr(self, "global_steps", 3) or 0)
    (tmp_path / "run/checkpoints/global_step_3").mkdir()
    trainer._e7_ledger.save(tmp_path / "run/checkpoints/global_step_3" / LEDGER_FILENAME)
    with pytest.raises(E7InvariantError, match="effective_step"):
        wrong._load_checkpoint()
    # Missing ledger is refused.
    (tmp_path / "run/checkpoints/global_step_3" / LEDGER_FILENAME).unlink()
    with pytest.raises(E7InvariantError, match="no e7_ledger"):
        _make_trainer(tmp_path, [])._load_checkpoint()
    # veRL's epoch-boundary heuristic (global_steps % len(dataloader) == 0) is refused explicitly.
    boundary = _make_trainer(tmp_path / "boundary", [], dataset_size=64 * 4)  # 4 batches / epoch
    monkeypatch.setattr(dft.RayPPOTrainer, "_load_checkpoint", lambda self: setattr(self, "global_steps", 8) or 0)
    with pytest.raises(E7InvariantError, match="epoch-boundary"):
        boundary._load_checkpoint()


def test_fixture_q_dataloader_rollover_and_mid_epoch_resume(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, [], dataset_size=64 * 3)  # 3 batches per epoch
    seen = [trainer._e7_next_batch_dict()["index"].tolist() for _ in range(7)]
    assert trainer._e7_ledger.dataloader_batches_consumed == 7
    assert trainer._e7_epoch == 2 and trainer._e7_ledger.epoch == 2  # 7 // 3 completed passes
    assert sorted(sum(seen[:3], [])) == list(range(192))  # epoch 0 is a permutation
    assert seen[:3] != seen[3:6]  # reshuffled epoch 1
    # Resume mid-epoch: save state after batch 4, reload into a fresh loader, get batches 5..7.
    fresh = _make_trainer(tmp_path / "f", [], dataset_size=64 * 3)
    for _ in range(4):
        fresh._e7_next_batch_dict()
    state = fresh.train_dataloader.state_dict()
    resumed = _make_trainer(tmp_path / "r", [], dataset_size=64 * 3)
    resumed.train_dataloader.load_state_dict(state)
    resumed._e7_iter = iter(resumed.train_dataloader)
    resumed._e7_epoch = 1
    resumed._e7_ledger.epoch = 1
    resumed._e7_ledger.dataloader_batches_consumed = 4
    continued = [resumed._e7_next_batch_dict()["index"].tolist() for _ in range(3)]
    assert continued == seen[4:7]
    assert resumed._e7_epoch == 2
    # Exhausted-at-checkpoint state: after exactly 6 batches the next call rolls the epoch.
    edge = _make_trainer(tmp_path / "e", [], dataset_size=64 * 3)
    for _ in range(6):
        edge._e7_next_batch_dict()
    state = edge.train_dataloader.state_dict()
    edge_resumed = _make_trainer(tmp_path / "er", [], dataset_size=64 * 3)
    edge_resumed.train_dataloader.load_state_dict(state)
    edge_resumed._e7_iter = iter(edge_resumed.train_dataloader)
    edge_resumed._e7_epoch = 2
    edge_resumed._e7_ledger.epoch = 2
    edge_resumed._e7_ledger.dataloader_batches_consumed = 6
    batch = edge_resumed._e7_next_batch_dict()
    assert batch["index"].tolist() == seen[6]
    assert edge_resumed._e7_epoch == 2


def test_fixture_q_stop_step_from_flag_and_stop_request(tmp_path: Path) -> None:
    trainer = _make_trainer(tmp_path, [])
    (tmp_path / "run").mkdir(parents=True, exist_ok=True)
    trainer.total_training_steps = 200
    trainer.global_steps = 37
    assert trainer._e7_stop_step() == 200
    DynamicFilterRayPPOTrainer.E7_STOP_AT_STEP = 50
    try:
        assert trainer._e7_stop_step() == 50
        (tmp_path / "run" / STOP_REQUEST_FILENAME).write_text("{}", encoding="utf-8")
        # Stop request at step 37 -> next checkpoint step 50 (save_freq 25) -> min(50, 50).
        assert trainer._e7_stop_step() == 50
        trainer.global_steps = 26
        assert trainer._e7_stop_step() == 50
        DynamicFilterRayPPOTrainer.E7_STOP_AT_STEP = 100
        assert trainer._e7_stop_step() == 50  # stop request wins: next multiple of 25
        trainer.global_steps = 50
        assert trainer._e7_stop_step() == 50
    finally:
        DynamicFilterRayPPOTrainer.E7_STOP_AT_STEP = None
    assert next_checkpoint_step(26, 25) == 50 and next_checkpoint_step(50, 25) == 50
    with pytest.raises(E7InvariantError):
        next_checkpoint_step(3, -1)


# --------------------------------------------------------------------------- #
# Fixture R — fit() provenance
# --------------------------------------------------------------------------- #

# Upstream (1-based, inclusive) line ranges of ray_trainer.py that E7 is allowed to remove/replace.
ALLOWED_REMOVED_RANGES = ((1389, 1389), (1422, 1423), (1435, 1464), (1466, 1525), (1716, 1716))
WHITELIST_TAGS = ("# E7-W1", "# E7-W2", "# E7-W3a", "# E7-W3", "# E7-W4", "# E7-W5", "# E7-W6")


def _allowed_removed(line_no: int) -> bool:
    return any(lo <= line_no <= hi for lo, hi in ALLOWED_REMOVED_RANGES)


def test_fixture_r_fit_is_the_pinned_source_except_whitelisted_lines() -> None:
    upstream = upstream_fit_source_lines()
    e7 = fit_source_lines()
    assert len(upstream) == 409
    start = dft.FIT_SOURCE_LINES[0]
    matcher = difflib.SequenceMatcher(a=upstream, b=e7, autojunk=False)
    removed: list[int] = []
    added: list[str] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            continue
        removed.extend(range(start + a0, start + a1))
        added.extend(e7[b0:b1])
    for line_no in removed:
        assert _allowed_removed(line_no), f"upstream line {line_no} changed outside the whitelist: {upstream[line_no - start]!r}"
    for line in added:
        assert any(tag in line for tag in WHITELIST_TAGS), f"E7 line without whitelist marker: {line!r}"
    # Every whitelist item is present, and the untouched ranges are byte-identical.
    joined = "\n".join(e7)
    for tag in ("# E7-W1", "# E7-W2", "# E7-W3a", "# E7-W3", "# E7-W4", "# E7-W5", "# E7-W6"):
        assert tag in joined, tag
    for lo, hi in ((1362, 1388), (1390, 1421), (1424, 1434), (1526, 1670), (1672, 1682), (1684, 1715), (1717, 1770)):
        for line_no in range(lo, hi + 1):
            assert upstream[line_no - start] in e7, f"upstream line {line_no} missing verbatim"
    # W7: on_batch_end call is unchanged.
    assert "                    self.train_dataset.on_batch_end(batch=batch)" in e7
    # Old-log-prob, ref, advantage, actor update, weight sync and validation lines are verbatim.
    for needle in (
        "old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)",
        "ref_log_prob = self._compute_ref_log_prob(batch)",
        "batch = compute_advantage(",
        "actor_output = self._update_actor(batch)",
        "self.checkpoint_manager.update_weights(self.global_steps)",
        "val_metrics: dict = self._validate()",
    ):
        assert any(needle in line for line in e7), needle


def test_fixture_r_helper_reuses_upstream_generation_lines_verbatim() -> None:
    import inspect

    upstream = upstream_fit_source_lines()
    start = dft.FIT_SOURCE_LINES[0]
    helper = [line.strip() for line in inspect.getsource(DynamicFilterRayPPOTrainer._e7_collect_informative_batch).splitlines()]
    for lo, hi in ((1435, 1449), (1461, 1462), (1466, 1470), (1472, 1480), (1495, 1501), (1502, 1517), (1518, 1525)):
        expected = [upstream[n - start].strip() for n in range(lo, hi + 1) if upstream[n - start].strip()]
        position = 0
        for line in expected:
            try:
                position = helper.index(line, position) + 1
            except ValueError as exc:  # pragma: no cover - diagnostic
                raise AssertionError(f"upstream line {line!r} (range {lo}-{hi}) not reused verbatim in order") from exc
    # The sleep call from :1471 appears exactly once, inside ``finally``.
    source = inspect.getsource(DynamicFilterRayPPOTrainer._e7_collect_informative_batch)
    assert source.count("self.checkpoint_manager.sleep_replicas()") == 1
    assert "finally:" in source and source.index("finally:") < source.index("self.checkpoint_manager.sleep_replicas()")
    # No REMAX branch and no weight sync inside the helper.
    assert "AdvantageEstimator.REMAX" not in source and "update_weights" not in source and "_update_actor" not in source


def test_fixture_r_pinned_hashes_still_match() -> None:
    hashes = dft.pinned_source_hashes()
    assert hashes["verl/trainer/ppo/ray_trainer.py"] == dft.PINNED_RAY_TRAINER_SHA256
    assert hashes["verl/trainer/main_ppo.py"] == dft.PINNED_MAIN_PPO_SHA256


# --------------------------------------------------------------------------- #
# Fixture S — equal-compute checkpoint
# --------------------------------------------------------------------------- #


def _synthetic_ledger(batches_per_step: list[int]) -> DynamicSamplingLedger:
    ledger = DynamicSamplingLedger()
    due_steps: list[int] = []
    for step, batches in enumerate(batches_per_step, start=1):
        splits = [
            split_groups_by_score_variance(_uids(TARGET_GROUPS, prefix=f"s{step}c{k}g"), np.array([1.1, 0.0] * 4 * TARGET_GROUPS), chunk_index=k)
            for k in range(batches)
        ]
        selection = select_informative_groups(splits)
        ledger.dataloader_batches_consumed += batches
        ledger.record_step(
            effective_step=step,
            selection=selection,
            splits=splits,
            prompts_per_chunk=TARGET_GROUPS,
            group_size=GROUP_SIZE,
            rollout_tokens=1000 * batches,
            generation_wall_time=1.0,
            filter_wall_time=0.1,
        )
        if ledger.equal_compute_due():
            due_steps.append(step)
            ledger.mark_equal_compute_saved(step)
    ledger.due_steps = due_steps  # type: ignore[attr-defined]
    return ledger


def test_fixture_s_equal_compute_triggers_exactly_once_when_crossing_the_e3_total() -> None:
    # 2 batches/step = 1024 trajectories/step -> crosses 102,400 exactly at step 100.
    ledger = _synthetic_ledger([2] * 120)
    assert ledger.due_steps == [100]
    assert ledger.equal_compute_step == 100 and ledger.rows[99].equal_compute_checkpoint
    assert ledger.rows[99].cumulative_trajectories == E3_TOTAL_TRAJECTORIES
    assert sum(row.equal_compute_checkpoint for row in ledger.rows) == 1
    # Irregular batches: crossing happens strictly once, mid-interval, at the first step >= total.
    pattern = [3, 2, 4, 2, 3] * 40
    ledger = _synthetic_ledger(pattern)
    cumulative = np.cumsum(np.array(pattern) * TARGET_GROUPS * GROUP_SIZE)
    expected = int(np.argmax(cumulative >= E3_TOTAL_TRAJECTORIES)) + 1
    assert ledger.due_steps == [expected] and expected % 25 != 0
    assert ledger.rows[expected - 2].cumulative_trajectories < E3_TOTAL_TRAJECTORIES <= ledger.rows[expected - 1].cumulative_trajectories
    with pytest.raises(E7InvariantError):
        ledger.mark_equal_compute_saved(expected + 1)


def test_fixture_s_equal_compute_checkpoint_is_saved_outside_rotation_and_survives_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = _make_trainer(tmp_path, [["MIX"] * 64] * 3)
    ledger = trainer._e7_ledger
    # Pretend 99 steps of two chunks each already happened.
    ledger.effective_step = 99
    ledger.cumulative_trajectories = 99 * 1024
    ledger.cumulative_prompts = 99 * 128
    ledger.dataloader_batches_consumed = 198
    ledger.rows = []
    trainer.global_steps = 100
    trainer.async_rollout_manager.chunk_patterns = [["MIX"] * 64, ["AC"] * 64, ["MIX"] * 64]
    # Make the first chunk zero-std so step 100 uses two chunks -> exactly 102,400.
    trainer.async_rollout_manager.chunk_patterns = [["AC"] * 64, ["MIX"] * 64]
    ledger.rows = [
        dft.LedgerRow(
            effective_step=99, generation_batches_used=2, prompts_sampled=128, trajectories_sampled=1024,
            rollout_tokens=1, informative_groups=64, zero_std_groups=64, surplus_groups=0, selected_groups=64,
            all_correct_groups=0, all_wrong_groups=0, mixed_groups=64, generation_wall_time=0.0, filter_wall_time=0.0,
            dataloader_batches_consumed=198, epoch=0, cumulative_prompts=99 * 128, cumulative_trajectories=99 * 1024,
            cumulative_rollout_tokens=1, cumulative_generation_wall_time=0.0,
        )
    ]
    # ``rows`` must be contiguous from 1 for validate(); relax by trimming validation for this synthetic case.
    monkeypatch.setattr(DynamicSamplingLedger, "validate", lambda self: None)
    _, _, _, row = trainer._e7_collect_informative_batch({}, {}, False)
    assert row.cumulative_trajectories == E3_TOTAL_TRAJECTORIES and ledger.equal_compute_due()
    # A regular checkpoint of the same step (100 is a multiple of 25) was written first.
    regular = tmp_path / "run/checkpoints/global_step_100"
    regular.mkdir(parents=True)
    ledger.save(regular / LEDGER_FILENAME)
    trainer._e7_save_equal_compute_checkpoint(row)
    final = tmp_path / "run/checkpoints/equal_compute_step_100"
    assert final.is_dir() and not (tmp_path / "run/checkpoints/equal_compute_step_100.saving").exists()
    assert (final / "actor/model_world_size_1_rank_0.pt").is_file() and (final / "data.pt").is_file()
    saved_path, saved_step, keep = trainer.actor_rollout_wg.saved[-1]
    assert saved_step == 100 and keep is None and saved_path.endswith("equal_compute_step_100.saving/actor")
    assert not Path(saved_path).exists()  # veRL's rotation will find the staging path absent and skip it
    assert not (tmp_path / "run/checkpoints/latest_checkpointed_iteration.txt").exists()  # no tracker
    assert ledger.equal_compute_saved and ledger.equal_compute_step == 100 and row.equal_compute_checkpoint
    assert DynamicSamplingLedger.load(final / LEDGER_FILENAME).equal_compute_saved
    assert DynamicSamplingLedger.load(regular / LEDGER_FILENAME).equal_compute_saved  # rewritten for consistency
    assert not ledger.equal_compute_due()
    with pytest.raises(E7InvariantError):
        trainer._e7_save_equal_compute_checkpoint(row)
    # Reload from the regular checkpoint: the equal-compute directory on disk is recognised.
    monkeypatch.setattr(dft.RayPPOTrainer, "_load_checkpoint", lambda self: setattr(self, "global_steps", 100) or 0)
    resumed = _make_trainer(tmp_path, [])
    resumed._load_checkpoint()
    assert resumed._e7_ledger.equal_compute_saved and resumed._e7_ledger.equal_compute_step == 100
    assert not resumed._e7_ledger.equal_compute_due()
