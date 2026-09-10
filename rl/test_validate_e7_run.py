"""M10 / E7 run-gate tests: the validator over a synthetic run directory (no GPU).

Builds a small but structurally faithful run: ledger JSONL, per-step candidates with roles,
veRL-style train JSONL for the selected groups, checkpoint ledgers, proofs, TensorBoard
scalars, and an equal-compute directory; then checks every fail-closed path.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest
from torch.utils.tensorboard import SummaryWriter

from rl.custom.dynamic_filter_trainer import (
    E3_TOTAL_TRAJECTORIES,
    GROUP_SIZE,
    LEDGER_FILENAME,
    LEDGER_JSONL,
    TARGET_GROUPS,
    TRAINER_ACTIVATIONS_JSONL,
    TRAINER_CLASS_FQN,
    TRAINER_PROOF_FILENAME,
    DynamicSamplingLedger,
    LedgerRow,
)
from rl.launch.e7_dynamic_filtering import BOUNDARY_ACTIVATIONS_JSONL, BOUNDARY_PROOF_FILENAME
from rl.validate_e7_run import (
    E7GateError,
    full_batch_streak_gate,
    validate_candidates,
    validate_equal_compute,
    validate_ledger,
    validate_metrics,
    validate_proofs,
)

ROWS = TARGET_GROUPS * GROUP_SIZE


def _mixed_scores(rng: random.Random) -> list[float]:
    scores = [rng.choice([0.0, 0.1, 1.0, 1.1]) for _ in range(GROUP_SIZE)]
    if max(scores) == min(scores):
        scores[0] = 1.1 if scores[0] < 1.0 else 0.0
    return scores


def build_run(root: Path, *, steps: int, batches: list[int], equal_compute: bool = True, trajectories_per_step: int | None = None) -> Path:
    """Write a synthetic E7 run with ``batches[i]`` generation batches at step i+1."""
    rng = random.Random(7)
    run = root / "e7_synth"
    (run / "predictions/train").mkdir(parents=True)
    (run / "predictions/candidates").mkdir(parents=True)
    (run / "checkpoints").mkdir()
    (run / "tensorboard").mkdir()
    writer = SummaryWriter(str(run / "tensorboard"))
    ledger = DynamicSamplingLedger()
    if trajectories_per_step:
        ledger.equal_compute_threshold = trajectories_per_step * 3
    for step in range(1, steps + 1):
        n_batches = batches[step - 1]
        candidates: list[dict[str, Any]] = []
        train: list[dict[str, Any]] = []
        informative_seen = 0
        counts = {"selected": 0, "surplus": 0, "zero_std": 0}
        patterns = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
        for chunk in range(n_batches):
            for g in range(TARGET_GROUPS):
                uid = f"s{step}c{chunk}g{g}"
                sample_id = f"math_train_{rng.randrange(5203):06d}"
                # ~55% informative in chunk 0/1, all zero-std in later chunks to force 4-batch steps.
                informative = rng.random() < 0.55 if chunk < 2 else False
                if step == steps and n_batches == 4:
                    informative = chunk == 3 and g < 64  # last chunk fills the batch
                if informative:
                    scores = _mixed_scores(rng)
                    role = "selected" if informative_seen < TARGET_GROUPS else "surplus"
                    informative_seen += 1
                    patterns["mixed"] += 1
                else:
                    value = rng.choice([0.0, 1.1])
                    scores = [value] * GROUP_SIZE
                    role = "zero_std"
                    patterns["all_correct" if value > 1 else "all_wrong"] += 1
                counts[role] += 1
                for k, score in enumerate(scores):
                    row = {"step": step, "chunk": chunk, "uid": uid, "role": role, "sample_id": sample_id, "score": score, "acc": 1.0 if score >= 1 else 0.0, "response_length": 10 + k, "acc_pattern": "mixed" if informative else "all"}
                    candidates.append(row)
                    if role == "selected":
                        train.append({"sample_id": sample_id, "score": score, "acc": row["acc"], "step": step})
        if counts["selected"] != TARGET_GROUPS:
            # Top up deterministically so every synthetic step selects exactly 64 groups.
            deficit = TARGET_GROUPS - counts["selected"]
            for g in range(deficit):
                uid = f"s{step}fill{g}"
                sample_id = f"math_train_{rng.randrange(5203):06d}"
                scores = _mixed_scores(rng)
                for k, score in enumerate(scores):
                    candidates.append({"step": step, "chunk": n_batches - 1, "uid": uid, "role": "selected", "sample_id": sample_id, "score": score, "acc": 1.0 if score >= 1 else 0.0, "response_length": 10 + k, "acc_pattern": "mixed"})
                    train.append({"sample_id": sample_id, "score": score, "acc": 1.0 if score >= 1 else 0.0, "step": step})
                # keep 512 x batches rows: demote one zero_std group from the last chunk into nothing
                victim = next(c["uid"] for c in candidates if c["role"] == "zero_std" and c["chunk"] == n_batches - 1)
                candidates = [c for c in candidates if c["uid"] != victim]
                counts["zero_std"] -= 1
                counts["selected"] += 1
                patterns["mixed"] += 1
        rng.shuffle(train)
        (run / "predictions/candidates" / f"{step}.jsonl").write_text("".join(json.dumps(c) + "\n" for c in candidates), encoding="utf-8")
        (run / "predictions/train" / f"{step}.jsonl").write_text("".join(json.dumps(t) + "\n" for t in train), encoding="utf-8")
        ledger.effective_step = step
        ledger.dataloader_batches_consumed += n_batches
        ledger.cumulative_prompts += TARGET_GROUPS * n_batches
        ledger.cumulative_trajectories += ROWS * n_batches
        ledger.cumulative_rollout_tokens += 1000
        row = LedgerRow(
            effective_step=step, generation_batches_used=n_batches, prompts_sampled=TARGET_GROUPS * n_batches,
            trajectories_sampled=ROWS * n_batches, rollout_tokens=1000, informative_groups=counts["selected"] + counts["surplus"],
            zero_std_groups=counts["zero_std"], surplus_groups=counts["surplus"], selected_groups=counts["selected"],
            all_correct_groups=patterns["all_correct"], all_wrong_groups=patterns["all_wrong"], mixed_groups=patterns["mixed"],
            generation_wall_time=50.0, filter_wall_time=0.1, dataloader_batches_consumed=ledger.dataloader_batches_consumed,
            epoch=ledger.dataloader_batches_consumed // 81, cumulative_prompts=ledger.cumulative_prompts,
            cumulative_trajectories=ledger.cumulative_trajectories, cumulative_rollout_tokens=ledger.cumulative_rollout_tokens,
            cumulative_generation_wall_time=50.0 * step,
        )
        ledger.rows.append(row)
        if equal_compute and ledger.equal_compute_due():
            ledger.mark_equal_compute_saved(step)
            folder = run / "checkpoints" / f"equal_compute_step_{step}"
            (folder / "actor").mkdir(parents=True)
            for name in ("model_world_size_1_rank_0.pt", "optim_world_size_1_rank_0.pt", "extra_state_world_size_1_rank_0.pt"):
                (folder / "actor" / name).write_bytes(b"x")
            (folder / "data.pt").write_bytes(b"x")
            ledger.save(folder / LEDGER_FILENAME)
        for key, value in row.metrics().items():
            writer.add_scalar(key, value, step)
        writer.add_scalar("actor/kl_loss", 0.001, step)
        if step % 25 == 0 or step == steps:
            folder = run / "checkpoints" / f"global_step_{step}"
            folder.mkdir()
            ledger.save(folder / LEDGER_FILENAME)
            (run / "checkpoints/latest_checkpointed_iteration.txt").write_text(str(step), encoding="utf-8")
    writer.close()
    ledger.rewrite_jsonl(run / LEDGER_JSONL)
    trainer_proof = {
        "trainer_class": TRAINER_CLASS_FQN, "actual_class": TRAINER_CLASS_FQN,
        "filter_groups": {"enable": True, "metric": "score", "max_num_gen_batches": 4},
        "offload": {"actor_param_offload": False, "actor_optimizer_offload": False, "ref_param_offload": False},
        "pinned_sha256": {"verl/trainer/ppo/ray_trainer.py": "de58"},
    }
    (run / TRAINER_PROOF_FILENAME).write_text(json.dumps(trainer_proof), encoding="utf-8")
    (run / TRAINER_ACTIVATIONS_JSONL).write_text(json.dumps(trainer_proof) + "\n", encoding="utf-8")
    boundary = {"installed": True, "restored": True, "bound_to": TRAINER_CLASS_FQN, "stop_at_step": None}
    (run / BOUNDARY_PROOF_FILENAME).write_text(json.dumps(boundary), encoding="utf-8")
    (run / BOUNDARY_ACTIVATIONS_JSONL).write_text(json.dumps(boundary) + "\n", encoding="utf-8")
    return run


def test_ledger_candidates_proofs_metrics_pass_on_a_consistent_run(tmp_path: Path) -> None:
    run = build_run(tmp_path, steps=6, batches=[2, 3, 2, 1, 4, 2], equal_compute=True, trajectories_per_step=ROWS * 2)
    summary = validate_ledger(run, 6, threshold=ROWS * 2 * 3)
    assert summary["generation_batches_histogram"] == {"1": 1, "2": 3, "3": 1, "4": 1}
    assert summary["steps_with_resampling"] == 5 and summary["max_full_batch_streak"] == 1
    assert summary["equal_compute_step"] == 3  # threshold = 3 x 1024 crossed at step 3 (2+3 batches... 5120 >= 3072)
    assert validate_candidates(run, range(1, 7)) == {"steps_checked": 6}
    proofs = validate_proofs(run, min_activations=1)
    assert proofs["trainer_activations"] == 1
    assert validate_metrics(run, range(1, 7))["e7_metrics_present"]
    equal = validate_equal_compute(run, 3)
    assert equal["present"] and equal["step"] == 3
    assert full_batch_streak_gate(run) == {"current_streak": 0, "limit": 10, "pause": False, "steps": 6}


def test_ledger_gate_rejects_gaps_underfill_and_bad_accounting(tmp_path: Path) -> None:
    run = build_run(tmp_path, steps=3, batches=[2, 2, 2], equal_compute=False)
    import copy

    rows = [json.loads(l) for l in (run / LEDGER_JSONL).read_text().splitlines()]
    with pytest.raises(E7GateError, match="1..4"):
        validate_ledger(run, 4)
    broken = copy.deepcopy(rows)
    broken[1]["selected_groups"] = 63
    (run / LEDGER_JSONL).write_text("".join(json.dumps(r) + "\n" for r in broken))
    with pytest.raises(E7GateError, match="underfilled"):
        validate_ledger(run, 3)
    broken = copy.deepcopy(rows)
    broken[2]["cumulative_trajectories"] += 1
    (run / LEDGER_JSONL).write_text("".join(json.dumps(r) + "\n" for r in broken))
    with pytest.raises(E7GateError, match="cumulative"):
        validate_ledger(run, 3)
    broken = copy.deepcopy(rows)
    broken[0]["equal_compute_checkpoint"] = True
    (run / LEDGER_JSONL).write_text("".join(json.dumps(r) + "\n" for r in broken))
    with pytest.raises(E7GateError, match="equal-compute"):
        validate_ledger(run, 3)


def test_candidates_gate_detects_leaks_and_role_lies(tmp_path: Path) -> None:
    run = build_run(tmp_path, steps=2, batches=[2, 2], equal_compute=False)
    cands_path = run / "predictions/candidates/1.jsonl"
    cands = [json.loads(l) for l in cands_path.read_text().splitlines()]
    # A zero-std group relabelled as selected has no variance -> refused.
    zero = next(c["uid"] for c in cands if c["role"] == "zero_std")
    lying = [dict(c, role="selected") if c["uid"] == zero else c for c in cands]
    cands_path.write_text("".join(json.dumps(c) + "\n" for c in lying))
    with pytest.raises(E7GateError, match="no score variance"):
        validate_candidates(run, [1])
    # Train JSONL containing a dropped group -> refused.
    cands_path.write_text("".join(json.dumps(c) + "\n" for c in cands))
    train_path = run / "predictions/train/1.jsonl"
    train = [json.loads(l) for l in train_path.read_text().splitlines()]
    dropped = [c for c in cands if c["uid"] == zero]
    train[:GROUP_SIZE] = [{"sample_id": d["sample_id"], "score": d["score"], "acc": d["acc"], "step": 1} for d in dropped]
    train_path.write_text("".join(json.dumps(t) + "\n" for t in train))
    with pytest.raises(E7GateError, match="selected != trained"):
        validate_candidates(run, [1])
    # Row count mismatch -> refused.
    train_path.write_text("".join(json.dumps(t) + "\n" for t in train[:-1]))
    with pytest.raises(E7GateError):
        validate_candidates(run, [1])


def test_proof_and_equal_compute_gates_fail_closed(tmp_path: Path) -> None:
    run = build_run(tmp_path, steps=2, batches=[2, 2], equal_compute=False)
    with pytest.raises(E7GateError, match="activations"):
        validate_proofs(run, min_activations=2)
    bad = json.loads((run / TRAINER_PROOF_FILENAME).read_text())
    bad["offload"]["actor_param_offload"] = True
    (run / TRAINER_PROOF_FILENAME).write_text(json.dumps(bad))
    with pytest.raises(E7GateError, match="offload"):
        validate_proofs(run, min_activations=1)
    assert validate_equal_compute(run, None) == {"present": False}
    (run / "checkpoints/equal_compute_step_2").mkdir()
    with pytest.raises(E7GateError):
        validate_equal_compute(run, None)
    with pytest.raises(E7GateError):
        validate_equal_compute(run, 2)


def test_streak_gate_counts_consecutive_full_batch_steps(tmp_path: Path) -> None:
    run = build_run(tmp_path, steps=12, batches=[4] * 12, equal_compute=False)
    streak = full_batch_streak_gate(run)
    assert streak["current_streak"] == 12 and streak["pause"] is True
    with pytest.raises(E7GateError, match="consecutive"):
        from rl.validate_e7_run import segment_gate

        (run / "status.json").write_text(json.dumps({"state": "segment_completed"}))
        (run / "resolved_config.yaml").write_text("trainer: {}\n")
        segment_gate(run, 12)


def test_threshold_constant_is_the_e3_total() -> None:
    assert E3_TOTAL_TRAJECTORIES == 200 * 64 * 8 == 102_400
