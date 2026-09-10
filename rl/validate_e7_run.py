"""Fail-closed gates for M10 / E7 runs: smoke, resume check, segment end, formal completion.

All evidence is read from what the run persists (plans/M10.md §3.3, §4, §8; STEP0 §4–§6):

* ``e7_ledger.jsonl`` — one row per effective step (contiguous, ≤ 4 generation batches, exactly
  64 selected groups, conservation ``selected + surplus + zero_std == 64 x batches``);
* ``predictions/candidates/<step>.jsonl`` — every candidate group with its role; zero-std groups
  really have zero score variance, selected groups do not, and the selected multiset of
  ``(sample_id, score)`` equals what veRL dumped to ``predictions/train/<step>.jsonl``;
* ``checkpoints/global_step_<N>/e7_ledger.json`` — ledger step identity next to each checkpoint;
* ``checkpoints/equal_compute_step_<N>/`` — exactly one, at the ledger's flagged step;
* proof files — the driver binding was installed and restored, the trainer class was active;
* TensorBoard ``e7/*`` scalars for every step.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rl.custom.dynamic_filter_trainer import (
    E3_TOTAL_TRAJECTORIES,
    GROUP_SIZE,
    LEDGER_FILENAME,
    LEDGER_JSONL,
    MAX_GENERATION_BATCHES,
    TARGET_GROUPS,
    TRAINER_ACTIVATIONS_JSONL,
    TRAINER_CLASS_FQN,
    TRAINER_PROOF_FILENAME,
    DynamicSamplingLedger,
)
from rl.launch.e7_dynamic_filtering import (
    BOUNDARY_ACTIVATIONS_JSONL,
    BOUNDARY_PROOF_FILENAME,
    E3_DIFF_PATHS,
    RESUME_CHECK_STEPS,
    RESUME_CHECK_STOP_STEP,
    SEGMENT_STOP_STEPS,
    SMOKE_DIFF_PATHS,
    SMOKE_STEPS,
    compose_e7_config,
)
from rl.launch.e5v2_conserved_credit import diff_paths, resolved
from rl.monitor_run import read_native_scalars

ROWS_PER_STEP = TARGET_GROUPS * GROUP_SIZE
FORMAL_STEPS = 200
FORMAL_VALIDATION_STEPS = list(range(0, FORMAL_STEPS + 1, 25))
MAX_FULL_BATCH_STREAK = 10  # plan §4: ten consecutive steps at 4 batches pauses the run
REQUIRED_E7_METRICS = (
    "e7/generation_batches_used",
    "e7/informative_groups",
    "e7/zero_std_groups",
    "e7/surplus_groups",
    "e7/trajectories_sampled",
    "e7/cumulative_trajectories",
    "e7/rollout_tokens",
    "e7/generation_wall_time",
)
SCORE_TOL = 1e-6


class E7GateError(RuntimeError):
    pass


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise E7GateError(f"missing {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _status(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "status.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --------------------------------------------------------------------------- #
# Ledger
# --------------------------------------------------------------------------- #


def validate_ledger(run_dir: Path, expected_steps: int, *, threshold: int = E3_TOTAL_TRAJECTORIES) -> dict[str, Any]:
    rows = _rows(run_dir / LEDGER_JSONL)
    steps = [int(row["effective_step"]) for row in rows]
    if steps != list(range(1, expected_steps + 1)):
        raise E7GateError(f"ledger steps are not 1..{expected_steps}: {steps[:5]}...{steps[-3:]}")
    cumulative = 0
    batches_hist: Counter[int] = Counter()
    streak = best_streak = 0
    flagged: list[int] = []
    for row in rows:
        batches = int(row["generation_batches_used"])
        if not 1 <= batches <= MAX_GENERATION_BATCHES:
            raise E7GateError(f"step {row['effective_step']} used {batches} generation batches")
        if int(row["selected_groups"]) != TARGET_GROUPS or bool(row["underfilled"]):
            raise E7GateError(f"step {row['effective_step']} is underfilled or did not select 64 groups")
        total = int(row["informative_groups"]) + int(row["zero_std_groups"])
        if total != TARGET_GROUPS * batches:
            raise E7GateError(f"step {row['effective_step']} groups {total} != 64 x {batches}")
        if int(row["informative_groups"]) != TARGET_GROUPS + int(row["surplus_groups"]):
            raise E7GateError(f"step {row['effective_step']} informative/surplus mismatch")
        if int(row["trajectories_sampled"]) != ROWS_PER_STEP * batches or int(row["prompts_sampled"]) != TARGET_GROUPS * batches:
            raise E7GateError(f"step {row['effective_step']} trajectory accounting mismatch")
        cumulative += int(row["trajectories_sampled"])
        if int(row["cumulative_trajectories"]) != cumulative:
            raise E7GateError(f"step {row['effective_step']} cumulative trajectories {row['cumulative_trajectories']} != {cumulative}")
        batches_hist[batches] += 1
        streak = streak + 1 if batches == MAX_GENERATION_BATCHES else 0
        best_streak = max(best_streak, streak)
        if bool(row["equal_compute_checkpoint"]):
            flagged.append(int(row["effective_step"]))
    if len(flagged) > 1:
        raise E7GateError(f"more than one equal-compute flag: {flagged}")
    due = next((int(r["effective_step"]) for r in rows if int(r["cumulative_trajectories"]) >= threshold), None)
    if due is not None and flagged != [due]:
        raise E7GateError(f"equal-compute flag {flagged} does not match the first step reaching the E3 total ({due})")
    if due is None and flagged:
        raise E7GateError("equal-compute flagged before the E3 total was reached")
    return {
        "steps": expected_steps,
        "generation_batches_histogram": {str(k): v for k, v in sorted(batches_hist.items())},
        "mean_generation_batches": sum(int(r["generation_batches_used"]) for r in rows) / len(rows),
        "steps_with_resampling": sum(1 for r in rows if int(r["generation_batches_used"]) >= 2),
        "max_full_batch_streak": best_streak,
        "cumulative_trajectories": cumulative,
        "equal_compute_step": flagged[0] if flagged else None,
        "equal_compute_due_step": due,
        "zero_std_frac_mean": sum(int(r["zero_std_groups"]) / (int(r["informative_groups"]) + int(r["zero_std_groups"])) for r in rows) / len(rows),
    }


def full_batch_streak_gate(run_dir: Path) -> dict[str, Any]:
    """Plan §4: consecutive steps at the 4-batch cap (used by the watcher on a partial ledger)."""
    rows = _rows(run_dir / LEDGER_JSONL) if (run_dir / LEDGER_JSONL).is_file() else []
    streak = 0
    for row in rows:
        streak = streak + 1 if int(row["generation_batches_used"]) == MAX_GENERATION_BATCHES else 0
    return {"current_streak": streak, "limit": MAX_FULL_BATCH_STREAK, "pause": streak >= MAX_FULL_BATCH_STREAK, "steps": len(rows)}


# --------------------------------------------------------------------------- #
# Candidates vs. training JSONL
# --------------------------------------------------------------------------- #


def validate_candidates(run_dir: Path, steps: Iterable[int]) -> dict[str, Any]:
    ledger = {int(r["effective_step"]): r for r in _rows(run_dir / LEDGER_JSONL)}
    checked = 0
    for step in steps:
        row = ledger.get(step)
        if row is None:
            raise E7GateError(f"ledger has no row for step {step}")
        cands = _rows(run_dir / "predictions/candidates" / f"{step}.jsonl")
        batches = int(row["generation_batches_used"])
        if len(cands) != ROWS_PER_STEP * batches:
            raise E7GateError(f"step {step}: {len(cands)} candidate rows != 512 x {batches}")
        if any(int(c["step"]) != step for c in cands):
            raise E7GateError(f"step {step}: candidate rows from another step")
        by_uid: dict[str, list[dict[str, Any]]] = {}
        for c in cands:
            by_uid.setdefault(str(c["uid"]), []).append(c)
        roles: Counter[str] = Counter()
        for uid, members in by_uid.items():
            if len(members) != GROUP_SIZE:
                raise E7GateError(f"step {step}: uid {uid} has {len(members)} rows")
            role_set = {m["role"] for m in members}
            chunk_set = {int(m["chunk"]) for m in members}
            if len(role_set) != 1 or len(chunk_set) != 1:
                raise E7GateError(f"step {step}: uid {uid} spans roles/chunks {role_set}/{chunk_set}")
            role = role_set.pop()
            scores = [float(m["score"]) for m in members]
            zero_var = (max(scores) - min(scores)) < 1e-8
            if role == "zero_std" and not zero_var:
                raise E7GateError(f"step {step}: zero_std group {uid} has score variance")
            if role in ("selected", "surplus") and zero_var:
                raise E7GateError(f"step {step}: {role} group {uid} has no score variance")
            roles[role] += 1
        if roles["selected"] != int(row["selected_groups"]) or roles["surplus"] != int(row["surplus_groups"]) or roles["zero_std"] != int(row["zero_std_groups"]):
            raise E7GateError(f"step {step}: candidate roles {dict(roles)} disagree with the ledger")
        chunks_seen = {int(c["chunk"]) for c in cands}
        if chunks_seen != set(range(batches)):
            raise E7GateError(f"step {step}: candidate chunks {sorted(chunks_seen)} != 0..{batches - 1}")
        # Selected groups are exactly what veRL trained on and dumped.
        train = _rows(run_dir / "predictions/train" / f"{step}.jsonl")
        if len(train) != ROWS_PER_STEP:
            raise E7GateError(f"step {step}: train JSONL has {len(train)} rows, expected {ROWS_PER_STEP}")
        selected = Counter((str(c["sample_id"]), round(float(c["score"]), 6)) for c in cands if c["role"] == "selected")
        trained = Counter((str(t["sample_id"]), round(float(t["score"]), 6)) for t in train)
        if selected != trained:
            missing = selected - trained
            extra = trained - selected
            raise E7GateError(f"step {step}: selected != trained (missing {list(missing)[:3]}, extra {list(extra)[:3]})")
        excluded = Counter((str(c["sample_id"]), round(float(c["score"]), 6)) for c in cands if c["role"] != "selected")
        # A dropped group can only coincide with a trained (sample_id, score) pair if the same
        # prompt was drawn twice in one step (epoch boundary); otherwise it must be absent.
        overlap = excluded & trained
        if overlap and sum(overlap.values()) > len(train):
            raise E7GateError(f"step {step}: dropped groups appear in the train JSONL: {list(overlap)[:3]}")
        checked += 1
    return {"steps_checked": checked}


# --------------------------------------------------------------------------- #
# Proofs, metrics, checkpoints
# --------------------------------------------------------------------------- #


def validate_proofs(run_dir: Path, *, min_activations: int) -> dict[str, Any]:
    trainer_proof = json.loads((run_dir / TRAINER_PROOF_FILENAME).read_text(encoding="utf-8"))
    if trainer_proof.get("actual_class") != TRAINER_CLASS_FQN or trainer_proof.get("trainer_class") != TRAINER_CLASS_FQN:
        raise E7GateError(f"trainer proof does not name {TRAINER_CLASS_FQN}")
    if trainer_proof["filter_groups"] != {"enable": True, "metric": "score", "max_num_gen_batches": MAX_GENERATION_BATCHES}:
        raise E7GateError(f"trainer proof filter_groups drifted: {trainer_proof['filter_groups']}")
    if any(trainer_proof["offload"].values()):
        raise E7GateError("trainer proof reports FSDP offload still on")
    boundary = json.loads((run_dir / BOUNDARY_PROOF_FILENAME).read_text(encoding="utf-8"))
    if not (boundary.get("installed") and boundary.get("restored")):
        raise E7GateError("driver binding proof is not installed+restored")
    if boundary.get("bound_to") != TRAINER_CLASS_FQN:
        raise E7GateError("driver binding proof names the wrong trainer")
    trainer_acts = _rows(run_dir / TRAINER_ACTIVATIONS_JSONL)
    boundary_acts = _rows(run_dir / BOUNDARY_ACTIVATIONS_JSONL)
    if len(trainer_acts) < min_activations or len(boundary_acts) < min_activations:
        raise E7GateError(f"expected >= {min_activations} activations, got trainer {len(trainer_acts)} / boundary {len(boundary_acts)}")
    if any(not act.get("restored") for act in boundary_acts):
        raise E7GateError("a driver binding activation was never restored")
    return {
        "trainer_activations": len(trainer_acts),
        "boundary_activations": len(boundary_acts),
        "stop_at_steps": [act.get("stop_at_step") for act in boundary_acts],
        "pinned_sha256": trainer_proof["pinned_sha256"],
    }


def validate_metrics(run_dir: Path, steps: Iterable[int]) -> dict[str, Any]:
    native = read_native_scalars(run_dir / "tensorboard")
    missing: list[str] = []
    for tag in REQUIRED_E7_METRICS:
        have = {int(item["step"]) for item in native.get(tag, [])}
        for step in steps:
            if step not in have:
                missing.append(f"{tag}@{step}")
    if missing:
        raise E7GateError(f"missing e7 metrics: {missing[:10]}")
    kl = native.get("actor/kl_loss", [])
    return {"e7_metrics_present": True, "kl_loss_points": len(kl)}


def validate_checkpoint_ledgers(run_dir: Path) -> dict[str, Any]:
    root = run_dir / "checkpoints"
    steps: list[int] = []
    candidates = [
        (int(path.name.removeprefix("global_step_")), path)
        for path in root.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    ]
    for step, path in sorted(candidates):
        ledger = DynamicSamplingLedger.load(path / LEDGER_FILENAME)
        if ledger.effective_step != step:
            raise E7GateError(f"{path.name}: ledger step {ledger.effective_step} != {step}")
        steps.append(step)
    tracker = root / "latest_checkpointed_iteration.txt"
    tracked = int(tracker.read_text(encoding="utf-8").strip()) if tracker.is_file() else None
    if tracked is not None and tracked not in steps:
        raise E7GateError(f"tracker points at step {tracked} without a ledger-bearing checkpoint")
    return {"checkpoint_steps": steps, "tracker": tracked}


def validate_equal_compute(run_dir: Path, expected_step: int | None) -> dict[str, Any]:
    root = run_dir / "checkpoints"
    dirs = sorted(p for p in root.glob("equal_compute_step_*") if p.is_dir()) if root.exists() else []
    if expected_step is None:
        if dirs:
            raise E7GateError(f"unexpected equal-compute checkpoints: {dirs}")
        return {"present": False}
    if len(dirs) != 1 or dirs[0].name != f"equal_compute_step_{expected_step}":
        raise E7GateError(f"equal-compute checkpoint dirs {dirs} != step {expected_step}")
    folder = dirs[0]
    if not (folder / LEDGER_FILENAME).is_file():
        raise E7GateError(f"equal-compute checkpoint {folder.name} has no {LEDGER_FILENAME}")
    ledger = DynamicSamplingLedger.load(folder / LEDGER_FILENAME)
    if ledger.effective_step != expected_step or ledger.equal_compute_step != expected_step or not ledger.equal_compute_saved:
        raise E7GateError("equal-compute ledger disagrees with the directory step")
    if ledger.cumulative_trajectories < ledger.equal_compute_threshold:
        raise E7GateError("equal-compute ledger below the E3 trajectory total")
    if len(ledger.rows) > 1 and ledger.rows[-2].cumulative_trajectories >= ledger.equal_compute_threshold:
        raise E7GateError("equal-compute checkpoint is not the first step reaching the E3 total")
    required = ["actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt", "actor/extra_state_world_size_1_rank_0.pt", "data.pt"]
    missing = [name for name in required if not (folder / name).is_file() or (folder / name).stat().st_size <= 0]
    if missing:
        raise E7GateError(f"equal-compute checkpoint incomplete: {missing}")
    if (root / f"equal_compute_step_{expected_step}.saving").exists():
        raise E7GateError("equal-compute staging directory still exists")
    return {
        "present": True,
        "step": expected_step,
        "cumulative_trajectories": ledger.cumulative_trajectories,
        "total_bytes": sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()),
    }


def validate_validation_cadence(run_dir: Path, expected_steps: list[int]) -> dict[str, Any]:
    validation = run_dir / "predictions/validation"
    actual = sorted(int(p.stem) for p in validation.glob("*.jsonl")) if validation.exists() else []
    if actual != expected_steps:
        raise E7GateError(f"validation cadence {actual} != {expected_steps}")
    curve: dict[str, float] = {}
    for step in expected_steps:
        rows = _rows(validation / f"{step}.jsonl")
        if len(rows) != 100:
            raise E7GateError(f"validation step {step} has {len(rows)} rows")
        values = [float(r["acc"]) for r in rows]
        if not all(math.isfinite(v) for v in values):
            raise E7GateError(f"non-finite validation acc at step {step}")
        curve[str(step)] = sum(values) / len(values)
    return {"steps": actual, "pass_at_1": curve}


def validate_resolved_config(run_dir: Path, mode: str) -> dict[str, Any]:
    from omegaconf import OmegaConf

    saved = resolved(OmegaConf.load(run_dir / "resolved_config.yaml"))
    expected = resolved(compose_e7_config(run_dir.name, mode=mode))
    drift = diff_paths(saved, expected)
    if drift:
        raise E7GateError(f"resolved config drifted from the {mode} composition: {sorted(drift)}")
    formal = resolved(compose_e7_config(run_dir.name, mode="formal"))
    smoke_diff = diff_paths(formal, saved)
    if mode == "formal" and smoke_diff:
        raise E7GateError(f"formal run differs from the formal composition: {sorted(smoke_diff)}")
    if mode != "formal" and not smoke_diff <= SMOKE_DIFF_PATHS:
        raise E7GateError(f"{mode} differs from formal beyond the smoke set: {sorted(smoke_diff - SMOKE_DIFF_PATHS)}")
    return {"mode": mode, "diff_vs_formal": sorted(smoke_diff), "e3_diff_paths": sorted(E3_DIFF_PATHS)}


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #


def smoke_gate(run_dir: Path) -> dict[str, Any]:
    status = _status(run_dir)
    if status.get("state") != "completed":
        raise E7GateError(f"smoke status is {status.get('state')!r}")
    report: dict[str, Any] = {"gate": "smoke", "run_dir": str(run_dir)}
    report["config"] = validate_resolved_config(run_dir, "smoke")
    report["ledger"] = validate_ledger(run_dir, SMOKE_STEPS)
    if report["ledger"]["steps_with_resampling"] < 1:
        raise E7GateError("smoke never resampled: no step used >= 2 generation batches")
    report["candidates"] = validate_candidates(run_dir, range(1, SMOKE_STEPS + 1))
    report["proofs"] = validate_proofs(run_dir, min_activations=1)
    report["metrics"] = validate_metrics(run_dir, range(1, SMOKE_STEPS + 1))
    report["validation"] = validate_validation_cadence(run_dir, [0, SMOKE_STEPS])
    leftovers = sorted(p.name for p in (run_dir / "checkpoints").glob("*step_*")) if (run_dir / "checkpoints").exists() else []
    if leftovers:
        raise E7GateError(f"smoke left checkpoints behind: {leftovers}")
    report["checkpoints_left"] = leftovers
    report["equal_compute"] = validate_equal_compute(run_dir, None)
    return report


def resume_check_gate(run_dir: Path) -> dict[str, Any]:
    status = _status(run_dir)
    if status.get("state") != "completed":
        raise E7GateError(f"resume-check status is {status.get('state')!r}")
    report: dict[str, Any] = {"gate": "resume_check", "run_dir": str(run_dir)}
    report["config"] = validate_resolved_config(run_dir, "resume_check")
    report["ledger"] = validate_ledger(run_dir, RESUME_CHECK_STEPS)
    report["candidates"] = validate_candidates(run_dir, range(1, RESUME_CHECK_STEPS + 1))
    report["proofs"] = validate_proofs(run_dir, min_activations=2)
    if report["proofs"]["stop_at_steps"][0] != RESUME_CHECK_STOP_STEP:
        raise E7GateError("first segment of the resume check did not stop at step 1")
    report["metrics"] = validate_metrics(run_dir, range(1, RESUME_CHECK_STEPS + 1))
    report["validation"] = validate_validation_cadence(run_dir, [0, RESUME_CHECK_STOP_STEP, RESUME_CHECK_STEPS])
    report["checkpoints"] = validate_checkpoint_ledgers(run_dir)
    if report["checkpoints"]["checkpoint_steps"] != [RESUME_CHECK_STOP_STEP, RESUME_CHECK_STEPS] or report["checkpoints"]["tracker"] != RESUME_CHECK_STEPS:
        raise E7GateError(f"resume-check checkpoints {report['checkpoints']} unexpected")
    archives = sorted(p.name for p in (run_dir / "recovery").glob("resume_from_*")) if (run_dir / "recovery").exists() else []
    if len(archives) != 1 or not archives[0].startswith(f"resume_from_{RESUME_CHECK_STOP_STEP}_"):
        raise E7GateError(f"resume-check recovery archives unexpected: {archives}")
    recovery = json.loads((run_dir / "recovery" / archives[0] / "recovery.json").read_text(encoding="utf-8"))
    if recovery.get("recomputed_train_steps"):
        raise E7GateError("a planned segment stop must not leave train steps to recompute")
    report["recovery_archives"] = archives
    report["equal_compute"] = validate_equal_compute(run_dir, None)
    return report


def segment_gate(run_dir: Path, stop_step: int) -> dict[str, Any]:
    status = _status(run_dir)
    expected_state = "completed" if stop_step >= FORMAL_STEPS else "segment_completed"
    if status.get("state") != expected_state:
        raise E7GateError(f"segment status is {status.get('state')!r}, expected {expected_state!r}")
    report: dict[str, Any] = {
        "gate": "segment",
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "status": status.get("state"),
        "stop_step": stop_step,
    }
    report["ledger"] = validate_ledger(run_dir, stop_step)
    if report["ledger"]["max_full_batch_streak"] >= MAX_FULL_BATCH_STREAK:
        raise E7GateError(f"{report['ledger']['max_full_batch_streak']} consecutive steps at 4 batches (plan §4 pause)")
    report["config"] = validate_resolved_config(run_dir, "formal")
    report["candidates"] = validate_candidates(run_dir, range(1, stop_step + 1))
    segments_so_far = sum(1 for s in SEGMENT_STOP_STEPS if s <= stop_step) or 1
    report["proofs"] = validate_proofs(run_dir, min_activations=1)
    report["expected_segments_so_far"] = segments_so_far
    report["metrics"] = validate_metrics(run_dir, range(1, stop_step + 1))
    report["validation"] = validate_validation_cadence(run_dir, [s for s in FORMAL_VALIDATION_STEPS if s <= stop_step])
    report["checkpoints"] = validate_checkpoint_ledgers(run_dir)
    if report["checkpoints"]["tracker"] != stop_step:
        raise E7GateError(f"tracker {report['checkpoints']['tracker']} != segment stop {stop_step}")
    report["equal_compute"] = validate_equal_compute(run_dir, report["ledger"]["equal_compute_step"])
    return report


def formal_completion(run_dir: Path) -> dict[str, Any]:
    from rl.validate_e5_canonical_smoke import validate_checkpoint

    report = segment_gate(run_dir, FORMAL_STEPS)
    report["gate"] = "formal_completion"
    if report["status"] != "completed":
        raise E7GateError(f"formal completion needs status 'completed', got {report['status']!r}")
    if report["ledger"]["equal_compute_step"] is None:
        raise E7GateError("formal run never reached the equal-compute trajectory total")
    proofs = validate_proofs(run_dir, min_activations=len(SEGMENT_STOP_STEPS))
    report["proofs"] = proofs
    archives = sorted(p.name for p in (run_dir / "recovery").glob("resume_from_*")) if (run_dir / "recovery").exists() else []
    if len(archives) < len(SEGMENT_STOP_STEPS) - 1:
        raise E7GateError(f"expected >= {len(SEGMENT_STOP_STEPS) - 1} recovery archives, got {archives}")
    report["recovery_archives"] = archives
    report["final_checkpoint"] = validate_checkpoint(run_dir, FORMAL_STEPS)
    report["five_artifacts"] = {
        name: (run_dir / name).is_file() for name in ("resolved_config.yaml", "preflight.json", "metrics.json", "summary.md")
    }
    if not all(report["five_artifacts"].values()):
        raise E7GateError(f"missing run artifacts: {report['five_artifacts']}")
    return report


# The v6 evaluation freeze binds ``analysis/formal_completion_gate.json`` by name, matching the
# M8/M9 convention; the other gates keep the ``<gate>_gate.json`` shape.
GATE_OUTPUT_NAMES = {
    "smoke": "smoke_gate.json",
    "resume-check": "resume_check_gate.json",
    "segment": "segment_gate.json",
    "formal": "formal_completion_gate.json",
}

GATES = {
    "smoke": lambda run_dir, args: smoke_gate(run_dir),
    "resume-check": lambda run_dir, args: resume_check_gate(run_dir),
    "segment": lambda run_dir, args: segment_gate(run_dir, int(args.stop_step)),
    "formal": lambda run_dir, args: formal_completion(run_dir),
    "streak": lambda run_dir, args: full_batch_streak_gate(run_dir),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gate", choices=sorted(GATES))
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--stop-step", type=int, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.gate == "segment" and args.stop_step is None:
        raise SystemExit("segment gate needs --stop-step")
    report = GATES[args.gate](run_dir, args)
    if args.gate != "streak":
        out = run_dir / "analysis" / GATE_OUTPUT_NAMES[args.gate]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
