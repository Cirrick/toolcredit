"""M10 / E7: DAPO-style zero-variance group filtering with same-snapshot resampling.

Approved boundary: ``plans/M10.md`` §3/§6 and ``plans/M10_STEP0_BOUNDARY.md`` §3–§5.  Nothing in
veRL's site-packages is modified.  This module provides

* ``split_groups_by_score_variance`` / ``select_informative_groups`` — pure functions that
  classify every UID group of a generation chunk by the *scalar* ``score`` that enters the GRPO
  advantage (``rm_scores.sum(-1)``, answer + 0.1 format), never by ``acc``;
* ``DynamicSamplingLedger`` — the per-effective-step bookkeeping (plan §3.3) that is serialised
  next to every checkpoint and drives the equal-compute checkpoint (plan §3.4);
* ``DynamicFilterRayPPOTrainer`` — a subclass of the pinned ``RayPPOTrainer`` whose ``fit()`` is a
  copy of ``ray_trainer.py:1362–1770`` (SHA ``de58d295…6d66``) with exactly the eight whitelisted
  differences W1–W7, each marked ``# E7-W<n>`` and proved by Fixture R
  (``rl/custom/test_dynamic_filter.py``).  The only data-flow change is W3: the upstream
  prompt → generation → reward segment is executed per chunk (verbatim lines) inside
  ``_e7_collect_informative_batch``; everything after ``extract_reward`` (old/ref log prob,
  advantage, actor update, weight sync, validation, metrics) is untouched.

Frozen semantics (plan §3.1): 64 prompts × 8 rollouts per chunk, at most 4 chunks per effective
update, the first 64 informative groups in sampling order form the actor batch, surplus and
zero-std groups are recorded as evidence and never trained on, surplus is never recycled across
snapshots, fewer than 64 informative groups after 4 chunks fails explicitly before any actor
update, and rollout replicas stay awake between chunks with exactly one ``sleep_replicas()`` in
``finally``.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
    compute_spec_decode_metrics,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import Role
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip

# --------------------------------------------------------------------------- #
# Frozen constants (plans/M10.md §3)
# --------------------------------------------------------------------------- #

PINNED_RAY_TRAINER_SHA256 = "de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66"
PINNED_MAIN_PPO_SHA256 = "e3fe6e73b18d63367402a2570c5ba054a2b05443ac40a168524dbf545b1da392"
FIT_SOURCE_LINES = (1362, 1770)  # inclusive, 409 lines of the pinned ``RayPPOTrainer.fit``
GROUP_SIZE = 8
TARGET_GROUPS = 64
MAX_GENERATION_BATCHES = 4
FILTER_METRIC = "score"
SCORE_TOLERANCE = 1e-8
SCORE_IDENTITY_TOLERANCE = 1e-6
# E3 total rollout trajectories: 200 effective steps x 64 prompts x 8 rollouts (plan §3.4).
E3_TOTAL_TRAJECTORIES = 200 * TARGET_GROUPS * GROUP_SIZE
LEDGER_SCHEMA_VERSION = "toolcredit_e7_ledger_v1"
LEDGER_FILENAME = "e7_ledger.json"
LEDGER_JSONL = "e7_ledger.jsonl"
CANDIDATES_DIRNAME = "candidates"
STOP_REQUEST_FILENAME = "e7_stop_request.json"
TRAINER_PROOF_FILENAME = "e7_trainer_active.json"
TRAINER_ACTIVATIONS_JSONL = "e7_trainer_activations.jsonl"
LOG_MARKER = "TOOLCREDIT_E7_TRAINER_ACTIVE "
TRAINER_CLASS_FQN = "rl.custom.dynamic_filter_trainer.DynamicFilterRayPPOTrainer"
ROLE_SELECTED = "selected"
ROLE_SURPLUS = "surplus"
ROLE_ZERO_STD = "zero_std"

E7_DEBUG = os.environ.get("TOOLCREDIT_E7_DEBUG", "0") not in {"", "0", "false", "False"}


class E7UnderfilledError(RuntimeError):
    """Fewer than ``TARGET_GROUPS`` informative groups after ``MAX_GENERATION_BATCHES`` chunks."""


class E7InvariantError(RuntimeError):
    """A frozen-semantics invariant was violated; the run must stop (plan §8 hard stop)."""


# --------------------------------------------------------------------------- #
# Pure filtering functions (Fixture N / O)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GroupSplit:
    """One generation chunk classified by the scalar ``score`` variance of each UID group."""

    chunk_index: int
    group_uids: tuple[str, ...]
    informative_uids: tuple[str, ...]
    zero_std_uids: tuple[str, ...]
    group_scores: dict[str, tuple[float, ...]]
    group_acc: dict[str, tuple[float, ...]]

    @property
    def n_groups(self) -> int:
        return len(self.group_uids)

    def acc_pattern(self, uid: str) -> str:
        values = self.group_acc.get(uid)
        if not values:
            return "unknown"
        if all(value == 1.0 for value in values):
            return "all_correct"
        if all(value == 0.0 for value in values):
            return "all_wrong"
        return "mixed"


def _group_is_zero_std(scores: np.ndarray, tolerance: float) -> bool:
    spread = float(scores.max() - scores.min())
    std = float(scores.std())
    return spread < tolerance and std < tolerance


def split_groups_by_score_variance(
    uids: Any,
    scores: Any,
    *,
    chunk_index: int,
    group_size: int = GROUP_SIZE,
    acc: Any | None = None,
    tolerance: float = SCORE_TOLERANCE,
) -> GroupSplit:
    """Classify contiguous UID groups of a chunk as informative or zero-std by ``score``.

    ``uids``/``scores`` are in rollout order (``repeat(interleave=True)``), so every group of
    ``group_size`` consecutive rows must share one UID and no UID may recur elsewhere.
    Zero variance means ``max - min < tolerance`` *and* ``std < tolerance`` (plan §3.1).
    """
    uid_list = [str(value) for value in np.asarray(uids, dtype=object).tolist()]
    score_array = np.asarray(
        scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else scores, dtype=np.float64
    ).reshape(-1)
    if len(uid_list) != len(score_array):
        raise E7InvariantError(f"uid/score length mismatch: {len(uid_list)} != {len(score_array)}")
    if len(uid_list) == 0 or len(uid_list) % group_size != 0:
        raise E7InvariantError(f"chunk of {len(uid_list)} rows is not a multiple of group_size={group_size}")
    if not np.all(np.isfinite(score_array)):
        raise E7InvariantError("non-finite score in chunk")
    acc_array = None
    if acc is not None:
        acc_array = np.asarray(
            acc.detach().cpu().numpy() if isinstance(acc, torch.Tensor) else acc, dtype=np.float64
        ).reshape(-1)
        if len(acc_array) != len(uid_list):
            raise E7InvariantError("acc length mismatch")

    group_uids: list[str] = []
    informative: list[str] = []
    zero_std: list[str] = []
    group_scores: dict[str, tuple[float, ...]] = {}
    group_acc: dict[str, tuple[float, ...]] = {}
    for start in range(0, len(uid_list), group_size):
        members = uid_list[start : start + group_size]
        uid = members[0]
        if any(member != uid for member in members):
            raise E7InvariantError(f"group starting at row {start} is not a contiguous UID block")
        if uid in group_scores:
            raise E7InvariantError(f"UID {uid} appears in more than one group of the chunk")
        group_values = score_array[start : start + group_size]
        group_uids.append(uid)
        group_scores[uid] = tuple(float(value) for value in group_values)
        if acc_array is not None:
            group_acc[uid] = tuple(float(value) for value in acc_array[start : start + group_size])
        if _group_is_zero_std(group_values, tolerance):
            zero_std.append(uid)
        else:
            informative.append(uid)
    if len(informative) + len(zero_std) != len(group_uids):
        raise E7InvariantError("group classification is not a partition")
    return GroupSplit(
        chunk_index=int(chunk_index),
        group_uids=tuple(group_uids),
        informative_uids=tuple(informative),
        zero_std_uids=tuple(zero_std),
        group_scores=group_scores,
        group_acc=group_acc,
    )


@dataclass(frozen=True)
class Selection:
    """Which groups of the accumulated chunks form the actor batch (plan §3.1)."""

    selected_uids: tuple[str, ...]
    surplus_uids: tuple[str, ...]
    zero_std_uids: tuple[str, ...]
    generation_batches_used: int
    total_groups: int

    @property
    def informative_groups(self) -> int:
        return len(self.selected_uids) + len(self.surplus_uids)

    @property
    def zero_std_groups(self) -> int:
        return len(self.zero_std_uids)

    @property
    def surplus_groups(self) -> int:
        return len(self.surplus_uids)

    @property
    def selected_groups(self) -> int:
        return len(self.selected_uids)

    def underfilled(self, target_groups: int = TARGET_GROUPS) -> bool:
        return len(self.selected_uids) < target_groups


def select_informative_groups(splits: list[GroupSplit], *, target_groups: int = TARGET_GROUPS) -> Selection:
    """Take the first ``target_groups`` informative groups in sampling order across chunks.

    Conservation: ``selected + surplus + zero_std == sum(chunk groups)`` and every UID is unique
    across chunks.  Surplus is evidence only; the caller never trains on it or carries it to
    the next actor snapshot.
    """
    if not splits:
        raise E7InvariantError("select_informative_groups requires at least one chunk")
    expected_index = 0
    seen: set[str] = set()
    informative: list[str] = []
    zero_std: list[str] = []
    total = 0
    for split in splits:
        if split.chunk_index != expected_index:
            raise E7InvariantError(f"chunks are not consecutive: {split.chunk_index} != {expected_index}")
        expected_index += 1
        for uid in split.group_uids:
            if uid in seen:
                raise E7InvariantError(f"UID {uid} is not unique across chunks")
            seen.add(uid)
        total += split.n_groups
        informative.extend(split.informative_uids)
        zero_std.extend(split.zero_std_uids)
    selected = tuple(informative[:target_groups])
    surplus = tuple(informative[target_groups:])
    selection = Selection(
        selected_uids=selected,
        surplus_uids=surplus,
        zero_std_uids=tuple(zero_std),
        generation_batches_used=len(splits),
        total_groups=total,
    )
    if selection.selected_groups + selection.surplus_groups + selection.zero_std_groups != total:
        raise E7InvariantError("selection does not conserve the candidate groups")
    return selection


# --------------------------------------------------------------------------- #
# Ledger (plan §3.3 / §3.4) — Fixture Q / S
# --------------------------------------------------------------------------- #


@dataclass
class LedgerRow:
    effective_step: int
    generation_batches_used: int
    prompts_sampled: int
    trajectories_sampled: int
    rollout_tokens: int
    informative_groups: int
    zero_std_groups: int
    surplus_groups: int
    selected_groups: int
    all_correct_groups: int
    all_wrong_groups: int
    mixed_groups: int
    generation_wall_time: float
    filter_wall_time: float
    dataloader_batches_consumed: int
    epoch: int
    cumulative_prompts: int
    cumulative_trajectories: int
    cumulative_rollout_tokens: int
    cumulative_generation_wall_time: float
    equal_compute_checkpoint: bool = False
    underfilled: bool = False
    recorded_at: str = ""

    def metrics(self) -> dict[str, float]:
        """Flat ``e7/*`` training metrics (whitelist W6)."""
        total_groups = self.informative_groups + self.zero_std_groups
        acc_groups = self.all_correct_groups + self.all_wrong_groups + self.mixed_groups
        return {
            "e7/generation_batches_used": float(self.generation_batches_used),
            "e7/informative_groups": float(self.informative_groups),
            "e7/zero_std_groups": float(self.zero_std_groups),
            "e7/surplus_groups": float(self.surplus_groups),
            "e7/selected_groups": float(self.selected_groups),
            "e7/informative_frac": self.informative_groups / total_groups if total_groups else 0.0,
            "e7/prompts_sampled": float(self.prompts_sampled),
            "e7/trajectories_sampled": float(self.trajectories_sampled),
            "e7/rollout_tokens": float(self.rollout_tokens),
            "e7/cumulative_prompts": float(self.cumulative_prompts),
            "e7/cumulative_trajectories": float(self.cumulative_trajectories),
            "e7/cumulative_rollout_tokens": float(self.cumulative_rollout_tokens),
            "e7/all_correct_frac": self.all_correct_groups / acc_groups if acc_groups else 0.0,
            "e7/all_wrong_frac": self.all_wrong_groups / acc_groups if acc_groups else 0.0,
            "e7/mixed_frac": self.mixed_groups / acc_groups if acc_groups else 0.0,
            "e7/generation_wall_time": float(self.generation_wall_time),
            "e7/filter_wall_time": float(self.filter_wall_time),
            "e7/dataloader_batches_consumed": float(self.dataloader_batches_consumed),
            "e7/equal_compute_checkpoint": 1.0 if self.equal_compute_checkpoint else 0.0,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class DynamicSamplingLedger:
    """Cumulative dynamic-sampling state; serialised with every checkpoint (plan §3.3)."""

    schema_version: str = LEDGER_SCHEMA_VERSION
    effective_step: int = 0
    dataloader_batches_consumed: int = 0
    epoch: int = 0
    cumulative_prompts: int = 0
    cumulative_trajectories: int = 0
    cumulative_rollout_tokens: int = 0
    cumulative_generation_wall_time: float = 0.0
    equal_compute_saved: bool = False
    equal_compute_step: int | None = None
    equal_compute_threshold: int = E3_TOTAL_TRAJECTORIES
    rows: list[LedgerRow] = field(default_factory=list)

    def record_step(
        self,
        *,
        effective_step: int,
        selection: Selection,
        splits: list[GroupSplit],
        prompts_per_chunk: int,
        group_size: int,
        rollout_tokens: int,
        generation_wall_time: float,
        filter_wall_time: float,
    ) -> LedgerRow:
        if effective_step != self.effective_step + 1:
            raise E7InvariantError(
                f"ledger step discontinuity: recording {effective_step} after {self.effective_step}"
            )
        batches = selection.generation_batches_used
        if batches != len(splits):
            raise E7InvariantError("generation_batches_used disagrees with the chunk count")
        prompts = prompts_per_chunk * batches
        trajectories = prompts * group_size
        patterns = {"all_correct": 0, "all_wrong": 0, "mixed": 0}
        for split in splits:
            for uid in split.group_uids:
                pattern = split.acc_pattern(uid)
                if pattern in patterns:
                    patterns[pattern] += 1
        self.effective_step = effective_step
        self.cumulative_prompts += prompts
        self.cumulative_trajectories += trajectories
        self.cumulative_rollout_tokens += int(rollout_tokens)
        self.cumulative_generation_wall_time += float(generation_wall_time)
        row = LedgerRow(
            effective_step=effective_step,
            generation_batches_used=batches,
            prompts_sampled=prompts,
            trajectories_sampled=trajectories,
            rollout_tokens=int(rollout_tokens),
            informative_groups=selection.informative_groups,
            zero_std_groups=selection.zero_std_groups,
            surplus_groups=selection.surplus_groups,
            selected_groups=selection.selected_groups,
            all_correct_groups=patterns["all_correct"],
            all_wrong_groups=patterns["all_wrong"],
            mixed_groups=patterns["mixed"],
            generation_wall_time=float(generation_wall_time),
            filter_wall_time=float(filter_wall_time),
            dataloader_batches_consumed=self.dataloader_batches_consumed,
            epoch=self.epoch,
            cumulative_prompts=self.cumulative_prompts,
            cumulative_trajectories=self.cumulative_trajectories,
            cumulative_rollout_tokens=self.cumulative_rollout_tokens,
            cumulative_generation_wall_time=self.cumulative_generation_wall_time,
            underfilled=selection.underfilled(),
            recorded_at=_utc_now(),
        )
        self.rows.append(row)
        return row

    def equal_compute_due(self) -> bool:
        """Plan §3.4: first time cumulative trajectories reach the E3 total, exactly once."""
        return self.cumulative_trajectories >= self.equal_compute_threshold and not self.equal_compute_saved

    def mark_equal_compute_saved(self, step: int) -> None:
        if self.equal_compute_saved:
            raise E7InvariantError("equal-compute checkpoint already saved")
        if not self.rows or self.rows[-1].effective_step != step:
            raise E7InvariantError("equal-compute checkpoint must be marked on the just-recorded step")
        self.equal_compute_saved = True
        self.equal_compute_step = int(step)
        self.rows[-1].equal_compute_checkpoint = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [asdict(row) for row in self.rows]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DynamicSamplingLedger":
        if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
            raise E7InvariantError(f"unexpected ledger schema: {payload.get('schema_version')!r}")
        rows = [LedgerRow(**row) for row in payload.get("rows", [])]
        ledger = cls(
            effective_step=int(payload["effective_step"]),
            dataloader_batches_consumed=int(payload["dataloader_batches_consumed"]),
            epoch=int(payload["epoch"]),
            cumulative_prompts=int(payload["cumulative_prompts"]),
            cumulative_trajectories=int(payload["cumulative_trajectories"]),
            cumulative_rollout_tokens=int(payload["cumulative_rollout_tokens"]),
            cumulative_generation_wall_time=float(payload["cumulative_generation_wall_time"]),
            equal_compute_saved=bool(payload["equal_compute_saved"]),
            equal_compute_step=payload.get("equal_compute_step"),
            equal_compute_threshold=int(payload.get("equal_compute_threshold", E3_TOTAL_TRAJECTORIES)),
            rows=rows,
        )
        ledger.validate()
        return ledger

    def validate(self) -> None:
        steps = [row.effective_step for row in self.rows]
        if steps != list(range(1, len(steps) + 1)):
            raise E7InvariantError(f"ledger rows are not contiguous from 1: {steps[:5]}...")
        if self.rows and self.rows[-1].effective_step != self.effective_step:
            raise E7InvariantError("ledger effective_step disagrees with its last row")
        if not self.rows and self.effective_step != 0:
            raise E7InvariantError("ledger without rows must be at step 0")
        if self.rows:
            last = self.rows[-1]
            if (
                last.cumulative_trajectories != self.cumulative_trajectories
                or last.cumulative_prompts != self.cumulative_prompts
                or last.dataloader_batches_consumed != self.dataloader_batches_consumed
            ):
                raise E7InvariantError("ledger cumulative totals disagree with the last row")
        flagged = [row.effective_step for row in self.rows if row.equal_compute_checkpoint]
        if len(flagged) > 1 or (flagged and not self.equal_compute_saved) or (
            self.equal_compute_saved and flagged != [self.equal_compute_step]
        ):
            raise E7InvariantError(f"equal-compute flags inconsistent: {flagged} vs {self.equal_compute_step}")

    def save(self, path: Path) -> None:
        self.validate()
        _atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> "DynamicSamplingLedger":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def rewrite_jsonl(self, path: Path) -> None:
        text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in self.rows)
        _atomic_write_text(path, text)


def append_ledger_row(path: Path, row: LedgerRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def next_checkpoint_step(global_step: int, save_freq: int) -> int:
    """The smallest multiple of ``save_freq`` that is >= ``global_step``."""
    if save_freq <= 0:
        raise E7InvariantError("stop requests need a positive save_freq")
    return int(math.ceil(global_step / save_freq) * save_freq)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_source_hashes() -> dict[str, str]:
    """Re-verify the two pinned veRL files whose line numbers this module copies."""
    import verl

    root = Path(verl.__file__).resolve().parent
    actual = {
        "verl/trainer/ppo/ray_trainer.py": _sha256_file(root / "trainer/ppo/ray_trainer.py"),
        "verl/trainer/main_ppo.py": _sha256_file(root / "trainer/main_ppo.py"),
    }
    expected = {
        "verl/trainer/ppo/ray_trainer.py": PINNED_RAY_TRAINER_SHA256,
        "verl/trainer/main_ppo.py": PINNED_MAIN_PPO_SHA256,
    }
    for name, digest in actual.items():
        if digest != expected[name]:
            raise RuntimeError(f"pinned veRL source drift for {name}: {digest} != {expected[name]}")
    return actual


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


class DynamicFilterRayPPOTrainer(RayPPOTrainer):
    """E7 trainer: pinned ``fit()`` plus bounded same-snapshot resampling (W1–W7)."""

    # Set on the class by the launcher inside the driver actor process (plan §3.5 segments).
    E7_STOP_AT_STEP: int | None = None
    _e7_epoch: int = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._e7_validate_config()
        self._e7_ledger = DynamicSamplingLedger()
        self._e7_iter: Any = None
        self._e7_epoch = 0
        self._e7_generation_phase = False
        self._e7_candidate_rows_cache: list[dict[str, Any]] = []
        self._e7_write_activation_proof()

    # ---- configuration & proof ------------------------------------------------ #

    def _e7_filter_groups(self) -> dict[str, Any]:
        """``algorithm.filter_groups`` as a plain dict (DictConfig before, dict after dataclass conversion)."""
        raw = self.config.algorithm.get("filter_groups", None)
        if raw is None:
            return {}
        if hasattr(raw, "items"):
            items = dict(raw.items())
        else:
            items = {key: getattr(raw, key) for key in ("enable", "metric", "max_num_gen_batches")}
        return {
            "enable": bool(items.get("enable", False)),
            "metric": None if items.get("metric") is None else str(items.get("metric")),
            "max_num_gen_batches": int(items.get("max_num_gen_batches", 0)),
        }

    def _e7_validate_config(self) -> None:
        cfg = self.config
        filter_groups = self._e7_filter_groups()
        if not filter_groups.get("enable", False):
            raise E7InvariantError("E7 trainer requires algorithm.filter_groups.enable=true")
        if filter_groups.get("metric") != FILTER_METRIC:
            raise E7InvariantError(f"E7 filters on {FILTER_METRIC!r}, got {filter_groups.get('metric')!r}")
        if filter_groups.get("max_num_gen_batches", 0) != MAX_GENERATION_BATCHES:
            raise E7InvariantError("E7 allows exactly 4 generation batches per effective step")
        if cfg.algorithm.adv_estimator != AdvantageEstimator.GRPO:
            raise E7InvariantError("E7 requires the GRPO advantage estimator")
        if int(cfg.actor_rollout_ref.rollout.n) != GROUP_SIZE:
            raise E7InvariantError("E7 rollout.n must be 8")
        if int(cfg.data.train_batch_size) != TARGET_GROUPS:
            raise E7InvariantError("E7 train_batch_size must be 64")
        if cfg.data.get("gen_batch_size", None) not in (None, TARGET_GROUPS):
            raise E7InvariantError("E7 does not support a separate gen_batch_size")
        marker = cfg.trainer.get("toolcredit_trainer_class", None)
        if marker != TRAINER_CLASS_FQN:
            raise E7InvariantError(f"trainer.toolcredit_trainer_class must be {TRAINER_CLASS_FQN}, got {marker!r}")
        if self.use_rm:
            raise E7InvariantError("E7 expects rm_scores from the agent loop, not a colocated reward model")
        if int(cfg.trainer.save_freq) <= 0 and self.E7_STOP_AT_STEP is not None:
            raise E7InvariantError("--stop-at-step needs save_freq > 0 to leave a resumable checkpoint")

    def _e7_run_dir(self) -> Path:
        return Path(self.config.trainer.default_local_dir).parent

    def _e7_write_activation_proof(self) -> None:
        cfg = self.config
        payload = {
            "boundary_version": "e7_dynamic_filter_trainer_v1",
            "trainer_class": TRAINER_CLASS_FQN,
            "actual_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "pinned_sha256": pinned_source_hashes(),
            "fit_source_lines": list(FIT_SOURCE_LINES),
            "filter_groups": self._e7_filter_groups(),
            "prompts_per_chunk": int(cfg.data.train_batch_size),
            "rollout_n": int(cfg.actor_rollout_ref.rollout.n),
            "target_groups": TARGET_GROUPS,
            "equal_compute_threshold": E3_TOTAL_TRAJECTORIES,
            "offload": {
                "actor_param_offload": bool(cfg.actor_rollout_ref.actor.fsdp_config.param_offload),
                "actor_optimizer_offload": bool(cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload),
                "ref_param_offload": bool(cfg.actor_rollout_ref.ref.fsdp_config.param_offload),
            },
            "stop_at_step": self.E7_STOP_AT_STEP,
            "pid": os.getpid(),
            "activated_at": _utc_now(),
        }
        run_dir = self._e7_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(run_dir / TRAINER_PROOF_FILENAME, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        with (run_dir / TRAINER_ACTIVATIONS_JSONL).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        print(LOG_MARKER + json.dumps(payload, sort_keys=True))

    # ---- segment stop (W3a) ----------------------------------------------------- #

    def _e7_stop_step(self) -> int:
        """Planned segment end: ``--stop-at-step`` and/or a watcher stop request (plan §3.5, §4)."""
        stop = int(self.total_training_steps)
        if self.E7_STOP_AT_STEP is not None:
            stop = min(stop, int(self.E7_STOP_AT_STEP))
        request = self._e7_run_dir() / STOP_REQUEST_FILENAME
        if request.exists():
            stop = min(stop, next_checkpoint_step(int(self.global_steps), int(self.config.trainer.save_freq)))
        return stop

    # ---- dataloader iteration (W2) ---------------------------------------------- #

    def _e7_next_batch_dict(self) -> dict[str, Any]:
        """Explicit ``next()`` on the StatefulDataLoader; epoch rollover on exhaustion."""
        if self._e7_iter is None:
            raise E7InvariantError("dataloader iterator was not initialised by fit()")
        steps_per_epoch = len(self.train_dataloader)
        for _attempt in range(2):
            if self._e7_epoch >= int(self.config.trainer.total_epochs):
                raise E7InvariantError("train dataloader exhausted before total_training_steps")
            try:
                batch_dict = next(self._e7_iter)
            except StopIteration:
                # ``drop_last=True``: exhaustion happens exactly at a multiple of len(dataloader),
                # so the epoch counter below has already advanced; only rebuild the iterator.
                self._e7_iter = iter(self.train_dataloader)
                continue
            self._e7_ledger.dataloader_batches_consumed += 1
            # W1 definition: epoch == completed passes == batches // len(dataloader).
            self._e7_epoch = self._e7_ledger.dataloader_batches_consumed // steps_per_epoch
            self._e7_ledger.epoch = self._e7_epoch
            return batch_dict
        raise E7InvariantError("train dataloader yielded no batch after an epoch rollover")

    # ---- guard: no actor update while candidates are being generated ------------- #

    def _update_actor(self, batch: DataProto) -> DataProto:
        if self._e7_generation_phase:
            raise E7InvariantError("actor update attempted while a candidate batch was being generated")
        return super()._update_actor(batch)

    # ---- W3: bounded resampling loop -------------------------------------------- #

    def _e7_split_chunk(
        self,
        batch: DataProto,
        reward_tensor: torch.Tensor,
        reward_extra_infos_dict: dict[str, Any],
        chunk_index: int,
    ) -> GroupSplit:
        expected_rows = int(self.config.data.train_batch_size) * int(self.config.actor_rollout_ref.rollout.n)
        if len(batch) != expected_rows:
            raise E7InvariantError(f"chunk {chunk_index} has {len(batch)} rows, expected {expected_rows}")
        scalar = reward_tensor.sum(dim=-1).detach().cpu().to(torch.float64)
        if E7_DEBUG:
            print(f"[E7 debug] chunk {chunk_index}: rm_scores {tuple(reward_tensor.shape)} -> score {tuple(scalar.shape)}")
        if scalar.shape != (expected_rows,):
            raise E7InvariantError(f"scalar score shape {tuple(scalar.shape)} != ({expected_rows},)")
        if FILTER_METRIC not in reward_extra_infos_dict:
            raise E7InvariantError(f"reward extra info lacks {FILTER_METRIC!r}; cannot filter by score")
        reported = np.asarray(reward_extra_infos_dict[FILTER_METRIC], dtype=np.float64).reshape(-1)
        if len(reported) != expected_rows or not np.all(np.isfinite(reported)):
            raise E7InvariantError("reported score column is malformed")
        if not np.allclose(reported, scalar.numpy(), atol=SCORE_IDENTITY_TOLERANCE, rtol=0.0):
            raise E7InvariantError("score != rm_scores.sum(-1); the filter metric would not be the GRPO input")
        acc = reward_extra_infos_dict.get("acc", None)
        return split_groups_by_score_variance(
            batch.non_tensor_batch["uid"],
            scalar.numpy(),
            chunk_index=chunk_index,
            group_size=int(self.config.actor_rollout_ref.rollout.n),
            acc=None if acc is None else np.asarray(acc, dtype=np.float64),
        )

    def _e7_candidate_rows(self, chunks: list[DataProto], splits: list[GroupSplit], selection: Selection) -> list[dict[str, Any]]:
        role_of: dict[str, str] = {}
        role_of.update({uid: ROLE_SELECTED for uid in selection.selected_uids})
        role_of.update({uid: ROLE_SURPLUS for uid in selection.surplus_uids})
        role_of.update({uid: ROLE_ZERO_STD for uid in selection.zero_std_uids})
        rows: list[dict[str, Any]] = []
        for chunk, split in zip(chunks, splits, strict=True):
            uids = [str(value) for value in chunk.non_tensor_batch["uid"].tolist()]
            scores = chunk.batch["rm_scores"].sum(dim=-1).detach().cpu().tolist()
            lengths = chunk.batch["response_mask"].sum(dim=-1).detach().cpu().tolist()
            accs = chunk.non_tensor_batch.get("acc", None)
            sample_ids = chunk.non_tensor_batch.get("sample_id", None)
            for index, uid in enumerate(uids):
                rows.append(
                    {
                        "step": int(self.global_steps),
                        "chunk": split.chunk_index,
                        "uid": uid,
                        "role": role_of[uid],
                        "sample_id": None if sample_ids is None else str(sample_ids[index]),
                        "score": float(scores[index]),
                        "acc": None if accs is None else float(accs[index]),
                        "response_length": int(lengths[index]),
                        "acc_pattern": split.acc_pattern(uid),
                    }
                )
        return rows

    def _e7_candidates_dir(self, rollout_data_dir: str | None = None) -> Path:
        base = Path(rollout_data_dir) if rollout_data_dir else self._e7_run_dir() / "predictions/train"
        return base.parent / CANDIDATES_DIRNAME

    def _e7_dump_rows(self, path: Path, rows: list[dict[str, Any]]) -> None:
        _atomic_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))

    def _e7_collect_informative_batch(
        self, metrics: dict[str, Any], timing_raw: dict[str, float], curr_step_profile: bool
    ) -> tuple[DataProto, torch.Tensor, dict[str, Any], LedgerRow]:
        """Generate up to four 64x8 chunks from one frozen actor snapshot and select 64 informative groups.

        Per chunk the upstream lines ``:1435–1449, 1460–1462, 1466–1470, 1472–1480, 1495–1501,
        1518–1525`` run verbatim (REMAX branch dropped; ``sleep_replicas()`` from ``:1471`` moved
        to ``finally``).  After selection ``:1502–1517`` and ``:1518–1525`` run once on the
        selected batch.
        """
        assert self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO, "E7 is a GRPO-only ablation"
        max_batches = int(self._e7_filter_groups()["max_num_gen_batches"])
        prompts_per_chunk = int(self.config.data.train_batch_size)
        snapshot_step = int(self.global_steps)
        chunks: list[DataProto] = []
        splits: list[GroupSplit] = []
        selection: Selection | None = None
        generation_wall = 0.0
        filter_wall = 0.0
        rollout_tokens = 0
        self._e7_generation_phase = True
        try:
            for chunk_index in range(max_batches):
                if int(self.global_steps) != snapshot_step:
                    raise E7InvariantError("global_steps changed during candidate generation")
                batch_dict = self._e7_next_batch_dict()
                started = time.perf_counter()
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                rollout_n = self.config.actor_rollout_ref.rollout.n
                gen_batch_output = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                combined_gen_batch = gen_batch_output
                num_sampled_prompts = len(gen_batch_output)

                # generate a batch
                with marked_timer("gen", timing_raw, color="red"):
                    if curr_step_profile:
                        self.llm_server_manager.start_profile()
                    combined_gen_output = self.async_rollout_manager.generate_sequences(combined_gen_batch)
                    if curr_step_profile:
                        self.llm_server_manager.stop_profile()

                    timing_raw.update(combined_gen_output.meta_info["timing"])
                    combined_gen_output.meta_info.pop("timing", None)

                gen_batch_output = combined_gen_output.slice(0, num_sampled_prompts)
                if "__do_sample__" in gen_batch_output.non_tensor_batch:
                    gen_batch_output.pop(non_tensor_batch_keys=["__do_sample__"])

                del combined_gen_batch, combined_gen_output
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                generation_wall += time.perf_counter() - started
                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        batch_reward = self._compute_reward_colocate(batch)
                        batch = batch.union(batch_reward)

                    # extract reward_tensor and reward_extra_infos_dict for training
                    reward_tensor, reward_extra_infos_dict = extract_reward(batch)

                filter_started = time.perf_counter()
                if chunks and set(batch.batch.keys()) != set(chunks[0].batch.keys()):
                    raise E7InvariantError("chunk tensor keys drifted between generation batches")
                if chunks and batch.batch["responses"].shape[1:] != chunks[0].batch["responses"].shape[1:]:
                    raise E7InvariantError("chunk response padding drifted between generation batches")
                split = self._e7_split_chunk(batch, reward_tensor, reward_extra_infos_dict, chunk_index)
                chunks.append(batch)
                splits.append(split)
                rollout_tokens += int(batch.batch["response_mask"].sum().item())
                selection = select_informative_groups(splits, target_groups=TARGET_GROUPS)
                filter_wall += time.perf_counter() - filter_started
                print(
                    f"[E7] step {snapshot_step} chunk {chunk_index + 1}/{max_batches}: "
                    f"informative {selection.informative_groups}/{selection.total_groups} groups "
                    f"(zero-std {selection.zero_std_groups}), selected {selection.selected_groups}/{TARGET_GROUPS}"
                )
                if not selection.underfilled(TARGET_GROUPS):
                    break
            assert selection is not None
            if selection.underfilled(TARGET_GROUPS):
                rows = self._e7_candidate_rows(chunks, splits, selection)
                path = self._e7_candidates_dir(self.config.trainer.get("rollout_data_dir", None))
                self._e7_dump_rows(path / f"{snapshot_step}.underfilled.jsonl", rows)
                raise E7UnderfilledError(
                    f"step {snapshot_step}: only {selection.informative_groups} informative groups after "
                    f"{selection.generation_batches_used} generation batches; candidates saved to {path}"
                )
        finally:
            self.checkpoint_manager.sleep_replicas()
            self._e7_generation_phase = False

        if int(self.global_steps) != snapshot_step:
            raise E7InvariantError("global_steps changed during candidate generation")
        candidates = DataProto.concat(chunks) if len(chunks) > 1 else chunks[0]
        selected_set = set(selection.selected_uids)
        mask = np.array([str(uid) in selected_set for uid in candidates.non_tensor_batch["uid"].tolist()], dtype=bool)
        batch = candidates.select_idxs(mask)
        expected_rows = TARGET_GROUPS * int(self.config.actor_rollout_ref.rollout.n)
        if len(batch) != expected_rows:
            raise E7InvariantError(f"selected batch has {len(batch)} rows, expected {expected_rows}")
        selected_uids = [str(uid) for uid in batch.non_tensor_batch["uid"].tolist()]
        if len(set(selected_uids)) != TARGET_GROUPS or any(
            uid in selection.surplus_uids or uid in selection.zero_std_uids for uid in selected_uids
        ):
            raise E7InvariantError("selected batch contains surplus or zero-std groups")
        if E7_DEBUG:
            print(f"[E7 debug] selected batch responses {tuple(batch.batch['responses'].shape)}")
        self._e7_candidate_rows_cache = self._e7_candidate_rows(chunks, splits, selection)
        del candidates, chunks

        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        # get images_seqlens
        images_seqlens_all = []
        for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
            if "image_grid_thw" not in multi_modal_input.keys():
                continue
            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
        batch.meta_info["images_seqlens"] = images_seqlens_all
        with marked_timer("reward", timing_raw, color="yellow"):
            # compute reward model score
            if self.use_rm and "rm_scores" not in batch.batch.keys():
                batch_reward = self._compute_reward_colocate(batch)
                batch = batch.union(batch_reward)

            # extract reward_tensor and reward_extra_infos_dict for training
            reward_tensor, reward_extra_infos_dict = extract_reward(batch)

        step_row = self._e7_ledger.record_step(
            effective_step=snapshot_step,
            selection=selection,
            splits=splits,
            prompts_per_chunk=prompts_per_chunk,
            group_size=int(self.config.actor_rollout_ref.rollout.n),
            rollout_tokens=rollout_tokens,
            generation_wall_time=generation_wall,
            filter_wall_time=filter_wall,
        )
        return batch, reward_tensor, reward_extra_infos_dict, step_row

    # ---- W4: equal-compute checkpoint ------------------------------------------- #

    def _e7_equal_compute_dirs(self) -> list[Path]:
        root = Path(self.config.trainer.default_local_dir)
        if not root.exists():
            return []
        return sorted(path for path in root.glob("equal_compute_step_*") if path.is_dir() and not path.name.endswith(".saving"))

    def _e7_save_equal_compute_checkpoint(self, step_row: LedgerRow) -> None:
        """Plan §3.4: one extra checkpoint outside the ``max_actor_ckpt_to_keep`` rotation.

        The worker saves with ``max_ckpt_to_keep=None`` into a staging directory that is then
        renamed to ``equal_compute_step_<N>``; veRL's rotation later finds the staging path
        absent and skips it (``remove_previous_save_local_path`` tolerates missing paths), so
        this checkpoint is never deleted.  No tracker is written.
        """
        step = int(self.global_steps)
        if step != step_row.effective_step:
            raise E7InvariantError("equal-compute checkpoint step disagrees with the ledger row")
        existing = self._e7_equal_compute_dirs()
        if existing:
            raise E7InvariantError(f"equal-compute checkpoint already exists: {existing}")
        root = Path(self.config.trainer.default_local_dir)
        staging = root / f"equal_compute_step_{step}.saving"
        final = root / f"equal_compute_step_{step}"
        if staging.exists() or final.exists():
            raise E7InvariantError(f"equal-compute checkpoint path already exists: {final}")
        staging.mkdir(parents=True, exist_ok=False)
        print(f"[E7] cumulative trajectories {step_row.cumulative_trajectories} >= {E3_TOTAL_TRAJECTORIES}: saving {final}")
        self.actor_rollout_wg.save_checkpoint(str(staging / "actor"), None, step, max_ckpt_to_keep=None)
        torch.save(self.train_dataloader.state_dict(), staging / "data.pt")
        self._e7_ledger.mark_equal_compute_saved(step)
        self._e7_ledger.save(staging / LEDGER_FILENAME)
        os.replace(staging, final)
        regular = root / f"global_step_{step}" / LEDGER_FILENAME
        if regular.exists():
            # A regular checkpoint written earlier in this step predates the flag; keep it consistent.
            self._e7_ledger.save(regular)

    # ---- W5: candidates evidence ------------------------------------------------ #

    def _e7_log_candidates(self, step_row: LedgerRow, rollout_data_dir: str) -> None:
        rows = getattr(self, "_e7_candidate_rows_cache", None)
        if not rows:
            raise E7InvariantError("no candidate rows cached for this effective step")
        step = int(self.global_steps)
        if any(row["step"] != step for row in rows):
            raise E7InvariantError("candidate rows belong to a different step")
        self._e7_dump_rows(self._e7_candidates_dir(rollout_data_dir) / f"{step}.jsonl", rows)
        self._e7_candidate_rows_cache = []
        append_ledger_row(self._e7_run_dir() / LEDGER_JSONL, step_row)

    # ---- checkpoint / resume (STEP0 §5) ---------------------------------------- #

    def _save_checkpoint(self) -> None:
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        if self._e7_ledger.effective_step != int(self.global_steps):
            raise E7InvariantError(
                f"ledger step {self._e7_ledger.effective_step} != global_steps {self.global_steps} at checkpoint"
            )
        self._e7_ledger.save(folder / LEDGER_FILENAME)
        super()._save_checkpoint()

    def _load_checkpoint(self) -> Any:
        result = super()._load_checkpoint()
        if int(self.global_steps) == 0:
            self._e7_ledger = DynamicSamplingLedger()
            self._e7_epoch = 0
            return result
        steps_per_epoch = len(self.train_dataloader)
        if steps_per_epoch > 0 and int(self.global_steps) % steps_per_epoch == 0:
            # Upstream skips the dataloader restore here (ray_trainer.py:1095–1103), which
            # would silently reset E7's prompt order.  Refuse instead (STEP0 §5).
            raise E7InvariantError(
                f"global_steps={self.global_steps} hits veRL's epoch-boundary heuristic "
                f"(steps_per_epoch={steps_per_epoch}); resume from a different checkpoint"
            )
        folder = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        ledger_path = folder / LEDGER_FILENAME
        if not ledger_path.is_file():
            raise E7InvariantError(f"checkpoint {folder} has no {LEDGER_FILENAME}; refusing to resume")
        ledger = DynamicSamplingLedger.load(ledger_path)
        if ledger.effective_step != int(self.global_steps):
            raise E7InvariantError(
                f"ledger effective_step {ledger.effective_step} != checkpoint step {self.global_steps}"
            )
        computed_epoch = ledger.dataloader_batches_consumed // steps_per_epoch if steps_per_epoch else 0
        if computed_epoch != ledger.epoch:
            raise E7InvariantError(f"ledger epoch {ledger.epoch} != batches // steps_per_epoch {computed_epoch}")
        equal_dirs = self._e7_equal_compute_dirs()
        if equal_dirs:
            saved_step = int(equal_dirs[-1].name.removeprefix("equal_compute_step_"))
            if len(equal_dirs) != 1 or saved_step > int(self.global_steps):
                raise E7InvariantError(f"unexpected equal-compute checkpoints on disk: {equal_dirs}")
            if not ledger.equal_compute_saved:
                # The regular checkpoint of that step was written before the flag; trust the disk.
                ledger.equal_compute_saved = True
                ledger.equal_compute_step = saved_step
                for row in ledger.rows:
                    if row.effective_step == saved_step:
                        row.equal_compute_checkpoint = True
                ledger.validate()
            elif ledger.equal_compute_step != saved_step:
                raise E7InvariantError("ledger equal-compute step disagrees with the directory on disk")
        elif ledger.equal_compute_saved:
            raise E7InvariantError("ledger claims an equal-compute checkpoint that is missing on disk")
        self._e7_ledger = ledger
        self._e7_epoch = ledger.epoch
        ledger.rewrite_jsonl(self._e7_run_dir() / LEDGER_JSONL)
        print(
            f"[E7] resumed ledger at effective step {ledger.effective_step}: "
            f"{ledger.dataloader_batches_consumed} dataloader batches, epoch {ledger.epoch}, "
            f"{ledger.cumulative_trajectories} cumulative trajectories"
        )
        return result

    # ---- fit(): pinned copy of ray_trainer.py:1362–1770 ------------------------- #

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if self._dump_executor._shutdown:
            self._init_dump_executor()

        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self._e7_ledger.dataloader_batches_consumed // len(self.train_dataloader)  # E7-W1

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        if self.config.actor_rollout_ref.rollout.skip.get("enable", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        self._e7_epoch = current_epoch  # E7-W2: explicit iterator; one effective step may consume several batches
        self._e7_iter = iter(self.train_dataloader)  # E7-W2
        while self._e7_epoch < self.config.trainer.total_epochs:  # E7-W2
            while True:  # E7-W2: one iteration per effective step; chunks are pulled in _e7_collect_informative_batch
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                is_last_step = self.global_steps >= min(self.total_training_steps, self._e7_stop_step())  # E7-W3a
                with marked_timer("step", timing_raw):
                    batch, reward_tensor, reward_extra_infos_dict, step_row = self._e7_collect_informative_batch(  # E7-W3
                        metrics, timing_raw, curr_step_profile  # E7-W3
                    )  # E7-W3

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                raise ValueError(
                                    "Detected conflicting router replay configuration: "
                                    "router_replay.mode='R2' and enable_rollout_routing_replay=True "
                                    "cannot be enabled simultaneously. "
                                    "The enable_rollout_routing_replay option is only used in R3 mode; "
                                    "it should not be set when using R2 mode."
                                )
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup > self.global_steps:
                        # Still in critic warmup, only update weights to wake up rollout replicas.
                        self.checkpoint_manager.update_weights(self.global_steps)
                    else:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()
                        if self._e7_ledger.equal_compute_due():  # E7-W4: plan §3.4 equal-compute checkpoint
                            with marked_timer("save_checkpoint", timing_raw, color="green"):  # E7-W4
                                self._e7_save_equal_compute_checkpoint(step_row)  # E7-W4

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights(self.global_steps)

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                        self._e7_log_candidates(step_row, rollout_data_dir)  # E7-W5

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": self._e7_epoch,  # E7-W2
                    }
                )
                metrics.update(step_row.metrics())  # E7-W6
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                # GDPO per-component reward metrics
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for key in gdpo_reward_keys:
                        if key in batch.non_tensor_batch:
                            vals = np.asarray(batch.non_tensor_batch[key], dtype=np.float32)
                            metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                            metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                            metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                            metrics[f"gdpo/{key}/min"] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # Per-request spec decode metrics.
                metrics.update(
                    compute_spec_decode_metrics(
                        batch.non_tensor_batch.get("spec_num_draft_tokens", None),
                        batch.non_tensor_batch.get("spec_num_accepted_tokens", None),
                        batch.non_tensor_batch.get("spec_num_verify_steps", None),
                    )
                )

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dump_executor()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

        # Ensure dump executor is shut down when training loop ends without reaching is_last_step
        self._shutdown_dump_executor()


def fit_source_lines() -> list[str]:
    """The E7 ``fit()`` source as written in this file (used by Fixture R)."""
    source = inspect.getsource(DynamicFilterRayPPOTrainer.fit)
    return source.splitlines()


def upstream_fit_source_lines() -> list[str]:
    """Lines ``FIT_SOURCE_LINES`` of the pinned ``ray_trainer.py`` (verified by hash)."""
    import verl

    pinned_source_hashes()
    path = Path(verl.__file__).resolve().parent / "trainer/ppo/ray_trainer.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    start, end = FIT_SOURCE_LINES
    return lines[start - 1 : end]
