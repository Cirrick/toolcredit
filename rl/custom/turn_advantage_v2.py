"""M8 / E5-v2: conserved within-trajectory turn-level credit (plans/M8.md §3).

The frozen E5 v1 module ``rl/custom/turn_advantage.py`` is hashed by the M6/M7 freeze
closures and is therefore not modified.  This module imports its validated helpers and
adds only the v2 semantics: the token-weighted within-trajectory baseline ``s_tilde``,
the ``|T| >= 2`` and informative-group gates, the §3.2 ``s_t`` conditions and the
per-trajectory conservation assertion ``sum_t c_t * n_t == 0``.
"""

from __future__ import annotations

import hashlib
import math
from functools import wraps
from typing import Any, Callable, Sequence

import numpy as np
import torch

from rl.custom.turn_advantage import (
    AUDIT_KEY,
    LEDGER_KEY,
    WRAPPER_PROOF_KEY,
    _native_scalar,
    _python_scalar,
    _validate_turn_spans,
)

ADOPTION_VERSION_V2 = "toolcredit_adoption_v2"
LEDGER_SCHEMA_VERSION_V2 = "toolcredit_turn_ledger_v2"
VARIANT_CONSERVED_V2 = "conserved_v2"
FROZEN_BETA = 0.5
CONSERVATION_TOLERANCE = 1e-6
BOXED_MARKER = "\\boxed{"


def conserved_turn_corrections(
    s_values: Sequence[int],
    token_counts: Sequence[int],
    eligible: Sequence[bool],
    *,
    beta: float,
    informative_group: bool,
) -> tuple[list[float], float | None, bool]:
    """Return per-turn corrections ``c_t`` under the frozen E5-v2 formula (plan §3.1).

    ``c_t = beta * (s_t - s_tilde)`` on eligible (single-call) turns when at least two
    eligible turns exist and the GRPO group is informative; ``0`` everywhere else.
    ``s_tilde`` is the token-weighted mean of ``s_t`` over eligible turns, so
    ``sum_t c_t * n_t == 0`` for every trajectory.  Returns ``(c, s_tilde, gated)``.
    """
    if not len(s_values) == len(token_counts) == len(eligible):
        raise ValueError("s_values/token_counts/eligible lengths differ")
    for s_t in s_values:
        if int(s_t) not in {0, 1}:
            raise ValueError(f"s_t must be binary, got {s_t}")
    for n_t in token_counts:
        if int(n_t) <= 0:
            raise ValueError(f"turn token count must be positive, got {n_t}")
    indices = [index for index, flag in enumerate(eligible) if bool(flag)]
    if len(indices) < 2 or not informative_group:
        return [0.0] * len(s_values), None, True
    total = float(sum(int(token_counts[index]) for index in indices))
    s_tilde = float(sum(int(s_values[index]) * int(token_counts[index]) for index in indices)) / total
    corrections = [
        beta * (float(int(s_values[index])) - s_tilde) if bool(eligible[index]) else 0.0
        for index in range(len(s_values))
    ]
    residual = sum(corrections[index] * int(token_counts[index]) for index in range(len(s_values)))
    if abs(residual) > CONSERVATION_TOLERANCE:
        raise ValueError(f"E5-v2 correction is not conserved: residual={residual}")
    return corrections, s_tilde, False


def normalized_code_hash(code: str) -> str:
    """AST-normalized code identity shared with the E4 and budget diagnostics."""
    from rl.analyze_e4 import code_hash

    return code_hash(code)


def v2_turn_conditions(
    codes_by_turn: Sequence[Sequence[str]],
    assistant_texts: Sequence[str],
    s_t_v1: Sequence[int],
    eligible: Sequence[bool],
) -> list[dict[str, Any]]:
    """Apply plan §3.2 conditions 3 (non-repeat) and 4 (before boxed) on top of v1 ``s_t``.

    Returns one dict per turn with ``s_t_v1``, ``repeat_of_turn``, ``after_boxed``,
    ``credit_eligible`` and the final v2 ``s_t``.  ``repeat_of_turn`` is the earliest
    earlier turn whose AST-normalized code hash equals this turn's single code; a
    turn that carries several codes still registers all of them as seen.
    """
    if not len(codes_by_turn) == len(assistant_texts) == len(s_t_v1) == len(eligible):
        raise ValueError("per-turn sequences have different lengths")
    seen_hashes: dict[str, int] = {}
    boxed_seen = False
    results: list[dict[str, Any]] = []
    for index, codes in enumerate(codes_by_turn):
        hashes = [normalized_code_hash(str(code)) for code in codes]
        repeat_of_turn: int | None = None
        if bool(eligible[index]) and len(hashes) == 1 and hashes[0] in seen_hashes:
            repeat_of_turn = seen_hashes[hashes[0]]
        after_boxed = boxed_seen
        s_v1 = int(s_t_v1[index])
        if s_v1 not in {0, 1}:
            raise ValueError(f"s_t_v1 must be binary, got {s_v1}")
        s_t = int(bool(eligible[index]) and s_v1 == 1 and repeat_of_turn is None and not after_boxed)
        results.append(
            {
                "s_t_v1": s_v1,
                "repeat_of_turn": repeat_of_turn,
                "after_boxed": after_boxed,
                "credit_eligible": bool(eligible[index]),
                "s_t": s_t,
            }
        )
        for digest in hashes:
            seen_hashes.setdefault(digest, index)
        if BOXED_MARKER in str(assistant_texts[index]):
            boxed_seen = True
    return results


def _as_ledger_v2(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise TypeError(f"{LEDGER_KEY} must contain dictionaries, got {type(value).__name__}")
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION_V2:
        raise ValueError(f"unexpected turn ledger schema: {value.get('schema_version')!r}")
    if value.get("adoption_version") != ADOPTION_VERSION_V2:
        raise ValueError(f"E5-v2 requires {ADOPTION_VERSION_V2} ledgers")
    if not isinstance(value.get("assistant_turns"), list):
        raise TypeError("turn ledger assistant_turns must be a list")
    return value


def _group_informative_flags(
    token_scores: torch.Tensor, group_members: dict[str, list[int]]
) -> tuple[dict[str, bool], dict[str, float], list[float]]:
    """Gate on the exact scalar that enters native GRPO: the summed token-level score."""
    scores = token_scores.sum(dim=-1).detach().cpu().to(torch.float64)
    per_row = [float(value) for value in scores]
    informative: dict[str, bool] = {}
    group_std: dict[str, float] = {}
    for uid, rows in group_members.items():
        values = torch.tensor([per_row[row] for row in rows], dtype=torch.float64)
        group_std[uid] = float(values.std(unbiased=False)) if len(rows) > 1 else 0.0
        informative[uid] = bool(values.max() > values.min())
    return informative, group_std, per_row


def compute_conserved_turn_credit_data(
    base_compute_advantage: Callable[..., Any],
    data: Any,
    *args: Any,
    beta: float,
    enabled: bool,
    proof_id: str,
    **kwargs: Any,
) -> Any:
    """Call native GRPO first, then redistribute credit within each trajectory (plan §3)."""
    data = base_compute_advantage(data, *args, **kwargs)
    if not enabled:
        return data
    if not math.isclose(beta, FROZEN_BETA, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"E5-v2 beta is frozen at {FROZEN_BETA}, got {beta}")
    if LEDGER_KEY not in data.non_tensor_batch:
        raise KeyError(f"missing required rollout metadata: {LEDGER_KEY}")
    if "uid" not in data.non_tensor_batch:
        raise KeyError("missing native GRPO uid")
    token_scores = data.batch.get("token_level_scores")
    if token_scores is None:
        raise KeyError("E5-v2 gating requires token_level_scores")

    response_mask = data.batch["response_mask"]
    responses = data.batch["responses"]
    base_advantages = data.batch["advantages"].clone()
    base_returns = data.batch["returns"].clone()
    if response_mask.shape != responses.shape or base_advantages.shape != responses.shape:
        raise ValueError("responses/response_mask/advantages shapes differ")
    if token_scores.shape != responses.shape:
        raise ValueError("token_level_scores shape differs from responses")

    batch_size, response_width = responses.shape
    ledgers = [_as_ledger_v2(value) for value in data.non_tensor_batch[LEDGER_KEY]]
    uids = [str(_python_scalar(value)) for value in data.non_tensor_batch["uid"]]
    if len(ledgers) != batch_size or len(uids) != batch_size:
        raise ValueError("uid/ledger batch length mismatch")

    num_repeat = int(kwargs.get("num_repeat", 1))
    group_members: dict[str, list[int]] = {}
    for row, uid in enumerate(uids):
        group_members.setdefault(uid, []).append(row)
    if any(len(rows) != num_repeat for rows in group_members.values()):
        sizes = {uid: len(rows) for uid, rows in group_members.items()}
        raise ValueError(f"every GRPO uid must have num_repeat={num_repeat} rows, got {sizes}")
    informative, group_std, row_scores = _group_informative_flags(token_scores, group_members)

    turn_ids = torch.full_like(response_mask, -1, dtype=torch.long)
    native_scalars: list[float] = []
    trajectory_ids: set[tuple[str, str]] = set()
    for row, (uid, ledger) in enumerate(zip(uids, ledgers, strict=True)):
        trajectory_uid = str(ledger.get("trajectory_uid", ""))
        if not trajectory_uid or (uid, trajectory_uid) in trajectory_ids:
            raise ValueError("missing or duplicate trajectory_uid within GRPO group")
        trajectory_ids.add((uid, trajectory_uid))
        turn_ids[row] = _validate_turn_spans(ledger, response_mask[row], response_width)
        native_scalars.append(_native_scalar(base_advantages[row], response_mask[row]))
        if not informative[uid] and not math.isclose(native_scalars[-1], 0.0, abs_tol=1e-6):
            raise ValueError("zero-variance GRPO group has a nonzero native advantage")

    corrected_advantages = base_advantages.clone()
    corrected_returns = base_returns.clone()
    audits = np.empty(batch_size, dtype=object)
    for row, (uid, ledger) in enumerate(zip(uids, ledgers, strict=True)):
        turns = ledger["assistant_turns"]
        s_values: list[int] = []
        token_counts: list[int] = []
        eligible: list[bool] = []
        for turn in turns:
            for field in ("s_t", "s_t_v1", "credit_eligible", "after_boxed", "repeat_of_turn"):
                if field not in turn:
                    raise ValueError(f"E5-v2 ledger turn is missing {field}")
            s_values.append(int(turn["s_t"]))
            token_counts.append(int(turn["policy_token_end"]) - int(turn["policy_token_start"]))
            eligible.append(bool(turn["credit_eligible"]))
        corrections, s_tilde, gated = conserved_turn_corrections(
            s_values, token_counts, eligible, beta=beta, informative_group=informative[uid]
        )
        residual = 0.0
        rows: list[dict[str, Any]] = []
        for turn, correction, n_t in zip(turns, corrections, token_counts, strict=True):
            start = int(turn["policy_token_start"])
            end = int(turn["policy_token_end"])
            if correction != 0.0:
                corrected_advantages[row, start:end] += correction
                corrected_returns[row, start:end] += correction
            residual += correction * n_t
            rows.append(
                {
                    "turn_index": int(turn["turn_index"]),
                    "A_traj": native_scalars[row],
                    "execution_success": int(turn.get("execution_success", 0)),
                    "execution_status": turn.get("execution_status"),
                    "adopted": int(turn.get("adopted", 0)),
                    "adoption_status": turn.get("adoption_status"),
                    "adoption_status_v2": turn.get("adoption_status_v2"),
                    "adoption_evidence": turn.get("adoption_evidence"),
                    "credit_eligible": bool(turn["credit_eligible"]),
                    "s_t_v1": int(turn["s_t_v1"]),
                    "repeat_of_turn": turn["repeat_of_turn"],
                    "after_boxed": bool(turn["after_boxed"]),
                    "s_t": int(turn["s_t"]),
                    "n_t": n_t,
                    "s_tilde": s_tilde,
                    "gated": gated,
                    "correction": correction,
                    "A_turn": native_scalars[row] + correction,
                    "policy_token_span": [start, end],
                    "tool_token_span": (
                        [int(turn["tool_token_start"]), int(turn["tool_token_end"])]
                        if turn.get("tool_token_start") is not None
                        else None
                    ),
                    "assistant_text": turn.get("assistant_text"),
                    "visible_tool_response_text": turn.get("visible_tool_response_text"),
                    "final_score": row_scores[row],
                }
            )
        if abs(residual) > CONSERVATION_TOLERANCE:
            raise ValueError(f"E5-v2 trajectory {ledger['trajectory_uid']} violates conservation")
        if gated and not torch.equal(corrected_advantages[row], base_advantages[row]):
            raise ValueError("gated trajectory must keep the native advantage exactly")
        audits[row] = {
            "schema_version": LEDGER_SCHEMA_VERSION_V2,
            "adoption_version": ADOPTION_VERSION_V2,
            "variant": VARIANT_CONSERVED_V2,
            "wrapper_proof": proof_id,
            "prompt_uid": uid,
            "trajectory_uid": ledger["trajectory_uid"],
            "response_length_unpadded": int(ledger["response_length_unpadded"]),
            "response_mask_sha256_uint8": hashlib.sha256(
                response_mask[row].detach().cpu().numpy().astype(np.uint8).tobytes()
            ).hexdigest(),
            "policy_token_count": int(response_mask[row].sum().detach().cpu()),
            "masked_tool_or_padding_token_count": int(
                response_mask.shape[1] - response_mask[row].sum().detach().cpu()
            ),
            "trajectory_truncated": bool(ledger.get("trajectory_truncated", False)),
            "termination_reason": ledger.get("termination_reason"),
            "group_score_std": group_std[uid],
            "informative_group": informative[uid],
            "eligible_turn_count": int(sum(eligible)),
            "gated": gated,
            "treated": any(value != 0.0 for value in corrections),
            "conservation_residual": residual,
            "turns": rows,
        }

    corrected_advantages *= response_mask
    corrected_returns *= response_mask
    if corrected_advantages.shape != responses.shape or turn_ids.shape != responses.shape:
        raise ValueError("turn-credit tensor shape mismatch")
    if bool(torch.any(corrected_advantages[~response_mask.bool()] != 0)):
        raise ValueError("tool/padding advantages must remain zero")

    data.batch["turn_ids"] = turn_ids
    data.batch["turn_credit_base_advantages"] = base_advantages
    data.batch["advantages"] = corrected_advantages
    data.batch["returns"] = corrected_returns
    data.non_tensor_batch[AUDIT_KEY] = audits
    proof_values = np.empty(batch_size, dtype=object)
    proof_values[:] = proof_id
    data.non_tensor_batch[WRAPPER_PROOF_KEY] = proof_values
    return data


def make_conserved_compute_advantage_wrapper(
    base_compute_advantage: Callable[..., Any], *, beta: float, enabled: bool, proof_id: str
) -> Callable[..., Any]:
    """Identity-preserving wrapper around veRL's module-level hook, v2 variant."""

    @wraps(base_compute_advantage)
    def wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
        return compute_conserved_turn_credit_data(
            base_compute_advantage,
            data,
            *args,
            beta=beta,
            enabled=enabled,
            proof_id=proof_id,
            **kwargs,
        )

    setattr(wrapped, "__toolcredit_wrapper_proof__", proof_id)
    setattr(wrapped, "__toolcredit_variant__", VARIANT_CONSERVED_V2)
    return wrapped
