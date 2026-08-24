"""Tier-A turn-level credit correction for M6 / E5.

The native veRL GRPO function remains the sole producer of trajectory-level
advantages.  This module validates the structured rollout ledger and adds only
the preregistered ``beta * (s_t - s_bar_t)`` correction on policy-token spans.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import wraps
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

ADOPTION_VERSION = "toolcredit_adoption_v1"
LEDGER_SCHEMA_VERSION = "toolcredit_turn_ledger_v1"
LEDGER_KEY = "turn_credit_ledger"
AUDIT_KEY = "turn_credit_audit"
WRAPPER_PROOF_KEY = "turn_credit_wrapper_proof"

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_LATEX_FRAC_RE = re.compile(r"\\frac\s*\{\s*([+-]?\d+)\s*\}\s*\{\s*([+-]?\d+)\s*\}")
_NUMBER_RE = re.compile(
    r"(?<![\w.])(?:[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\.\d+)(?:/[+-]?\d+)?(?!\w|\.\d)"
)
_BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
_CUE_RE = re.compile(
    r"\b(?:therefore|thus|hence|so|we\s+get|gives?|using|substitut(?:e|ing)|result|answer)\b",
    flags=re.IGNORECASE,
)
_MENTION_RE = re.compile(r"\b(?:tool|output|returned|returns|says?|reported)\b", flags=re.IGNORECASE)
_EXPRESSION_RE = re.compile(r"(?:=|\+|-|\*|/|<|>|≤|≥|\^)")
_NOISE_LINES = {
    "ok",
    "success",
    "done",
    "completed",
    "(no output)",
}


@dataclass(frozen=True)
class Candidate:
    """A normalized value extracted from model-visible tool text."""

    kind: str
    raw: str
    normalized: str


@dataclass(frozen=True)
class AdoptionDecision:
    """Deterministic adoption outcome and its auditable evidence."""

    status: str
    s_t: int
    evidence: dict[str, Any] | None
    candidate_count: int


def truncate_tool_response_text(text: str | None, max_length: int, side: str) -> str | None:
    """Mirror veRL 0.8.0's pre-tokenization string truncation exactly."""
    if text is None or len(text) <= max_length:
        return text
    if max_length <= 0:
        raise ValueError("max_tool_response_length must be positive")
    if side == "left":
        return "(truncated)..." + text[-max_length:]
    if side == "right":
        return text[:max_length] + "...(truncated)"
    length = max_length // 2
    return text[:length] + "...(truncated)..." + text[-length:]


def normalize_visible_text(text: str) -> str:
    """Normalize visible content without introducing text absent from rollout."""
    normalized = unicodedata.normalize("NFKC", _ANSI_RE.sub("", text))
    normalized = _CONTROL_RE.sub("", normalized)
    return "\n".join(" ".join(line.split()) for line in normalized.splitlines()).strip()


def _parse_number(raw: str) -> Fraction | Decimal | None:
    latex = _LATEX_FRAC_RE.fullmatch(raw.strip())
    if latex:
        denominator = int(latex.group(2))
        return None if denominator == 0 else Fraction(int(latex.group(1)), denominator)
    compact = raw.replace(",", "")
    if "/" in compact:
        numerator, denominator = compact.split("/", maxsplit=1)
        try:
            denominator_int = int(denominator)
            return None if denominator_int == 0 else Fraction(int(numerator), denominator_int)
        except ValueError:
            return None
    try:
        value = Decimal(compact)
    except InvalidOperation:
        return None
    return value if value.is_finite() else None


def _number_key(value: Fraction | Decimal) -> str:
    if isinstance(value, Fraction):
        return f"fraction:{value.numerator}/{value.denominator}"
    return f"decimal:{value.normalize()}"


def _numbers_equal(left: Fraction | Decimal, right: Fraction | Decimal) -> bool:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return left == right
    left_decimal = Decimal(left.numerator) / Decimal(left.denominator) if isinstance(left, Fraction) else left
    right_decimal = Decimal(right.numerator) / Decimal(right.denominator) if isinstance(right, Fraction) else right
    tolerance = max(Decimal("1e-9"), Decimal("1e-6") * max(abs(left_decimal), abs(right_decimal)))
    return abs(left_decimal - right_decimal) <= tolerance


def _numeric_occurrences(text: str) -> list[tuple[str, Fraction | Decimal, int, int]]:
    occurrences: list[tuple[str, Fraction | Decimal, int, int]] = []
    occupied: list[tuple[int, int]] = []
    for match in _LATEX_FRAC_RE.finditer(text):
        value = _parse_number(match.group(0))
        if value is not None:
            occurrences.append((match.group(0), value, match.start(), match.end()))
            occupied.append((match.start(), match.end()))
    for match in _NUMBER_RE.finditer(text):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        value = _parse_number(match.group(0))
        if value is not None:
            occurrences.append((match.group(0), value, match.start(), match.end()))
    return sorted(occurrences, key=lambda item: (item[2], item[3]))


def extract_candidates(visible_tool_response_text: str | None) -> tuple[list[Candidate], str | None]:
    """Extract candidates only from the post-truncation text encoded for the model."""
    if visible_tool_response_text is None:
        return [], "unobservable_truncated"
    normalized = normalize_visible_text(visible_tool_response_text)
    if not normalized or normalized.casefold() == "(no output)":
        return [], "no_output"
    numeric = _numeric_occurrences(normalized)
    if len(numeric) > 32:
        return [], "ambiguous_output"

    candidates: list[Candidate] = []
    seen: set[tuple[str, str]] = set()
    for raw, value, _, _ in numeric:
        key = ("number", _number_key(value))
        if key not in seen:
            seen.add(key)
            candidates.append(Candidate(kind="number", raw=raw, normalized=key[1]))

    for line in normalized.splitlines():
        folded = line.casefold()
        if not (4 <= len(line) <= 80) or folded in _NOISE_LINES:
            continue
        if _parse_number(line) is not None:
            continue
        key = ("string", folded)
        if key not in seen:
            seen.add(key)
            candidates.append(Candidate(kind="string", raw=line, normalized=folded))
    return candidates, None


def _candidate_number(candidate: Candidate) -> Fraction | Decimal:
    if candidate.kind != "number":
        raise TypeError("candidate is not numeric")
    kind, value = candidate.normalized.split(":", maxsplit=1)
    parsed = _parse_number(value)
    if parsed is None:
        raise ValueError(f"invalid normalized numeric candidate: {candidate.normalized}")
    if kind not in {"fraction", "decimal"}:
        raise ValueError(f"unknown numeric candidate kind: {kind}")
    return parsed


def _integration_kind(text: str, start: int, end: int) -> str | None:
    window = text[max(0, start - 96) : min(len(text), end + 96)]
    if _CUE_RE.search(window):
        return "reasoning_cue"
    occurrence_start = min(96, start)
    outside_occurrence = window[:occurrence_start] + window[occurrence_start + (end - start) :]
    if _EXPRESSION_RE.search(outside_occurrence):
        return "expression"
    if _MENTION_RE.search(window):
        return None
    return None


def _boxed_numeric_match(candidate: Candidate, text: str) -> dict[str, Any] | None:
    target = _candidate_number(candidate)
    for boxed in _BOXED_RE.finditer(text):
        value = _parse_number(boxed.group(1).strip())
        if value is not None and _numbers_equal(target, value):
            return {"kind": "boxed", "candidate": candidate.raw, "match": boxed.group(0), "offset": boxed.start()}
    return None


def _boxed_string_match(candidate: Candidate, text: str) -> dict[str, Any] | None:
    for boxed in _BOXED_RE.finditer(text):
        if normalize_visible_text(boxed.group(1)).casefold() == candidate.normalized:
            return {"kind": "boxed", "candidate": candidate.raw, "match": boxed.group(0), "offset": boxed.start()}
    return None


def _code_uses_numeric_candidate(candidate: Candidate, codes: Iterable[str]) -> dict[str, Any] | None:
    target = _candidate_number(candidate)
    for code_index, code in enumerate(codes):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                value = _parse_number(repr(node.value))
                if value is not None and _numbers_equal(target, value):
                    return {
                        "kind": "later_tool_code_ast",
                        "candidate": candidate.raw,
                        "code_index": code_index,
                        "line": getattr(node, "lineno", None),
                    }
    return None


def classify_adoption(
    visible_tool_response_text: str | None,
    later_assistant_texts: Sequence[str],
    later_tool_codes: Sequence[str] = (),
) -> AdoptionDecision:
    """Classify whether a successful call's visible result is later integrated."""
    candidates, terminal_status = extract_candidates(visible_tool_response_text)
    if terminal_status is not None:
        return AdoptionDecision(terminal_status, 0, None, 0)

    mentioned: dict[str, Any] | None = None
    for turn_offset, later_text in enumerate(later_assistant_texts, start=1):
        normalized_later = normalize_visible_text(later_text)
        occurrences = _numeric_occurrences(normalized_later)
        for candidate in candidates:
            boxed = (
                _boxed_numeric_match(candidate, normalized_later)
                if candidate.kind == "number"
                else _boxed_string_match(candidate, normalized_later)
            )
            if boxed is not None:
                boxed["later_turn_offset"] = turn_offset
                return AdoptionDecision("adopted", 1, boxed, len(candidates))

            if candidate.kind == "number":
                target = _candidate_number(candidate)
                for raw, value, start, end in occurrences:
                    if not _numbers_equal(target, value):
                        continue
                    integration = _integration_kind(normalized_later, start, end)
                    evidence = {
                        "kind": integration or "mentioned",
                        "candidate": candidate.raw,
                        "match": raw,
                        "offset": start,
                        "later_turn_offset": turn_offset,
                    }
                    if integration is not None:
                        return AdoptionDecision("adopted", 1, evidence, len(candidates))
                    mentioned = mentioned or evidence
            else:
                pattern = re.compile(rf"(?<!\w){re.escape(candidate.normalized)}(?!\w)", re.IGNORECASE)
                for match in pattern.finditer(normalized_later.casefold()):
                    integration = _integration_kind(normalized_later, match.start(), match.end())
                    evidence = {
                        "kind": integration or "mentioned",
                        "candidate": candidate.raw,
                        "match": match.group(0),
                        "offset": match.start(),
                        "later_turn_offset": turn_offset,
                    }
                    if integration is not None:
                        return AdoptionDecision("adopted", 1, evidence, len(candidates))
                    mentioned = mentioned or evidence

    for candidate in candidates:
        if candidate.kind == "number":
            code_evidence = _code_uses_numeric_candidate(candidate, later_tool_codes)
            if code_evidence is not None:
                return AdoptionDecision("adopted", 1, code_evidence, len(candidates))
    if mentioned is not None:
        return AdoptionDecision("mentioned_only", 0, mentioned, len(candidates))
    return AdoptionDecision("not_found", 0, None, len(candidates))


def _python_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def _as_ledger(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if not isinstance(value, dict):
        raise TypeError(f"{LEDGER_KEY} must contain dictionaries, got {type(value).__name__}")
    if value.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError(f"unexpected turn ledger schema: {value.get('schema_version')!r}")
    if not isinstance(value.get("assistant_turns"), list):
        raise TypeError("turn ledger assistant_turns must be a list")
    return value


def _validate_turn_spans(
    ledger: dict[str, Any], response_mask: torch.Tensor, response_width: int
) -> torch.Tensor:
    response_length = int(ledger.get("response_length_unpadded", -1))
    if not 0 <= response_length <= response_width:
        raise ValueError(f"invalid response_length_unpadded={response_length}")
    turn_ids = torch.full((response_width,), -1, dtype=torch.long, device=response_mask.device)
    previous_end = 0
    for expected_index, turn in enumerate(ledger["assistant_turns"]):
        if int(turn.get("turn_index", -1)) != expected_index:
            raise ValueError("assistant turn indices must be contiguous and zero-based")
        start = int(turn.get("policy_token_start", -1))
        end = int(turn.get("policy_token_end", -1))
        if not 0 <= start < end <= response_length or start < previous_end:
            raise ValueError(f"invalid policy token span for turn {expected_index}: [{start}, {end})")
        if not bool(torch.all(response_mask[start:end] == 1)):
            raise ValueError(f"turn {expected_index} policy span contains masked tool/padding tokens")
        if bool(torch.any(turn_ids[start:end] != -1)):
            raise ValueError(f"overlapping policy span for turn {expected_index}")
        turn_ids[start:end] = expected_index
        tool_start = turn.get("tool_token_start")
        tool_end = turn.get("tool_token_end")
        if (tool_start is None) != (tool_end is None):
            raise ValueError("tool span must provide both start and end")
        if tool_start is not None:
            tool_start, tool_end = int(tool_start), int(tool_end)
            if not end <= tool_start < tool_end <= response_length:
                raise ValueError(f"invalid tool span for turn {expected_index}: [{tool_start}, {tool_end})")
            if not bool(torch.all(response_mask[tool_start:tool_end] == 0)):
                raise ValueError(f"tool span for turn {expected_index} contains policy tokens")
            previous_end = tool_end
        else:
            previous_end = end
    if not torch.equal(turn_ids.ge(0), response_mask.bool()):
        raise ValueError("assistant turn spans do not cover every and only policy-generated token")
    if bool(torch.any(response_mask[response_length:] != 0)):
        raise ValueError("response padding contains nonzero policy mask")
    return turn_ids


def _native_scalar(advantages: torch.Tensor, response_mask: torch.Tensor) -> float:
    values = advantages[response_mask.bool()]
    if values.numel() == 0:
        raise ValueError("trajectory has no policy-generated tokens")
    first = values[0]
    if not torch.allclose(values, first.expand_as(values), atol=1e-6, rtol=1e-6):
        raise ValueError("native GRPO advantage is not a single trajectory-level scalar")
    return float(first.detach().cpu())


def compute_turn_credit_data(
    base_compute_advantage: Callable[..., Any],
    data: Any,
    *args: Any,
    beta: float,
    enabled: bool,
    proof_id: str,
    **kwargs: Any,
) -> Any:
    """Call native GRPO first, then apply the frozen Tier-A correction."""
    data = base_compute_advantage(data, *args, **kwargs)
    if not enabled:
        return data
    if not math.isclose(beta, 0.5, rel_tol=0.0, abs_tol=0.0):
        raise ValueError(f"E5 beta is frozen at 0.5, got {beta}")
    if LEDGER_KEY not in data.non_tensor_batch:
        raise KeyError(f"missing required rollout metadata: {LEDGER_KEY}")
    if "uid" not in data.non_tensor_batch:
        raise KeyError("missing native GRPO uid")

    response_mask = data.batch["response_mask"]
    responses = data.batch["responses"]
    base_advantages = data.batch["advantages"].clone()
    base_returns = data.batch["returns"].clone()
    if response_mask.shape != responses.shape or base_advantages.shape != responses.shape:
        raise ValueError("responses/response_mask/advantages shapes differ")

    batch_size, response_width = responses.shape
    ledgers = [_as_ledger(value) for value in data.non_tensor_batch[LEDGER_KEY]]
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

    s_by_position: dict[tuple[str, int], list[int]] = {}
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
        for turn in ledger["assistant_turns"]:
            s_t = int(turn.get("s_t", -1))
            if s_t not in {0, 1}:
                raise ValueError(f"s_t must be binary, got {s_t}")
            position = int(turn["turn_index"])
            s_by_position.setdefault((uid, position), []).append(s_t)

    position_stats = {
        key: (sum(values) / len(values), len(values)) for key, values in s_by_position.items()
    }
    corrected_advantages = base_advantages.clone()
    corrected_returns = base_returns.clone()
    audits = np.empty(batch_size, dtype=object)
    token_scores = data.batch.get("token_level_scores")
    for row, (uid, ledger) in enumerate(zip(uids, ledgers, strict=True)):
        rows: list[dict[str, Any]] = []
        final_score = float(token_scores[row].sum().detach().cpu()) if token_scores is not None else None
        for turn in ledger["assistant_turns"]:
            position = int(turn["turn_index"])
            start = int(turn["policy_token_start"])
            end = int(turn["policy_token_end"])
            s_t = int(turn["s_t"])
            s_bar, denominator = position_stats[(uid, position)]
            correction = beta * (s_t - s_bar)
            corrected_advantages[row, start:end] += correction
            corrected_returns[row, start:end] += correction
            rows.append(
                {
                    "turn_index": position,
                    "A_traj": native_scalars[row],
                    "execution_success": int(turn.get("execution_success", 0)),
                    "execution_status": turn.get("execution_status"),
                    "adopted": int(turn.get("adopted", 0)),
                    "adoption_status": turn.get("adoption_status"),
                    "adoption_evidence": turn.get("adoption_evidence"),
                    "s_t": s_t,
                    "s_bar_t": s_bar,
                    "position_denominator": denominator,
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
                    "final_score": final_score,
                }
            )
        audits[row] = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "adoption_version": ADOPTION_VERSION,
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


def make_compute_advantage_wrapper(
    base_compute_advantage: Callable[..., Any], *, beta: float, enabled: bool, proof_id: str
) -> Callable[..., Any]:
    """Build an identity-preserving wrapper around veRL's module-level hook."""

    @wraps(base_compute_advantage)
    def wrapped(data: Any, *args: Any, **kwargs: Any) -> Any:
        return compute_turn_credit_data(
            base_compute_advantage,
            data,
            *args,
            beta=beta,
            enabled=enabled,
            proof_id=proof_id,
            **kwargs,
        )

    setattr(wrapped, "__toolcredit_wrapper_proof__", proof_id)
    return wrapped


def decision_to_dict(decision: AdoptionDecision) -> dict[str, Any]:
    """Return a JSON-safe decision dictionary for the rollout ledger."""
    return asdict(decision)
