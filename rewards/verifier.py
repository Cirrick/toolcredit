"""Answer verifier (M2, PLAN §6.2).

Chain (first method that can render a judgement wins):
  1. normalized string exact match — a certain positive certificate, cheap;
  2. math-verify equivalence (primary path);
  3. sympy normalization fallback — for expressions math-verify fails to parse.

Two extraction regimes:
  - strict_boxed=True (TRAINING reward): the prediction must carry a `\\boxed{...}`;
    otherwise it is simply wrong (anti-hacking: prevents "spray candidate numbers
    and let a lenient extractor pick the right one", see reports/qa_log.md Q3).
  - strict_boxed=False (evaluation/probe): fall back to math-verify's lenient
    whole-text extraction when no boxed is present.

No failure is silent: machinery errors set invalid=True and are logged
(禁止事项 #2); callers aggregate invalid_rate.
"""

import logging
from typing import TypedDict

logger = logging.getLogger("verifier")


class VerifyResult(TypedDict):
    correct: bool
    method: str  # "string" | "math_verify" | "sympy" | "no_boxed" | "none"
    extracted: str | None
    invalid: bool  # machinery failed (all methods raised) — not the same as wrong


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST \\boxed{...} with balanced braces, else None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start - 1, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i]
    return None  # unbalanced braces


def _normalize_string(s: str) -> str:
    for tok in ("\\left", "\\right", "\\!", "\\,", "\\;", "$", " "):
        s = s.replace(tok, "")
    return s.strip("{}").strip()


def _try_math_verify(pred_expr: str, gold: str) -> bool | None:
    from math_verify import parse, verify

    try:
        gold_parsed = parse("\\boxed{" + gold + "}")
        pred_parsed = parse("\\boxed{" + pred_expr + "}")
        if not gold_parsed or not pred_parsed:
            return None
        return bool(verify(gold_parsed, pred_parsed))
    except Exception as e:
        logger.debug("math_verify failed on %r vs %r: %r", pred_expr, gold, e)
        return None


def _try_sympy(pred_expr: str, gold: str) -> bool | None:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr

    def to_expr(s: str):
        s = _normalize_string(s).replace("^", "**")
        return parse_expr(s, evaluate=True)

    try:
        return bool(sympy.simplify(to_expr(pred_expr) - to_expr(gold)) == 0)
    except Exception as e:
        logger.debug("sympy fallback failed on %r vs %r: %r", pred_expr, gold, e)
        return None


def _lenient_extract(pred_text: str, gold: str) -> bool | None:
    """Evaluation-only: let math-verify extract an answer from the whole text."""
    from math_verify import parse, verify

    try:
        gold_parsed = parse("\\boxed{" + gold + "}")
        pred_parsed = parse(pred_text)
        if not pred_parsed:
            return None
        return bool(verify(gold_parsed, pred_parsed))
    except Exception as e:
        logger.debug("lenient extraction failed: %r", e)
        return None


def verify_answer(pred_text: str, gold: str, strict_boxed: bool = True) -> VerifyResult:
    """Judge a model response against the gold answer. See module docstring."""
    extracted = extract_boxed(pred_text)
    if extracted is None:
        if strict_boxed:
            return VerifyResult(correct=False, method="no_boxed", extracted=None, invalid=False)
        lenient = _lenient_extract(pred_text, gold)
        if lenient is None:
            return VerifyResult(correct=False, method="no_boxed", extracted=None, invalid=False)
        return VerifyResult(correct=lenient, method="math_verify", extracted=None, invalid=False)

    if _normalize_string(extracted) == _normalize_string(gold):
        return VerifyResult(correct=True, method="string", extracted=extracted, invalid=False)

    for method, fn in (("math_verify", _try_math_verify), ("sympy", _try_sympy)):
        judged = fn(extracted, gold)
        if judged is not None:
            return VerifyResult(correct=judged, method=method, extracted=extracted, invalid=False)

    logger.warning("verifier could not judge pred=%r gold=%r — counted invalid", extracted, gold)
    return VerifyResult(correct=False, method="none", extracted=extracted, invalid=True)
