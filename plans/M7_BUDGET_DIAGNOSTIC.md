# M7 E3 budget/truncation post-hoc diagnostic

## Scope and authorization

On 2026-08-28 the user authorized one small diagnostic rerun of the 70 canonical E3 greedy failures whose frozen coarse proposal was `budget_exhaustion_or_truncation`. This was generation and analysis only: no model was trained, the M7 canonical denominator and artifacts were not changed, and these held-out evaluation questions are permanently forbidden from training use.

Any future training data must be constructed only from the original training split and must pass the repository contamination check again. This diagnostic does not authorize M5/E7, additional seeds, tuning, or further evaluation reruns.

## Frozen comparison

- Parent: `eval/runs/m7_unified_eval_20260824_201032/`, E3, FULL760, greedy sample 0.
- Diagnostic run: `eval/runs/e3_budget_diagnostic_20260828_095658/`.
- Model, tokenizer, prompt, tool schema and sandbox, greedy decoding, sample seeds, and strict verifier were held fixed.
- Only budgets changed: response tokens `3072→6144`, user/tool turns `4→6`, assistant turns `5→7`.
- Parent `hashes.sha256` and `analysis_hashes.sha256` passed before and after the run. The diagnostic ledger also passes `sha256sum --quiet -c hashes.sha256`.

## Original 70: mechanistic split

All 70 were truly truncated in the interaction ledger. The stopping signal was response length for 63 and assistant/tool-turn budget for 7. To produce the requested mutually exclusive split, normalized repeated Python code takes precedence over the stopping signal:

| Exclusive class | Count |
|---|---:|
| response token limit | 58 |
| assistant/tool-turn limit | 2 |
| repeated normalized code | 10 |

The repeat detector uses the existing AST-normalized code hash. The orthogonal counts remain 63 response-length stops, 7 turn stops, and 10 trajectories with repeated normalized code; these should not be added together.

## Longer-budget result

| Outcome | Count |
|---|---:|
| wrong→correct; budget direct-cause supported | 16 |
| completed final answer but still wrong | 14 |
| still wrong and again hit 6144-token limit | 38 |
| still wrong and again exhausted turn budget | 2 |

The requested binary rule therefore labels 16 as budget-direct and 54 as still-wrong semantic/policy failures after the budget increase. Mechanistically, only 14/54 are uncensored completed semantic errors; 40/54 still terminate at the expanded budget and should not be represented as completed reasoning that was proven semantically wrong.

Wrong→correct by original exclusive class was 9/58 response-token, 2/2 turn-limit, and 5/10 repeated-code. Fifteen corrected trajectories terminated normally; one reached the correct boxed answer before a later assistant-turn stop and is recorded as `correct_despite_truncation` rather than hidden.

## Intermediate-step diagnosis

- Of the 38 persistent response-length failures, 34 made no tool call. A deterministic audit of normalized nontrivial output lines found exact line repetition at least three times in 30/38. Representative trajectories repeatedly restart the same explanation or sentence, so simple budget expansion mostly gives the language loop more room.
- Both persistent turn-budget failures repeatedly attempted tools. The original repeated-code group was only partially repaired: 5/10 became correct, while the other five split into two completed-wrong, two persistent-token, and one persistent-turn failure.
- Among the 14 completed-wrong trajectories, 6 contained at least one sandbox/timeout execution error. Manual review found the recurring patterns to be unsupported guessing after tool failure, misuse or blind trust of a numerical result, and conceptual/algebraic/constraint errors even when the tool executed successfully.

The exact-output prefix matched the original shorter trajectory for only 17/70 reruns, despite greedy settings and fixed prompt/seed. Numerical serving nondeterminism therefore limits the 16 wrong→correct transitions to strong diagnostic evidence, not a strict counterfactual continuation proof.

## Evidence and validation

- Config: `eval/e3_budget_diagnostic_config.json` (`61b4ca54…71596`).
- Selection/raw/scored/comparisons: exactly 70 rows each; no missing or duplicate question IDs.
- Raw generation SHA-256: `d03742d2…fbb00`; metrics SHA-256: `9f704ce0…9756`.
- Primary result: `metrics.json`; per-question result: `comparisons.jsonl`; full steps: `scored.jsonl`.
- Tests: `18 passed` for `eval/test_budget_diagnostic.py`, `eval/test_m7_protocol.py`, and `eval/test_m7_metrics.py`.
- Generation ran in a unique tmux run; the SGLang process exited and GPU memory was released after completion.

