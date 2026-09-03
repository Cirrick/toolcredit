# E3 longer-budget diagnostic

This run is diagnostic-only. Its 70 held-out questions are forbidden from training use and do not alter the M7 canonical denominator.

## Original selection

- Selected canonical E3 greedy failures: **70**
- Stopping mechanisms: `{"assistant_or_tool_turn_limit": 7, "response_token_limit": 63}`
- Exclusive classes: `{"assistant_or_tool_turn_limit": 2, "repeated_normalized_code": 10, "response_token_limit": 58}`
- Repeated normalized code tag: **10**

## Longer-budget result

Budget changed from 3072 tokens / 4 tool turns / 5 assistant turns to 6144 / 6 / 7; all other generation and scoring semantics were held fixed.

- Strict project-correct after expansion: **16/70 (0.2286)**
- Correct answer reached before a later budget stop: **1**
- Still wrong: **54**
- Residual subtypes: `{"completed_wrong_semantic": 14, "persistent_assistant_or_tool_turn_limit": 2, "persistent_response_token_limit": 38}`
- Long-run termination reasons: `{"assistant_turn_budget": 3, "final_answer": 29, "response_length": 38}`

A wrong-to-correct transition supports budget as the direct cause for that trajectory. A residual wrong is reported both under the requested binary reclassification and under a mechanistic subtype; persistent re-truncation is not silently called a completed semantic answer.
