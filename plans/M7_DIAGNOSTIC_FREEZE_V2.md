# M7 diagnostic protocol v2 freeze ledger

- 冻结时间（UTC）：`2026-08-22T23:03:14Z`
- Git HEAD：`87464202c987311d77e93e72b107918407dab10f`
- 状态：`FROZEN_V2_HUMAN_ADOPTION_REVIEW_PENDING`。
- v2唯一协议变更：修正E3 locator并新增所有checkpoint locator的role-aware fail-closed gate；panel、taxonomy、generation、tool、verifier、pairing与bootstrap沿用v1。
- E3 locator：`rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200`。
- checkpoint gate：raw/SFT/E3非空locator必须存在并匹配冻结role/结构/identity；E5在pre-training freeze中必须显式为`state=future, locator=null`，未来诊断执行前另行version bump解析step-200 locator。
- v1 smoke只保留为noncanonical wiring evidence；Codex preliminary adoption labels不计入human precision gate。canonical v2 smoke在真实human review前禁止启动。

panel仍为760题；v1 panel/taxonomy文件按相同hash复用。

| artifact | SHA256 | Git blob ID |
|---|---|---|
| `eval/diagnostic_protocol_v2.json` | `a3cdf6bf98efe5c214b970e2c78b5cede9bcfa560c726f1b4394ea3b83198831` | `9434f87c53e8e39028b04d977daad65936e3475b` |
| `eval/diagnostic_panel_v1.jsonl` | `0a01ebfe258c92da9f7a06f3767e514f4a853d9d32f7956940a2966f5a9654df` | `4d497b5280815b2fee1da234a5dd309ae923e2fb` |
| `eval/diagnostic_taxonomy_v1.json` | `4ba98088f8f5a9987a9986534c0cc2e85e26ad5b33fe40114e793c5145fe805a` | `104eddf3a5eb8fdbd128c4bcd5465e2eac237185` |

完整闭包由`eval/diagnostic_freeze_v2.sha256`锁定；manifest不自哈希，其hash由preflight单独记录。
