# M7 diagnostic protocol v3 freeze ledger

- 冻结时间（UTC）：`2026-08-24T18:45:31Z`
- Git HEAD：`7ee169ff90d8d204fffd96be227c88b4ba6101cc`
- 状态：`FROZEN_V3_READY_FOR_GPU_SMOKE_REVIEW`；尚未运行80条GPU smoke、canonical evaluation或E7。
- parent v2 manifest SHA256：`6fa185e18c17761cf9c238cb2a9754e245cda8179136e59cb69c729c9b07f3f0`（37-file closure保持逐字节不变）。
- v3唯一语义范围：解析M6完成后的E5 step-200 role、绑定E3/E5官方FSDP materialization与统一evaluator closure；panel/generation/tool/verifier/taxonomy/pairing/bootstrap/data-isolation继续沿用v2。
- 用户批准解释约束：metadata-only M7 loop必须通过与ToolCreditAgentLoop的behavioral-identity fixture，只有audit metadata可不同。
- 诊断证据层级：greedy paired failure transitions为primary；sampled matched-index transitions为secondary/exploratory，不从matched index作强因果主张。
- E3 materialization manifest SHA256：`b543647a661837e713c2a0bdf8b23a3a78084a60b61ad111ab0a68698c695db1`。
- E5 materialization manifest SHA256：`78441f518ff55fa8b46a523427730c3cb6486eb6e4527ebf9e928a7827b65ae3`。
- evaluator config SHA256：`34f7608e37948d94e07d2d6e239a8e5d5ead644ac1f3dbfb93d3bfb750efcaa7`；trajectory schema SHA256：`7068aa7904cca30daca0947aff2fc0dfd8c8b0bbda2491551217fd09868097e5`。

| artifact | SHA256 | Git blob ID |
|---|---|---|
| `eval/diagnostic_protocol_v3.json` | `c461d22bb5750e2986f26ce825104f5f803148573767625644d8143a5aa36212` | `22702532eedbc131e7f6bfc6aa38ab734f34e278` |
| `eval/m7_eval_config.json` | `34f7608e37948d94e07d2d6e239a8e5d5ead644ac1f3dbfb93d3bfb750efcaa7` | `aab2d07321cd1639f6cb9cc9111821fc3a3bf64e` |
| `eval/m7_trajectory_schema.json` | `7068aa7904cca30daca0947aff2fc0dfd8c8b0bbda2491551217fd09868097e5` | `ebc6d03f4a4853f11e0bbf6179126964cb646682` |

完整闭包由`eval/diagnostic_freeze_v3.sha256`锁定；manifest不自哈希，verification report另行记录其hash。
