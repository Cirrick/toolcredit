# M7 diagnostic protocol v1 freeze ledger

- 冻结时间（UTC）：`2026-08-22T22:13:35Z`
- Git HEAD：`87464202c987311d77e93e72b107918407dab10f`
- 决定：冻结并批准用于 M6 E5 smoke 前门禁及后续 E3→E5/M7 配对诊断。
- 批准依据：用户于 2026-08-22 明确授权创建、验证、hash 并冻结完整 M7 bundle；所有门禁通过后可继续 5-step E5 smoke。
- 状态：`FROZEN_V1`。任何协议、panel、taxonomy、generation、tool、verifier、pairing 或 bootstrap 变更都必须 version bump，并对 raw/SFT/E3/E5 全部重跑。

## 冻结内容

M6 primary 是 fixed MATH500-100 的全部 100 题；M7 universe 是四个冻结 test source 的全部 760 题。panel 与 E3 train ID 交集为 0；既有 normalized exact + 13-gram contamination report 命中为 0。

`max_tool_response_length=512` 冻结为 pin veRL 0.8.0 的 Python 字符串长度截断：tokenization 前检查 `len(text)`，超过时使用 `text[:256] + "...(truncated)..." + text[-256:]`。它不是 512-token budget。adoption 只能读取实际编码进 rollout、模型可见的这个 post-processed/post-truncation 文本；raw sandbox stdout 仅供 audit。

## Canonical component ledger

JSON 使用 UTF-8、sorted keys、compact separators、单个尾随换行；JSONL 每行按同一规则 canonicalize。

| artifact | SHA256 | Git blob ID |
|---|---|---|
| `eval/diagnostic_protocol_v1.json` | `351c35b44d9663aff5b1a8780ecc0008e50a562b1b8e412839833c9e7906a5cd` | `e1cadf938e35a6916302383ff4a3a31b1695e6de` |
| `eval/diagnostic_panel_v1.jsonl` | `0a01ebfe258c92da9f7a06f3767e514f4a853d9d32f7956940a2966f5a9654df` | `4d497b5280815b2fee1da234a5dd309ae923e2fb` |
| `eval/diagnostic_taxonomy_v1.json` | `4ba98088f8f5a9987a9986534c0cc2e85e26ad5b33fe40114e793c5145fe805a` | `104eddf3a5eb8fdbd128c4bcd5465e2eac237185` |

完整闭包由 `eval/diagnostic_freeze_v1.sha256` 锁定；manifest 不自哈希。其 SHA256 必须由每次 smoke/formal preflight 单独记录。
