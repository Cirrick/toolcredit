# M8 diagnostic protocol v4 freeze ledger

- 冻结时间（UTC）：`2026-09-05T13:44:45Z`
- Git HEAD：`1def127b1676155c02abb29ee71a90026a22fc16`
- 状态：`FROZEN_V4_READY_FOR_E5V2_GENERATION`；E5-v2 的 3,800 条评测轨迹尚未生成。
- parent v3 manifest SHA256：`b765ec2c121ce8bf166a5bcb43373334aeb42c1b2b796de2b02998f2a05b05a4`（66-file closure 保持逐字节不变）。
- v4 唯一语义范围：新增 role `e5v2_conserved_credit_step_200`（run `e5v2_conserved_credit_20260905_003041`）及其官方 FSDP materialization，绑定 M8 evaluator 文件，并记录复用的 M7 canonical 产物；panel/generation/tool/verifier/taxonomy/pairing/bootstrap/persistence/prompt/data-isolation 继续沿用 v3。
- 配对：E3→E5-v2 为 primary，E5-v1→E5-v2 为 secondary；greedy paired transitions 仍为 primary diagnostic，sampled matched-index 为 secondary/exploratory。
- E5-v2 materialization manifest SHA256：`4db573b41aae016baf26bf070bb657b4772d73c59837d4328f69af6eb18436e3`。
- 复用的 M7 canonical run：`m7_unified_eval_20260824_201032`，hashes.sha256 `491b4d66a539123d0ade1e9b37a25a9ee269207e318898dc0b9b8a91acf176f9`，full exact-key digest `c43ce71901f61c24ca0510b1af6fd70a149d4b4692793cff0326c57285efa8b8`。

| artifact | SHA256 | Git blob ID |
|---|---|---|
| `eval/diagnostic_protocol_v4.json` | `187c627551cb5b652b3e7f856db930b53d451978e5a442f304ad28ad5de12ba4` | `dffc6dc5b42e8e946d27aead24f3d17cb0a68d40` |
| `eval/materialization/e5v2_conserved_credit_step_200.json` | `4db573b41aae016baf26bf070bb657b4772d73c59837d4328f69af6eb18436e3` | `7a2f99868f9af1b57f6522686a397b55e4277c65` |

完整闭包由 `eval/diagnostic_freeze_v4.sha256` 锁定；manifest 不自哈希，verification report 另行记录其 hash。
