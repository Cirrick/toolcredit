# M10 diagnostic protocol v6 freeze ledger

- 冻结时间（UTC）：`2026-09-10T13:11:59Z`
- Git HEAD：`83bbd62c2abc6c5585bfd1c670d5901f1b375808`
- 状态：`FROZEN_V6_READY_FOR_E7_GENERATION`；两个 E7 角色的 7,600 条评测轨迹尚未生成。
- parent v5 manifest SHA256：`5238d08e73be1d7aef59ca89f3cb5f6a66425ccab208ade476d63f2e6ffb2e4d`（107-file closure 保持逐字节不变）。
- v6 唯一语义范围：新增两个 `tir` 角色 `e7_dynamic_filter_step_200`、`e7_equal_compute`（均由 run `e7_dynamic_filtering_20260909_200152` 的官方 FSDP materialization 得到），绑定 M10 evaluator 文件，记录复用的 M7/M8/M9 canonical 产物，并声明两组 E3→E7 primary 配对；panel/generation/tool/verifier/taxonomy/pairing/bootstrap/persistence/prompt/data-isolation 与 notool 模式块继续沿用 v5。
- 等算力角色：有效 step `95`，累计 rollout 轨迹 `102400`（E3 全程为 102400）。该 step 由 run ledger 判定后从磁盘发现，计划只冻结规则不冻结数值。
- 判读：`Δ_step`（fixed-100 greedy 曲线按有效 step 的 0–200 归一化 AUC 之差）与 `Δ_compute`（等算力 checkpoint 与 E3 step-200 在 FULL760 greedy 上的配对 Δ 与 bootstrap CI）构成 §9 四格矩阵；greedy paired transitions 仍为 primary diagnostic，sampled matched-index 为 secondary/exploratory。
- e7_dynamic_filter_step_200 materialization manifest SHA256：`f9ff56ff972beee45bdb20e3ee6a84ba25dc016c588015bc0bf0d00f1491b68f`。
- e7_equal_compute materialization manifest SHA256：`b861a5041ee14d14a485c96bb3e1d161d884238a821cc64f205aadd9032a01fb`。
- 复用的 canonical run：M7 `m7_unified_eval_20260824_201032`（`491b4d66a539123d0ade1e9b37a25a9ee269207e318898dc0b9b8a91acf176f9`）、M8 `m8_e5v2_eval_20260905_134737`（`135e05ee2f63f3c5f4b776347662f22e9f3f5b63692f2d9a94c2767b12c80332`）、M9 `m9_notool_eval_20260907_175726`（`7c65cd21f8afb7b363a783f28fb763506fdfc389bc9073e2ed8ddc1fddc7224c`）。

| artifact | SHA256 | Git blob ID |
|---|---|---|
| `eval/diagnostic_protocol_v6.json` | `fb60f3ed374dae834256f3f00254112f227b1a88991d8fa3930c338b27cf0836` | `e471358690df1de2471a86765eb363ef5f3d2365` |
| `eval/materialization/e7_dynamic_filter_step_200.json` | `f9ff56ff972beee45bdb20e3ee6a84ba25dc016c588015bc0bf0d00f1491b68f` | `cc4073411dbaa05bd43076e0d695331048db8dc6` |
| `eval/materialization/e7_equal_compute.json` | `b861a5041ee14d14a485c96bb3e1d161d884238a821cc64f205aadd9032a01fb` | `f2444a9cda3e794170409e9eef2ebfd2730b774f` |

完整闭包由 `eval/diagnostic_freeze_v6.sha256` 锁定；manifest 不自哈希，verification report 另行记录其 hash。
