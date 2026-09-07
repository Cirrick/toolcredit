# M9 diagnostic protocol v5 freeze ledger

- 冻结时间（UTC）：`2026-09-07T17:54:08Z`
- Git HEAD：`a4bd7b359b933be4ca486b877d844f581dd31458`
- 状态：`FROZEN_V5_READY_FOR_NOTOOL_GENERATION`；四个 notool 角色的 15,200 条评测轨迹尚未生成。
- parent v4 manifest SHA256：`2a0d214d337038b9151a7e2930405f69912b7e9f79f8c1ad76a3eb8a132684f2`（87-file closure 保持逐字节不变）。
- v5 唯一语义范围：新增 `generation_mode`（`tir` = 现有五角色逐字段等于 v4；`notool` = chat template 不传 tools、单轮生成、tool 计数恒 0、新增 `tool_tag_emitted/tool_tag_count`），新增四个 notool 角色 `raw_base_model_notool`、`sft_start_notool`、`e3_grpo_baseline_step_200_notool`、`e3notool_grpo_step_200`（前三者复用冻结的 raw/SFT/E3 权重，`e3notool_grpo_step_200` 为 run `e3notool_grpo_20260906_220907` 的官方 FSDP materialization），绑定 M9 evaluator 文件，并记录复用的 M7/M8 canonical 产物；panel/generation/tool/verifier/taxonomy/pairing/bootstrap/persistence/prompt/data-isolation 继续沿用 v4。
- 配对：E3→e3notool_grpo_step_200 为 primary；五组 secondary 与一组 descriptive 见 protocol `evaluator.m9_extension.pairings`；greedy paired transitions 仍为 primary diagnostic，sampled matched-index 为 secondary/exploratory。
- E3-NoTool materialization manifest SHA256：`ea34d468f84ec72a10313bf61f85a0631f31d5429d1b4efa79565fa4aa41556c`。
- 复用的 M7 canonical run：`m7_unified_eval_20260824_201032`，hashes.sha256 `491b4d66a539123d0ade1e9b37a25a9ee269207e318898dc0b9b8a91acf176f9`，full exact-key digest `c43ce71901f61c24ca0510b1af6fd70a149d4b4692793cff0326c57285efa8b8`；M8 canonical run：`m8_e5v2_eval_20260905_134737`，hashes.sha256 `135e05ee2f63f3c5f4b776347662f22e9f3f5b63692f2d9a94c2767b12c80332`。

| artifact | SHA256 | Git blob ID |
|---|---|---|
| `eval/diagnostic_protocol_v5.json` | `6377c4288ac3b8399e4b108bf4ae87cb38f5b6235414a9095f445ae26daf57c2` | `f77305253a3d2319d44b332ce45eaad183137936` |
| `eval/materialization/e3notool_grpo_step_200.json` | `ea34d468f84ec72a10313bf61f85a0631f31d5429d1b4efa79565fa4aa41556c` | `eee3c3f7fce46cf9c4ded54da849ea09ddafe28c` |

完整闭包由 `eval/diagnostic_freeze_v5.sha256` 锁定；manifest 不自哈希，verification report 另行记录其 hash。
