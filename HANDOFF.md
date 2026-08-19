# 交接状态

## 当前状态

- **M3 验收通过**（2026-07-14，tag `m3`；格式成功率与 pass@1 提升两项边缘指标经用户确认接受）。
- **RL 统一起点：`sft/checkpoints/qwen3-1.7b-sft`**（merged 完整 ckpt，verl `model.path` 直接用）。
- SFT 数据：`sft/data/sft_traces.jsonl`（6056 条，五条件拒绝采样，含 797 条报错恢复）；
  held-out：`sft/data/heldout_200.jsonl`（未参与蒸馏）。
- 关键机制：SFT 样本经**真实 ToolAgentLoop 重放**构造（`sft/trace_tokenizer.py`），与 M4
  rollout tokenization 逐 token 同构。
- SFT 后画像（500 题，temp0.6 n=4）：TIR 0.704 / 增益 −0.032（L4–5 平价/转正，负增益在
  不入池的 L1–3）/ 报错率 19% / 弃用率 7% / 组内混合结果 L3–5 = 0.39/0.54/0.48。
- 教师覆盖率缺口：L5 41% 题目无合格轨迹（拒绝采样难度偏置），交给 on-policy RL。
- 2026-08-19 最小复核已完成：30B teacher 的 yield/L5 覆盖未胜 8B（52%/63% vs
  53%/65%）；full SFT held-out 的 TIR/CoT 为 0.5188/0.5900，但工具错误率/弃用率
  0.2654/0.1150 均差于 LoRA 的 0.2013/0.1050。两项均拒绝切换，服务已停止；详见
  `plans/M3.md` 与 `sft/experiments/m3_minimal/`。

## 下一步（M4，PLAN §8）——计划待用户批准

1. `rl/configs/e3_grpo_baseline.yaml` + `rl/launch/`：字段以 PLAN §8.1 骨架为基准、对照
   M0 官方示例校正（sglang async + tool_agent + 我们的 sandbox 工具 + rewards/composite）；
   数据 = train_subset.jsonl 除 held-out 外的 5203 条转 verl parquet 格式。
2. 监控面板（PLAN §8.2 十项指标）+ 异常处置手册进 README。
3. 200 step，save/test_freq 25（MATH500 子集）；tmux 长跑 + 日志落盘；预算 4 天含 2–3 次重调。
4. 决策点：reward 用 rewards/composite_reward 注入 verl 的方式（custom reward fn 路径）
   需按禁止事项 #1 最小侵入并记 CHANGES.md。

## 记账

- M0 遗留：DataLoader worker 收尾阶段被杀（自愈）——**M4 长跑重点监控**；NFS ckpt IO 未实测。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项）。
- 工具行为注意：M1 的 AST auto-print 在 `env/sandbox.py:prepare_tool_code`（单一来源）；
  M4 的训练工具应复用它而非官方 SandboxTool 的行版启发式。
- verifier 存疑项（单位省略等价按对）用户未表异议，维持现状。
- qa_log 持续追加（约定 #8）；技术报告每 Milestone 更新（约定 #5）。
