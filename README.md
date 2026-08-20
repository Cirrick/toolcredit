# ToolCredit：多轮工具调用 RL 中的信用分配

**研究问题**：在多轮工具调用（TIR）的 RL 训练中，把 outcome 奖励以轨迹级优势广播到全部轮次
（标准 GRPO），与引入轮级（turn-level）信用分配相比，对训练效率、最终性能和模型行为的影响。

完整实施文档见 [PLAN.md](PLAN.md)；项目约定见 [CLAUDE.md](CLAUDE.md)；环境记录见
[environment.md](environment.md)；**逐 Milestone 的动机/方法/结果叙事见
[reports/technical_report.md](reports/technical_report.md)**（living document，面试复盘入口）。

## 非目标（防 scope creep）

- 不追求 SOTA 分数，不与论文数字对齐。
- 不声明 novelty，不写 related work 综述。
- 不做分布式多机训练，不深度 fork veRL。
- 不做 VLM、不做 web agent、不做 SWE 任务。

## 结果总表

| Milestone | 状态 | 结果 |
|---|---|---|
| M0 环境与 smoke test | ✅ 服务器侧完成（2026-07-12） | veRL 0.8.0 官方 agent-loop 示例原样跑通（Qwen3-1.7B+MATH+沙箱，val acc 0.76，5/5 GRPO 步）；smoke test（20 条+1 梯度步）约 2 分钟 < 10 分钟预算。5090 本地侧待用户复跑 |
| M1 数据与工具增益预实验 | ✅ 完成（2026-07-13） | 训练/评测数据五件套 + 污染检查（剔 216 条）；probe（Qwen3-1.7B，500 题分层，CoT vs TIR）：**零样本工具增益全 level 为负（总 −10pt），机制分解显示工具成功执行时 L3–5 增益 +8~+23pt**——SFT 冷启动必要性的定量证据；训练子集 = L3–5 共 5403 条（详见 [reports/01_tool_gain.md](reports/01_tool_gain.md)） |
| M2 沙箱/verifier/masking | ✅ 完成（2026-07-14） | **56 tests 全绿**：沙箱加固（真禁网 netns + 资源限制 + 恶意 payload 全过）；verifier 双口径 + 200 例人工审计（**数学等价判定假阳性/语义假阴性均 0 检出；严格 boxed 口径另有 4/200 格式漏分**，顺带修复 M1 判分 2 处假阴性）；verl ToolAgentLoop mask 构建验证无误（详见 [reports/02_appendix_verifier_audit.md](reports/02_appendix_verifier_audit.md)） |
| M3 SFT 冷启动 | ✅ 完成（2026-07-14；2026-08-19 最小复核） | Qwen3-8B 本地蒸馏 + 五条件拒绝采样 **6056 条**（含 797 条报错恢复）；工具报错率 34%→19%、弃用率 27%→7%、**增益 −0.101→−0.032**。30B teacher 未提高 yield/L5 覆盖；full SFT 仅小幅提高 held-out pass@1 且工具错误率恶化 6.4pt，均拒绝切换；LoRA SFT-6k 仍为全部 RL 实验统一起点（详见 [plans/M3.md](plans/M3.md)） |
| M4 E3 GRPO baseline | ✅ 完成（2026-08-20） | 标准轨迹级 GRPO 完成 200/200 step；固定 MATH500-100 greedy pass@1 **0.60→0.76**（峰值 0.77），KL、entropy 与长度曲线健康；训练截断、工具错误和格式无效率均下降。五件套、step-200 完整 checkpoint 与 74 项测试验收通过（详见 [plans/M4.md](plans/M4.md)） |

## 复现

M4 数据与配置预检：

```bash
conda run --no-capture-output -n toolcredit python -m rl.prepare_data
conda run --no-capture-output -n toolcredit python -m rl.launch.e3_grpo_baseline \
  --run-name e3_config_check --dry-run
```

M4 长任务必须在 tmux 中运行；脚本自动生成唯一 run name，也可显式传入：

```bash
conda run --no-capture-output -n toolcredit tmux new-session -d -s m4-e3-smoke \
  'cd /home/jovyan/toolcredit && bash scripts/m4/run_e3.sh smoke'
conda run --no-capture-output -n toolcredit tmux new-session -d -s m4-e3 \
  'cd /home/jovyan/toolcredit && bash scripts/m4/run_e3.sh full'
```

失败 run 仅可从已有完整 veRL checkpoint 且 resolved config 完全一致时恢复。恢复入口会追加日志，
把 checkpoint 后将被重算的旧训练预测保存到 run 内 `recovery/`，并跳过 checkpoint step 已完成的
启动验证：

```bash
conda run --no-capture-output -n toolcredit tmux new-session -d -s m4-e3-resume \
  'cd /home/jovyan/toolcredit && RUN_NAME=<failed_run_name> RESUME=1 bash scripts/m4/run_e3.sh full'
```

按需做一次轻量监控汇总（不常驻轮询）：

```bash
conda run --no-capture-output -n toolcredit python -m rl.monitor_run \
  --run-dir rl/runs/<run_name>
```

正式长跑可另开一个稀疏 watcher；它每 10 分钟只检查状态和验证文件名，仅在出现新的验证 step
或终态时刷新汇总，不轮询 tmux/GPU/完整日志：

```bash
conda run --no-capture-output -n toolcredit tmux new-session -d -s m4-e3-monitor \
  'cd /home/jovyan/toolcredit && bash scripts/m4/watch_e3.sh <run_name> 600'
```

## M4 训练稳定性监控与异常处置

固定面板包含：`reward/mean`、`pass@1(eval)`、`actor/ppo_kl`、`actor/entropy`、
`response_length/mean`、`n_tool_calls/mean`、`tool_error_rate`、`invalid_format_rate`、
`tool_parse_error_rate`、`group_all_correct_frac`、`group_all_wrong_frac`。veRL 原生指标写入 TensorBoard；轨迹级辅助指标
由 `rl.monitor_run` 从逐 step JSONL 计算并写入同一 run 的 `m4/*` 面板与 `metrics.json`。

- KL 快速上飙且 eval 下降：先保留问题 run，核对 reward/轨迹后升 `kl_loss_coef` 或降学习率；
- entropy 单调降到 0.3 以下：核对 clip-higher 是否生效，再考虑提高 rollout temperature；
- reward 上升但 pass@1 不动：立即抽查已落盘轨迹，优先排查 verifier/格式套利；
- response length 或截断率上升：核对四次工具预算、最终答案和 3072-token 边界；
- DataLoader worker 被杀：保留日志并检查宿主内存；若持续发生，将 worker 数降级并记录偏差；
- NFS checkpoint 写入异常：不覆盖已有恢复点，保留失败状态后从最近完整 step 恢复。
