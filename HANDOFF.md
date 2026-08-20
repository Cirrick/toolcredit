# 交接状态

## 当前状态

- **M4 / E3 已完成并验收通过**（2026-08-20，tag `m4`）：正式 run
  `e3_grpo_baseline_20260819_224555` 完成 200/200 step，固定 MATH500-100 greedy pass@1
  **0.60→0.76**（峰值 0.77）。KL、entropy 与响应长度无病态；训练前/后 25-step 的工具
  错误率 12.84%→7.66%、格式无效率 10.30%→3.14%、截断率 6.30%→0.55%。
- 最终 checkpoint `global_step_200` 含 model/optimizer/scheduler/RNG/DataLoader 状态；
  200 份训练预测、9×100 条验证预测、resolved config、metrics、summary 与 TensorBoard 齐全。
  数据 schema/哈希/split/污染检查通过，真实 pod 权限下 **74 tests passed**。
- JupyterHub 中断恢复证据保留在 run 的
  `recovery/resume_from_150_20260820_100834/`：中断前 step 168 完成、169 未完成，从最新完整
  checkpoint 150 恢复并重算 151–168；旧预测已归档，未与恢复后正式曲线混用。
- **M3 验收通过**（2026-07-14，tag `m3`）；RL 统一起点继续是
  `sft/checkpoints/qwen3-1.7b-sft`。2026-08-19 的 30B teacher/full SFT 最小复核均未支持
  切换起点，产物与结论见 `plans/M3.md`。

## 下一步（M5，PLAN §9）——先计划、获批后开工

1. 为 M5 新建并提交 `plans/M5.md` 供用户批准；未获明确授权前不启动训练或批量评测。
2. E6 必做：复制 E3 config，只关闭工具返回 token mask，跑 50–80 step，记录 loss、复读工具
   输出与 eval 崩溃点；不得改动或覆盖 E3 run。
3. E4 必做：优先执行 λ_exec=0.2 shaping run，保持数据、SFT 起点、seed、步数和评测协议与 E3
   一致，关注早期收敛及“空刷执行成功”的 reward hacking。
4. E7 仅在时间富余且 veRL 有现成动态过滤开关时做；否则按 PLAN 如实跳过。

## M4 复现与证据指针

- 配置：`rl/configs/e3_grpo_baseline.yaml`；launcher：`rl/launch/e3_grpo_baseline.py`；
  后台入口：`scripts/m4/run_e3.sh`；监控：`rl/monitor_run.py`。
- 正式 run：`rl/runs/e3_grpo_baseline_20260819_224555/`；详细验收与执行偏差：`plans/M4.md`；
  技术叙事：`reports/technical_report.md` §6。
- M4 数据：源训练池 5203 条、有效训练池 5195 条、固定验证 100 条；manifest 位于
  `rl/data/manifest.json`，污染报告位于 `data/contamination_report_m4.md`。
- 核心验收命令：

  ```bash
  conda run --no-capture-output -n toolcredit pytest -q \
    env/test_sandbox.py rewards/test_rewards.py sft/test_trace_tokenizer.py \
    rl/custom/test_masking.py rl/custom/test_m4_adapters.py \
    rl/launch/test_e3_config.py rl/test_monitor_run.py rl/test_prepare_data.py
  ```

## 风险与边界

- M4 验收是固定 MATH500-100、greedy、n=1；完整 MATH500/AIME 泛化与 bad-case 分析留给 M7。
- 最终验证 invalid rate 仍为 12%；parser 错误为 0，剩余主要是 verifier 无法判定或格式问题，
  后续实验必须保持同一口径并持续记录，不得静默忽略。
- M4 已验证 NFS 约 21 GB checkpoint 写入；JupyterHub 仍可能重启，长跑继续使用 conda
  `toolcredit` 环境中的 tmux、唯一 run name、完整 checkpoint 与低频 watcher。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项），不阻塞服务器侧后续 Milestone。
- 不提交或改动 `reports/qa_log.md`、`.vscode/` 和 M3 未跟踪 checkpoint/tensorboard 产物。
