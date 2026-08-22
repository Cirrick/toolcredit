# 交接状态

## 当前状态

- **M5 已完成 E6、E4-A、E4-B，处于阶段验收状态**（2026-08-22）。E7 exact implementation
  boundary 设计审查已经完成；用户明确将 E7 有意 defer 到 M6/E5 完成之后。E7 trainer/config/launcher
  未创建，smoke/full run 未启动；这不是实现失败、blocked 或取消。M5 不 tag。
- E6 `e6_nomask_20260820_235621` 完成 80/80：mask 操纵实际覆盖 4.662% loss token，固定 panel
  `0.61→0.71→0.74→0.68→0.70`（step 0/25/50/75/80），未触发 early stop，也未观察到灾难性退化。
- E4-A `e4a_exec_only_20260821_040632` 与 E4-B `e4b_joint_shaping_20260821_153100` 均完成
  200/200。A/B final pass@1 为 0.77/0.76，0–100 AUC 为 0.66625/0.68500；E3 为
  0.76/0.67625。A 明显增加工具调用和 hacking candidates；B 相对 A 小幅缓解这些启发式信号，
  但 penalty 暴露很弱，不能解释为稳定能力提升。
- A/B 各经历一次 JupyterHub 中断，分别从最后完整 checkpoint 125/150 恢复；旧重算区间和半写
  checkpoint 已归档，未混入正式曲线。E4-B 通过 smoke 后，其 21 GiB smoke checkpoint 已按用户
  指令删除，resolved config、轨迹审计、metrics、summary 和 cleanup ledger 保留。
- 最终分组回归为 **105 passed, 1 skipped**；pod 对 `/etc/hostname` 的拒绝由 EACCES 变成 EROFS，
  sandbox test 现接受两种内核错误，同时仍断言删除失败且文件存在。
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

## 下一步（M6）

1. 新会话按 `PLAN.md` 与新的 `plans/M6.md` 进入 E5 turn-level credit；开始前重新执行只读 preflight。
2. 不在 M6 开工前顺带实现 E7。E7 设计审查保存在 `plans/M5_E7_IMPLEMENTATION_REVIEW.md`，待 M6/E5
   完成后由用户决定是否恢复并重新专项批准。
3. 保持 E6、E4-A、E4-B 正式结果和历史产物不变，不重跑、不 tag M5。

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

## M5 证据指针

- 权威计划与偏差：`plans/M5.md`；E7 边界审查：`plans/M5_E7_IMPLEMENTATION_REVIEW.md`。
- 正式 runs：`rl/runs/e6_nomask_20260820_235621/`、
  `rl/runs/e4a_exec_only_20260821_040632/`、`rl/runs/e4b_joint_shaping_20260821_153100/`。
- 三方比较：`rl/runs/m5_e3_e4_comparison/e3_e4a_e4b_comparison.json` 与 `summary.md`。
- 汇总叙事：`README.md` 结果总表；`reports/technical_report.md` §7（动机、接线、结果、限制、
  面试叙事和证据索引）。
- 当前回归命令见各 test 文件；涉及真实 tokenizer/veRL import 的测试需在 pod 权限下使用
  `conda run --no-capture-output -n toolcredit pytest ...`。

## 风险与边界

- M4 验收是固定 MATH500-100、greedy、n=1；完整 MATH500/AIME 泛化与 bad-case 分析留给 M7。
- 最终验证 invalid rate 仍为 12%；parser 错误为 0，剩余主要是 verifier 无法判定或格式问题，
  后续实验必须保持同一口径并持续记录，不得静默忽略。
- M4 已验证 NFS 约 21 GB checkpoint 写入；JupyterHub 仍可能重启，长跑继续使用 conda
  `toolcredit` 环境中的 tmux、唯一 run name、完整 checkpoint 与低频 watcher。
- 当前磁盘余量约 170 GiB（96% used）；E7 启动前必须重新核对，不能假定足够容纳最多四批的 raw
  candidate evidence 与正式 checkpoints。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项），不阻塞服务器侧后续 Milestone。
- 不提交或改动 `reports/qa_log.md`、`.vscode/` 和 M3 未跟踪 checkpoint/tensorboard 产物。
