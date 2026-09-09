# M10 执行会话启动 prompt（2026-09-09 生成，复制整段给新 session）

---

你在 `/home/jovyan/toolcredit/` 执行 **M10 / E7：DAPO 式零方差 group 动态过滤与补采样**。先读 `CLAUDE.md`，再按顺序读：
`plans/M10.md`（草案 v2，**已批准，原文不得追改**，偏差写 §12、结果写 §13）、
`plans/M10_STEP0_BOUNDARY.md`（exact boundary，**已批准**，§3 的 8 处白名单、§4 方案 (b)、§5 resume 语义、§8 决定）、
`plans/M5_E7_IMPLEMENTATION_REVIEW.md`、`HANDOFF.md` 2026-09-09 三条、`reports/qa_log.md` Q17/Q22、
`rl/launch/e5_turn_credit.py:334–440`（驱动边界先例）、`rl/launch/e3notool_grpo.py`（config diff gate 先例）、
`rl/launch/e3_offload_check.py`（offload 三条 diff 先例）、`rl/custom/CHANGES.md`。

执行 `plans/M10.md` §8 步骤 1–6，步骤 0 已完成。硬约束：

1. 不改 veRL site-packages、不 fork；所有新代码在 `rl/custom/`、`rl/launch/`、`rl/configs/`、`scripts/m10/`、`analysis/`、`eval/`；
   同步更新 `rl/custom/CHANGES.md`（方法、行号、pin SHA、行为差异、回退路径）。
2. 开工前复核 pin SHA：`ray_trainer.py` = `de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66`，
   `main_ppo.py` = `e3fe6e73b18d63367402a2570c5ba054a2b05443ac40a168524dbf545b1da392`；不符即停并报告。
3. `fit()` 复制 pin `:1362–1770` 全部 409 行，只允许 STEP0 §3 的 8 处白名单差异，每处以 `# E7-W<n>` 标记；Fixture R 逐行 diff 证明。
4. 驱动边界用 STEP0 §4 方案 (b)：驱动进程内临时把 `verl.trainer.main_ppo.RayPPOTrainer` 绑定到 `DynamicFilterRayPPOTrainer`，
   `try/finally` 恢复，proof 文件 + 启动打印 `TOOLCREDIT_E7_TRAINER_ACTIVE`。
5. config diff gate：E3 → E7 只允许 `algorithm.filter_groups.{enable,metric,max_num_gen_batches}`、trainer 标记、
   `project_name`/`experiment_name`、以及 `actor.fsdp_config.{param_offload,optimizer_offload}=false`、`ref.fsdp_config.param_offload=false`。
6. 零方差按标量 `score`（`rm_scores.sum(-1)`，答案 + 0.1 格式）判，不按 acc；每 chunk 64 题×8；最多 4 chunk；surplus 不跨 snapshot 回收；
   underfilled 显式失败；补采样期间不 update actor、不 sync 权重、replicas 保持 awake，`sleep_replicas()` 只在 finally 一次。
7. 等算力 checkpoint：累计轨迹数首次 ≥ 102,400 时额外保存到 `checkpoints/equal_compute_step_<N>/`，不参与轮换、不写 tracker。
8. formal 分 4 段（`--stop-at-step` 50/100/150/200），每段 tmux 后台 + watcher + memwatch v2（60 s，按进程 `Pss_Anon` 增量，修复 `step` 字段）；
   anon > 110 GiB 提前分段；第一段后按斜率决定是否放宽段长（记 §12）。
9. 长任务一律 tmux，日志落盘到 `rl/runs/`；会话内只提交任务与轮询；`conda` 在本 shell 里可能是坏函数，用 `/opt/conda/bin/conda run --no-capture-output -n toolcredit ...`。
10. 每个门禁（Fixture 全绿、smoke、每段结束、评测、判读）向用户报告后再进入下一步；硬停点见 `plans/M10.md` §8。
11. 评测协议 bump 到 v6：新增 `e7_dynamic_filter_step_200` 与 `e7_equal_compute` 两个角色，7,600 条，其余复用 canonical；
    配对 E3→E7 两组 + bootstrap；判读按 §9 四格（Δ_step × Δ_compute）。
12. 完成后：`reports/02_main_results.md` M10 节、`reports/technical_report.md`、README 总表、LOG/HANDOFF、resume、qa_log（含泄漏进程诊断）；
    commit + tag `m10`。

已知环境事实：GH200 142 GB 单卡；容器 cgroup `memory.max`=128 GiB 且 `oom.group=1`（超限整 pod 被杀，无 Traceback）；
E3 单步约 161 s（gen 55 s）；E3 informative 比例早期 .5593、后期 .4725；E3 总轨迹 102,400 = 200×64×8；
E3 run `rl/runs/e3_grpo_baseline_20260819_224555`；SFT 起点 `sft/checkpoints/qwen3-1.7b-sft`；
offload-off 验证 `rl/runs/e3_offload_check_20260909_150849`（GPU max_allocated 43.5 GB，anon 约 +0.5 GB/step）。

先做的第一件事：`git status`，把 M10 相关的未提交文件（`plans/M10.md`、`plans/M10_STEP0_BOUNDARY.md`、`plans/M10_KICKOFF_PROMPT.md`、
`rl/launch/e3_offload_check.py`、`scripts/infra/run_offload_check.sh`、`analysis/offload_check_compare.py`、`HANDOFF.md`、`LOG.md`、
`reports/qa_log.md`）作为 `docs: M10 plan v2, step-0 boundary, offload-off check` 提交；**不要**提交 `PLAN.md`、`Untitled.ipynb`、`.vscode/`、
`analysis/tool_use_conditional_accuracy.*`、`reports/05_data_qa.md`、`today_research_workflow.md`（用户的其他改动）。然后进入步骤 1。
