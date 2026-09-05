# 简历 bullets（2026-09-05 更新，数字均来自 canonical 产物）

> 证据指针：M8 `eval/runs/m8_e5v2_eval_20260905_134737/metrics/m8_headline.json`；M7 `eval/runs/m7_unified_eval_20260824_201032/metrics/`、`transitions/bootstrap.json`；
> M6 `rl/runs/e5_turn_credit_20260823_082012/analysis/`；M4 `rl/runs/e3_grpo_baseline_20260819_224555/`；
> M2 `reports/02_appendix_verifier_audit.md`；M5 `reports/04_reward_hacking.md`。
> 待办：如仓库公开，在项目标题行补 GitHub 链接；模型是 **Qwen3-1.7B**（不是 PLAN §14 模板里的 Qwen2.5-3B）。

## 中文版（4 条，可按岗位删到 3 条）

**多轮工具调用 RL 中的信用分配研究（个人项目，veRL + GRPO）** 2026.07–2026.09（M8 完成于 2026-09-05）

- 基于 veRL 0.8.0 + SGLang 在 GH200/aarch64 单卡上搭建“数学推理 + Python 沙箱”多轮工具调用 RL 全栈：
  netns 隔离沙箱、200 例人工审计的 math-verify 验证器（等价判定 0 假阳性）、针对真实 veRL 状态机的
  loss-mask 单测、教师蒸馏 + 五条件拒绝采样 SFT 冷启动（6k 轨迹）；标准 GRPO 200 step 将 Qwen3-1.7B 的
  MATH500 greedy pass@1 从 0.594 提升到 0.766，SFT→RL 在 760 题冻结评测集上 +12.8pt（95% CI [9.7, 15.9]）。
- 以零侵入 wrapper（不 fork 框架）实现轮级信用分配 `A_turn = A_traj + β(s_t − s̄_t)`，对 10.2 万条训练轨迹、
  36 万个 turn 做机制审计确认修正方向零违约；发现其早期收敛更快（0–100 step AUC +3pt）但终点无增益
  （−1.05pt，CI 跨 0），且诱发工具过度调用（4 次调用占比 1.3%→92.9%）；定位根因为修正项非零和、
  等价于隐式稠密奖励，并在约半数组内零方差 group 中成为唯一梯度来源；随后用最小反事实（只把修正项改为
  轨迹内 token 加权零和并门控零方差组）验证：过度调用完全消失（4-call 0.8%、截断 100%→8.7%），
  终点 .741 vs .733（CI 跨 0），证明行为漂移来自非守恒结构而非轮级信用本身。
- 完成 reward shaping / no-mask / 预算诊断三组消融：人工标注出执行奖励下的常量执行、重复失败调用、
  忽略工具证据三类 reward exploitation 案例；证明工具 token 未 mask 时 4.7% loss token 来自环境但 80 step
  内未崩溃；对 70 条截断失败做长预算反事实重跑，仅 16 条转正、30/38 为语言循环，否定“加预算”方向。
- 预注册并 hash 冻结评测协议：4 个 checkpoint × 760 题 × (greedy + 4 采样) 共 15,200 条轨迹全量落盘，
  逐题配对 failure-transition 矩阵 + 1 万次分层 bootstrap 置信区间；训练 run 五件套、中断恢复 ledger、
  205 项测试保证可复现。

## English version

**Credit Assignment in Multi-Turn Tool-Use RL (personal project, veRL + GRPO)** Jul–Sep 2026

- Built an end-to-end multi-turn tool-integrated RL stack (math reasoning + sandboxed Python) on veRL 0.8.0 /
  SGLang on a single GH200 (aarch64): netns-isolated sandbox, math-verify verifier audited on 200 cases
  (0 false positives), loss-mask unit tests driving the real veRL agent-loop state machine, and a
  distillation + rejection-sampling SFT cold start (6k trajectories). Standard GRPO (200 steps) raised
  Qwen3-1.7B MATH500 greedy pass@1 from 0.594 to 0.766; SFT→RL gain on a frozen 760-question panel was
  +12.8pt (95% CI [9.7, 15.9]).
- Implemented turn-level credit assignment `A_turn = A_traj + β(s_t − s̄_t)` as a non-invasive wrapper
  (no framework fork) and audited it on 102k trajectories / 363k turns (zero sign violations). Found faster
  early learning (+3pt AUC over steps 0–100) but no endpoint gain (−1.05pt, CI crossing 0) and severe
  tool over-calling (4-call rate 1.3%→92.9%); traced the root cause to a non-zero-sum correction that acts as
  an implicit dense reward and is the sole gradient in the ~45% of GRPO groups with zero outcome variance;
  confirmed it with a minimal counterfactual (token-weighted zero-sum redistribution within each trajectory,
  gated on group variance): over-calling vanished (4-call 0.8%, truncation 100%→8.7%) with endpoint accuracy
  .741 vs .733 (CI crossing 0), isolating the non-conservation as the cause.
- Ran reward-shaping, no-mask, and budget ablations: hand-labeled three reward-exploitation patterns under an
  execution bonus (constant execution, repeated failing calls, ignored tool evidence); showed 4.7% of loss
  tokens come from environment text without masking yet no collapse within 80 steps; a longer-budget
  counterfactual rerun of 70 truncated failures fixed only 16 and exposed verbal loops in 30/38.
- Pre-registered, hash-frozen evaluation: 4 checkpoints × 760 questions × (greedy + 4 samples) = 15,200
  trajectories, paired failure-transition matrices, and 10k stratified bootstrap CIs; every training run
  ships resolved config, checkpoints, predictions, metrics, recovery ledgers, and 205 tests.
