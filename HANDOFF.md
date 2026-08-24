# 交接状态

## 当前状态

- **M6 / E5 Tier A 已完整验收（2026-08-24）。** formal run
  `e5_turn_credit_20260823_082012`完成200/200；status/tracker=completed/200，step-200完整checkpoint可读。
  102,400条canonical train trajectory、363,350个turn、两个checkpoint-bound wrapper proof segment全部
  通过formula/span/mask/UID和restore检查；旧126–138及受控停止的旧126均留在recovery archive，没有混入。
- E5 fixed-100 curve为`0.61/0.69/0.73/0.73/0.74/0.75/0.71/0.70/0.76`；E3为
  `0.60/0.67/0.67/0.70/0.73/0.73/0.74/0.77/0.76`。E5 0–100 AUC `0.70625`比E3
  `0.67625`高3pt，但final同为0.76，0–200 AUC只高0.005625。
- step-200 paired结果为5 fixed、5 new、71 unchanged-correct、19 unchanged-failed；accuracy delta 0，
  10,000次paired bootstrap 95% CI=`[-0.06,0.06]`。conditional fix 5/24，new failure 5/76。
- mechanism audit覆盖全部rollout并固定抽30条mixed-quality trajectory。native `A_traj`同轨迹恒定；
  213,403个useful turn中99,996个获正correction，47,806个非no-tool问题turn中41,444个获负correction，
  方向违约为0。treatment确实生效，不是接线空开关。
- 结论为**mixed/negative**：early更快但final无增益，并出现显著over-calling。训练mean calls
  `0.963→2.551`、4-call fraction `2.71%→44.05%`、repeated code `2.02%→39.71%`、truncation
  `1.70%→3.42%`；error recovery `41.17%→40.98%`基本不变。fixed panel E5有93/100条4-call tag，
  E3仅2/100。
- canonical smoke、人类adoption gate（14 TP/1 FP，precision 0.9333）、freeze v2 closure
  `6fa185e1…b07f3f0`、formal completion、30条mechanism audit、checkpoint-blinded 48 failure裁决、
  paired matrix/secondary transitions/10k bootstrap和M6文档全部齐全。Codex taxonomy裁决没有冒充human audit。
- M6收尾时first-party全量回归为**150 passed, 1 skipped**；freeze v2语义验证和37-file
  `sha256sum -c`再次通过，analysis JSON可解析且30/48/100行审计产物数量吻合。
- 按冻结协议没有调β、改heuristic、追加seed、实施Tier B/GiGPO、启动E7或运行M7 generation/evaluation。
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

## 下一步（需用户另行授权）

1. 当前停在M6完成边界。不要自动启动M7、E7、Tier B、额外seed或β sweep。
2. 若用户选择M7，先基于frozen v2语义version evaluator/checkpoint locator，把E5 role绑定到
   `rl/runs/e5_turn_credit_20260823_082012/checkpoints/global_step_200`并做role/structure/identity gate；然后
   才能对raw/SFT/E3/E5统一运行760题评测，不能复用本次旧validation拼接四阶段主表。
3. 若用户选择恢复E7，先重新review `plans/M5_E7_IMPLEMENTATION_REVIEW.md` 的pin源码、exact boundary、
   compute/storage和专项授权；M6完成不自动批准E7。
4. 保持E3/E4/E5/E6正式产物、M6 freeze与recovery archives不变；不删除checkpoint腾空间。

## M6 复现与证据指针

- formal run：`rl/runs/e5_turn_credit_20260823_082012/`；计划、偏差与验收：`plans/M6.md`。
- 只读重验：`python -m rl.validate_e5_formal_completion <run_dir>`；分析：`python -m rl.finalize_m6`。
- 主结论：`reports/02_main_results.md`；paired taxonomy：`reports/03_badcase_taxonomy.md`；技术叙事：
  `reports/technical_report.md` §8。
- 机器证据：run内`analysis/formal_completion_gate.json`、`m6_e5_completion_manifest.json`、
  `e3_e5_performance_behavior.json`、`mechanism_analysis.json`、`mixed_quality_mechanism_audit.jsonl`、
  `paired_failure_transition_analysis.json`与`paired_failure_transitions.jsonl`。
- failure labels：`rl/m6_failure_adjudications.jsonl`；这是checkpoint-blinded Codex manual adjudication，
  不算独立human audit。smoke adoption gate才是用户确认的人类审计。

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

- M6结论仍是固定MATH500-100、greedy、n=1、单seed；完整MATH500/AIME/GSM8K四阶段泛化留给M7。
  任何category分母≤6都标记evidence insufficient，不能外推为某类已解决。
- E5 over-calling是主要新风险：position-aligned final/no-tool比较和visible-text adoption可能奖励重复调用/复述。
  未来方案若修改此语义必须独立version和全套对照，不能回写M6 frozen result。
- M4 已验证 NFS 约 21 GB checkpoint 写入；JupyterHub 仍可能重启，长跑继续使用 conda
  `toolcredit` 环境中的 tmux、唯一 run name、完整 checkpoint 与低频 watcher。
- M6 completion gate时磁盘约189 GiB free；未来任务仍须按`remaining_peak_write + 30 GiB`现场重算，
  不能依赖该快照，也不能自行删除历史正式证据。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项），不阻塞服务器侧后续 Milestone。
- 不提交或改动 `reports/qa_log.md`、`.vscode/` 和 M3 未跟踪 checkpoint/tensorboard 产物。
