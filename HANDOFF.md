# 交接状态

## 当前状态

- **2026-08-28完成E3 70条budget/truncation post-hoc诊断，不训练。** 原70条按互斥机制拆为
  response-token 58、assistant/tool-turn 2、repeated normalized code 10（正交停止信号仍为63 token、7 turn）。
  唯一长预算run `eval/runs/e3_budget_diagnostic_20260828_095658/`仅把3072/4/5提高到6144/6/7，其他
  prompt/model/greedy/seed/tool/verifier保持不变；16/70转正确，54仍错。54中只有14条正常结束后答错，38条
  再次撞token上限、2条再次撞turn上限；38条中34条零工具、30条有至少三次完全相同的非短输出行，主模式是
  verbal loop而非预算略短。诊断题永久禁止回灌训练；未来训练数据只能来自train split并重新污染检查。详细方法、
  边界和证据见`plans/M7_BUDGET_DIAGNOSTIC.md`；diagnostic与canonical两套hash ledger均通过。
- **M7 canonical generation、strict scoring、downstream analysis与documentation synthesis已完成并由用户
  最终验收（2026-08-25）；completion commit与tag `m7`已获授权。** 唯一run
  `eval/runs/m7_unified_eval_20260824_201032/`在frozen v3下完成greedy 3,040、sampled 12,160，raw/scored
  均为exact 15,200/15,200；missing/unexpected/duplicate/conflict/infra failure均为0。full exact-key digest为
  `c43ce719…fa8b8`，3,800个跨role pairing group与15,200个UID完整。canonical `hashes.sha256`及17项
  downstream `analysis_hashes.sha256`均通过。
- M7 FULL760 greedy raw/SFT/E3/E5为426/460/557/549 correct，即`.561/.605/.733/.722`。paired
  Raw→SFT为+4.47pt、95% CI `[+1.05,+7.89]`；SFT→E3为+12.76pt `[+9.74,+15.92]`；E3→E5为
  −1.05pt `[−3.68,+1.58]`。sampled matched-index结论同向。**E3是最强endpoint；E5无final accuracy
  benefit。** M7 endpoint-only不能验证M6 faster-early-learning。
- E5有强烈跨source behavioral treatment effect：E3→E5 mean calls greedy `.888→3.755`、sampled
  `.958→3.935`；4-call `1.3%→92.9%`与`1.2%→97.2%`，四source均出现，per-call success约升至98%。
  定性为credit-signal/proxy gaming，不自动等同classic reward hacking；E4是更干净的reward-shaping
  exploitation案例。
- frozen taxonomy覆盖15,200行（10,085 correct、5,115 wrong、0 unclassified）；192条stable-hash
  checkpoint-blinded manual Codex audit不是独立human audit，总agreement 151/192。E5 agreement仅greedy
  6/24、sampled 3/24，因此**不得科学解释E5 coarse taxonomy migration**；exact outcome/behavior/four-cell仍有效。
  三段×两setting共11,400 paired rows与10,000次source-stratified question-cluster bootstrap齐全。
- required docs已更新：`reports/02_main_results.md`、`03_badcase_taxonomy.md`、新建
  `04_reward_hacking.md`与`interview_outline.md`，以及README、technical report、M7 plan、LOG/HANDOFF。
  canonical `summary.md`因受`hashes.sha256`保护保持不变，新增`final_summary.md`记录下游结论。
- behavioral-identity fixture通过真实veRL state machine证明M7 metadata-only loop与`ToolCreditAgentLoop`的
  prompt/response IDs、mask、turn、metrics和baseline extras完全相同，唯一新增`turn_credit_ledger` audit
  metadata。greedy paired transitions为`primary_diagnostic`；sampled matched-index为
  `secondary_exploratory`并禁止强因果解释。
- E3/E5 step-200已用pin veRL 0.8.0官方FSDP merger物化到`eval/checkpoints/`下两个唯一HF inference路径；
  每个export为11文件、310 tensors、1,720,574,976参数。外部identity manifests分别为
  `eval/materialization/e3_grpo_baseline_step_200.json`（`b543647a…95db1`）和
  `eval/materialization/e5_turn_credit_step_200.json`（`78441f51…65ae3`）。
- v3 runtime freeze状态为`FROZEN_V3_SMOKE_PASSED_AWAITING_CANONICAL_REVIEW`：66-file manifest
  `b765ec2c…5b05a4`、protocol `c461d22b…36212`。semantic verifier证明parent v2 37-file manifest
  `6fa185e1…07f3f0`及panel/generation/tool/verifier/taxonomy/pairing/bootstrap/data-isolation语义未变，
  `gpu_generation_performed=false`（freeze时事实）。smoke完整保留resolved bindings、materialization manifests、32+32 shards、
  strict scores、server/driver logs、exact closure、descriptive-only metrics与hash ledger。E5在四题smoke中20/20均到达
  assistant-turn budget，与M6 over-calling风险一致，但不得从四题外推prevalence或performance effect。
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

1. M7 canonical仍冻结；70题budget diagnostic只作post-hoc诊断，不改变M7 denominator，也不进入训练数据。
2. E7、Tier B、额外seed、β sweep、post-hoc tuning、新训练和继续提高诊断预算仍禁止。
3. 若用户未来选择恢复E7，先重新review `plans/M5_E7_IMPLEMENTATION_REVIEW.md` 的pin源码、exact boundary、
   compute/storage和专项授权；M6完成不自动批准E7。
4. 保持E3/E4/E5/E6正式产物、M6 freeze与recovery archives不变；不删除checkpoint腾空间。

## M7 复现与证据指针

- canonical run：`eval/runs/m7_unified_eval_20260824_201032/`；计划、执行偏差与验收：`plans/M7.md`。
- canonical closure：`metrics/completeness.json`、`metrics/full_scored_completeness.json`、`hashes.sha256`。
- performance/behavior：`metrics/pass_at_k.json`、`metrics/tool_behavior.json`。
- taxonomy/audit：`taxonomy/coarse_metrics.json`、`taxonomy/audit_metrics.json`、`taxonomy/adjudications.jsonl`。
- paired/bootstrap：`transitions/{raw_to_sft,sft_to_e3,e3_to_e5}.json`、`paired_rows.jsonl`、
  `bootstrap.json`；downstream integrity：`analysis_hashes.sha256`。
- synthesis：`final_summary.md`、`reports/02_main_results.md`、`reports/03_badcase_taxonomy.md`、
  `reports/04_reward_hacking.md`与`reports/interview_outline.md`。
- 只读重验：run目录执行`sha256sum --quiet -c hashes.sha256`及
  `sha256sum --quiet -c analysis_hashes.sha256`；分析单测使用conda `toolcredit`环境运行
  `python -m pytest -q -p no:cacheprovider analysis/test_m7_analysis.py`。

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

- M7已把M6的over-calling/no-endpoint-gain扩展到完整MATH500/AIME/GSM8K与两setting，但仍是单seed、
  fixed β和endpoint-only；不能用M7声称faster-early-learning泛化。E5 coarse taxonomy因blind agreement低不得
  作category migration科学解释，任何小分母category也不能外推为某类已解决。
- E5 over-calling仍是主要风险：position-aligned final/no-tool比较和visible-text adoption可能奖励重复调用/复述。
  未来方案若修改此语义必须独立version和全套对照，不能回写M6/M7 frozen result。
- M4 已验证 NFS 约 21 GB checkpoint 写入；JupyterHub 仍可能重启，长跑继续使用 conda
  `toolcredit` 环境中的 tmux、唯一 run name、完整 checkpoint 与低频 watcher。
- M6 completion gate时磁盘约189 GiB free；未来任务仍须按`remaining_peak_write + 30 GiB`现场重算，
  不能依赖该快照，也不能自行删除历史正式证据。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项），不阻塞服务器侧后续 Milestone。
- 不提交或改动 `reports/qa_log.md`、`.vscode/` 和 M3 未跟踪 checkpoint/tensorboard 产物。
