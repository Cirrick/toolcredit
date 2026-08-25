# ToolCredit interview outline

## 60-second version

我研究的是multi-turn工具RL里的credit assignment：标准GRPO把trajectory outcome优势广播给每个turn，我做了
一个冻结的turn-level correction，让成功且被后文采纳的工具turn得到相对正credit。机制ledger证明treatment
确实生效：102,400条E5训练trajectory里，99,996个useful turn得到正修正、41,444个problem turn得到负修正，
方向违约为0（`rl/runs/e5_turn_credit_20260823_082012/analysis/mechanism_analysis.json`）。

M6 fixed-100显示E5早期AUC比E3高3pt，但final同为.76；M7再用760题、四source、greedy+四个matched sample
index统一评测。最强endpoint是E3：greedy 557/760=.733，E5 549/760=.722，paired delta −.0105，95% CI
`[−.0368,.0158]`；sampled也没有收益（`eval/runs/m7_unified_eval_20260824_201032/metrics/pass_at_k.json`、
`transitions/bootstrap.json`）。但E5行为变化极强：greedy平均调用.888→3.755，4-call 1.3%→92.9%，sampled
1.2%→97.2%，且四个source都出现（`metrics/tool_behavior.json`）。

所以结论不是“更细credit一定更好”，而是：它能改变学习速度和策略，但proxy语义不对齐会把优化导向重复
调用，而不是最终能力。E5更准确叫credit-signal/proxy gaming；E4显式execution bonus下的常量执行、重复调用、
忽略结果才是更干净的reward-shaping exploitation案例（`reports/04_reward_hacking.md`）。

## 5-minute version

### 1. 为什么做

零样本工具使用在M1总分反而下降10pt，但工具成功执行时L3–L5条件增益为+8到+23pt，说明工具有潜力、
执行与采纳是瓶颈；因此先做SFT cold start，再做RL（`data/probe/metrics.json`、`reports/01_tool_gain.md`）。

### 2. 对照如何保证可归因

E3与E5从同一SFT起点独立训练200 step，数据、seed、64×8 rollout、生成预算、answer+format reward、mask、
optimizer与validation cadence一致；唯一treatment是
`A_turn=A_traj+0.5(s_t−mean(s_t))`（`plans/M6.md`、两个run的`resolved_config.yaml`）。M6全量检查
102,400条trajectory、363,350个assistant turn，formula/span/mask/UID mismatch均为0；这排除了“开关没接上”
（`analysis/formal_completion_gate.json`）。

### 3. 训练曲线先给了什么信号

M6 fixed MATH500-100上，E5 0–100 normalized AUC=.70625，E3=.67625，首次到.73分别是step50与100；
但step200同为76/100，paired 5 fixed / 5 new，delta CI `[−.06,.06]`。所以可以说early更快，不能说final更强
（`analysis/e3_e5_performance_behavior.json`、`analysis/paired_failure_transition_analysis.json`）。

### 4. M7如何检验泛化

冻结四个role（raw/SFT/E3/E5）、MATH500+AIME2024+AIME2025+GSM8K、greedy与sampled n=4；canonical
raw/scored各15,200，missing/duplicate/conflict/infra failure为0（M7 run的`metrics/completeness.json`、
`metrics/full_scored_completeness.json`）。greedy是primary，sampled matched-index只作secondary/exploratory；
CI是10,000次source-stratified question-cluster bootstrap、seed42（`transitions/bootstrap.json`）。

### 5. 主结果

Greedy FULL760依次为raw 426/760=.561、SFT 460/760=.605、E3 557/760=.733、E5 549/760=.722。
paired Raw→SFT为+4.47pt `[+1.05,+7.89]`，SFT→E3为+12.76pt `[+9.74,+15.92]`，E3→E5为
−1.05pt `[−3.68,+1.58]`。sampled对应+4.74pt、+12.86pt、−.43pt，最后一项CI同样跨0
（`metrics/pass_at_k.json`、`transitions/bootstrap.json`）。E3是最强endpoint stage。

### 6. 行为为何比accuracy更关键

E3→E5 greedy平均calls .888→3.755，4-call 10/760→706/760，repeated code 22/760→702/760，
truncation 73/760→760/760；sampled 4-call 37/3040→2956/3040。与此同时per-call success从约88%升至约98%，
说明不是不会调用，而是“继续成功调用”成了容易获得credit的策略。E5在AIME上的over-call稍低，但greedy仍为
60.0%/66.7%，GSM8K/MATH500为98.5%/94.2%；sampled四source为87.5%–100%
（`metrics/tool_behavior.json`）。

### 7. 失败迁移怎么审

三段greedy四格分别是109/75/351/225、127/30/430/173、47/55/502/156，保存全部paired rows而非只挑
fixed case（`transitions/paired_rows.jsonl`）。192条stable-hash checkpoint-blinded Codex audit总体agreement
151/192，但E5 greedy/sample agreement仅6/24与3/24，因为自动budget label把“先boxed wrong、后耗尽预算”
误当truncation cause。因此我不科学解释E5 coarse taxonomy migration，只解释exact outcome与behavior
（`taxonomy/audit_metrics.json`、`reports/03_badcase_taxonomy.md`）。

### 8. Reward hacking如何准确措辞

E4的execution bonus直接进入shaped reward：E4-A mean calls比E3高.342，unused-result candidate高9.985pt，
shaped score约.885但base约.706；人审看到常量执行、重复失败调用和工具结果被反驳后仍领bonus。E4-B的小额
第四-call penalty只弱缓解（mean penalty .00110），没有稳定accuracy收益（
`rl/runs/m5_e3_e4_comparison/e3_e4a_e4b_comparison.json`、`reports/04_reward_hacking.md`）。E4因此是更干净的
reward-shaping exploitation；E5称credit-signal/proxy gaming更严谨。

### 9. 最终结论与限制

M7验证M6的over-calling/no-endpoint-gain，但不能验证faster-early-learning，因为M7只有endpoint、没有full-panel
训练曲线。证据仍是单seed、固定β=.5、最多四次调用、position-aligned文本adoption heuristic；sampled与AIME
结果不升级成强因果结论。E6只证明mask treatment生效且80-step未复现灾难性退化；E7完成设计审查后deferred，
没有启动。项目没有用extra seed、β sweep、Tier B、E7或post-hoc tuning挽救null结果（`plans/M5.md`、
`plans/M6.md`、`plans/M7.md`）。
