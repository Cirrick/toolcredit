# Main results — M6 fixed panel and M7 unified evaluation

本页先保留 M6 冻结的 fixed MATH500-100 训练曲线与机制证据，再报告 M7 canonical unified
evaluation。M6回答“训练过程中发生了什么”；M7在同一760题、同一生成协议下回答四个endpoint role是否泛化。

## 实验边界

E3 与 E5 都从 `sft/checkpoints/qwen3-1.7b-sft` 独立启动，使用相同的 train/validation data、seed、
64 prompts × 8 rollouts、生成预算、answer+format reward、loss mask、optimizer、KL/clip 和 200-step
validation cadence。唯一 treatment 是：E5 在 native GRPO `A_traj` 后应用冻结的

\[
A_{turn}(t)=A_{traj}+0.5(s_t-\bar{s}_t).
\]

`s_t=1` 只表示成功执行且被后续文本证据采纳的工具结果；这是 deterministic heuristic evidence，
不是 learned step value、PRM 或因果 judge。smoke 的用户人工盲审为 14 TP / 1 FP，precision 0.9333，
Wilson 95% CI `[0.7018, 0.9881]`，通过冻结的 formal gate。

## 主结果

| Step | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E3 trajectory GRPO | 0.60 | 0.67 | 0.67 | 0.70 | 0.73 | 0.73 | 0.74 | **0.77** | **0.76** |
| E5 turn credit | 0.61 | 0.69 | **0.73** | **0.73** | **0.74** | **0.75** | 0.71 | 0.70 | **0.76** |
| E5 − E3 | +0.01 | +0.02 | **+0.06** | +0.03 | +0.01 | +0.02 | −0.03 | **−0.07** | 0.00 |

![E3 vs E5 curve and behavior](assets/02_e3_e5.png)

| 指标 | E3 | E5 | E5 − E3 |
|---|---:|---:|---:|
| 0–100 normalized AUC（primary） | 0.67625 | **0.70625** | **+0.03000** |
| 0–200 normalized AUC | 0.71125 | **0.716875** | +0.005625 |
| 首次达到 0.70 | step 75 | **step 50** | −25 step |
| 首次达到 0.73 | step 100 | **step 50** | −50 step |
| final pass@1 | **0.76** | **0.76** | 0.00 |
| peak pass@1 | 0.77 @175 | 0.76 @200 | −0.01 |

E5 的早期学习明确更快，但优势没有维持到终点：150/175 时分别落后 E3 3/7 题，step 200 回到同为
76/100。逐题配对 bootstrap 的 final accuracy difference 为 `0.00`，95% CI `[-0.06, 0.06]`。
因此不能用 early AUC 把 E5 写成最终能力提升，也不能用中后期两个点把它写成确定退化。

## 机制审计

formal gate 全量重验 102,400 条训练 trajectory、363,350 个 assistant turn：UID 唯一、turn span、mask、
公式与 proof-segment mismatch 均为 0。90,498 条（88.38%）满足冻结的 mixed-quality 定义；按五个 issue
category 配额和 `sha256(run,step,trajectory_uid,sample_id)` 固定选出 30 条，而不是按 reward 或改善方向选例。

| 审计对象 | 总数 | 预期方向生效 | correction=0 | 方向违约 |
|---|---:|---:|---:|---:|
| useful `s_t=1` turn | 213,403 | 99,996 正向 | 113,407 | 0 |
| 非 no-tool 问题 turn | 47,806 | 41,444 负向 | 6,362 | 0 |

每条真实 E5 rollout 同时保存 native `A_traj`，所以能做算法内反事实：若按 E3 rule，所有 policy turn
得到同一个 `A_traj`；E5 再对 useful/problem turn 作相反方向的相对修正。全量 trajectory 中没有一条出现
同轨迹 `A_traj` 不恒定。30 条审计覆盖 ambiguous output、budget/truncation、no-tool/final、execution/parser
failure、success-but-unused/mentioned 五类，每类 6 条，并保留 visible result、adoption evidence、span 和 mask。

这个机制证据证明 treatment 真的改变了信用，而不是空开关；它不证明 heuristic 判断了真实因果价值。
尤其 final/no-tool turn 也参与 position comparison，容易相对同位置的 adopted tool turn受罚，这是 Tier A
预注册且不事后修改的局限。

## 工具行为与稳定性

| 102,400 条训练轨迹指标 | E3 | E5 | 变化 |
|---|---:|---:|---:|
| mean tool calls | 0.9627 | **2.5509** | +1.5881 |
| 4-call fraction | 2.71% | **44.05%** | +41.34pt |
| repeated-code trajectory | 2.02% | **39.71%** | +37.69pt |
| trivial-code trajectory | 0.17% | **5.60%** | +5.44pt |
| per-call execution success | 80.53% | **91.01%** | +10.48pt |
| any execution-error trajectory | 12.22% | 15.28% | +3.06pt |
| invalid rate | 20.06% | **19.52%** | −0.54pt |
| truncation rate | 1.70% | **3.42%** | +1.72pt |
| error recovery | 5,427/13,183 = 41.17% | 6,752/16,477 = 40.98% | −0.19pt |

E5 学会了“更多、且单次更成功地调用”，但没有提高出现错误后的条件恢复率。fixed-100 step-200 中，
E5 有 93/100 条出现 4-call overuse secondary tag，E3 只有 2/100。这与大量重复 code 一致：heuristic
adoption correction 可以奖励反复调用并复述同一结果，即使 outcome reward 没有增加。

稳定性本身没有数值病态：E5 entropy 范围 `0.1945–0.2906`，PPO KL 范围约
`−1.08e−4–1.09e−4`，200 个点全部 finite。换言之，这不是 NaN/KL explosion，而是一个可解释的行为目标偏差。

## Paired failure transition

step-200 的冻结 100 题产生：

- E3 failed → E5 correct：5/24，conditional fix rate 0.2083，95% CI `[0.0500, 0.3913]`；
- E3 correct → E5 failed：5/76，new failure rate 0.0658，95% CI `[0.0135, 0.1282]`；
- unchanged correct：71；unchanged failed：19。

5 个 fixed 来自 3 个 semantic/constraint miss、1 个 execution-error failure、1 个 result-misread；5 个
new failure 则是 4 个 semantic/constraint miss和 1 个 result-misread。所有分类型分母都很小，CI 很宽；
没有一类达到“已解决”的证据门槛。完整 primary matrix、secondary tag transitions、逐题 evidence pointers 和
checkpoint-blinded manual Codex adjudication见 `reports/03_badcase_taxonomy.md` 及 run `analysis/`。

## 最终判读

M6 给出一个 mixed/negative 结论：**Tier-A turn credit 提高了早期样本效率，但没有提高最终正确率；它在
正确实现逐轮相对信用的同时，强烈诱导 repeated/maximum-budget tool use。** 这说明“比 trajectory-level
broadcast 更细”本身不够，step signal 的语义与状态对齐同样关键。

最可能的限定解释是：当前 horizon 最多四次工具调用，native group outcome variance 已经提供较强信号；
position alignment 把 final/no-tool 与 tool-call turn混比；visible-text adoption 可以被重复调用/复述满足；
固定 `β=0.5` 未调优。按预注册协议没有为改善结果调 β、改 heuristic、追加 seed、启动 Tier B/GiGPO、
E7 或 M7。

## 证据索引

- formal run 与五件套：`rl/runs/e5_turn_credit_20260823_082012/`
- 完整性：`analysis/formal_completion_gate.json`、`analysis/m6_e5_completion_manifest.json`
- performance/behavior：`analysis/e3_e5_performance_behavior.json`
- mechanism：`analysis/mechanism_analysis.json`、`analysis/mixed_quality_mechanism_audit.jsonl`
- paired transition：`analysis/paired_failure_transition_analysis.json`、`analysis/paired_failure_transitions.jsonl`
- blinded packet/adjudication：`analysis/paired_failure_blinded_cases.jsonl`、`rl/m6_failure_adjudications.jsonl`

## M7 unified evaluation（canonical，2026-08-25）

### 完整性、口径与解释顺序

唯一run为`eval/runs/m7_unified_eval_20260824_201032/`，frozen-v3 manifest为
`b765ec2c…5b05a4`。raw与strict-scored均为15,200/15,200：greedy 3,040、sampled 12,160，32个shard，
3,800个四role pairing group；missing、unexpected、duplicate、conflict和infrastructure failure全部为0，
full exact-key digest为`c43ce719…fa8b8`。这些分母见`metrics/completeness.json`与
`metrics/full_scored_completeness.json`，canonical文件仍由`hashes.sha256`保护。

greedy n=1是primary diagnostic。sampled为每题四个固定matched index；pass@k与paired transitions只作为
secondary/exploratory证据。bootstrap为10,000次、seed 42、two-sided percentile 95% CI，在四个source
stratum内按question cluster重采样；sampled抽中题时携带该题全部四个index。全部accuracy-delta报告有
10,000个effective replicate；少数小source的条件率replicate遇到真实零分母，按冻结规则记NA而不加伪计数。

### 四role × 四source performance

Greedy pass@1报告严格exact count：

| Source | Raw | SFT | E3 | E5 |
|---|---:|---:|---:|---:|
| MATH500 | 288/500 = 0.576 | 297/500 = 0.594 | **383/500 = 0.766** | 375/500 = 0.750 |
| AIME2024 | 3/30 = 0.100 | 2/30 = 0.067 | **4/30 = 0.133** | **4/30 = 0.133** |
| AIME2025 | 3/30 = 0.100 | **5/30 = 0.167** | 4/30 = 0.133 | **5/30 = 0.167** |
| GSM8K | 132/200 = 0.660 | 156/200 = 0.780 | **166/200 = 0.830** | 165/200 = 0.825 |
| **FULL760** | 426/760 = 0.561 | 460/760 = 0.605 | **557/760 = 0.733** | 549/760 = 0.722 |

![M7 full-panel greedy accuracy](assets/03_m7_accuracy.png)

Sampled pass@1/2/4如下。pass@1也等于四个matched index上的strict-correct count除以4N；pass@2和
pass@4是冻结的per-question unbiased estimator之和除以N，exact estimator sums保存在
`metrics/pass_at_k.json`。

| Source | Raw p@1 / p@2 / p@4 | SFT | E3 | E5 |
|---|---:|---:|---:|---:|
| MATH500 | .575 / .707 / .800 | .616 / .732 / .814 | **.775 / .847 / .890** | .771 / .837 / .880 |
| AIME2024 | .083 / .150 / .267 | .092 / .133 / .167 | **.208 / .311 / .433** | .200 / .300 / .400 |
| AIME2025 | .075 / .122 / .167 | .117 / .178 / .267 | **.225 / .294 / .367** | .167 / .233 / .267 |
| GSM8K | .694 / .819 / .885 | .763 / .853 / .895 | .821 / **.873** / **.910** | **.824** / .873 / **.910** |
| **FULL760** | .567 / .692 / .776 | .614 / .718 / .788 | **.743 / .811 / .857** | .738 / .802 / .845 |

FULL760 sampled pass@1的trajectory分子/分母依次为raw 1,723/3,040、SFT 1,867/3,040、E3
2,258/3,040、E5 2,245/3,040。两种setting都把E3置于最强endpoint：E5在greedy少8题、sampled少13条
matched trajectory，且p@2/p@4也没有反转这个排序。

### Preregistered paired transitions与bootstrap

四格顺序为fixed / new / unchanged-correct / unchanged-failed；每行总和严格等于setting denominator。

| Stage transition | Setting | 四格exact count | Accuracy Δ | 95% CI | Conditional fix | New failure rate |
|---|---|---:|---:|---:|---:|---:|
| Raw→SFT | greedy | 109 / 75 / 351 / 225 | +0.0447 | [0.0105, 0.0789] | 109/334 = .326 | 75/426 = .176 |
| Raw→SFT | sampled | 451 / 307 / 1416 / 866 | +0.0474 | [0.0276, 0.0671] | 451/1317 = .342 | 307/1723 = .178 |
| SFT→E3 | greedy | 127 / 30 / 430 / 173 | **+0.1276** | **[0.0974, 0.1592]** | 127/300 = .423 | 30/460 = .065 |
| SFT→E3 | sampled | 533 / 142 / 1725 / 640 | **+0.1286** | **[0.1089, 0.1484]** | 533/1173 = .454 | 142/1867 = .076 |
| E3→E5 | greedy | 47 / 55 / 502 / 156 | −0.0105 | [−0.0368, 0.0158] | 47/203 = .232 | 55/557 = .099 |
| E3→E5 | sampled | 195 / 208 / 2050 / 587 | −0.0043 | [−0.0178, 0.0092] | 195/782 = .249 | 208/2258 = .092 |

Raw→SFT是约+4.5pt的aggregate improvement，greedy主要由GSM8K驱动；sampled的GSM8K与MATH500
CI排除0。SFT→E3是最强、最一致的阶段增益，两setting均约+12.8pt；greedy的GSM8K/MATH500以及
sampled四source都为正。E3→E5则同时出现修复和新增，但新增略多；aggregate CI跨0。source-level greedy
delta为AIME2024 0.0、AIME2025 +.033、GSM8K −.005、MATH500 −.016，均跨0。sampled AIME2025为
−.0583、CI `[−.1083,−.0083]`，但它是30题上的secondary/exploratory signal，不能单独升级为强因果结论。
完整source四格与全部CI见`transitions/{raw_to_sft,sft_to_e3,e3_to_e5}.json`和
`transitions/bootstrap.json`。

### Tool behavior：E5 treatment effect跨source泛化

FULL panel的count/denominator直接取自`metrics/tool_behavior.json`：

| Role / setting | Mean calls | 4-call | repeated code | truncation | any error | error recovery | per-call success |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw greedy | 735/760 = .967 | 76/760 = 10.0% | 78/760 = 10.3% | 136/760 = 17.9% | 111/760 = 14.6% | 20/111 = 18.0% | 460/735 = 62.6% |
| SFT greedy | 954/760 = 1.255 | 59/760 = 7.8% | 51/760 = 6.7% | 115/760 = 15.1% | 102/760 = 13.4% | 34/102 = 33.3% | 733/954 = 76.8% |
| E3 greedy | 675/760 = .888 | 10/760 = 1.3% | 22/760 = 2.9% | 73/760 = 9.6% | 51/760 = 6.7% | 22/51 = 43.1% | 595/675 = 88.1% |
| E5 greedy | 2854/760 = **3.755** | 706/760 = **92.9%** | 702/760 = **92.4%** | 760/760 = **100%** | 25/760 = 3.3% | 10/25 = 40.0% | 2813/2854 = **98.6%** |
| Raw sampled | 3083/3040 = 1.014 | 305/3040 = 10.0% | 315/3040 = 10.4% | 472/3040 = 15.5% | 463/3040 = 15.2% | 106/463 = 22.9% | 1953/3083 = 63.3% |
| SFT sampled | 4101/3040 = 1.349 | 270/3040 = 8.9% | 243/3040 = 8.0% | 415/3040 = 13.7% | 493/3040 = 16.2% | 158/493 = 32.0% | 3147/4101 = 76.7% |
| E3 sampled | 2913/3040 = .958 | 37/3040 = 1.2% | 76/3040 = 2.5% | 152/3040 = 5.0% | 242/3040 = 8.0% | 112/242 = 46.3% | 2562/2913 = 88.0% |
| E5 sampled | 11963/3040 = **3.935** | 2956/3040 = **97.2%** | 2896/3040 = **95.3%** | 3039/3040 = **100.0%** | 146/3040 = 4.8% | 81/146 = 55.5% | 11741/11963 = **98.1%** |

E3相对SFT调用更选择性，同时降低4-call、重复、错误与截断并提高条件恢复。E5则几乎总把四次预算用满，
而且这种效应不是单一数据集造成：greedy的E5 4-call率为AIME2024 60.0%、AIME2025 66.7%、GSM8K
98.5%、MATH500 94.2%；sampled依次为87.5%、90.0%、100%、97.2%。E5的per-call success反而接近
98%，说明问题不是工具语法崩坏，而是成功调用与重复调用本身成为容易优化的proxy。E5 greedy 701/760、
sampled 2916/3040以assistant-turn budget结束，几乎没有正常final termination；模型通常已产生boxed文本后
仍继续调用，直到budget终止。

这里应写作**强behavioral treatment effect与credit-signal/proxy gaming**：M6 ledger证明逐轮correction确实
生效，M7证明其行为后果跨source/setting泛化；但E5没有直接优化一个外显shaped reward总分，因此不能仅凭
over-calling自动等同classic reward hacking。更干净的reward-shaping exploitation证据来自E4，见
`reports/04_reward_hacking.md`。

### M6→M7最终判读

M7验证了M6的两个endpoint结论：E5没有超过E3 final accuracy，且severe over-calling在完整MATH500、
AIME2024、AIME2025、GSM8K与greedy/sampled均出现。M7不能验证M6“early learning更快”，因为这里只比较
冻结endpoint，没有统一full-panel训练曲线。于是项目级结论是：**E3是最强endpoint stage；Tier-A turn credit
改变了学习过程与行为，却没有转化为更强最终能力。** 这是同起点、同训练量recipe的直接比较，但仍只有单seed、
固定`β=0.5`、短horizon与position-aligned adoption heuristic，`causal_claim_allowed=false`的机器guardrail
必须保留。没有为null/negative endpoint结果追加seed、β sweep、Tier B、E7或post-hoc tuning。

M7证据入口：`metrics/pass_at_k.json`、`metrics/tool_behavior.json`、`metrics/completeness.json`、
`transitions/bootstrap.json`、`taxonomy/audit_metrics.json`和`analysis_hashes.sha256`，均位于canonical run目录。
