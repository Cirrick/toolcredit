# Bad-case taxonomy — M6 fixed panel and M7 full-panel transitions

本报告先保留M6预注册的E3→E5 fixed MATH500-100机制对照，再报告M7在760题full panel上的三段冻结
paired transition。M6与M7使用不同canonical输出和分母；以下不会把两者混算。

## 协议与裁决

- panel：E3 与 E5 step-200 的同一 100 个 question ID，greedy n=1；ID 完全配对。
- taxonomy：`toolcredit_failure_taxonomy_v1`，在任何 canonical E5 output 前冻结；没有结果后新增 label。
- pairing/bootstrap：question-level paired bootstrap，10,000 次、seed 42、percentile 95% CI；零分母不加伪计数。
- persistence：保存全部 100 对，而不是只保存 fixed cases。
- 裁决：先生成不含 checkpoint role 的 48 条 failure packet，再由 Codex 逐条人工式语义裁决并回连私有 role
  index。它是 **checkpoint-blinded manual Codex adjudication，不是 human audit**；coarse rules 只提供 proposal。

## Headline transition

| E3→E5 | exact count | rate / denominator | paired bootstrap 95% CI |
|---|---:|---:|---:|
| failed→correct（fixed） | 5 | 5/24 = 0.2083 | [0.0500, 0.3913] |
| correct→failed（new） | 5 | 5/76 = 0.0658 | [0.0135, 0.1282] |
| correct→correct | 71 | 71/100 | — |
| failed→failed | 19 | 19/100 | — |
| accuracy delta | 0 | 0/100 = 0.0000 | [−0.0600, 0.0600] |

净正确数不变并不表示逐题行为不变：10 道题交换了成败状态，另有多道 failure 在不同 taxonomy 类之间迁移。

## Primary failure distribution

| Primary failure | E3 | E5 | E3 failure 的 fixed | E3→E5 主要去向 |
|---|---:|---:|---:|---|
| boxed/verifier format | 6 | 7 | 0/6 | 6/6 仍为同类 |
| budget exhaustion/truncation | 1 | 0 | 0/1 | 1→semantic miss |
| exact→low precision | 1 | 0 | 0/1 | 1→result misread |
| no-tool wrong | 4 | 2 | 0/4 | 1仍同类，3→semantic miss |
| Python/sandbox execution error | 3 | 0 | 1/3 | 1→correct，2→semantic miss |
| tool-correct semantic/constraint miss | 6 | 12 | 3/6 | 3→correct，1→format，2仍同类 |
| tool result not adopted/misread | 3 | 3 | 1/3 | 1→correct，1→no-tool，1仍同类 |

其他冻结 primary 类在两端分母均为 0。所有非零类分母最多为 6，category-level bootstrap CI 普遍覆盖很宽；
这些数字只能描述观察到的迁移，不能宣称某类已被解决。

值得注意的是，“execution error primary 从 3 降到 0”并不等于三题都修好：只有 1 题变正确，另 2 题只是
迁移成 semantic miss。同样，truncation 与 low-precision primary 消失的两题仍然失败，只是换了失败类型。

## Fixed、regressed 与 unchanged 例子

### Fixed（5）

| question | E3 primary | E5 状态 | 观察 |
|---|---|---|---|
| `math500_000018` | result not adopted/misread | correct | E3 工具给 28 后仍答 124；E5答对，但有 4-call overuse |
| `math500_000046` | execution error | correct | E3 verification broadcasting error后答12；E5答对6，但仍满调用预算 |
| `math500_000115` | semantic miss | correct | E3误数 ELLIPSE 重复字母；E5修正 |
| `math500_000212` | semantic miss | correct | E3错误枚举得48；E5修正，但出现重复调用/unused heuristic tag |
| `math500_000248` | semantic miss | correct | E3把所求矩阵降成 trace；E5给出正确矩阵，但仍4 calls |

### New failures（5）

| question | E3 状态 | E5 primary | 观察 |
|---|---|---|---|
| `math500_000226` | correct | semantic miss | 球盒分拆漏计，重复调用只打印错误的2 |
| `math500_000232` | correct | semantic miss | 无推导地让工具反复打印0，正确答案为1997/2 |
| `math500_000312` | correct | result not adopted/misread | 工具给出 `x>5`，模型反而覆盖为两侧并集 |
| `math500_000323` | correct/no-tool | semantic miss | 过度调用后把交叉概率答成1/4而非1/3 |
| `math500_000401` | correct | semantic miss | symbolic tool没有求最大值，模型仍断言sqrt(5) |

### Unchanged failures

- `math500_000025`：两端数学集合都正确，但冻结 verifier 对列表顺序/terminal form 仍判 invalid；失败未修复。
- `math500_000019`：E3 的 execution-error/no-recovery 迁移为 E5 semantic miss，仍未得到最小值 3。
- `math500_000224`：E3 把正确 decimal 映射成 1/32，E5 映射成 1/27；failure type保持 result/exact-value误读。
- `math500_000151`：两端均陷入无工具重复文本且无 boxed final。

## Secondary behavior transitions

secondary tags 用来呈现 primary cause之外的行为事实。最显著的是 `tool_overuse`：

| transition | count |
|---|---:|
| E3 no overuse → E5 overuse | 92 |
| overuse → overuse | 1 |
| no overuse → no overuse | 6 |
| overuse → no overuse | 1 |

因此 E5 step-200 有 93/100 条达到四次调用，E3 为 2/100。这个现象同时存在于 fixed、new、unchanged
correct 和 unchanged failed，不能被解释为“只在难题上多做必要计算”。训练全量的 4-call rate 也从
2.71% 升到 44.05%，与 fixed panel方向一致。

`tool_result_not_adopted_or_misread` secondary tag 的迁移为：20 条消失、10 条两端都有、6 条新增；自动
secondary detection偏高召回，不能当总体真实语义率。primary labels 才使用逐条语义裁决。

## 结论与限制

E5 没有表现出针对某个关键 failure class 的净清除：5 个修复被 5 个新增抵消，semantic miss由 6 增至12，
boxed/verifier failure由6增至7；部分 execution/truncation label 的下降只是迁移而非正确。与 mechanism audit
合看，turn correction确实降低了未采纳/错误 turn的相对 credit，但 heuristic 也使“再调一次工具并复述结果”
成为容易获得正 correction 的行为，最终以 over-calling而非更高 final accuracy体现。

局限包括 fixed panel只有100题、单 seed、manual Codex而非独立 human taxonomy adjudication，以及某些
category分母为0或≤6。M7必须按 frozen v2 bundle 对 raw/SFT/E3/E5 全部重新生成和统一裁决，不能把本页
旧 E3/E5 validation与未来 full-panel输出混成四阶段主表。

## Evidence pointers

- `rl/runs/e5_turn_credit_20260823_082012/analysis/paired_failure_transition_analysis.json`
- `rl/runs/e5_turn_credit_20260823_082012/analysis/paired_failure_transitions.jsonl`
- `rl/runs/e5_turn_credit_20260823_082012/analysis/paired_failure_blinded_cases.jsonl`
- `rl/runs/e5_turn_credit_20260823_082012/analysis/paired_failure_private_index.json`
- `rl/m6_failure_adjudications.jsonl`
- `eval/diagnostic_taxonomy_v1.json`

## M7 frozen full-panel taxonomy workflow

M7对15,200条strict-scored trajectory运行`toolcredit_failure_taxonomy_v1`：10,085条correct的primary为null，
5,115条wrong各有且仅有一个coarse primary，unclassified为0。全量proposal见canonical run的
`taxonomy/coarse_proposals.jsonl`与`coarse_metrics.json`；这些是frozen deterministic proposals，不自动等于
人工语义真值。

盲审packet用stable hash在每个role×setting最多取24条wrong，共192条；packet不含checkpoint role、stage、
direction或sample index，随后才通过`private_index.json`回连。reviewer记录为
`manual_codex_agent_adjudication`，`independent_human_audit=false`，不能冒充独立human audit。总体proposal
agreement为151/192 = 78.65%，uncertain 48/192 = 25.0%。唯一系统性confusion是41条
`budget_exhaustion_or_truncation → tool_correct_semantic_or_constraint_miss`：这些轨迹在后续重复调用耗尽预算前
已经给出boxed-but-wrong答案，所以truncation并未“直接阻止完成”，不满足冻结taxonomy rule 1。

| Role / setting | Agreement | 科学使用边界 |
|---|---:|---|
| Raw greedy / sampled | 24/24 / 24/24 | 可描述coarse proposal与迁移，但audit仍是分层抽样而非prevalence estimator |
| SFT greedy / sampled | 24/24 / 24/24 | 同上 |
| E3 greedy / sampled | 24/24 / 22/24 | 同上；sampled保留两条边界不确定性 |
| E5 greedy / sampled | **6/24 / 3/24** | **不得科学解释coarse category prevalence或E3→E5 category migration** |

因此E5的正确/错误、工具调用、termination和四格paired outcome仍是exact machine fact；但E5大量失败被自动
归为budget/truncation，而blind adjudication认为不少已先发生semantic wrong。后文有意不把E5 coarse label
迁移写成机制结论。完整confusion、uncertain flag与evidence pointer见`taxonomy/audit_metrics.json`和
`taxonomy/adjudications.jsonl`。

## 三段paired outcome transition

Greedy是primary diagnostic，sampled matched-index是secondary/exploratory。四格顺序均为fixed / new /
unchanged-correct / unchanged-failed：

| Transition | Greedy（N=760） | Greedy Δ [95% CI] | Sampled（N=3040） | Sampled Δ [95% CI] |
|---|---:|---:|---:|---:|
| Raw→SFT | 109 / 75 / 351 / 225 | +.0447 [.0105, .0789] | 451 / 307 / 1416 / 866 | +.0474 [.0276, .0671] |
| SFT→E3 | 127 / 30 / 430 / 173 | **+.1276 [.0974, .1592]** | 533 / 142 / 1725 / 640 | **+.1286 [.1089, .1484]** |
| E3→E5 | 47 / 55 / 502 / 156 | −.0105 [−.0368, .0158] | 195 / 208 / 2050 / 587 | −.0043 [−.0178, .0092] |

Raw→SFT的coarse failure-category migration rate为greedy .3832 `[.3323,.4359]`、sampled .3971
`[.3652,.4298]`；SFT→E3为greedy .3533 `[.3007,.4077]`、sampled .3606 `[.3248,.3966]`。
结合高agreement的raw/SFT/E3 audit，这两段可作受限的coarse机制描述：SFT显著减少raw的no-tool与format
failure，但增加调用；E3进一步把大量SFT failure变正确，并同时使调用更选择性、执行更可靠。任何单个小类仍
受分母和自动proposal限制，不能宣称某类被“解决”。E3→E5虽然机器文件也计算了migration statistic，因E5
blind agreement极低，本文**不解释该数值和category matrix**。

source四格进一步防止aggregate掩盖异质性：

| Transition / setting | AIME2024 | AIME2025 | GSM8K | MATH500 |
|---|---:|---:|---:|---:|
| Raw→SFT greedy | 1/2/1/26 | 2/0/3/25 | 37/13/119/31 | 69/60/228/143 |
| Raw→SFT sampled | 8/7/3/102 | 11/6/3/100 | 125/70/485/120 | 307/224/925/544 |
| SFT→E3 greedy | 3/1/1/25 | 2/3/2/23 | 17/7/149/27 | 105/19/278/98 |
| SFT→E3 sampled | 16/2/9/93 | 19/6/8/87 | 97/50/560/93 | 401/84/1148/367 |
| E3→E5 greedy | 1/1/3/25 | 3/2/2/23 | 8/9/157/26 | 35/43/340/82 |
| E3→E5 sampled | 10/11/14/85 | 4/11/16/89 | 42/40/617/101 | 139/146/1403/312 |

每格仍按fixed/new/unchanged-correct/unchanged-failed排序。AIME每个greedy source只有30题，不能从一两题
差异外推；全部source-level bootstrap在`transitions/bootstrap.json`。

## Stable-hash paired examples

下表不按改善方向挑选：对每段greedy transition的四个cell，按
`sha256([comparison,setting,cell,key])`字典序最小值固定选一条。raw output hash使例子可直接回到canonical
evidence；完整路径和两侧key在`transitions/paired_rows.jsonl`。

| Transition / cell | Question | Before → after primary state | selection hash | before / after output hash |
|---|---|---|---|---|
| Raw→SFT fixed | `gsm8k_000174` | semantic/constraint miss → correct | `00d37d5e4623` | `9341988fedbc` / `a4c2fc7a6aee` |
| Raw→SFT new | `math500_000415` | correct → budget/truncation | `08a97f2ccf20` | `1abe6b5e99f1` / `ea977c5fed80` |
| Raw→SFT unchanged-correct | `math500_000203` | correct → correct | `0280547551fa` | `855b76c7e371` / `9bb1401bbd7b` |
| Raw→SFT unchanged-failed | `math500_000249` | budget/truncation → semantic/constraint miss | `0094085c5d42` | `729e9ca48ca6` / `00e6232b1aa8` |
| SFT→E3 fixed | `math500_000415` | budget/truncation → correct | `00889ca3d604` | `ea977c5fed80` / `4424b9cc2706` |
| SFT→E3 new | `math500_000128` | correct → semantic/constraint miss | `1445de40f485` | `9824189ba4aa` / `36c1d5a3156b` |
| SFT→E3 unchanged-correct | `math500_000091` | correct → correct | `01a273d4048f` | `8c306baa77ce` / `a01db32afbe7` |
| SFT→E3 unchanged-failed | `aime24_000011` | budget/truncation → semantic/constraint miss | `0076759f5bb8` | `3ae151674f61` / `4d61ffc04588` |
| E3→E5 fixed | `math500_000400` | budget/truncation proposal → correct | `00ab045175e1` | `d2041b5d0348` / `210bfb02c461` |
| E3→E5 new | `math500_000299` | correct → budget/truncation proposal | `0055c17689bc` | `7fe96ec3a27c` / `f652c3821d7a` |
| E3→E5 unchanged-correct | `math500_000386` | correct → correct | `0094e42b54b7` | `188f85722d72` / `7dcdadd2a174` |
| E3→E5 unchanged-failed | `math500_000320` | no-tool wrong → budget/truncation proposal | `02f9b2e259a0` | `6d1eb2763c40` / `cc7092445cc0` |

E3→E5表中的failure名明确标为proposal，只用于定位轨迹，不用于科学迁移叙事。科学上可说的是：greedy有
47个failure被修复，同时55个原本correct变wrong；sampled为195与208。它与M6机制ledger共同说明treatment
改变了具体轨迹和行为，但没有产生净endpoint accuracy收益。

## M7结论与限制

阶段迁移不是同一种因果问题：Raw→SFT与SFT→E3描述训练阶段；E3与E5才是同起点、同训练量的recipe直接
对照。M7在full panel验证M6的over-calling/no-endpoint-gain结论，但endpoint-only设计不能验证M6的
faster-early-learning。sampled matched-index、AIME小分母、单seed、固定β与Codex而非独立human audit限制仍在。
没有因CI或taxonomy不利而换seed、并类、改分母或补跑。

M7 evidence pointers：

- `eval/runs/m7_unified_eval_20260824_201032/taxonomy/{coarse_metrics,audit_metrics}.json`
- `eval/runs/m7_unified_eval_20260824_201032/taxonomy/{blinded_audit_packet,adjudications}.jsonl`
- `eval/runs/m7_unified_eval_20260824_201032/transitions/{raw_to_sft,sft_to_e3,e3_to_e5}.json`
- `eval/runs/m7_unified_eval_20260824_201032/transitions/{paired_rows.jsonl,bootstrap.json}`
- `eval/runs/m7_unified_eval_20260824_201032/analysis_hashes.sha256`
