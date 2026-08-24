# Bad-case taxonomy — M6 E3→E5 paired analysis

本报告只完成 M6 预注册的 E3→E5 fixed MATH500-100 机制对照。M7 才会按同一 frozen v2 protocol 对
raw/SFT/E3/E5 和 760 题 full panel 全量重新生成；本次没有提前运行 M7。

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
