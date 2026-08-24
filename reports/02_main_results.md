# Main results — M6 E3 vs E5

本页记录 M6 冻结的 fixed MATH500-100 对照。它不是 M7 的 MATH500/AIME/GSM8K 四阶段全量评测；
raw/SFT/E3/E5 的统一 full-panel generation 尚未启动。

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
