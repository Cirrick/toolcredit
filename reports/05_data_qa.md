# 报告 05 — 数据专题 Q&A

日期：2026-09-07｜范围：ToolCredit 全流程用到的每一份数据——来源、切分、去污染、各自承担的职责、
已知弱点。本文是数据问题的**单一入口**；实验结论见 `02_main_results.md`，工具增益见 `01_tool_gain.md`。

同源问答留痕：`reports/qa_log.md` Q18 / Q19 / Q20。

---

## 0. 一张图：数据谱系

```
MATH train split (lighteval 预处理 parquet)            HuggingFaceH4/MATH-500   Maxwell-Jia/AIME_2024
        7500 条                                              500 条            math-ai/aime25
          │                                                    │               openai/gsm8k test[:200]
          │ download_convert.py                                │                        │
          │ 剔 4 条(Level? / 空答案)、剥离 prompt 后缀           │                        │
          ▼                                                    │                        │
   math_train.jsonl  7496                                      │                        │
          │                                                    ▼                        ▼
          │ dedup_check.py  ←──── 精确匹配 + 词级 13-gram ──── 四个评测集(共 760 条)
          │ 剔 216 条命中                                            │
          ▼                                                         │
   math_train_clean.jsonl  7280                                     │
          │                                                         │
          │ select_train_subset.py  取 L3–5（M1 结论）               │
          ▼                                                         │
   train_subset.jsonl  5403   (L3 1558 / L4 1654 / L5 2191)         │
          │                                                         │
          │ make_splits.py  seed 42 分层抽 200                       │
          ├─────────────► heldout_200.jsonl  200   [M3 验收，不进任何训练]
          ▼                                                         │
   sft_pool.jsonl  5203  (L3 1500 / L4 1593 / L5 2110)              │
          │                                                         │
          ├──► 教师 Qwen3-8B 蒸馏(temp .7, n=2) ──► sft_traces.jsonl 6056  [SFT 训练]
          │                                                         │
          └──► rl/prepare_data.py ──► e3_train.parquet 5203         │
                                       (verl 长度过滤后 5195)  [RL 训练]
                                                 │                  │
                                                 │  训练内 val ◄─────┤ MATH500 分层 100 题
                                                 ▼                  ▼
                                       所有 RL run(E3/E5/E5v2/  最终统一评测 panel
                                       E3-NoTool) 共用同一池        760 条 × 5 setting
```

---

## Q1. 原始训练集是什么？

**MATH（Hendrycks et al.）的 train split**，7500 条，取自 M0 阶段已下载到本地的
`~/verl-team/lighteval-MATH-preprocessed/train.parquet`（源 `DigitalLearningGmbH/MATH-lighteval`）。

`data/download_convert.py` 把它转成统一 schema：

```json
{"id": "math_train_000001", "question": "...", "answer": "\\frac{3}{4}",
 "level": 4, "subject": "Algebra", "source": "MATH", "split": "train"}
```

- 剔除 4 条：2 条 `Level ?`（无法分层）、2 条空 ground_truth。→ **7496 条**。
- **剥离 prompt 后缀**：verl-team 的预处理给每条 question 追加了固定指令
  `" Let's think step by step and output the final answer within \boxed{}."`。转换时断言全表匹配后剥离，
  在 SFT / RL / 评测三处再统一拼回。这保证三个阶段 prompt 逐字符一致——否则训练与评测的
  prompt 分布会悄悄错开，是这类项目里很常见但很难查的 bug。

Level 分布（7280 条净池）：L1 556 / L2 1321 / L3 1558 / L4 1654 / L5 2191。

## Q2. 怎么去污染？

`data/dedup_check.py`（PLAN §5.1）。训练池对四个评测集逐条比，两个判据，命中即剔除、不做人工豁免：

1. **归一化精确匹配**：转小写 → 非字母数字字符全替换成空格 → 压空白。
2. **词级 13-gram 重叠**（GPT-3 论文的做法）：任一 13 元组命中即算。

| 评测集 | 条数 | 命中训练条目 |
|---|---|---|
| MATH500 | 500 | 172 |
| AIME24 | 30 | 66 |
| AIME25 | 30 | 69 |
| GSM8K200 | 200 | 0 |

去重后共剔 **216** 条 → `math_train_clean.jsonl` **7280**。

**怎么读这些数字**：AIME 只有 30 题却命中 66/69 条训练条目——因为绝大多数命中的是
**答案格式模板句**（如 `where m and n are relatively prime positive integers find m n`），
不是真题目重复。这是 13-gram 判据的已知性质：宁可过度剔除，不冒漏检风险。
选集后复跑（`contamination_report_subset.md`）：5403 条 vs 四个评测集命中 **0**。
脚本带 `--self-test`，构造精确 / n-gram / 干净三种行验证检出逻辑。

### 一个已核实的假阳性（2026-09-07 新发现）

报告里唯一那条标记 `exact` 的命中 `math_train_007163 ↔ math500_000046` **不是真重复**：

| | 题目 | 答案 |
|---|---|---|
| 训练 | `z^4 - z^2 + 1 = 0` 的所有根都是 n 次单位根的最小正整数 n | 12 |
| 评测 | `z^4 + z^2 + 1 = 0` 的所有根都是 n 次单位根的最小正整数 n | 6 |

归一化把 `+` / `-` 一并当标点去掉，两题才撞在一起。方向是**保守的**（过度剔除，不是漏检），
不影响任何已发布结论。准确表述应为：**MATH train 与 MATH500 之间真实的精确重复数为 0**。
（是否收紧 normalizer 保留数学符号：待定，收益是报告更精确，风险是可能降低召回，未改动。）

## Q3. held-out 200 是什么？

`sft/make_splits.py`（seed 42）：从 **5403 条 L3–5 选集**里按 level 比例分层随机抽 200 题，
实得 **L3 58 / L4 61 / L5 81**；剩余 **5203** 作 `sft_pool.jsonl`。

**职责：M3 的 SFT 验收集**——测"SFT 到底有没有教会用工具"：工具格式成功率、pass@1 vs 零样本。
实测 89.5%（目标 >90%，边缘未达）/ +1.9pt（n=800，不显著）。两项都是边缘通过，这也是
Q18 里"SFT 买到的是可靠性、不是准确率"最早的信号。

**隔离保证**：它既不参与蒸馏轨迹生成，也不进 RL。`rl/prepare_data.py::validate_split` 里是硬断言——
`heldout_ids & train_ids` 必须为空、train 必须恰好 5203 行，不满足直接抛异常。

**注意**：它**不在最终 760 题评测 panel 里**，只服务 M3 验收，不要在结果表里找它。

## Q4. SFT 的训练数据（蒸馏轨迹）怎么来的？

对 `sft_pool.jsonl` 的 5203 题，用教师 **Qwen3-8B**（temp 0.7、n=2）生成 TIR 轨迹，
拒绝采样后保留（`sft/data/gen/gen_stats.json`）：

| 项 | 值 |
|---|---|
| 生成 | 10406（5203 × 2） |
| 保留 | **6056**（yield 58.2%） |
| 丢弃原因 | 无成功工具调用 2083 / 答案错 1884 / 截断 374 / 过长 8 / 工具参数非法 1 |
| 覆盖题目 | 3716 / 5203 = 71.4% |
| 含错误恢复的轨迹 | 797 |
| 平均工具调用 / response tokens | 1.28 / 580.2 |

**按 level 的题目覆盖率**：L3 1248/1500 = 83.2%、L4 1217/1593 = 76.4%、**L5 1251/2110 = 59.3%**。

这个 L5 缺口是 M3 的关键限制：**拒绝采样教不了教师自己不会的题**。M3 报告里"残余负增益"的
两个原因之一就是它，该缺口由 M4 的 on-policy RL 接手。

PLAN 原定 1.5k–2.5k 条（基于 API 成本假设），实际本地教师无此约束；敏感性实验显示数据翻倍
单调改善（工具增益 −0.056 → −0.032、报错率 22% → 19%），故**采用全量 6056**。

## Q5. RL 的 prompt 池怎么来的？

**就是 `sft_pool.jsonl` 那 5203 题，一题不多一题不少。**

`rl/prepare_data.py` 转成 veRL parquet：`prompt` 拼回后缀、`reward_model.ground_truth` 放答案、
`extra_info` 保留 id/level/subject/source/split。veRL 按 `max_prompt_tokens=1024` 过滤掉 8 条超长题，
**实际训练 5195 条**。产物带 sha256 落在 `rl/data/manifest.json`。

含义（面试会问）：**RL 与 SFT 用同一批题**。RL 是在 SFT 已见过的题上做 on-policy 改进，
MATH500 上 SFT→GRPO 的 +17.2pt 不来自新数据。这是刻意设计——要隔离"信用分配"这一个变量，
数据必须固定。**E3 / E5-v1 / E5-v2 / E3-NoTool 四条臂全部共用这一个 parquet**，
臂之间只差 reward/advantage 计算与是否挂工具。

训练内 val：`e3_val_math500_100.parquet`，MATH500 的 100 题分层子集（20/level，seed 42），只画曲线。

## Q6. 最终评测的 760 题是怎么构成的？

`eval/diagnostic_panel_v1.jsonl`，全部 `split=test`，每条带 `question_sha256` 用于冻结校验：

| 来源 | 条数 | 分层 |
|---|---|---|
| MATH500 | 500 | L1 43 / L2 90 / L3 105 / L4 128 / L5 134 |
| GSM8K test[:200] | 200 | — |
| AIME 2024 | 30 | — |
| AIME 2025 | 30 | — |
| **合计** | **760** | 8 个 stratum |

每个 checkpoint role 跑两套协议：**greedy**（n=1, temp 0）与 **sampled**（n=4, temp 0.6）→
760 × 5 = 3800 条轨迹/role。M7 四个 role = 15,200 条，M8 新增 E5-v2 一个 role = 3,800 条，
合计 **19,000 条** canonical 轨迹。bootstrap 按 question_id 重采样（4 个 sample 保持成簇），
10,000 replicates，seed 42。

另有 **70 条预算诊断子集**（`eval/e3_budget_diagnostic_config.json`）：从 E3 greedy 的
"预算耗尽/截断"失败题里选出，用更长预算复跑做 post-hoc 诊断。配置里写死
`training_use_forbidden: true`，不得回流训练。

## Q7. MATH500 与训练集是什么关系？

三层，分开说才准确。

**(a) split 级天然隔离。** MATH500 = `HuggingFaceH4/MATH-500`，是 MATH **test** split 的 500 题子集
（Lightman et al. / PRM800K 的选取）；训练池是 **train** split。构造上不相交；实测归一化精确重叠 0
（Q2 那条是假阳性），13-gram 保守剔除后训练子集命中 0。

**(b) 分布上是同源 in-distribution 评测，但 level 不对齐。**

| | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|
| MATH500 | 43 | 90 | 105 | 128 | 134 |
| RL 训练池 | — | — | 1558 | 1654 | 2191 |

MATH500 里有 133 题（L1+L2）落在训练分布之外的"更简单"档。这正好解释了两件事：
工具增益在 L1–3 为负（那些题本来不需要工具，且不在训练池），而 RL 的提升集中在 L4/L5
（E3−SFT：L4 +23.5pt、L5 +20.1pt）。subject 分布两边接近，没有额外偏移。

**(c) 有 100 题被"看过曲线"。** RL 训练内 val 取自 MATH500。缓解：全程无早停、不按 val 选 checkpoint、
一律固定 step 200，所以泄漏风险有限；但汇报必须说明，**760 题合并结果才是主口径**。
更干净的做法是把训练内 val 换成 held-out 200 的子集——需要重跑训练，未做。

## Q8. 为什么测 AIME 却说"单点差异不要读"？GSM8K 为什么不用来训练？

不矛盾，因为 panel 里每个数据集承担**不同职责**。

**AIME = 污染兜底，不是效应量测量。**
- 测它的理由：AIME 2024/2025 时间上晚于基座（Qwen3-1.7B）知识截止，是全 panel 污染风险最低的一档。
  它回答"MATH500 上的 +17.2pt 是不是背题背出来的"。实测 Raw .100 → E3 .133 → E5-v2 .200（AIME24），
  方向一致、没塌成随机 → **兜底通过**。这是定性健全性检查，通过即可。
- 不读单点差异的理由：n=30，1 题 = 3.3pt。

| | 正确数 | pass@1 | Wilson 95% CI | 宽度 |
|---|---|---|---|---|
| AIME24 E3 | 4/30 | .133 | [.053, .297] | 24.4pt |
| AIME24 E5-v2 | 6/30 | .200 | [.095, .373] | 27.8pt |
| FULL760 E3 | 557/760 | .733 | [.700, .763] | 6.3pt |

两者只差 **2 道题**，CI 几乎完全重叠；而待检效应量本身约 1pt。用 30 题分辨 1pt，等于拿尺子量原子。
所以 AIME 只进 760 题合并 bootstrap，单列展示、不单独下结论。

**GSM8K = 退化哨兵。** 心算即可解，工具增益趋近零——任何信用分配差异都会被噪声淹没。
PLAN §5.2 据此**否决用 GSM8K 训练**（这是 M1 工具增益预实验先于选数据的直接产出）。
它回答的是"有没有掉"，不是"有没有涨"。

## Q9. 每份数据承担什么职责（一览）

| 数据 | 条数 | 职责 | 可以下什么结论 |
|---|---|---|---|
| `math_train_clean.jsonl` | 7280 | 全量净池 | 选集的来源，不直接使用 |
| `train_subset.jsonl` | 5403 | L3–5 选集 | — |
| `heldout_200.jsonl` | 200 | **M3 SFT 验收** | 工具格式成功率、SFT vs 零样本 |
| `sft_traces.jsonl` | 6056 | **SFT 训练** | — |
| `e3_train.parquet` | 5203/5195 | **所有 RL 臂训练** | — |
| `e3_val_math500_100.parquet` | 100 | 训练内 val | 只看曲线趋势，不作报告结论 |
| MATH500 | 500 | **效应量主战场** | 同分布、n 够、有 level 分层 → 主结论 |
| AIME24/25 | 60 | **污染与泛化兜底** | 只看方向有没有塌，不看单点差异 |
| GSM8K200 | 200 | **退化哨兵** | 只看有没有掉 |
| 预算诊断 70 题 | 70 | post-hoc 机制诊断 | 禁止回流训练 |

## Q10. 已知的数据层面弱点（诚实清单）

1. **MATH500 的 100 题做过训练内 val**——无早停无选点，风险有限但存在；主口径用 760 题。
2. **13-gram 去污染过度剔除**：216 条里绝大多数是模板句而非真重复；唯一的 "exact" 是
   normalizer 丢失 `+/-` 造成的假阳性（Q2）。方向保守，不影响结论。
3. **教师覆盖缺口**：L5 只覆盖 59.3% 的题，SFT 数据在最难档上系统性偏薄。
4. **RL 与 SFT 同题**：这是设计选择（隔离变量），但意味着无法回答"更多新题能否继续提升"。
5. **没有 CoT-only 的 RL 对照臂**：SFT→E3 的 +12.8pt（FULL760）分不出"学会用工具"与
   "RL 本身提升推理"。M9 的 E3-NoTool 就是补这个洞，用完全相同的 5203 题池，只去掉工具。
6. **AIME n=30**：只能做定性兜底，无法支撑任何效应量声明。

---

## 复现命令

```bash
python data/download_convert.py                      # → data/processed/*.jsonl
python data/dedup_check.py                           # → math_train_clean.jsonl + contamination_report.md
python data/dedup_check.py --self-test               # 检出逻辑自测
python data/select_train_subset.py                   # → train_subset.jsonl (L3-5, 5403)
python sft/make_splits.py                            # → heldout_200.jsonl + sft_pool.jsonl
python rl/prepare_data.py                            # → rl/data/*.parquet + manifest.json
python analysis/tool_use_conditional_accuracy.py     # 分 level / 分工具使用的条件准确率
```

## 文件索引

`data/README.md`（schema 与文件清单）、`data/contamination_report{,_m4,_subset}.md`、
`rl/data/manifest.json`（sha256 与切分断言结果）、`sft/data/gen/gen_stats.json`、
`eval/m7_eval_config.json`、`eval/diagnostic_panel_v1.jsonl`、
`analysis/tool_use_conditional_accuracy.json`、`reports/qa_log.md` Q18–Q20。


项目比较过约 2,500 条和 6,000 条数据，工具错误率从约 22% 进一步降到 19%
我先从完成去污染的数学训练子集中留出 200 题做验证，用剩下约 5,200 题生成教师轨迹。教师是本地部署的 Qwen3-8B，每题采样两次，生成过程中真实执行 Python 沙箱，把执行结果或报错返回模型，让它继续推理。  
总共生成约 1 万条候选，再用数学 verifier 和工具交互规则做拒绝采样：保留最终答案正确、至少有一次成功工具调用、未截断、长度符合学生训练预算且调用参数合法的轨迹，最终得到约 6,000 条。之后制作 SFT 样本，屏蔽工具返回的 loss，进行 LoRA 冷启动。  
我没有把出现过工具错误的轨迹全部删除，只要后续恢复并最终答对就可以保留，因此数据里也包含报错恢复样本。