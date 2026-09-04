# GRPO 成功率、失败轨迹与预算诊断会话总结

> 日期：2026-08-28 至 2026-09-04  
> 范围：E3 GRPO baseline 的训练后表现、失败轨迹、组内零梯度问题，以及 70 条预算/截断失败的长预算诊断  
> 性质：结果总结与诊断记录；不包含新模型训练

## 1. 本次会话回答的问题

本次会话依次讨论了以下问题：

1. GRPO 训练完成后的模型成功率是多少，失败轨迹有多少？
2. 是否检查过失败轨迹的中间步骤，主要失败原因是什么，是否有统计？
3. 针对这些错误可以采用哪些改进方法？
4. GRPO 训练中组内全对、全错以及无梯度 rollout 是否常见？
5. 冻结的 `budget_exhaustion_or_truncation` 类是否真的由预算不足直接造成？

最后一个问题通过一次限定范围的长预算生成进行了验证。该诊断只覆盖既有 E3 greedy 的 70 条失败轨迹，没有训练新模型，也没有改变 M7 canonical 结果。

## 2. GRPO 模型成功率与失败数

需要区分训练期间的固定验证集结果和训练完成后的统一全量评测结果。

| 评测口径 | 正确数 | 总数 | 成功率 | 失败数 |
|---|---:|---:|---:|---:|
| M4 训练期间固定 MATH500-100，step 200 greedy | 76 | 100 | 76.0% | 24 |
| M7 统一 FULL760，E3 step 200 greedy | 557 | 760 | 73.29% | 203 |

FULL760 的分来源结果如下：

| 来源 | 正确数 | 总数 | 成功率 | 失败数 |
|---|---:|---:|---:|---:|
| AIME 2024 | 4 | 30 | 13.33% | 26 |
| AIME 2025 | 4 | 30 | 13.33% | 26 |
| GSM8K | 166 | 200 | 83.0% | 34 |
| MATH500 | 383 | 500 | 76.6% | 117 |
| 合计 | 557 | 760 | 73.29% | 203 |

采样评测的 FULL760 指标为 pass@1 74.28%、pass@2 81.07%、pass@4 85.66%。这些值描述多次采样下的覆盖能力，不能与单条 greedy 的 73.29% 直接混成同一个成功率。

## 3. 是否查看了失败轨迹的中间步骤

是。诊断不是只比较最终答案，而是保留并检查了：

- 完整 assistant/tool message 序列；
- 每轮 Python 工具代码、执行输出与错误；
- 工具调用次数、成功次数、解析错误和 sandbox/timeout 错误；
- response token 数、停止原因和真实截断标记；
- AST 归一化后的重复工具代码；
- strict verifier 提取结果和最终正确性。

M7 FULL760 greedy 的 203 条失败中，冻结 coarse taxonomy 将 70 条提出为 `budget_exhaustion_or_truncation`。由于 coarse proposal 可能把“先发生语义错误、随后耗尽预算”误当成预算原因，本次会话专门对这 70 条做了因果方向的诊断。

## 4. 原 70 条预算/截断失败的重新拆分

70 条轨迹在 interaction ledger 中都确实发生了截断。原始停止信号为：

- response length：63 条；
- assistant/tool turn budget：7 条。

为了得到互斥的主类，重复归一化工具代码优先于最终停止信号：

| 互斥主类 | 数量 | 定义 |
|---|---:|---|
| 真实 response-token 截断 | 58 | 到达 3072 response-token 上限，且未被重复调用主类覆盖 |
| assistant/tool 轮次耗尽 | 2 | 到达调用/助手轮次上限，且未被重复调用主类覆盖 |
| 重复归一化工具调用 | 10 | 至少两次工具代码的 AST 归一化 hash 相同 |

这里的 58/2/10 是互斥分类；63/7 是正交停止机制，二者不能相加或互相替代。

## 5. 长预算诊断设计

诊断使用同一个 E3 step-200 checkpoint，并固定以下因素：

- tokenizer、prompt 和 tool schema；
- Python sandbox 行为；
- greedy decoding 和 sample seed；
- strict verifier 与评分口径。

只改变预算：

| 预算 | 原值 | 诊断值 |
|---|---:|---:|
| response tokens | 3072 | 6144 |
| user/tool turns | 4 | 6 |
| assistant turns | 5 | 7 |

生成在独立、唯一命名的 tmux run 中完成。selection、raw、scored、comparisons 均为精确 70 条，没有缺失或重复 ID；父 run 和诊断 run 的 hash ledger 均通过。

## 6. 长预算诊断结果

| 长预算后的结果 | 数量 | 解释 |
|---|---:|---|
| wrong → correct | 16 | 支持预算是该次失败的直接原因 |
| 正常结束但答案仍错 | 14 | 已观察到完整的语义/策略错误 |
| 再次到达 6144-token 上限且仍错 | 38 | 仍受 token 截断，不能证明是完整推理后的纯语义错误 |
| 再次耗尽轮次且仍错 | 2 | 仍受 turn budget 截断 |

因此：

- 16/70，即 22.86%，增加预算后转为正确；
- 54/70 仍然错误；
- 按会话约定的二元规则，可将 16 条标为“预算直接原因”，54 条标为“增加预算后仍失败”；
- 但从机制上，54 条中只有 14 条是未被截断的 completed semantic error，另外 40 条仍是 censored trajectory，不应表述成“已完整推理并被证明语义错误”。

按原始互斥类统计修复率：

| 原始主类 | 转正确 | 分母 | 修复率 |
|---|---:|---:|---:|
| response-token 截断 | 9 | 58 | 15.52% |
| assistant/tool 轮次耗尽 | 2 | 2 | 100% |
| 重复归一化工具调用 | 5 | 10 | 50% |

轮次类只有 2 条，不能把 100% 外推为总体规律。16 条转正确轨迹中，15 条正常结束；1 条已经给出正确 boxed answer，随后才到达 assistant-turn 上限，因此同时记录为 correct 和 truncated。

## 7. 中间步骤揭示的主要失败原因

### 7.1 语言循环与重复推理

38 条再次触发 response-token 上限的失败中：

- 34 条完全没有调用工具；
- 30 条至少三次输出完全相同的非短文本行；
- 常见表现是重复同一句解释、反复“重新分析问题”，或者已经得到候选答案后继续机械复述。

这表明大量 token 截断并不是“只差一点推导空间”，而是生成进入了 verbal degeneration。继续单纯提高 token 上限，多数情况下只是给循环更多空间。

### 7.2 重复工具调用与无效恢复

两条再次耗尽 turn budget 的轨迹都在反复尝试工具。原始 10 条重复代码轨迹中，5 条转正确；剩余 5 条分别为：

- completed wrong：2；
- persistent token limit：2；
- persistent turn limit：1。

说明更长轮次能修复一部分过早停止，但不能解决重复代码、错误参数或没有吸收工具反馈的问题。

### 7.3 工具错误后的无依据作答

14 条正常结束但仍答错的轨迹中，6 条至少发生过一次 sandbox 或 timeout 错误。人工阅读发现的常见模式包括：

- 工具执行失败后没有修复代码，直接猜测最终答案；
- 多次执行相同或等价的错误代码；
- 数值结果与题目象限、约束或解析形式矛盾，却仍将其当作结论；
- 工具只验证了模型自己构造的错误公式，形成“错误推导—工具确认—错误答案”的闭环。

### 7.4 纯语义、代数和约束错误

其余 completed-wrong 轨迹主要包括：

- 组合计数遗漏或错误使用等可能性；
- 几何关系、角度或尺度关系错误；
- 递推、模运算和有理数表示理解错误；
- 忽略题目要求的象限、定义域、最值或表达形式；
- 工具返回值正确，但模型误读、覆盖或没有采用。

## 8. GRPO 组内全对、全错与无梯度 rollout

这个现象在 E3 中并不少。GRPO 对同一 prompt 的 8 条 rollout 做组内相对优势；如果组内用于 advantage 的标量 reward 完全相同，则标准化优势为零，该组不会提供有效 policy-gradient 信号。

E3 全部 200 个训练 step 的离线统计为：

| 按 answer accuracy 划分 | 平均比例 |
|---|---:|
| 组内全对 | 33.74% |
| 组内全错 | 17.70% |
| mixed | 48.55% |

但“全对/全错”等于“无梯度”并不完全准确。本项目 E3 的实际标量 `score` 还包含格式项，因此约 7.38% 的 group 虽然 answer accuracy 一致，`score` 仍有方差，仍可产生 format-gradient。按真正进入 GRPO advantage 的 `score` 统计：

- 零方差 group：44.07%；
- informative group：55.93%；
- 最后 25 step informative group：约 47.25%，即零方差约 52.75%；
- 历史最低单步 informative group：约 39.06%。

step 200 单步的 accuracy 口径为全对 43.75%、全错 18.75%。因此可以确认：无梯度 rollout 比例较高，平均接近一半，并且训练后期有上升压力。

## 9. 针对这些错误的改进方向

### 9.1 不把“继续加预算”作为主要方案

加倍 token 和提高轮次只修复了 16/70。更值得优先处理的是：

- 检测连续重复文本、重复 AST 代码和长期无进展状态；
- 达到可信 boxed answer 后及时结束；
- 对完全相同的工具调用设置重复上限；
- 把“仍在循环”与“真正需要更多推理 token”分开监控。

这些修改会改变推理协议，应作为新的对照实验，而不能回写 M7 frozen 结果。

### 9.2 改善工具错误恢复

- 将 syntax、timeout、sandbox 和空输出区分记录；
- 工具失败后要求先解释错误并修改代码，禁止无依据直接给答案；
- 判断新调用是否真正改变代码、输入或求解方法；
- 训练和评测都继续保留工具错误、未采用结果和重复调用的可见指标。

### 9.3 针对语义错误构造训练信号

可以从原训练 split 中构造以下样本：

- 工具结果与自然语言结论冲突时的纠错；
- 约束、象限、定义域和最值检查；
- 工具失败后的最小修复；
- “工具算对但最终答案采用错误”的反例。

本次 70 条评测题只能用于诊断，不能作为 SFT、RL、rejection sampling 或 hard-negative 数据回灌。任何新训练集都必须只从原训练 split 构造，并重新运行污染检查。

### 9.4 减少 GRPO 零方差计算浪费

动态过滤应以实际 `score` 的组内方差为准，而不能只过滤 accuracy 全对/全错组。可考虑 DAPO-style 流程：在 actor update 前丢弃 `std(score)=0` 的 group，并用同一 actor snapshot 补采样，直到获得足够的 informative groups。

该方向已经作为 E7 设计审查的一部分，但本次会话没有实现或启动 E7。离线估算 informative rate 为 55.93%，若保持有效 batch 大小，理论 prompt 生成倍率约为 1.79 倍，因此节省 update 浪费的同时会增加采样和调度复杂度。

### 9.5 谨慎设计 turn-level credit

只奖励“多调用一次工具”容易诱发 over-calling。已有 E5 结果显示 turn-level credit 确实改变了行为，却没有带来最终准确率提升，并显著增加工具调用。因此后续信用分配需要同时约束：

- 新调用是否提供新增信息；
- 工具结果是否被正确采用；
- 重复/无效调用是否获得正 credit；
- 最终 answer correctness 是否真实改善。

## 10. 结论边界

1. E3 的 canonical greedy 成功率仍是 557/760，即 73.29%；本次只重跑失败子集，不能据此改写整体成功率。
2. 70 条 coarse budget/truncation failure 中，只有 16 条因扩展预算转正确；预算不足是部分原因，不是主要的统一解释。
3. 40 条在更长预算下仍被截断，且多数表现为语言循环，因此不能简单重分类为纯语义错误，也不值得无限增加预算。
4. 14 条未截断但错误的轨迹提供了明确的语义、工具使用和错误恢复案例。
5. greedy rerun 中只有 17/70 与原输出保持完整前缀一致，说明 serving 数值非确定性限制了严格反事实解释；16 条转正确是强诊断证据，而非同一轨迹续写的数学证明。
6. GRPO 的零方差 group 平均为 44.07%，计算浪费显著；正确优化对象是实际 `score` 方差，而不是简单的全对/全错标签。

## 11. 证据指针

- M4 E3 训练总结：`rl/runs/e3_grpo_baseline_20260819_224555/summary.md`
- M4 E3 全量训练指标：`rl/runs/e3_grpo_baseline_20260819_224555/metrics.json`
- M7 FULL760 pass@k：`eval/runs/m7_unified_eval_20260824_201032/metrics/pass_at_k.json`
- M7 bad-case taxonomy：`reports/03_badcase_taxonomy.md`
- 长预算诊断计划和边界：`plans/M7_BUDGET_DIAGNOSTIC.md`
- 长预算诊断配置：`eval/e3_budget_diagnostic_config.json`
- 长预算诊断总指标：`eval/runs/e3_budget_diagnostic_20260828_095658/metrics.json`
- 逐题前后比较：`eval/runs/e3_budget_diagnostic_20260828_095658/comparisons.jsonl`
- 包含完整中间步骤的评分轨迹：`eval/runs/e3_budget_diagnostic_20260828_095658/scored.jsonl`
- 诊断完整性 ledger：`eval/runs/e3_budget_diagnostic_20260828_095658/hashes.sha256`

诊断实现相关回归最终为 **92 passed, 1 skipped**；未启动训练、额外 seed、全量评测或数据回灌。
