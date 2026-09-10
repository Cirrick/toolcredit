# 秋招简历与面试追问会话总结（2026-09-09）

本次范围：基于已有项目内容修改国内秋招简历，解释数据构建、SFT/RL 编码、轮级信用分配、token 与工具预算；离线统计既有教师及评测轨迹。未训练、未重新生成、未改变评测协议或 canonical 产物。用户先要求只在聊天中讨论，最后明确授权把会话总结写入 `reports/`；该授权不扩展到代码、实验或其他未提交工作。

## 1. 简历组织与措辞决定

当前三条文案维护在 [resume_bullets_autumn.md](../resume_bullets_autumn.md)，不在本文复制第二份。

- 不给每条 bullet 加“训练管线/数据构建”等小标题。第一条交代整体工程，第二条按 SFT→GRPO 写过程和效果，第三条写探索、问题、诊断、修正与效果。
- 保留约 6,000 条数据规模，教师型号、精确到个位的条数、760 题分母可留到面试。省略细节不改变事实：6,056 是轨迹数，覆盖 3,716 个独立题目，且并非每条都有多次工具调用。
- “数据去污染”足够用于简历，展开为归一化精确匹配 + **词级** 13-gram 重叠检测。它检查题面污染，与数学答案 verifier 不同。
- 用“针对训练中出现的……”替代“发现初版方案……”，不暗示迭代次数，也不声称做过未实施的方案。
- 第三条可以省略 92.9%→0.8%、100%→8.7%，写“大幅降低过度调用与轨迹截断比例”；不用“显著”暗示统计检验。追问时保留原始结果，不隐藏失败分支。
- 日期按真实起止月份填写。最早 Git 提交为 2026-07-12，本次建议 `2026.07—至今`；如此前确有直接相关调研，可以按事实说明，不人为凑月份。

三项指标来自不同对照：raw→SFT 探针工具错误率 0.3444→0.1925（错误调用数/总调用数）；MATH500 SFT→E3 为 297/500→383/500（59.4%→76.6%，+17.2pt）；FULL760 v1→v2 用满调用预算为 706/760→6/760，截断为 760/760→66/760。

## 2. 6,000 条如何构建，是否可以称蒸馏

这是工具调用冷启动 SFT 的数据规模，目标是改善交互可靠性，再进入 GRPO。不能仅靠数量判断是否足够。项目比较过 2,500 与约 6,000 条训练数据，探针工具错误率约 22%→19%，因此使用全量筛选结果；没有证明数据规模已经饱和或最优。

| 环节 | 实际做法 |
|---|---|
| 题目池 | 已去污染的 5,403 题 MATH L3–L5 子集 |
| 隔离 | 留出 200 题，剩余 5,203 题生成教师数据 |
| 教师 | 本地 Qwen3-8B，通过 SGLang 的 OpenAI-compatible 接口访问，不是购买托管 API 服务 |
| 采样 | 每题 2 条、temperature=0.7，共 10,406 条原始轨迹 |
| 交互 | 模型调用真实 Python 沙箱，执行结果/报错回填后继续生成 |
| 拒绝采样 | 排除轮数截断；要求至少一次成功工具调用、参数合法、严格 boxed verifier 判对、累计响应长度 ≤3,072 token |
| 保留 | 6,056 条，产率 58.2%，覆盖 3,716 题；L3/L4/L5 分别 2,128/1,995/1,933 条 |
| 恢复样本 | 797 条含工具报错，但后续满足成功调用与最终答对条件 |
| 学生训练 | LoRA SFT，工具返回不计 loss；保留结果为 RL 统一起点 |

过滤原因按顺序互斥记账：轮数截断 374、没有成功调用 2,083、参数错误 1、答案错误 1,884、超长度 8；合计剔除 4,350 条。不能把这些计数当作可重叠事件各自的总体发生率。

可以称“轨迹蒸馏/序列级蒸馏”：用教师生成的 assistant 序列作为学生监督目标，无教师 logits/KL 蒸馏。生成是流程的一部分，简历的更直白选项是“利用教师模型生成工具推理轨迹，经拒绝采样保留约 6,000 条，用于 LoRA SFT 冷启动”。不声称逐步验证了每个推理步骤的语义正确性。

面试回答：

> 我从去污染后的训练子集中留出验证题，对剩余约 5,200 题用本地教师每题采样两条，生成过程真实执行沙箱。再按答案正确、至少一次成功调用、无轮数截断、参数合法和学生长度预算筛选，得到约 6,000 条。报错后恢复且最终答对的轨迹也保留，帮助学生学习根据反馈继续推理。这些轨迹用于监督蒸馏，没有使用教师概率分布。

来源：`sft/data/gen/gen_stats.json`、`sft/data/gen/raw_traces.jsonl`、`sft/data/sft_traces.jsonl`、`sft/gen_trajectories.py`、`data/tool_gain_probe.py`、`plans/M3.md`。

外部术语参照：[Sequence-Level Knowledge Distillation](https://aclanthology.org/D16-1139/)。[DeepSeek-R1 论文](https://www.nature.com/articles/s41586-025-09422-z)也使用数千条初始冷启动数据，但该先例不证明本项目 6,000 条已经足够，更不是数据规模最优性的依据。

## 3. 统一 SFT/RL 编码是什么意思，为什么教师没有直接走 AgentLoop

模型接收的是带角色、轮次及工具标记的 token 序列。模型推理/调用/答案参与监督，prompt 和工具返回保留在上下文中但不计 loss。工具返回仍影响后续预测；loss mask 不等于 attention mask。

教师生成复用了 M1 已有的 `run_conversation`，Python 循环调用本地 SGLang、执行沙箱并保存 messages。M3 计划明确采用复用路线；不是教师不能接入 veRL AgentLoop，也没有比较并证明两条生成路线谁更优。

`sft/trace_tokenizer.py` 在制作 SFT 样本时调用真实 `ToolAgentLoop`：重放服务器按顺序返回已记录的教师回复，重放工具返回已保存的沙箱输出；不重新生成、不重新执行代码。由 loop 组装 `prompt_ids`、`response_ids`、`response_mask`，再映射为 labels。代码断言所有回复和工具结果被完整消费，测试覆盖单次及多次工具调用的监督位置。

这是减少自行拼接带来的格式/mask 差异的工程处理，不是原创 AgentLoop；也不代表教师与 RL 的所有预算配置、原始生成 token 序列或策略分布完全相同。简历可以保留“统一多轮交互编码格式”，篇幅紧时只保留“屏蔽工具返回内容的训练损失”，面试再展开。

面试回答：

> 教师采样复用前期的生成脚本，负责真实交互。制作 SFT 样本时，把记录好的回复与工具结果通过 veRL AgentLoop 重放，复用它的上下文拼接和 mask 逻辑，减少 SFT 与 RL 的编码差异。模型学习自己应该生成的内容，工具返回只作为上下文。

证据：`sft/trace_tokenizer.py`、`sft/test_trace_tokenizer.py`、`data/tool_gain_probe.py`、`plans/M3.md`。

## 4. 轮级信用如何解释，是否属于 reward hacking

初版公式的文字版：在轨迹级 GRPO 优势上，叠加该轮工具执行成功及结果采纳得分相对组内同轮位置均值的修正。结果采纳由启发式判断，不是 PRM，也不是准确测量每步真实因果贡献。

诊断发现非守恒修正可能使提前结束的轮次受到负修正，轨迹净修正与调用数相关，形成额外调用激励。v2 改为同一轨迹内按 policy token 数加权的零和修正，并加入零方差组门控、重复代码与答案之后调用的打分约束。通俗解释是“保持轨迹总信用不变，重新分配各轮信用”；严格含义是修正项的 token 加权和为零，不是策略梯度不变或最优策略不变定理。

仅凭调用次数增加不足以认定 reward hacking。本项目可讨论为隐式激励下的代理信号投机，简历优先写“非守恒信用修正引入隐式调用激励”。该修正在 advantage 中，不是显式 outcome reward 增长或 verifier 漏洞。无需也不能声称模型主观上有意作弊。E4 无信息执行仍获得 exec bonus 是更直接的显式奖励投机案例。术语参照：[DeepMind specification gaming](https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/)。

因果边界：v2 同时改变守恒、门控和打分条件，不能把行为改善单独归因于守恒这一项。相对 E3，v2 FULL760 greedy 准确率差为 +0.79pt，95% CI [−1.84,+3.42]pt，未检出显著提高，也没有证明等效。零方差时应说 GRPO 原始 policy-gradient 项为零，不能忽略仍存在的 KL loss。旧报告中“只改守恒/唯一梯度”的简写不能支持更强结论。

证据：`rl/custom/turn_advantage.py`、`rl/custom/turn_advantage_v2.py`、`plans/M8.md`、`reports/qa_log.md` Q12、`eval/runs/m8_e5v2_eval_20260905_134737/metrics/m8_headline.json`。

## 5. 教师 token 分布与截断：本次离线重算

读取 10,406 条原始轨迹，用当时的学生 tokenizer 和 `sft.gen_trajectories.response_token_len` 重算。该函数计算 chat-template 编码后的完整对话长度减去初始 prompt 长度，包含工具返回及格式标记；不是仅 assistant 的 loss token 数，也不是 AgentLoop 重放后序列长度的另一次测量。6,056 条保留轨迹与已有 `response_tokens` 逐条比较，**0 mismatch**；原始记录无空 messages。

| 指标（token） | 原始 10,406 条 | 保留 6,056 条 |
|---|---:|---:|
| 均值 | 751.9473 | 580.2483 |
| 中位数 | 564 | 492 |
| P90 | 1,432.5 | 981.5 |
| P95 | 2,050 | 1,229.5 |
| P99 | 3,078.8 | 1,874.15 |
| 最大值 | 9,556 | 2,980 |

分位数使用 `numpy.percentile` 默认线性插值。

| 累计响应阈值 | 原始超限数/比例 | 保留超限数/比例 |
|---|---:|---:|
| 1,024 | 1,884 / 18.10% | 521 / 8.60% |
| 1,536 | 910 / 8.74% | 138 / 2.28% |
| 2,048 | 549 / 5.28% | 34 / 0.56% |
| 3,072 | 109 / 1.05% | 0 |
| 4,096 | 28 / 0.27% | 0 |

上述是已记录轨迹的长度覆盖，不是更改生成预算后的实际截断率。保留数据本来经过长度筛选，不能仅凭其 0 超限声称预算没有限制。

- 原始 `truncated=True` 共 374/10,406 = **3.59%**，全部最终 `finish_reason=tool_calls`，对应轮数预算耗尽。
- 原始最终 `finish_reason=length` 共 213/10,406 = **2.05%**，对应最后一次模型请求触及生成长度限制（每次请求上限 2,048）。二者无重叠，合计 587/10,406 = **5.64%**。
- 完整 finish reason 分布：stop 9,804、tool_calls 389、length 213。原始数据只保存最后一次请求的原因，不能完整核查每个中间请求是否触顶。
- 当时过滤表中的 374 条“截断”不是累计响应 3,072 超限数。通过前置质量筛选、因累计长度超限而丢弃的只有 8 条；8/(6,056+8) = **0.13%**，分母来自顺序筛选记账，不是本次重跑 verifier。
- 保留数据 6,056 条最终均为 stop。不能因此推断所有中间步骤都正确，或中间请求从未触顶。

长度配置需分开回答：教师每次请求 2,048；SFT 筛选累计响应 3,072；SFT 完整序列上限 4,096；RL prompt 上限 1,024、累计 response 3,072。模型最大上下文窗口不等于本实验采用的预算。

复核命令（项目根目录，CPU、只读、输出到终端）：

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false conda run --no-capture-output -n toolcredit python - <<'PY'
import json
from collections import Counter
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from sft.gen_trajectories import response_token_len, STUDENT_TOKENIZER_PATH

raw = [json.loads(s) for s in Path('sft/data/gen/raw_traces.jsonl').open()]
kept = [json.loads(s) for s in Path('sft/data/sft_traces.jsonl').open()]
tokenizer = AutoTokenizer.from_pretrained(STUDENT_TOKENIZER_PATH, local_files_only=True)
lengths = [response_token_len(tokenizer, r['messages']) for r in raw]
by_key = {(r['id'], r['sample_idx']): n for r, n in zip(raw, lengths)}
assert len(by_key) == len(raw)
assert all(by_key[(r['id'], r['sample_idx'])] == r['response_tokens'] for r in kept)
for name, values in [('raw', lengths), ('kept', [r['response_tokens'] for r in kept])]:
    a = np.asarray(values)
    print(name, len(a), 'mean', a.mean(), 'max', a.max())
    print('P50/P90/P95/P99', np.percentile(a, [50, 90, 95, 99]))
    print('exceeds', {n: int((a > n).sum()) for n in [1024, 1536, 2048, 3072, 4096]})
print('finish reasons', Counter(r['finish_reason'] for r in raw))
print('truncated', sum(bool(r['truncated']) for r in raw))
print('truncated reasons', Counter(r['finish_reason'] for r in raw if r['truncated']))
print('kept finish reasons', Counter(r['finish_reason'] for r in kept))
PY
```

## 6. GRPO 训练后的评测截断与放宽预算结果

本节基于 E3 step-200 的已有 scored JSONL 逐条汇总；核对顶层和 `turn_credit_ledger` 的停止原因一致。不是 E5 的高截断实验，也不是训练期截断率。

| 评测 | 轨迹数 | 总截断 | token 原因 | 轮数原因 |
|---|---:|---:|---:|---:|
| FULL760 greedy | 760 | 73 / 9.61% | 66 / 8.68% | 7 / 0.92% |
| MATH500 greedy | 500 | 40 / 8.00% | 34 / 6.80% | 6 / 1.20% |
| FULL760 sampled | 3,040 | 152 / 5.00% | 122 / 4.01% | 30 / 0.99% |
| MATH500 sampled | 2,000 | 96 / 4.80% | 74 / 3.70% | 22 / 1.10% |

sampled 每题 4 条。token 原因合并 `response_length` 和 `tool_response_would_exceed_response_length`；FULL760 分别为 117+5，MATH500 为 71+3。greedy token 原因全部为 response_length，轮数原因全部为 assistant_turn_budget。FULL760 greedy 的 token 原因占全部截断 66/73 = **90.4%**，但停止原因不等于答错的唯一根因。

73 条截断中 3 条仍答对，所以后续重跑选取 70 条截断且答错的轨迹。其原停止原因是 token 63、轮数 7。旧报告中的 58/2/10 是“重复代码优先”的互斥行为分类，不能拿来替代 63/7 的停止机制统计。

后续诊断同时改变累计 response **3,072→6,144**、工具轮数 **4→6**、assistant 轮数 **5→7**；固定同一模型与其他协议。不是单独增加 token，也不是重新训练。

| 重跑后结果 | 数量/70 | 比例 |
|---|---:|---:|
| wrong→correct | 16 | 22.86% |
| 正常结束但仍答错 | 14 | 20.00% |
| 仍错且再次 token 截断 | 38 | 54.29% |
| 仍错且再次轮数耗尽 | 2 | 2.86% |

原 token 截断失败中 11/63 = **17.46%** 转正确，原轮数耗尽中 5/7 转正确。16 条转正确中 15 条正常结束，1 条先给出正确答案后又触及轮数上限；因此重跑总截断是 **41/70**，其中 40 条仍错，不能把 40 当作总截断数。

38 条持续 token 截断失败中，既有诊断发现 34 条未调用工具，30 条有规范化非平凡文本行至少重复三次。这支持部分预算消耗来自语言循环，但 40 条仍被预算终止的失败不能全部视为已完成推理后的纯语义错误。

因果边界：联合增加两类预算；greedy 重跑只有 17/70 与原输出保持完整前缀一致。结果支持“放宽预算观察到部分修复”，不是只增加 token 的独立效果，也不是严格原轨迹续写证明。只重跑选定失败题，不能把 16 条直接加到 canonical 总分或声称新的全量 pass@1。评测题永久不回灌训练。

证据：`eval/runs/m7_unified_eval_20260824_201032/scored/e3_grpo_baseline_step_200/{greedy,sampled}/*.jsonl`；`eval/runs/e3_budget_diagnostic_20260828_095658/{metrics.json,comparisons.jsonl,scored.jsonl}`；`plans/M7_BUDGET_DIAGNOSTIC.md`。更完整的行为诊断见 [既有 GRPO 诊断总结](grpo_failure_diagnostic_session_summary.md)。

## 7. token 和工具次数的面试回答

token 预算回答：

> 我设定 prompt 1,024、多轮累计 response 3,072，作为主要实验共享预算。后来复核一万多条原始教师轨迹，平均约 752，P95 约 2,050，P99 约 3,079，说明对教师数据有较充分覆盖。但学生经过 RL 后的行为不一定相同，所以我进一步检查评测截断：greedy 全量约 8.7% 因 token 上限停止。对 70 条截断失败同时放宽 token 和调用预算后，16 条转正确，其余不少仍在重复推理。因此该预算是有数据支持的成本与覆盖率折中，不能说最优，也不能靠无限增加长度解决全部错误。

不要把后验复核改写为“当初按 P99 精确选了 3,072”；原始教师数据本身已经受到单次生成和轮数预算限制，均值/分位数也不能证明无限制生成的需求。更系统的预算搜索应在训练/验证数据上比较准确率、截断、实际长度与耗时，再冻结正式评测配置；本项目没有完成这种完整搜索。

工具预算回答：

> 实际是最多 4 次工具调用、5 轮模型回复。第 4 次工具执行后，需要再给模型一轮读取结果并生成最终答案，因此 assistant 轮数多一轮。4 次是短程数学工具推理的工程预算，兼顾计算、验证、错误重试与 rollout 成本，并在各实验固定。它不是搜索出的最优值。后续 E3 greedy 中只有 10/760 条用满 4 次调用，7/760 条因轮数限制停止；但更长程任务可能需要更多轮。

E3 greedy 工具调用次数直方图：0/1/2/3/4 次分别为 **202/478/53/17/10**。用满预算不等于预算耗尽导致停止；第 5 轮允许生成答案，不保证模型一定停止调用。数字是后续观测支持，不能伪装成最初选定 4 次的先验依据。

配置证据：`rl/configs/e3_grpo_baseline.yaml`、`sft/train_sft.py`、`sft/gen_trajectories.py`、`data/tool_gain_probe.py`。本次仅做上述离线统计和文档整理，未运行新的 GPU 实验。
