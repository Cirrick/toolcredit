# 问答日志（Q&A Log）

> 项目进行中负责人提出的问题与解答记录，按时间追加。这些是**真实发生过的疑问**，
> 也是面试问答的一手素材（比 PLAN 里预设的 Q&A 表更贴近真实提问方式）。
> 约定：每次用户提问且解答有沉淀价值时，追加一条到本文件（见 CLAUDE.md 约定 #8）。

---

## Q1（2026-07-13，M1 模型选型）：不用 Qwen3 系列，面试官会不会质疑为什么不用最新模型？还是 Qwen2.5 就够了？

**答**：面试官更在意"你为什么这么选"有没有实验设计层面的理由，不在意新旧本身。
选基座的三段论（最终答案）：

1. **能力天花板是实验设计变量，不是越强越好**——GRPO 需要组内奖励方差，模型太强会把
   MATH 打满（全对组无梯度），被迫换更贵的 AIME 级数据。选"会做一半"的模型是有意为之；
2. **工具增益要存在**——模型心算越强，解释器的边际增益越小（与"不用 GSM8K"同一逻辑）；
3. **社区参照系**——训练曲线异常时有外部参照可对比。

最终决策 **Qwen3-1.7B**：家族新（消掉"为什么用旧模型"）、M0 已端到端验证（零管线风险）、
算力最省（M4 重调余量最大）、模型弱→含错轨迹多（对 E5 有利）。"为什么小"的回答：单卡
3 周预算下选择跑完完整对照矩阵而非在大模型上跑一半；研究问题与规模正交；余力时 4B 复跑。

**证据**：plans/M1.md Context、reports/technical_report.md §3.2。

---

## Q2（2026-07-13，M1 probe 结果）：为什么会出现 probe 全 level 负增益？什么是负增益？

**答**：工具增益 = 同批题上 `TIR pass@1 − CoT pass@1`；负增益 = 给了工具反而更差
（如 L4：纯推理 0.75，带工具 0.59）。直觉上工具"不用可以不用"只会加分，但零样本模型
一旦用工具就打开新的失败面：

1. 代码执行失败（报错率随难度 17%→46%）：漏 import、缩进错误、把无状态沙箱当 notebook
   （引用上次调用的变量）；
2. 报错→重试→烧光 4 次调用上限→截断判 0（"程序性死亡"，纯 CoT 没有）；
3. 代码跑通但逻辑错，模型盲信输出；
4. 温度采样放大代码脆性（代码错一个 token 即崩，文字推理更鲁棒；greedy 下报错率 48%→18%
   但增益仍负，说明是放大器不是根因）。

类比：给没学过计算器的人发计算器且按错扣分——裸算 75 分，带计算器 59 分。结论不是
"计算器没用"，是"还不会用"：同一难度层内，工具全部成功执行的采样正确率显著更高
（L5 翻倍：0.46 vs 0.23）。这正是 SFT 冷启动（M3）的定量依据。

**证据**：reports/01_tool_gain.md（图 B 分解）、data/probe/metrics.json。

---

## Q3（2026-07-14，M2 计划评审）：M2 会涉及 reward hacking 吗？还是后续计划才处理？

**答**：会，但角色不同——**M2 修防御工事、量化攻击面；hacking 现象在 M4–M6 训练中出现
（没有优化压力就没有 hacking）；M7 成文**。四段贯穿线：

- **M2**：verifier 200 例审计量化最大攻击面（假阳性是 hacking 入口——verifier 哪里判松，
  模型就朝哪里优化；假阴性只稀释信号）。具体设计决策：训练 reward 严格只认 `\boxed{}`，
  评测可宽松提取——否则模型会学"多写几个候选数字撞答案"。格式分压 0.1 防"只刷格式"；
  λ_exec 默认 0，E4 才打开这个已知风险面；
- **M4**：`reward/mean` 涨而 `pass@1(eval)` 不动 = 第一嫌疑信号，立即抽轨迹人检；
- **M5（E4）**：主动打开执行成功分当诱饵，观察是否被刷；
- **M7**：`04_reward_hacking.md`，≥3 案例，每例"现象→成因→修→复验"闭环。

---

## Q4（2026-07-14，M2 计划评审）：为什么 reward 要这么设计？为什么不设计成 sparse reward 或其它 reward？

**答**：我们的 reward **本质就是 sparse outcome reward**（主项 1.0·answer_correct 只在
轨迹末尾给），这是 RLVR 路线（不训 RM，避免 RM 偏置与更软的 hacking 面）。真正的设计点：

- **为什么加 0.1 格式分（而非纯 sparse）**：GRPO 组内全错→优势全 0→无梯度。全错里混着
  "数学错"（RL 要修的）和"接口没学会"（boxed 没写、tool call 吐错格式）两种失败；
  格式分把后者拆出一点早期信号，是对 sparse 冷启动问题的最小修补（与 M3 SFT 同题两解）；
- **为什么只有 0.1**：权重就是攻击面——定大了模型学会输出空 boxed 稳拿分；0.1 远小于
  做对题的 1.0，优化压力主方向始终指向正确性，模型会做题后该项近似常数；
- **为什么 λ_exec/λ_budget 默认 0**：过程性 shaping 改变最优策略（除非 potential-based，
  Ng 经典结果），执行成功分可被"刷无意义可跑代码"利用——所以从基础设施降级为 E4 的
  受控消融对象；
- **最关键**：核心实验 E5 研究"sparse outcome 下信用如何分配到轮次"。**改 reward**（shaping/
  PRM）= 注入新监督信号、改变优化目标；**改优势**（E5）= 零新信号，只改同一信号在轮次间的
  归因。baseline reward 若做稠密，E5 的问题就被别的手段偷解了，对照失效——保持 sparse 是
  实验设计的前提。三档取舍：PRM（学出来的过程信号，要训练+标注+自带 hacking 面）>
  shaping（规则化过程信号，改目标）> E5（信用重分配，零新信号）——成本与风险面依次降低。

**证据**：PLAN §6.3 复合奖励定义、§10 E5 设计；rewards/composite_reward.py（M2 实现）。

---

## Q5（2026-07-14，M3 前复查）：现有结果下，到底选 Qwen3 还是 Qwen2.5？数据集要改吗？

**答**：维持 Qwen3-1.7B + MATH L3–5，M1/M2 没有产生任何换的证据——反而验证了选型假设：
headroom（CoT L4 0.75/L5 0.43）、组内方差（TIR 混合结果占比 L3–5 0.44–0.51）、管线零风险
全部实测成立。零样本负增益不是 Qwen3 特有缺陷（机制是"未经 TIR 训练的小模型代码执行
不可靠"，Qwen2.5-3B 同样没经过 TIR 训练；社区所有 TIR 增益都来自训练而非 prompting）。
数据集不改，留一个应变点：若 SFT 后 L3 饱和（M3 复测 + M4 `group_all_correct_frac` 监控），
训练池收窄到 L4–5（一行配置）。

## Q6（2026-07-14，M3 前复查）：SFT 为什么需要教师模型？没有现成标注数据吗？这算 OPD 吗？

**答**：（1）MATH 人工解答是纯文本 CoT，零工具调用——拿它 SFT 会把模型往"别用工具"拉；
SFT 要教的是 M1 诊断出的具体行为（hermes tool_call 格式、无状态沙箱纪律、读输出继续推理），
必须以"多轮消息+真实沙箱返回"的轨迹存在，且与 M4 rollout 格式逐 token 一致。
（2）最接近的现成数据 NuminaMath-TIR（72k，GPT-4o）是 markdown 代码块约定而非 hermes，
要用需格式转换+沙箱重执行，工程量不小且题目分布不可控。教师的本质作用 =
在我们的格式/沙箱/选题上生成正确多轮轨迹。另有零依赖的自蒸馏路线（STaR/RFT 式，
用 M1 结论"工具成功时通过率高"支撑可行性）。
（3）**这不是 OPD**。区分两问：轨迹谁采样（M3：教师；OPD：学生自己）、信号是什么
（M3：硬序列 SFT loss；OPD：教师对学生轨迹逐 token 的稠密 reverse-KL）。M3 =
off-policy 蒸馏 = R1 式 cold start（加拒绝采样过滤 = rejection sampling distillation）；
OPD（GKD/Qwen3 OPD）训练时教师每步在场。面试高频区分点。

---

## Q7（2026-07-14，M3 教师选型）：API 太贵吗？能用 Claude/Codex 吗？Qwen2.5-Math-7B 准确率有保证吗、够新吗？

**答**：（1）成本重估 ~$10–20（只需 ~6k 原始轨迹），不贵但非必要。
（2）Claude/Codex 不推荐：Anthropic/OpenAI ToS 限制用输出训练竞争模型（项目要开源+进简历，
数据合规会被审视；DeepSeek/Qwen 明确允许蒸馏）；且订阅是交互授权，不是批量 API。
（3）**教师准确率不需要"保证"——rejection sampling 使教师准确率只影响产量、不影响数据
正确性**（verifier+沙箱三条件把关，错误轨迹进不了 SFT）。
（4）教师"新旧"不重要：教师不出现在最终产物里，只需"任务上强 + 格式可控"。格式维度上
Qwen3-8B（hermes 原生，零转换）优于 Qwen2.5-Math-7B（TIR 是代码块约定，需转换层）——
M1 的教训就是格式细节决定成败。**决策：本地 Qwen3-8B**，产量不足时用 Qwen2.5-Math-7B
补 L5（执行中小调整）。

---

## Q8（2026-07-14，M3 验收）：教师模型表现如何？能取得高分吗？

**答**：Qwen3-8B（非 thinking、bare TIR、temp 0.7）在训练池上：L3 85.4% / L4 76.3% /
L5 56.9%（整体 71.0%），比学生零样本 TIR 高 11–24pt——有足够教学落差但不是学霸
（thinking 模式会高很多，但轨迹与非 thinking 学生格式不符，弃用）。关键区分：
**教师准确率影响的是覆盖率而非质量**——质量由拒绝采样兜底（入选轨迹 100% 验证正确），
但 L5 有 41% 题目教师产不出一条合格轨迹 → SFT 数据缺席最难的题（拒绝采样的难度偏置）。
这解释了 SFT 后增益收窄却未完全转正的另一半；缺口天然由 M4 的 on-policy RL 接手
（训练池含全部题目，模型自己探索，不受教师上限约束）——"SFT 教格式、RL 提能力"的
教科书分工。抬教师上限（专家模型补 L5 / API / thinking 模式）收益不划算，未采纳。

**证据**：sft/data/gen/gen_stats.json、qa_log 本条内表格的计算脚本见会话记录、
reports/01_tool_gain.md 附录。

---

## Q9（2026-08-19，M3 最小复核）：为什么 30B teacher 和 Full SFT 没有明显更好？这合理吗？

**答**：先区分两个实验：30B 只作为**轨迹生成教师**，学生始终是 Qwen3-1.7B；Full SFT
则是对同一个 1.7B 学生更新全部参数，不是训练 30B。结果合理，但只能支持当前配置下的
工程决策，不能泛化为“30B teacher 更差”或“LoRA 永远优于 Full SFT”。

**30B teacher 实际更强，但不符合当前数据筛选目标。** 相对 8B，它的错误答案更少
（57 vs 79）、截断更少（8 vs 16）、工具错误率更低（10.3% vs 22.7%）；但完全不调用工具的
轨迹更多（125/400 vs 77/400），工具调用总数更少（369 vs 485），导致“无成功工具调用”过滤
增至 126 vs 90。我们的合格轨迹要求“数学正确 ∩ 至少一次成功工具调用 ∩ 未截断 ∩ 长度
合格”，因此数学能力收益被工具调用意愿下降抵消，最终 yield 52% vs 53%、L5 coverage
63% vs 65%。而且 Qwen3-30B-A3B 是 MoE，总参数 30.5B、每 token 激活约 3.3B，并非 30B
dense。probe 的近似 95% CI（yield 差 `[-7.9,+5.9]pt`、L5 coverage 差
`[-15.3,+11.3]pt`）也说明应解读为“未检测到值得全量切换的收益”，不是证明 30B 本质更差。

这里需要明确区分三个层次：**teacher capability ≠ trajectory suitability ≠ trajectory
learnability**。前者是教师能否把题做对；第二层是正确轨迹是否满足当前 TIR protocol（真实、
自然且有效地调用工具，并通过格式/执行/长度筛选）；第三层才是学生能否通过 SFT 学会该轨迹并
泛化到 held-out 问题。本次 probe 已直接证明前两层不等价：30B 能力更强，但合格 TIR 轨迹
没有变多。**当前 M3 尚未直接测量第三层 learnability**，不能仅凭轨迹更长、更复杂或教师更强
就断言学生更容易受益。

若要测 trajectory learnability，应在固定数据量下按属性分桶做小规模 SFT，例如短/长轨迹、
单次/多次 tool call、全程成功/error-recovery、L3/L4/L5，以及 8B/30B teacher 来源，然后用
同一 held-out TIR pass@1、工具错误率和弃用率比较。如果某类轨迹看似更强或更复杂，学生学完
却没有更好的 held-out 泛化，就说明该类轨迹对当前 1.7B student 的 learnability 较低。这是
合理的后续解释变量，但不是当前 M3 的必要验收项。

**Full SFT 的自由度更大，但不保证小数据泛化更好。** 同一 held-out 200 的题目级 bootstrap：
Full−LoRA 的 CoT pass@1 为 +1.75pt（95% CI `[-1.38,+4.88]`），TIR 为 +0.88pt
（`[-2.38,+4.25]`），均无法排除采样噪声；工具错误率却增加 +6.41pt
（`[+1.16,+11.65]`），是更可信的退化。具体原因有三点：① 这只是一次无调参对照，Full LR
为 `1e-5`、LoRA 为 `1e-4`，Full 最终训练 loss 反而更高（0.1906 vs 0.1683），没有证据表明
Full recipe 已优化充分；② 6056 条小数据主要教接口行为，LoRA 的低秩约束相当于正则化，
更能保留基座的代码/格式能力；③ 数据故意包含 797 条“报错后恢复”轨迹，错误 tool call 本身
仍是监督 token，而训练目标又不直接惩罚 tool error，Full SFT 可能更强地拟合这类副作用。

**一般解决方式不是继续放大 teacher，而是管理轨迹分布。** ① 从“只筛最终正确”升级为轨迹级
选择，同时考虑 tool interaction 是否必要、自然、可执行且适合学生；② 对每题多采样并显式控制
tool-use coverage，只保留调用格式正确、执行有效的轨迹，必要时按难度和调用形态配额；③ 保持
SFT 的职责是 cold start——教会基本 tool call、读取输出、错误恢复和最终作答，把“什么时候该
用工具、怎样用得更有效”交给 M4 之后的 on-policy RL。对 ToolCredit 而言，tool-behavior
distribution 已有直接证据，M4 又需要固定统一起点，因此没有必要为了尚未验证的 learnability
假说继续优化 M3 或扩大 teacher/SFT 搜索范围。

**公开研究中的位置**：这类非单调结果是正常现象。工具研究把“是否用工具”视为独立于通用
能力的行为维度（[MetaTool, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/bc12914d66b41b6bfc2d3a5decdb498b-Abstract-Conference.html)），
专门训练的较小模型也能达到接近更大闭源模型的工具表现
（[ToolLLM](https://arxiv.org/abs/2307.16789)）。原始
[LoRA 论文](https://arxiv.org/abs/2106.09685)报告过 LoRA 持平或超过 Full FT；后续
[LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673)则显示充分调参和更多
数据下 Full FT 常有更高目标域上限，但 LoRA 更少遗忘、正则化更强。因此本实验的严谨结论是：
**当前 bare-tool 生成协议下 30B 未通过产量/覆盖 gate；当前单次 Full-SFT recipe 没有可靠的
pass@1 收益且工具错误显著恶化；trajectory learnability 尚未直接测量，但不构成继续扩展 M3
的必要条件，所以继续保留 8B 教师数据和 LoRA SFT-6k 作为 M4 起点。**

**证据**：`sft/experiments/m3_minimal/teacher_probe/comparison.json`、
`sft/experiments/m3_minimal/full_sft/evaluation_comparison.json`、`plans/M3.md`；Qwen 参数口径见
[官方模型卡](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507)。

---

## Q10（2026-08-20，M4 指标口径复盘）：answer、parser、tool、truncation、invalid 和 reward 分别由谁产生？veRL 在模型与沙盒之间做什么？

**答**：这不是一个模块一次性算出的指标包，而是一条三层数据流：① `ToolCreditAgentLoop` 与
`ToolCreditSandboxTool` 在生成过程中把工具调用、成功、执行错误和解析错误累计到每条轨迹的
`extra_fields`；② `rl/custom/reward.py:compute_score` 读取这些状态，调用严格 boxed verifier、
format checker 和 composite reward，返回逐轨迹 `score/acc/format_ok/invalid/...`；③ veRL 将
返回值写进 rollout/validation JSONL，`rl/monitor_run.py` 再按 step 聚合成 mean/rate，并写入
`metrics.json` 和 TensorBoard。数据集原有的题目 ID 也沿同一个 `extra_info` 通道传到 reward。

**模型并不直接执行 Python，veRL 是中间的 agent 编排器。** 模型只生成 Hermes 文本，例如
`<tool_call>{"name":"code_interpreter","arguments":{"code":"1+1"}}</tool_call>`；veRL 的
Hermes parser 把合法文本转成 `FunctionCall(name, arguments)`，按 `tool_config.yaml` 找到工具，
调用 `ToolCreditSandboxTool.execute`，再把 `ToolResponse` 作为环境消息追加到对话并让模型继续
生成。veRL 同时维护轮数/长度状态、响应 token 与 mask、工具返回段 loss mask、rollout 和后续
GRPO 训练；项目适配器负责统计，`env/sandbox.py` 才负责受限子进程中的真实执行。

`parsed_count = len(agent_data.tool_calls)` 是 **veRL parser 成功构造出的结构化调用数**，不是
模型文本中 `<tool_call>` 标签的数量，也不是沙盒成功次数。在当前 `max_parallel_calls=1` 下，
一个解析阶段通常为 0 或 1：合法标签、合法 JSON、可构造成 `FunctionCall` 时为 1；没有调用时
为 0；模型明显想调用工具但 JSON/标签残缺时也为 0。项目用
`candidate_count - parsed_count`（下限 0）估算 `tool_parse_error_count`：前者数完整调用块及残留
的开/闭标签，后者来自 veRL 的实际解析结果。达到 response/turn budget 时不做这项比较，避免把
被截断的半个标签误报为 parser error。

错误类型按流水线阶段严格区分：

- **parser error**：沙盒前失败；模型生成了疑似 tool call，但 veRL 无法解析成结构化调用；
- **dispatch/argument error**：已能解析，但工具不存在，或参数不是唯一的字符串 `code`；计入
  tool error；
- **tool runtime error**：`env/sandbox.py` 返回 `error`/`timeout`；也计入 tool error；
- **普通答案错误**：verifier 能判定但不等价，`acc=0, invalid=0`；
- **verifier invalid**：已有 boxed 内容，但 math-verify 与 SymPy 都无法形成判断，表现为
  `verify_method="none"`；没有 boxed 是定义明确的 `no_boxed` 错误，不算 verifier invalid。

各逐轨迹字段的当前实现口径是：`acc = answer_correct`；`format_ok = 存在完整 boxed AND
tool_parse_error_count == 0`（不要求答案正确或工具执行成功）；`tool_error_rate =
tool_error_count / tool_call_counts`，零调用时为 0；显式 `truncated` 目前只表示“调用达到 4 次且
最终没有 boxed”，纯 response-length 截断没有进入这个字段；`invalid = verifier_invalid OR
parser_error`，是逻辑并集而非算术相加，tool error、普通答错、no-boxed 和 truncation 不自动
进入 invalid。若要从落盘 JSONL 单独统计 verifier invalid，应数 `verify_method == "none"`；当前
`invalid` 字段已经与 parser error 合并。

`invalid` 是**结果可信度/管线健康告警**，不是“答错”的同义词，也不是 E3 中独立的 reward
惩罚项。它让 verifier 无法判定和交互协议损坏不被静默混进普通错误，便于监控
`invalid_rate`、抽轨迹审计以及排查 reward hacking。E3 实际
`score = 1.0 * acc + 0.1 * format_ok`，截断强制为 0；parser error 会通过取消 0.1 格式分间接
影响 score，verifier invalid 会令 `acc=0`，tool error 本身不扣 E3 reward（E4 才可打开
execution shaping）。因此可能出现“答案正确 + parser error → `acc=1, format_ok=0,
invalid=1, score=1.0`”，也可能出现“工具执行失败但最终正确且格式合法 → `score=1.1`”。

监控端对 JSONL 做轨迹宏平均：`invalid_rate=mean(invalid)`、`truncated_rate=mean(truncated)`、
`invalid_format_rate=1-mean(format_ok)`；`tool_parse_error_rate` 是“至少一个解析错误的轨迹占比”，
不是平均解析错误次数；`tool_error_rate` 是先算每条轨迹的错误调用比例再平均，不等于全局
`sum(errors)/sum(calls)`。这些区别决定曲线应如何解释。

**证据**：`rl/custom/tool_agent_loop.py:19-79`、`rl/custom/sandbox_tool.py:73-106`、
`env/sandbox.py:87-128`、`rewards/verifier.py:99-119`、`rewards/format_reward.py:6-10`、
`rewards/composite_reward.py:38-67`、`rl/custom/reward.py:28-72`、
`rl/monitor_run.py:40-61`、`rl/custom/test_m4_adapters.py`。

---

## Q11（2026-08-21，M5 显存与吞吐复盘）：veRL 的训练/rollout 显存为何交替变化？offload、SGLang batch 和 micro-batch 分别是什么？36 GB 单卡如何适配？

**答**：先看结论：当前单卡采用 colocated/hybrid-engine 式复用——SGLang rollout 与 FSDP
actor 不是各自永久占一块 HBM，而是在同一张 GPU 上轮流成为主要使用者。`nvidia-smi` 的进程
显存是某一时刻的驻留/预留量，能辅助判断当前阶段，却不能据此推断时间占比，也不等于“有效
张量”大小；CUDA context、allocator cache、CUDA graph 和 workspace 也会计入。

本次观察到的两个稳定截面正好展示了这种切换：

| 截面 | actor/Ray worker | SGLang scheduler | 其他 Python | 合计 | 判读 |
|---|---:|---:|---:|---:|---|
| 训练侧活跃 | 43,024 MiB | 1,410 MiB | 596 MiB | 约 44.0 GiB | SGLang cache 已释放/休眠，actor 正在做 log-prob 或更新 |
| rollout 活跃 | 8,868 MiB | 91,086 MiB | 596 MiB | 约 98.2 GiB | actor 状态大部下放，SGLang 权重、KV cache 与推理工作区已唤醒 |

第二个截面所在 MIG instance 总容量为 146,210 MiB，仍余约 44.6 GiB，因此这两次观察本身没有
显示 OOM 或显存泄漏。SGLang 的约 89 GiB 绝大部分也不能解释成模型权重：Qwen3-1.7B 的 BF16
裸权重约 `1.7B × 2 bytes ≈ 3.4 GB`，额外空间主要是大量并发长序列的 KV cache、CUDA graph、
attention workspace 和预留池。显存占用大也不代表该阶段耗时一定更长。

**offload 是“暂时搬走”，不是删除或压缩。** 当前 actor 开启 `param_offload=true` 和
`optimizer_offload=true`，reference 也开启参数 offload；暂时不用的参数、Adam 状态等移到主机
内存，需要计算时再分块搬回 HBM。一个 GRPO step 的简化生命周期是：

```text
actor 参数/优化器下放 → 唤醒 SGLang、同步最新权重 → rollout 并维护 KV cache
       → 释放 rollout cache/SGLang sleep → actor/ref 参数按需回迁
       → old/ref log-prob → reward/advantage → actor forward/backward/update
       → 将新 actor 权重同步给 SGLang → 下一 step
```

训练时除了约 3.4 GB 的模型权重，还可能同时出现梯度、激活、Adam 一阶/二阶状态、old/ref
log-prob 临时张量和 allocator cache，所以 1.7B 模型仍可占数十 GiB。rollout 时不需要梯度和训练
激活，actor worker 因 offload 降到约 8.7 GiB，空出的空间交给 SGLang KV cache。配置
`free_cache_engine=true` 正是为了在两个阶段之间释放/恢复 rollout cache。

**实际时间分布要看 timing，而不是显存截图。** E4-A 前 149 step 平均每步 169.2 秒：rollout
生成 63.1 秒（37.3%），actor 更新 61.3 秒（36.2%），reference log-prob 24.1 秒（14.2%），
old log-prob 18.1 秒（10.7%）。所以只比生成与反向更新时两者接近；把 old/ref log-prob 算入
训练侧后，训练侧约 61%、rollout 约 37%。这也说明 PLAN 中“rollout 占 70%+”是预算估计，
当前实测不是 70%。

**当前 attention 口径不是 Hugging Face `flash_attention_2`。** veRL 默认值已被
`actor_rollout_ref.model.override_config.attn_implementation=sdpa` 覆盖，训练模型走 PyTorch
SDPA；已安装的 `flash_attn` 包只被 veRL 的 padding/unpadding helper 硬依赖。PyTorch SDPA 在
条件合适时仍可能内部选择 fused/flash SDP kernel，但这不等于显式使用 FlashAttention-2。
rollout 则走 SGLang/sgl-kernel 的推理路径，不受这个 Hugging Face attention 字段直接控制。

**三类 batch 必须分开理解：**

1. `data.train_batch_size=64` 是每个 GRPO update 的 64 个不同 prompt；`rollout.n=8` 是每个
   prompt 的 8 条组内采样，因此每步收集 `64 × 8 = 512` 条轨迹。它们决定有效训练数据和 GRPO
   组统计，随意改小会改变优化方差、采样预算和实验可比性。
2. `max_num_seqs=1024` 是 SGLang 同时运行/调度的序列数上限，不表示当前必有 1024 条；
   `max_num_batched_tokens=8192` 是一次模型执行最多调度的 token 总数，不是单条回答长度或整步
   token 总量。prefill 时 8 条各 1000-token 的 prompt 就接近 8192；decode 时 512 条活跃序列
   每条各生成一个 token，只贡献约 512 batched tokens。降低这两个上限会减少并发、KV cache 或
   单次 forward 峰值，但通常降低吞吐，且 SGLang 的静态预留池未必按相同比例下降。
3. `ppo_micro_batch_size_per_gpu=4` 是训练时一次真正送进单张 GPU 做 forward/backward 的样本数；
   一个较大的 mini-batch 会拆成多个 micro-batch，逐个反向并累积梯度后再完成相应更新。
   `log_prob_micro_batch_size_per_gpu=4` 同理用于分块计算 old/ref log-prob，通常只拼接输出、不做
   反向。micro-batch 从 4 降到 1 可明显降低激活/临时张量峰值，固定的权重和 runtime 开销不会
   下降，因此显存不会严格变成四分之一；代价是更多小 forward/backward 和更低吞吐。

一个便于理解的例子：把 512 条轨迹看成 512 份订单。`max_num_seqs` 是餐厅同时接待的座位数，
`max_num_batched_tokens` 是厨房每一轮最多处理的食材份量，二者只决定这些订单如何排队完成；
micro-batch 是训练后厨一次端上操作台做“前向+反向”的盘数。减少座位或每次上台的盘数可以
降低峰值空间，但只要最终仍处理同样 512 份订单并正确累积结果，有效 batch 可以保持不变。

**36 GB 单卡的优先适配顺序是先改执行切分，再改实验语义。** 对 1.7B 模型可先保持
`train_batch_size=64, n=8`，将 actor/ref/rollout log-prob micro-batch 从 4 降到 1，保留已有的
parameter/optimizer/reference offload、gradient checkpointing、remove padding 和
`free_cache_engine=true`；再把 SGLang `gpu_memory_utilization` 从 0.5 调到约 0.35–0.4，并把
`max_num_seqs` 限到 128/256、`max_num_batched_tokens` 限到约 4096。这样主要牺牲吞吐而尽量不改
GRPO 的 64×8 统计语义。

若仍 OOM，再依次考虑 activation offload；用同一冻结 actor snapshot 将每个 prompt 的 8 条采样
拆成 4+4、合并完整 8 条后才算组内 advantage 并只更新一次；最后才缩短 response 3072→2048、
减小 train batch、把 `n=8` 改成 4，或使用 LoRA/量化/更小模型。前两项主要改变执行方式，后几项
会改变截断率、有效样本量、GRPO 组统计或训练容量。尤其不能“先生成 4 条并更新，再生成后 4
条并更新”，那会变成两个 `n=4` group，后半还来自更新后的 policy，不再等价于原实验。

对本项目 M5 而言，batch、`n`、响应长度和 rollout 配置是 E3/E4/E6 可比性的冻结项；36 GB
适配若改变它们，应作为单独硬件 recipe 记录并重新建立对应 baseline，不能与 GH200 正式曲线
冒充严格同配置比较。

**证据**：`rl/configs/e4a_exec_only.yaml`、
`rl/runs/e4a_exec_only_20260821_040632/resolved_config.yaml`、
`rl/runs/e4a_exec_only_20260821_040632/metrics.json`、`environment.md:46-54`、`PLAN.md:269-301`。

## Q12（2026-09-04，M7 后复盘）：E5 比 E3 没有提升、还出现大量重复调用，这算不算 reward hacking？下一步该修 failure case 还是修 E5？

**答**：先纠正前提：E5 不是“显著更差”，是**终点无增益 + 行为严重偏移**。M7 FULL760 greedy
E3→E5 为 −1.05pt，95% CI `[−3.68, +1.58]` 跨 0；真正显著的是 4-call 占比 1.3%→92.9%、
truncation 9.6%→100%。

**可以叫 reward hacking，但要说清楚 hack 的是什么。** 狭义 reward hacking 指“可观测 reward 上涨而真实
能力不涨”，E5 的 outcome reward 没有异常上涨，所以报告里保守地叫 credit-signal/proxy gaming。更准确、也
更适合面试的说法是：**Tier-A 修正项不是零和的，等价于我在 advantage 里偷偷加了一个稠密奖励
“成功且被后文引用的调用”，策略最大化了这个隐式奖励。** 机制上有三条证据：

1. `s̄_t` 按组内同位置求均值，final/no-tool turn 的 `s_t=0` 会与其它轨迹同位置的 adopted call 比较：
   E5 全 run 的 102,141 个 no-tool turn 中 33,663 个拿到负修正——“不调用/早停”被直接惩罚。
2. 逐轨迹修正之和随调用次数单调：step 200 时 0/1/2/3/4 次调用的平均净修正为
   −0.41/−0.47/−0.23/−0.05/+0.01，token 加权 shift 与调用数相关系数 0.53。这是自增强的 ratchet：
   组内多数轨迹调到 4 次后，任何早停轨迹都会相对受罚。
3. E5 训练中组内 `score` 零方差 group 从 25%（step 1）升到 48%（step 200）；这些 group 的 `A_traj=0`，
   修正项是**唯一梯度**，即约一半 batch 在做与 outcome 无关的“多调用”优化。
   （E3 的零方差比例平均 44%，同量级。）

**下一步建议：优先修 E5 的设计，而不是修 E3 failure case。** 理由：E5 v1 的实验被上述混淆污染，
目前还没有干净地回答研究问题“轨迹内重新分配信用值不值”；failure case（语言循环、报错后乱猜）
主要是推理协议/基座能力问题，做了也不回答研究问题。E5-v2 最小改动：
(a) 修正项在轨迹内做零和中心化（token 加权），只保留轨迹内再分配；
(b) 只有真正发起工具调用的 turn 参与 `s̄_t`，final/no-tool turn 修正恒为 0；
(c) AST 归一化后重复的代码和 boxed 之后的调用 `s_t=0`；
(d) 只在 `std(score)>0` 的 group 施加修正。
一条 200-step run（约 1.5–2 GPU 日），同协议评测。无论正负都是干净结论。
E7（零方差过滤）与发现 3 直接相关，可作第二优先级；failure case 的循环检测作为 eval-only 对照放最后。

**证据**：`rl/custom/turn_advantage.py`（`s_by_position` 按 `(uid, position)` 聚合）、
`rl/runs/e5_turn_credit_20260823_082012/analysis/mechanism_analysis.json`
（`correction_sign_by_adoption_status/no_tool/negative = 33663`）、
`rl/runs/e5_turn_credit_20260823_082012/predictions/train/{1,25,50,100,150,200}.jsonl`
（本次会话即席统计，未落盘为脚本）、`eval/runs/m7_unified_eval_20260824_201032/metrics/tool_behavior.json`、
`reports/docs/grpo_failure_diagnostic_session_summary.md` §8。

## Q13（2026-09-05，M8 评测阶段）：为什么现在是"v4"？v2、v3、v4 有什么区别？

**答**：v* 是同一条评测协议 `eval/diagnostic_protocol_v*.json` 的连续版本，不是模型版本。每版由一份 sha256 清单锁住
闭包，后一版 `supersedes` 前一版，并有语义校验器证明"除声明的新增外逐字段相同"。**v2**（M6 前）锁"评什么、怎么评"：
760 题 panel、生成参数、工具/截断、verifier、taxonomy v1、配对与 bootstrap 定义、数据隔离、SFT/E3 身份，以及 E5 v1 训练代码
（E5 角色留空）；E5 v1 与 E5-v2 训练的 preflight 都校验它。**v3**（M7 前）在 v2 的 37 文件不动的前提下加 29 个：填入 E5 v1
step-200 角色、E3/E5 的 HF 物化 manifest、evaluator 代码与配置；M7 的 15,200 条 canonical 轨迹在 v3 下生成。**v4**（M8）在 v3
的 66 文件不动的前提下只加 E5-v2 角色、其物化、M8 evaluator 文件与"复用了哪些 M7 产物"的记录；只为 E5-v2 生成 3,800 条，
其余四角色直接复用 M7。因为 v3 把 evaluator 文件（`eval/generate.py` 等）hash 锁死且硬编码四角色，M8 的扩展全部放新文件
（`eval/m8_e5v2_eval.py` 等），并用等价测试证明 v4 评分在 M7 原始轨迹上逐字段复现 M7 评分。**面试角度**：这是"协议先冻结、
改动必须 bump 版本并证明语义不变"的纪律，它让 E5-v2 能与两周前的 E3/E5-v1 结果直接配对比较。

**证据**：`eval/build_diagnostic_freeze_v{2,3,4}.py`、`eval/verify_diagnostic_freeze_v{3,4}_semantics.py`、
`eval/test_m8_e5v2_eval.py::test_v4_scoring_reproduces_canonical_m7_scored_row`、`plans/M8_DIAGNOSTIC_FREEZE_V4.md`。
