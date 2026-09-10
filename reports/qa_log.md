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

## Q14（2026-09-06，M8 后复盘）：M8 证明了什么？信用分配有没有带来提升？项目初衷"学会用工具就更准"完成了吗？下一步做什么？

**答（M8 证明了什么）**：三件事，两件证明、一件没证明。(1) **v1 失效机制被反事实钉死**：同起点、同训练量、只把修正项
改成轨迹内守恒，over-calling 完全消失（greedy FULL760 4-call 92.9%→0.8%、截断 100%→8.7%），终点不比 E3 差；所以 v1
的行为漂移来自非守恒修正项等价的隐式稠密奖励，不是"轮级信用"本身。(2) **机制上实现了目标信用分配**：每条轨迹总信用
与 E3 逐条相同，只在多次调用之间按 `s_t` 再分配，守恒残差 1e-13 级、零方差组恒等于 E3，可审计。(3) **效果没检出**：
greedy Δ +0.79pt，CI `[−1.84,+3.42]`；早期 AUC +1.87pt 差 0.13pt 未过预注册 2pt 线，判情形 A，不事后改判。

**为什么不显著**（前两条有数据，后两条是判断）：
1. 作用面太小：守恒再分配只在"同轨迹 ≥2 次可区分调用"时非零；exposure 从前 25 步 10.4% 衰减到后 25 步 4.2%，
   即 96% 轨迹的梯度与 E3 完全相同。
2. **treatment 在缩小自己的作用面**（报告未明说，本次新观察）：E5-v2 评测零调用轨迹占 48.6%，E3 为 202/760 = 26.6%；
   mean calls 0.60 < E3 0.89。守恒修正对重复/boxed 后的调用给负值，策略学到"少发第二次调用"，多调用轨迹随之减少。
3. ≤4 轮 horizon 下轨迹几乎是 bandit，组内 outcome 方差已够用。
4. `s_t` 是"执行成功且被引用"的启发式代理，不是真实步骤价值。

**初衷完成了吗**：要分两个口径。PLAN §0.1 的研究问题是"轨迹级广播 vs 轮级信用"，工具增益只是前提检查（M1 已做：
零样本全负、SFT 后收窄到 −3pt）。按 PLAN 口径交付物齐全，研究问题得到否定性、带机制的回答。但**"模型学会用工具后解题
更准"没有被因果证明**：从未跑过 CoT-only 的 RL 对照臂，SFT→E3 的 +12.8pt 里多少来自学会用工具、多少来自 RL 本身提升
数学推理分不开。现有只是相关性证据（greedy FULL760，`analysis/tool_use_conditional_accuracy.json`，有选择偏置）：

| 阶段 | 零调用 (n, acc) | 有调用 (n, acc) | 有调用且全成功 | 有调用且有报错 |
|---|---|---|---|---|
| Raw | 299, .548 | 461, .568 | 365, .663 | 96, .208 |
| SFT | 67, .179 | 693, .646 | 593, .698 | 100, .340 |
| E3 | 202, .619 | 558, .774 | 509, .807 | 49, .429 |
| E5-v2 | 369, .672 | 391, .806 | 361, .834 | 30, .467 |

E3 分 level（MATH500 greedy）：L1–3 零调用与有调用准确率接近（.83/.90/.83 vs .97/.90/.85），L4–5 差距拉开
（.64 vs .84、.36 vs .58）——与 M1"潜在增益在 L4–5"一致，但同样是相关性。

**下一步排序**：(1) **E3-NoTool 对照臂**（`plans/M9.md`，"CoT"即无工具的纯文字推理，非 thinking 模式）：同起点、同数据、同 200 步，只去掉工具，另加 raw/SFT/E3 权重的无工具评测角色量化起点债务；补上因果空洞，面试必问，
约 1 GPU 日。(2) **E7 零方差过滤**（`plans/M10.md`）：E3 零方差组均值 44%、后期 53%，DAPO 式过滤把已有诊断数字变成曲线；
结果可预期（每步更快、终点持平或略高、rollout 成本约 1.8×），实现风险在 pin veRL `fit()` 耦合。(3) 不建议现在做：
额外 seed（效应约 1pt，3 seed 也压不出显著）、§3.4 `|A_traj|` 变体（exposure 未解决）、更长 horizon（需换任务或改沙箱让
多轮成为必要，E3 预算诊断显示单加预算只得到语言循环，属 scope 级决定）。

**证据**：`reports/02_main_results.md` M8 节、`eval/runs/m8_e5v2_eval_20260905_134737/metrics/{m8_headline,tool_behavior}.json`、
`analysis/tool_use_conditional_accuracy.py` 与其 JSON 输出（只读脚本，2026-09-06 新增）、
`reports/docs/grpo_failure_diagnostic_session_summary.md` §8、`reports/01_tool_gain.md` 附录。


## Q15（2026-09-06，秋招简历提炼）

**场景与问题**：基于项目已完成工作，提炼用于大厂秋招的简历 bullets。

**浓缩答案**：优先写训练管线与 MATH500 SFT→GRPO +17.2pt、6,056 条蒸馏轨迹及 token 对齐、轮级信用修正后四调用占比 92.9%→0.8%、19,000 条统一评测与配对 bootstrap。轮级方法准确率相对标准 GRPO 未检出显著增益；v2 同时改变守恒、门控与部分步骤信号，不能把行为改善宣称为单独守恒因素的因果证明。工具独立收益仍缺 NoTool RL 对照；E7 仍未实施。

**证据**：`reports/resume_bullets_autumn.md`（四条正文及逐项源文件）；M7 `metrics/pass_at_k.json`、M8 `metrics/m8_headline.json`、`sft/data/gen/gen_stats.json`、`rl/custom/turn_advantage_v2.py`。

## Q16（2026-09-06，HER 可行性讨论）

**场景与问题**：hindsight experience replay 的思想能否与 ToolCredit 结合，做一个有解释力的小实验？本次只讨论方案，未授权新训练或实现。

**浓缩答案**：可以，优先探索从 train split 的最终失败轨迹中提取可独立验证的成功子任务，重标注目标后用于辅助 SFT。HER 的核心是把目标改成轨迹实际达成的目标并重新计算奖励，不是反思文本，也不是把原题的错误答案改成 gold。原始 HER 面向 off-policy goal-conditioned RL；这里更准确称为 hindsight subgoal relabeling / supervised replay。

- 与当前主线的关系：E5-v2 在同一原题轨迹内守恒再分配优势，且对 `std(score)=0` 的组严格退化为 E3；HER 式辅助监督可能回收这些组中局部正确的技能，但这是新增训练目标和数据，不能解释为纯信用再分配。全错不必然零方差，因为 reward 还含 format；零优势只表示该项 policy gradient 为零，KL 等仍可能有梯度。
- 示例（示意，非实测轨迹）：原题求 `x²−5x+6=0` 两根平方和，工具正确求得 2、3，最终却答 12。可将子任务标注为“求该方程实根”，只保留与新目标一致的正确片段；若补写 `boxed{2,3}` 收尾，须记录为合成监督而非原始 replay。不能把原题答案从 13 改成 12。
- 数据门槛：子任务须独立、与原题相关且保留所有必要条件；用独立数学规格/参考解验证，沙箱重跑只能证明可执行性，不能证明代码解决了正确问题。排除常量打印、无关计算、重复和重标注后语义冲突；保留 parent sample ID、step、原始片段、目标、验证依据、拒绝原因及合成编辑记录。只用 train，衍生题重新污染检查；M7/M8 eval 和 70 条预算诊断题不得回灌。
- 最小先导建议：预先固定抽样规则，审计约 100–200 条 E3 train 失败轨迹（按 step/原题去重，并记录 score 零方差组身份），报告可恢复率、独立验证通过率、非平凡子任务比例和成本。当前仅检查了 E3 `predictions/train/1.jsonl` 首行 schema：有 input/output/gts/score/sample_id 和汇总工具指标，没有结构化 messages/逐轮 ledger；需先核实拼接文本能否无歧义恢复，不能假设已有 replay buffer。本次未执行该审计。
- 后续实验建议：各臂从同一 SFT checkpoint 出发，比较等 token 预算的原 SFT 数据续训、成功轨迹片段续训、失败轨迹的已验证子任务续训；再采用同一 E3 GRPO 配方和预算。保持原题评测、记录新增训练/验证成本、工具滥用与截断，不能只与没有额外 SFT 的旧 E3 比。先只组合标准 E3，避免同时改变 turn-credit。
- 暂不建议直接向 GRPO 注入历史重标注轨迹：旧策略采样和改写 prompt 同时改变概率条件，原 old log-prob、group UID、reward/advantage 不能直接复用；重新算新 prompt 下 log-prob 本身也不能消除采样偏差。可选替代是仅用重标注题目重新 on-policy 采样，但这是课程数据扩充，有额外 rollout 成本，仍须独立计划。

**证据**：`PLAN.md` §0.1/§9–10；`plans/M8.md` §3 与 §13；`rl/custom/turn_advantage_v2.py:conserved_turn_corrections`；`rewards/composite_reward.py`；`sft/trace_tokenizer.py`；M8 `metrics/m8_headline.json` 中实际 treatment exposure 首/末 25 步为 10.42%/4.20%；`plans/M9.md` 已完成 smoke、formal 待授权，`plans/M10.md` 仍草案。外部方法依据：[HER 原论文](https://arxiv.org/abs/1707.01495)、[HIR，ICML 2023](https://proceedings.mlr.press/v202/zhang23ab.html)。以上迁移方案是本项目设计建议，没有效果证据。

## Q17（2026-09-07，M9 formal 中断）：为什么每条 200-step formal run 都会在 5–9 小时后"pod 中断"？

**场景与问题**：E3-NoTool formal 在 step-150 checkpoint 写入时整个 pod 消失（进程树、tmux、/tmp 全没，无 Traceback），与 E3/E4-A/E4-B/E5/E5-v2 五次"JupyterHub 中断"表现相同。用户批准从 step 125 恢复后要求查明原因。

**浓缩答案**：不是 JupyterHub 空闲回收，而是**容器内存上限 128 GiB 被打满后的 cgroup 组级 OOM**。pod 的 `MEM_LIMIT=137438953472`（128 GiB），容器 cgroup `memory.max` 同值且 `memory.oom.group=1`——cgroup v2 下超限时内核一次杀死容器内全部进程，所以 tmux、watcher、Ray、sglang 同时消失、状态文件停在 running、日志没有任何 Python 异常；pod 直到用户下次登录才重建。六条 formal run 的日志给出一致的指纹：veRL 记录的 `perf/cpu_memory_used_gb`（`psutil.virtual_memory().used`，主机口径、不含 page cache）从起点 161–171 GB 单调涨 +51～+63 GB，**被杀前全部落在 220–225 GB**；恢复后新进程立刻回落约 45 GB（说明增长属于本容器）；恢复段最多 75 步，终点 206–219 GB，刚好没碰到死亡线，所以"恢复一次必定跑完"。E4-A 与本次都恰在 step-150 checkpoint 的 FSDP state-dict 聚合时刻被杀，对应额外约 20 GB 的瞬时峰值。时间假设（culler）无法解释"死亡时内存恒定而时长 5.5–8.7 h 不恒定"。

- 增长源尚未定位：恢复段起点容器 `memory.current` 117 GB，其中 anon 38.7 GB + shmem 29.3 GB 不可回收，page cache 可回收；单进程 RSS 最大的是 `ray::WorkerDict`（actor+ref，param/optimizer offload）49 GB。新增 `scripts/m9/memwatch.sh` 每 5 min 采样 cgroup current/anon/file/peak/oom_kill 与 top-RSS 进程到 run 的 `analysis/container_memory.jsonl`，跑完后可直接看哪个进程线性增长。
- 面试口径：分布式 RL 训练的"莫名其妙的 pod 重启"要先看 cgroup（`memory.max`/`memory.oom.group`/`memory.events`）而不是框架日志；oom.group=1 的特征就是"全体消失、无异常栈"。
- 可选对策（待用户决定）：① 200-step run 预设在 step 100 处计划性重启（现有 resume 机制已能精确从 checkpoint 续跑，成本约 5 min，只需把中断从"意外"改成"预期"）；② 定位泄漏后修（glibc arena 碎片可试 `MALLOC_ARENA_MAX=2`；若是 rollout dump/metrics 累积则改 veRL 配置）；③ 向平台申请更大内存 profile。

**证据**：`/proc/1/cgroup` 对应的 `memory.max`/`memory.oom.group`；六条 run 的 `rl/runs/<run>.log` 中 `perf/cpu_memory_used_gb` 序列；`rl/runs/*/recovery/resume_from_*/recovery.json`；`plans/M9.md` §11（2026-09-07 行）；`rl/runs/e3notool_grpo_20260906_220907/analysis/container_memory.jsonl`。

## Q18（2026-09-07，复核 M3 结论）：当初"SFT 有用"的依据是工具失败率下降 + L4/L5 用工具更准，这个观察对吗？E3 / E5-v2 用工具的准确率和分难度准确率有提升吗？

**答（SFT 那条观察：对了一半）**。同一 M1 probe（净训练池分层 100 题/level、temp 0.6、n=4、bare TIR），零样本 → SFT-6k：

| Level | 工具报错率 | 未用工具率 | TIR pass@1 | CoT pass@1 | 工具增益 |
|---|---|---|---|---|---|
| 3 | .317 → .155 | .228 → .070 | .745 → .775 | .845 → .838 | −.100 → −.062 |
| 4 | .392 → .232 | .268 → .065 | .588 → .642 | .750 → .693 | −.163 → −.050 |
| 5 | .456 → .272 | .270 → .150 | .333 → .367 | .425 → .338 | −.092 → **+.030** |

- **"工具失败率下降"成立且稳健**：probe 全 level 下降；M7 统一评测口径 per-call 执行成功率 Raw .626 → SFT .768，未用工具率 .393 → .088。
- **"L4/L5 用工具更准"成立但小且 protocol 依赖**：同 probe 上 L4 +5.4pt、L5 +3.4pt；但换到 M7 held-out MATH500 greedy，SFT 相对 raw 在 L4 是 −2.4pt（.586→.562）、L5 +0.8pt（.313→.321），基本没动。且 CoT 臂同时回落，**工具增益只有 L5 转正**，全局仍 −.032。
- 正确表述：**SFT 买到的是可靠性（会发调用、代码能跑、不循环重试），不是准确率；准确率的大跳在 RL 阶段。**

**答（E3 / E5-v2 用工具的准确率：明确提高）**。greedy、FULL760：

| 阶段 | per-call 执行成功 | 报错轨迹率 | 4 调用率 | 截断率 | P(对\|有调用) | P(对\|调用全成功) |
|---|---|---|---|---|---|---|
| Raw | .626 | .146 | .100 | .179 | .568 | .663 |
| SFT | .768 | .134 | .078 | .151 | .646 | .698 |
| E3 | .881 | .067 | .013 | .096 | .774 | .807 |
| E5-v2 | .906 | .042 | .008 | .087 | .806 | .834 |

（E5-v1 不参与比较：截断率 100%、4-call 92.9%，已确诊的 over-calling 失效臂，其 per-call .986 是"重复跑同一段能跑的代码"的产物。）

**答（分难度，greedy MATH500 整体准确率）**：

| Level | Raw | SFT | E3 | E5-v2 | E3−SFT | E5v2−E3 |
|---|---|---|---|---|---|---|
| 1 | .837 | .930 | .953 | 1.000 | +2.3 | +4.7 |
| 2 | .733 | .767 | .900 | .878 | +13.3 | −2.2 |
| 3 | .657 | .695 | .848 | .838 | +15.3 | −1.0 |
| 4 | .586 | .562 | .797 | .812 | **+23.5** | +1.5 |
| 5 | .313 | .321 | .522 | .537 | **+20.1** | +1.5 |

- **RL 的提升集中在 L4/L5**，正是 M1 预言的潜在工具增益区、也是 RL 训练池（L3–5）所在——这条对上了。
- **E5-v2 相对 E3 分 level 是 ±2pt 噪声**，与 FULL760 +0.79pt、CI `[−1.84,+3.42]`、情形 A 一致；分难度切开也没有藏着的效应。
- 分难度的条件准确率差距同样集中在 L4/L5：E3 有调用 vs 零调用 = .840/.643（L4）、.582/.361（L5）；E5-v2 = .861/.750、.644/.453；L1–3 两者持平。

**两条必挂的限定**：(1) 选择偏置——E5-v2 零调用率 48.6%（E3 26.6%），是策略自己挑题发调用，条件准确率不可作因果读；(2) 仍无 CoT-only RL 对照臂，SFT→E3 的 +12.8pt 分不出"学会用工具"与"RL 提升推理"，即 M9 要补的洞。

**证据**：`analysis/tool_use_conditional_accuracy.json`（分 level 分层已含 `MATH500_level_1..5`）、`data/probe/metrics.json`、`data/probe_sft/metrics.json`、两份 `metrics/tool_behavior.json`、`eval/runs/m8_e5v2_eval_20260905_134737/metrics/m8_headline.json`、`reports/01_tool_gain.md` 正文与 M3 附录。

## Q19（2026-09-07）：各阶段用的训练/评测数据集分别是什么？既然 AIME 结果"不要读"，为什么还测它？

**答（数据链条）**。全部源自一个池子 MATH train（lighteval 预处理），一次性切分后各阶段严格不交叉：

| 阶段 | 数据 | 条数 | 出处 |
|---|---|---|---|
| 原始训练池 | `data/processed/math_train.jsonl` | 7496 | MATH-lighteval train |
| 去污染后 | `math_train_clean.jsonl` | 7280 | `dedup_check.py` 剔 216 |
| M1 选集（L3–5） | `train_subset.jsonl` | 5403 | L3 1558 / L4 1654 / L5 2191 |
| M3 held-out | `sft/data/heldout_200.jsonl` | 200 | 从 5403 分层留出，不参与 SFT 与 RL |
| SFT prompt 池 | `sft/data/sft_pool.jsonl` | 5203 | 5403 − 200 |
| SFT 训练轨迹 | `sft/data/sft_traces.jsonl` | 6056 | Qwen3-8B 教师、temp 0.7、n=2，拒绝采样 yield 58.2%，覆盖 3716 题 |
| RL prompt 池 | `rl/data/e3_train.parquet` | 5203（verl 长度过滤后 5195） | 与 SFT 同池、同题 |
| RL 训练内 val | `e3_val_math500_100.parquet` | 100 | MATH500 分层 20/level，仅看曲线 |
| 最终统一评测 | MATH500 500 + GSM8K test 200 + AIME24 30 + AIME25 30 | **760** | greedy n=1 与 temp 0.6 n=4 两套 |

注意两点：**(a)** RL 与 SFT 用同一批题（5203），这是刻意的——RL 在 SFT 已见过的题上做 on-policy 改进，不是新数据带来的提升；**(b)** MATH500 是评测集，但 RL 训练内 val 用了它的 100 题子集，所以最终报告以 760 题全量为准，MATH500 那 100 题严格说是"调过曲线的"（无早停、无选 checkpoint，固定 step 200，风险有限但要如实说）。污染检查：5403 训练题 vs 四个评测集精确 + 13-gram 命中 **0**（`data/contamination_report_subset.md`）。

**答（为什么测 AIME 却不读单点差异）**。两件事不矛盾，因为 AIME 承担的是**污染兜底**职责，不是**效应量测量**职责：

1. **测它的理由**：AIME 2024/2025 时间上晚于基座知识截止，是全 panel 里污染风险最低的一档。它回答的是"MATH500 上的 +17pt 是不是背题背出来的"——如果模型在 MATH500 大涨而在 AIME 上塌成随机，就说明提升不可信。实际 Raw .100 → E3 .133 → E5-v2 .200（AIME24），方向一致、没塌，兜底通过。这是**定性的健全性检查**。
2. **不读单点差异的理由**：n=30。1 题 = 3.3pt。E3 4/30 vs E5-v2 6/30 只差 **2 题**，Wilson 95% CI 分别是 `[.053,.297]` 与 `[.095,.373]`，宽度约 25pt，完全重叠；而要检测的效应量本身只有约 1pt。用它做主结论等于用尺子量原子。所以 AIME 进 760 题的合并 bootstrap（按 question_id 重采样），单独一列只作展示，不单独下结论。
3. 同理 GSM8K 200 题是 sanity check（心算即可解，工具增益趋近零，PLAN §5.2 已据此否决用 GSM8K 训练），看的是"有没有退化"，不是"有没有提升"。
4. 面试口径：**评测 panel 的每个数据集要说清承担哪个职责**——MATH500 是效应量主战场（同分布、n 够、有 level 分层），AIME 是污染与泛化兜底（低污染、n 小），GSM8K 是退化 sanity。把兜底集的小样本差异当结论汇报，是评测设计里的典型错误。

**证据**：`data/README.md`、`rl/data/manifest.json`、`sft/data/gen/gen_stats.json`、`eval/m7_eval_config.json`、`data/contamination_report_subset.md`、PLAN §5.1/§5.2/§11。

## Q20（2026-09-07，数据谱系细节）：原始训练集是什么、怎么去污染、held-out 200 怎么来、RL prompt 池怎么来、MATH500 与训练集什么关系？

**原始训练集**：MATH（Hendrycks）**train split**，取自 M0 已下载的本地 `~/verl-team/lighteval-MATH-preprocessed/train.parquet`（源 `DigitalLearningGmbH/MATH-lighteval`），7500 条。`data/download_convert.py` 转统一 schema `{id,question,answer,level,subject,source,split}`，剔 4 条（2 条 `Level ?`、2 条空 ground_truth）→ **7496**。question 统一剥掉 verl-team 预处理追加的固定后缀 `" Let's think step by step and output the final answer within \boxed{}."`（断言全表匹配），训练/评测时再按需拼回——保证 SFT、RL、评测三处 prompt 完全一致。

**去污染**（`data/dedup_check.py`，PLAN §5.1）：训练池对四个评测集（MATH500 / AIME24 / AIME25 / GSM8K200）逐条比，两个判据——(1) 归一化（转小写、非字母数字→空格、压空白）后**精确匹配**；(2) 词级 **13-gram 重叠**（GPT-3 式）。命中即剔除，不做人工豁免。结果：MATH500 命中 172、AIME24 66、AIME25 69、GSM8K 0，去重后共剔 **216** 条 → `math_train_clean.jsonl` **7280**。绝大多数是答案格式模板句（如 "where m and n are relatively prime positive integers find m n"）而非真题目重复——AIME 只有 30 题却命中 66/69 条正是这个原因。选集后复跑（`contamination_report_subset.md`）：5403 条 vs 四个评测集命中 **0**。脚本带 `--self-test`（构造精确/n-gram/干净三种行验证检出）。

**一个已核实的假阳性（本次新发现，2026-09-07）**：报告里唯一那条 "exact" 命中 `math_train_007163 ↔ math500_000046` **不是真重复**——训练题是 `z^4 - z^2 + 1 = 0`（答案 12），评测题是 `z^4 + z^2 + 1 = 0`（答案 6）。归一化把 `+`/`-` 一并去掉才撞在一起。方向是保守的（过度剔除，不是漏检），不影响任何已发布结论；**MATH train 与 MATH500 之间真实的精确重复数为 0**。

**held-out 200**（`sft/make_splits.py`，seed 42）：从 **5403 条 L3–5 选集**里按 level 比例分层随机抽 200（实得 L3 58 / L4 61 / L5 81），剩余 **5203** 作 `sft_pool.jsonl`。它的作用是 M3 的**验收集**：SFT 后测工具格式成功率与 pass@1 vs 零样本（结果 89.5% / +1.9pt，均为边缘，见 plans/M3.md）。它既不参与蒸馏轨迹生成，也不进 RL——`rl/prepare_data.py::validate_split` 硬断言 `heldout_ids & train_ids == ∅` 且 train 必须恰好 5203 行，不满足直接抛错。注意它**不在最终 760 题 panel 里**，只服务 M3 验收。

**RL prompt 池**：就是 `sft_pool.jsonl` 那 5203 题，一题不多一题不少，`rl/prepare_data.py` 转成 veRL parquet（`prompt` 拼回后缀、`reward_model.ground_truth` 放答案、`extra_info` 保留 id/level/subject）。verl 按 `max_prompt_tokens=1024` 过滤掉 8 条超长题，**实际训练 5195**。这意味着 **RL 与 SFT 用同一批题**——RL 是在 SFT 已见过的题上做 on-policy 改进，+17.2pt 不来自新数据。训练内 val 是 MATH500 的 100 题分层子集（20/level，seed 42），只用于画曲线。

**MATH500 与训练集的关系**：三层。
1. **split 级天然隔离**：MATH500 = `HuggingFaceH4/MATH-500`，是 MATH **test** split 的 500 题子集（Lightman et al. PRM800K 选取），训练池是 **train** split，构造上不相交；实测归一化精确重叠 0（上面那条是假阳性），13-gram 保守剔除后训练子集命中 0。
2. **分布上是同源 in-distribution 评测**：同为 MATH、subject 分布接近，所以它是效应量主战场；但 **level 上不对齐**——训练只有 L3–5，MATH500 含 L1 43 / L2 90 共 133 题属训练分布外的"更简单"档。这正好解释了工具增益在 L1–3 为负而 RL 提升集中在 L4/L5（见 Q18）。
3. **有 100 题被"看过曲线"**：RL 训练内 val 取自 MATH500。缓解是全程无早停、不按 val 选 checkpoint、一律固定 step 200；但汇报时必须说明，760 题合并结果才是主口径。

**证据**：`data/download_convert.py`、`data/dedup_check.py`、`data/contamination_report{,_subset}.md`、`data/select_train_subset.py`、`sft/make_splits.py`、`rl/prepare_data.py`（`validate_split`）、`rl/data/manifest.json`、`data/README.md`。

## Q21（2026-09-07，M9 判读）：工具到底有没有用？SFT→E3 的 +12.76pt 该怎么讲？

**场景与问题**：M9 的 E3-NoTool 对照臂跑完并完成协议 v5 评测，落入预注册情形 B（无差异）。
需要给出可对外陈述的判读，以及"那 12.76pt 到底是什么"的解释。

**浓缩答案**：在 1.7B、≤4 轮、MATH 训练池、200 步这一配置下，**工具没有提供额外准确率**
（Δ = E3-NoTool − E3 = −0.92pt，95% CI [−3.55, +1.71]，跨 0）。但"工具学习"本身确实发生了，
只是它买到的是**回到无工具基线**而不是超过它。四个 pass@1 构成一条在同 760 题上精确闭合的路径：

- SFT-TIR .605 → SFT-NoTool .666：起点被工具**拖累** +6.05pt（47%）
- SFT-NoTool .666 → E3-NoTool-eval .713：RL 真正提升的**自身推理** +4.74pt（37%，CI [+2.11, +7.50]）
- E3-NoTool-eval .713 → E3-TIR .733：工具的**推理时边际价值** +1.97pt（15%，CI [−4.74, +0.79] 跨 0）
- 合计 = +12.76pt = SFT→E3

工具的代价随阶段单调收窄：raw −13.95pt → SFT −6.05pt → E3 −1.97pt。也就是说 RL 把工具从"净负担"训成了
"净中性"，这就是那 6.05pt 的来源；而模型自身推理提升了 4.74pt，且这部分**留在权重里**
（E3 权重拿掉工具仍比 SFT-NoTool 高 4.74pt），不依赖解释器。

- **null 不是"到处没差别"**：level 分解显示 E3 相对 E3-NoTool 的优势只在 MATH L4（+7.03pt）与 L5（+2.99pt），
  L1–2 完全打平、GSM8K −1.50pt、AIME 反向。少数难题上的正贡献被简单题上的负担抵消回 −0.92pt。
  与 M1"潜在增益在 L4–5"的先验一致。
- **预注册纪律的两处自我约束**：(i) §8 附加条款要求"SFT-NoTool→E3-NoTool 的增益与 +12.76pt 差 < 3pt 才支持 B"，
  实测 +5.79pt、差 6.97pt，**该条不成立**，必须写明不能据此加强 B——原因正是上面的分解（无工具臂的起点
  已免除那 6.05pt 负担）；(ii) §11 在看到终点**之前**记录了 smoke 显示起点其实偏向 NoTool，
  因此判读中不得引用"保守方向、结论只会被低估"。
- **要让工具净赚该怎么做**：换训练池（长程符号推导、多步数值、需要状态的任务）或换规模，
  而不是调 RL 超参。M9 的证据不支持"再训久一点工具就有用了"。
- **反直觉的副产品**：无工具臂的组内零方差比例 44.6%→56.3%（E3 step 1 为 26.6%），
  即约一半 batch 没有梯度信号却仍打平——这是 M10 零方差过滤的直接动机。

**证据**：`eval/runs/m9_notool_eval_20260907_175726/metrics/m9_headline.json`；
`reports/02_main_results.md` M9 节；`reports/01_tool_gain.md` M9 附录；
`rl/runs/e3notool_grpo_20260906_220907/analysis/formal_completion_gate.json`；`plans/M9.md` §8/§11/§12。

## Q22（2026-09-09，M10 批准前讨论）：M10 是 DAPO 还是 Dr. GRPO？下一步做哪个？为什么 M10 不上 DAPO 四件套？

**场景与问题**：用户阅读 `plans/M10.md` 草案后连问三题：(1) E7 属于 DAPO 还是 "Dr. DAPO"，算法为何提出、解决什么、为何有效；
(2) 下一步做 DAPO（M10）还是 Dr. GRPO；(3) M10 为什么只做动态采样而不上 DAPO 四件套。

**浓缩答案**：

1. **M10 = DAPO 的 Dynamic Sampling 单一组件，不是 Dr. GRPO**（学界无 "Dr. DAPO"，指的应是 Dr. GRPO）。
   GRPO 优势 `A_i=(r_i−mean)/std`，组内 8 条得分相同则分子恒 0，整组梯度贡献严格为零；被学会的题从 mixed 变全对，
   零方差比例随训练单调上升（E3 均值 44%、末 25 步 53%）。危害三层：rollout 算力白耗（占 step 时间 70%+）；
   有效 batch 随训练漂移，梯度方差后期变大；**token-mean 聚合下零优势序列不贡献分子却贡献分母，等价于一个悄悄衰减的学习率**。
   DAPO 做法：actor update 前按 UID 丢零方差组，用**同一冻结 snapshot** 补采样到满 batch。有效原因：被丢的组本来贡献为零，
   不改梯度方向，只保证有效样本数恒定并去掉稀释；副产品是自动课程（informative 集合向能力边界集中）。代价是补采样不免费，
   故 M10 §2 强制双 x 轴（有效 step / 累计 rollout）。本项目比原文更严格：按真正进入优势的标量 `score` 判零方差而非 accuracy，
   因约 7.4% 的组 accuracy 一致但 format 分有方差、仍有梯度。
   **Dr. GRPO** 攻击的是估计量偏置：除 std 的难度偏置（极易/极难组 std 小、优势被放大）与按长度归一化的长度偏置
   （错答越长每 token 罚越轻）。E3 已用 `token-mean`，长度偏置修正已生效；`norm_adv_by_std_in_grpo: true` 保留，
   故 M10 是纯 DAPO 动态采样消融，与 Dr. GRPO 正交。
2. **助手建议先做 Dr. GRPO**：在本项目它收缩成一个 YAML 开关（veRL `core_algos.py` 原生支持），约 1.5 天、不碰框架内部，
   结果不可预测（去 std 后 7:1 / 1:7 组权重压低 2–3×，而 E3 后期 informative 集合正向这类边缘组集中，效应可能随训练**变大**），
   且直接对应 `PLAN.md` 面试表第 312 行的 ⚠️；M10 约 4.5 天、复制 409 行 `fit()`、预期落情形 A。
   需先定口径：去 std 后优势量级缩小 2–3× 等价于学习率下降，"论文原样不补偿"与"补偿后只看难度偏置"要二选一并预注册。
   **用户决定：仍先做 M10**（2026-09-09）。
3. **为什么不上四件套**：Clip-Higher（0.2/0.28）与 token-level loss（veRL 默认 `token-mean`）**已在 E3 基线里**，是所有臂共享配置而非处理变量；
   单因子纪律——全部结论建立在 E3→Ex 配对分析与 resolved-config diff gate 上，M10 §3.2 只允许 `filter_groups` 三键不同，
   四件一起开则 Δ 无法归因；Overlong reward shaping 治截断奖励噪声，而 E3 后期截断率约 1%、E5-v2 把 4 轮截断压到 8.7%，
   病灶基本不存在，且它改 reward、属 E4 领地并被 M10 §1 划为范围外；去 KL 信息量太低（系数 0.001、low_var_kl，
   `PLAN.md` 313 行定位为"低系数折中并监控"）。动态采样是四件中唯一需改 trainer、也唯一对应本项目已量化病灶（44%→53%）的组件。
   DAPO 是系统论文，四件套是为 32B、2 万 token 长 CoT 配的药；本设置熵/长度在 M4 已验证健康，硬上四件回答的是"能否复现 DAPO"而非研究问题。

**证据**：`plans/M10.md` §1–§3；`rl/configs/e3_grpo_baseline.yaml` 第 6/37–41 行；veRL pin
`trainer/ppo/core_algos.py:296`（Dr. GRPO 开关注释）与 `trainer/config/_generated_ppo_trainer.yaml:82`（`loss_agg_mode: token-mean`）；
`PLAN.md` 312–316、479 行；`reports/technical_report.md` M8 行（截断 100%→8.7%）。

## Q23（2026-09-09，秋招简历）：三条如何组织，轮级信用问题能否称 reward hacking？

**场景与问题**：用户反复修订秋招简历，要求取消逐条小标题、按 SFT→GRPO 排列、简化教师与指标细节、不用“初版”暗示迭代次数，并准确解释工具过度调用。

**浓缩答案**：当前文案见 [简历三条版](resume_bullets_autumn.md)，完整会话见 [面试总结](docs/resume_interview_session_20260909.md)。第一条讲管线，第二条讲数据与 SFT→GRPO，第三条讲问题、优势诊断与守恒修正。约 6,000 是轨迹数；34%→19%、59.4%→76.6%、92.9%→0.8% 分属 raw/SFT 探针、MATH500、FULL760 对照。第三条可省略数字，写“大幅降低”，不以“显著”暗示检验。非守恒优势修正引入隐式调用激励，比笼统“解决 reward hacking”更准确；v2 联合调整守恒、门控与打分条件，不支持单项因果归因，相对 E3 准确率未检出显著增益。零和指 token 加权修正和为零，不是梯度/最优策略不变。项目日期按实际记录，可写 2026.07—至今。

**证据**：`plans/M8.md`；`rl/custom/turn_advantage_v2.py`；`eval/runs/m8_e5v2_eval_20260905_134737/metrics/m8_headline.json`；本文件 Q12；[面试总结](docs/resume_interview_session_20260909.md) §1/§4。

## Q24（2026-09-09，数据与编码）：6,000 条是否合理，能否称蒸馏，为什么教师生成与 AgentLoop 分开？

**场景与问题**：用户担心冷启动数据规模偏小、教师型号不够突出，并追问“统一 SFT/RL 多轮交互编码”的实际含义。

**浓缩答案**：面向工具使用可靠性的冷启动 SFT，6,000 条是合理的实验规模，但不能仅凭数量宣称足够。项目对比过 2,500 与约 6,000 条，工具错误率约 22%→19%，未证明饱和。5,403 题去污染子集留出 200 题，5,203 题各采样两次，得到 10,406 候选，经顺序筛选保留 6,056 条、覆盖 3,716 题，含 797 条报错后恢复轨迹。可以称序列/轨迹蒸馏，无教师 logits；更直白的写法为“教师生成轨迹，经拒绝采样用于学生 SFT”。教师生成复用 M1 的本地 SGLang+真实沙箱循环；SFT 制样再通过 veRL ToolAgentLoop 重放已有回复和输出，复用序列拼接及 mask，不重新生成或执行工具。工具返回保留为上下文但不计 loss。该处理是复用框架保证接线一致，不是原创 AgentLoop，也不保证教师/RL 所有预算或生成分布相同。

**证据**：`sft/data/gen/gen_stats.json`、`sft/gen_trajectories.py`、`sft/trace_tokenizer.py`、`sft/test_trace_tokenizer.py`、`plans/M3.md`；[面试总结](docs/resume_interview_session_20260909.md) §2/§3，含外部蒸馏术语参考。

## Q25（2026-09-09，预算）：教师和 GRPO 实际截断多少，为什么选 3,072 token 与 5 轮？

**场景与问题**：用户要求查看原始教师轨迹和 GRPO 评测记录，用实测分布解释长度选择，并核对此前放宽预算的效果及“5 次工具调用”的含义。

**浓缩答案**：

- 本次 CPU 离线重算 10,406 条教师轨迹，累计响应均值 751.95、P95 2,050、P99 3,078.8；超过 2,048/3,072 的分别为 549/109 条（5.28%/1.05%）。保留 6,056 条均值 580.25，逐条与既有长度记录比较 0 mismatch。长度包括工具返回及格式标记，沿用生成筛选 tokenizer 口径。
- 教师 `truncated=True` 374 条（3.59%）是轮数耗尽；最后一次请求 `finish_reason=length` 另有 213 条（2.05%），不重叠。两类合计 5.64%，但没有每轮 finish reason，不能证明完整捕获中间请求触顶。通过前置筛选后仅 8 条因累计长度 >3,072 被丢弃，8/6,064=0.13%。
- E3 FULL760 greedy 截断 73/760=9.61%，其中 token 66/760=8.68%，轮数 7/760=0.92%；3 条截断仍答对，因此长预算诊断选取 70 条失败（token 63、轮数 7）。MATH500 greedy 总截断 40/500=8%，token 34/500=6.8%。
- 诊断同时增加 response 3,072→6,144、工具次数 4→6、assistant 轮数 5→7。16/70=22.86% 转正确，其中原 token 失败修复 11/63=17.46%；其余 14 正常结束仍错、38 再次 token 截断仍错、2 再次轮数耗尽仍错。16 条正确中 1 条仍截断，因此长预算总截断 41/70，不是 40/70。前缀一致仅 17/70，不能归因于单独增加 token 或声称严格续写反事实；不可把选择性重跑结果并入 canonical 总分。
- “5”是 assistant 回复轮数；实际最多 4 次工具调用，额外一轮处理末次工具结果并作答。E3 greedy 用满 4 次调用 10/760，轮数停止 7/760。预算是短程任务的工程折中，不是搜索出的最优值；当前统计是后验复核支持，不能补写成当初按 P99 选参。后续完整预算搜索应在训练/验证数据上完成，正式评测保持冻结。

**证据**：[面试总结](docs/resume_interview_session_20260909.md) §5–§7（完整表格、只读复核命令及可复述回答）；`eval/runs/m7_unified_eval_20260824_201032/scored/e3_grpo_baseline_step_200/`；`eval/runs/e3_budget_diagnostic_20260828_095658/{metrics.json,comparisons.jsonl}`；`plans/M7_BUDGET_DIAGNOSTIC.md`。本次未训练/生成或重跑模型评测；用户最后授权仅将本会话沉淀到 reports 文档。

## Q26（2026-09-10，M10 步骤 4–6）：训练 200 步后容器匿名内存涨到 60 多 GiB，到底是哪个进程在漏？关掉 FSDP CPU offload 有没有用？

**场景与问题**：Q17 查明六条 200-step run 的"pod 中断"是 128 GiB cgroup 上限下的组级 OOM，但增长源没有定位；
2026-09-09 的 offload 关闭验证只跑 5 步，观察到 anon 仍以约 0.5 GB/step 增长，并**猜测**泄漏在 AgentLoop/sglang 侧的轨迹缓存或
reward 沙箱残留（`HANDOFF.md` 2026-09-09 条目）。M10 计划 §1 把"跑的过程中同步定位泄漏进程"列为目标（只诊断不修）。
E7 formal 全程用 memwatch v2（60 s 一次，按进程读 `/proc/<pid>/smaps_rollup` 的 `Pss_Anon`，`step` 字段读 trainer 每步 fsync 的 ledger）采样。

**浓缩答案**：**泄漏在 `ray::WorkerDict`（actor 与 ref 共置的那个 FSDP worker 进程）内部，占 cgroup anon 增长的 93%；
不在 AgentLoop、sglang、reward 沙箱或驱动进程。关闭 CPU offload 只把常驻基线从 49 GiB 降到 25 GiB，不改斜率。**

- 证据规模：`rl/runs/e7_dynamic_filtering_20260909_200152/analysis/memwatch_report.{json,md}`，828 个采样 / 16.63 h / 有效 step 1–200；
  cgroup anon 5.23 → 66.41 GiB（峰值 66.41），`oom_kill` 0，全程未触及 110 GiB 提前分段线。
- 斜率必须**按驱动 PID 分段拟合**：formal 分 4 段（每段重启、模型重新加载，anon 回落再爬升），逐段 **0.5047 / 0.5382 / 0.5431 / 0.5580 GiB/有效 step**，
  headline 取 post-startup 采样最多的第 4 段；跨段混拟会得到 0.038 GiB/step 的假数字（`analysis/m10_memwatch_report.py` 显式拒绝混拟）。
  拟合只用"段内 step 首次增加之后、driver 仍存活"的样本，避免把模型加载和收尾回收算进斜率（`plans/M10.md` §12 两条）。
- 按 `Pss_Anon` 增量排名，**每段的单个 `ray::WorkerDict` 稳居第一**：第 4 段 9.70 → 29.60 GiB，**0.5213 GiB/step = cgroup 斜率的 93.4%**，RSS 终值 48.6 GiB；
  四段分别 0.4885 / 0.5042 / 0.5138 / 0.5213 GiB/step。其余进程全部小两个数量级：驱动 actor `ray::E7DynamicF` 0.0024、`sglang::schedul` 0.0019、
  `gcs_server` 0.0027、`ray::AgentLoopW` 每个 ≤ 0.0011、`ray::SGLangHttp` ≤ 0.0004 GiB/step。
- 这**否定**了 2026-09-09 的猜测：AgentLoop worker 与 sglang HTTP/scheduler 进程的匿名内存 200 步只涨 0.1–0.5 GiB，轨迹缓存/沙箱残留不成立。
  E7 一个有效 step 生成 2–4 批（比 E3 多 2.53×），若泄漏与 rollout 数量成正比，E7 斜率应约为 E3 的 2.5×；实测 0.56 vs E3 约 0.35–0.45 GB/step（Q17 口径：psutil used 每 150 步 +51–63 GB），
  说明增长与**有效 update 次数**而非 rollout 数量挂钩，进一步把嫌疑指向 FSDP worker 内每次 `update_actor`/`compute_log_prob` 相关的宿主端分配
  （candidates：pinned/host 缓冲、优化器状态的 CPU 副本碎片、glibc arena 增长）。**这一条是推断，不是定位到代码行**；按计划 §1 不修。
- 工程含义：(1) 128 GiB 容器上，基线 25 GiB + 0.56 GiB/step 意味着约 150 步撞线，所以"每 50 有效 step 计划性分段恢复"是正确且必要的运行纪律，
  4 段全部无 OOM；(2) 关 offload 换来的是 24 GiB 基线余量与 GPU 余量（142 GB 卡上 max_allocated 43.5 GB），而不是根治；
  (3) 若要根治，下一步应在 `ray::WorkerDict` 内做 `tracemalloc`/`MALLOC_ARENA_MAX=2` 对照或 `torch.cuda.memory` host 端统计，不是再调 AgentLoop。
- 面试口径："分布式 RL 训练的匿名内存泄漏，先按 cgroup 看是不是组级 OOM，再按进程 `Pss_Anon` 分段拟合斜率找到进程，最后才进代码；
  中间任何一步跨段混拟或把启动期算进去都会给出错一个量级的数字。"

**证据**：`rl/runs/e7_dynamic_filtering_20260909_200152/analysis/{memwatch_report.json,memwatch_report.md,container_memory.jsonl}`；
`scripts/m10/memwatch_v2.sh`、`analysis/m10_memwatch_report.py`（+ `analysis/test_m10_memwatch_report.py` 6 项）；
`plans/M10.md` §12（2026-09-09 两条 memwatch 偏差、段长决定）与 §13 第 8 行；`rl/runs/e3_offload_check_20260909_150849/analysis/offload_check_compare.md`；本文件 Q17。

## Q27（2026-09-10，M10 判读）：DAPO 动态采样到底有没有用？等 step 赢 5 个点、等算力打平，该怎么讲？

**场景与问题**：M10 评测收工，`m10_headline.json` 机械判读为预注册情形 C（Δ_step +1.44pt CI 跨 0、Δ_compute −0.39pt CI 跨 0），
但附报的 step-200 终点配对 +5.13pt [+2.63, +7.63] 显著为正。两组数字如何同时成立、面试时怎么讲、能不能写"提升 5pt"。

**浓缩答案**：

- **两组数字不矛盾，差在分母**。E7 step-200 花了 259,072 条 rollout 轨迹，是 E3 的 2.53×；等算力（102,400 条）时 E7 只到有效 step 95，
  FULL760 greedy .729 vs E3 .733。动态采样在本设置下**没有提高每条 rollout 的学习效率**——把 fixed-100 曲线从"按 step"换成"按累计轨迹"，
  E7 的 AUC 从 +1.44pt 变成 −1.72pt；sampled 等算力配对 −2.11pt [−3.59, −0.62] 甚至显著为负。它做的事是把 E3 根本不会采的那 1.53 倍 rollout
  真的采出来训进去，换到一个更高的终点。所以口径是"**等算力无免费午餐，多算力换终点**"，不是"提升 5pt"。
- **为什么不能说 E7 更强**：没有 E3 在 259,072 条轨迹上的对照（例如从 step 200 续训约 306 步）。E3 的 fixed-100 曲线 step 100 后在 .73–.77
  震荡像平台，但那是 100 题的面板，分辨不了 5pt。这条缺失的臂是 M10 之后最直接的下一步，需另行授权。
- **为什么判读是 C 而不是 A**：§9 的 Δ_step 定义在 fixed-100 曲线的 0–200 归一化 AUC 上，+1.44pt 没过 2pt 阈值且 CI [−1.06, +4.13] 跨 0。
  面板 n=100、CI 宽 5pt，本质上分辨不了 2pt——这是设计时就接受的局限（§9 原文"CI 用验证点上的 bootstrap 近似"），不是事后借口。
  FULL760 上的等 step 终点 +5.13pt 是另一个量（终点、760 题），§9 明确列为附报、不参与落格。
- **情形 D 的影子**：等算力 checkpoint 在 MATH L2/L4/L5 各低约 3pt、sampled 低 2.1pt，与"补采样把训练分布推向更难的 mixed 题、95 次 update
  还没学会"一致；点估计有方向，没过阈值，按 §9 写 C 并注明。
- **成本发现是独立结论**：零方差比例 34.5%→61.4%、全对组 24%→50%，凑满 64 个 informative 组的批数 2.0→3.06，175 步后开始出现满 4 批的步；
  约 75% 时会 underfilled。"过滤越往后越贵"与 curriculum 是同一件事的两面：informative 集合向能力边界收缩，rollout 的边际信息量下降。
  DAPO 原文"总时间没有显著增加"在 1.7B/MATH 池/200 步上不成立。
- **行为侧**：E7 两个 checkpoint 都比 E3 少调工具（.63–.67 vs .89），step-200 的调用成功率 93% vs 88%、重复代码 0.8% vs 2.9%、截断 6.8% vs 9.6%；
  没有 E5-v1 式 over-calling。过滤改的是训练分布不是奖励，这些只报相关性。
- **面试一句话**："我复现了 DAPO 的动态采样并做了算力归一化：按训练步数看它赢 5 个点，按 rollout 数看它打平——论文只画了前一条曲线。
  它是一个用算力换终点的旋钮，而且越训越贵；要证明它比单纯多训更好，还差一条同预算的对照臂。"

**证据**：`eval/runs/m10_e7_eval_20260910_140124/metrics/{m10_headline,compute_axes}.json`、`transitions/e3_grpo_baseline_step_200__to__e7_{equal_compute,dynamic_filter_step_200}.bootstrap.json`；
`rl/runs/e7_dynamic_filtering_20260909_200152/e7_ledger.jsonl`；`plans/M10.md` §2/§9/§13；`reports/02_main_results.md` M10 节；本文件 Q22。
