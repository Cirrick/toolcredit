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
