# LLM RL / Agentic RL / 信用分配：趋势、开放问题与发表可行性（截至 2026-11-17）

> 调研范围说明：本笔记只能通过搜索引擎获得信息。arxiv.org、huggingface.co、alphaxiv.org、icml.cc 在本会话网络中**全部无法直接抓取**（DNS 被拒），因此下列结论多来自搜索结果摘要与二手页面，没有逐篇读全文。凡数字未能核对原文处均已标注。研究者画像：单卡 GH200、Qwen3-1.7B、veRL 多轮工具 GRPO；已有负结果包括 turn-level credit 只加速不提终点且导致过度调用工具、零和重分配消除过度调用但短视野无显著增益、DAPO 零方差过滤在等 rollout 算力下无增益。

## Q1. 热点主题：2025–2026 各会议在 LLM RL / agents 方向的主流是什么

### Takeaway
"LLM agents" 和 "RL for LLM" 是 NeurIPS 2026 增长最快的类别之一（第三方统计中 LLM agents 类从 105 篇增至 312 篇）。2026 年的研究热点正从"单轮 RLVR 数学推理"转向长视野多轮 agent，具体包括 credit assignment、memory 与 context 管理、self-evolving 或 continual learning，以及 agent 评测。NeurIPS 2026 的 workshop 列表也明显偏向 agent 评测、持续学习和资源受限 agent。

### Cited Findings
- NeurIPS 2026 主会据第三方统计收到 30,709 篇投稿，接收 7,900 篇，接收率约 25.7%（2025 年约 24.5%）。这不是官方数字，各统计站之间有出入（7,899 篇或 6,080 篇，口径不同）。 — [csconfstats](https://csconfstats.xoveexu.com/conferences/neurips/2026/)；[scime atlas](https://atlas.scime.ai/venues/neurips-2026)
- 第三方分类 scime atlas 给出的 NeurIPS 增长最快的类别如下（第三方自动分类，只能看方向）：
  - LLM agents（归在 multi-agent systems & games 下）：105→312 篇
  - Policy gradient / actor-critic RL：55→100 篇
  - RL from feedback：80→122 篇
  — [scime atlas NeurIPS 2026](https://atlas.scime.ai/venues/neurips-2026)
- NeurIPS 2026 官方 workshop 公告（2026-08-10，标注为 preliminary）中与本方向相关的 workshop：
  - "Foundations of LLM Post-Training in Changing Environments"
  - "Workshop on Evaluation of Interactive Agents"
  - "Towards Test-Time Continual Learning Agents"
  - "Continual Learning for Enterprise AI Agents"
  - "Workshop on Resource-Aware Agentic AI"
  - "Foundations of Agentic Systems Theory"
  - "First Workshop on Meta Agents"
  - "Reinforcement Learning for Experimental Sciences"

  NeurIPS 2026 分 Sydney、Paris、Atlanta 三地，workshop 日期为 2026-12-11 至 12-13。 — [NeurIPS Blog: Announcing the NeurIPS 2026 Workshops](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/)；非官方汇总见 [GitHub YLfan99/neurips2026-workshops](https://github.com/YLfan99/neurips2026-workshops)（统计 109 个 workshop）
- 截至 2026-08，agentic RL 中的 credit assignment 已有专门综述 "From Reasoning to Agentic: Credit Assignment in RL for LLMs"（arXiv 2604.09459）。该综述认为 agentic 场景的部分可观测性、有限 replay、中间步骤难以验证会让 credit assignment 更难。不同版本的覆盖截止时间不一致（2026-04 或 2026-07-31）。 — [arXiv 2604.09459](https://arxiv.org/html/2604.09459v1)
- 2026 年围绕长视野 agent 的综述与方法论文密集出现：
  - 综述：The Horizon Gap（arXiv 2608.06663，覆盖规划、记忆、执行、训练与评测）
  - 方法：BranPO（2602.03719，用于长视野 agentic RL 的对比式分支采样）、ECHO（2606.31650，选择性保留 turn 记忆）、Hindsight Credit Assignment for Long-Horizon LLM Agents（2603.08754）、Credit Assignment across Multiple Timescales（2608.08255）、LLM Value Functions for Credit Assignment in Agentic RL（2610.15840）

  — [Horizon Gap](https://arxiv.org/pdf/2608.06663)；[BranPO](https://arxiv.org/pdf/2602.03719)；[ECHO](https://arxiv.org/pdf/2606.31650)；[Hindsight CA](https://arxiv.org/pdf/2603.08754)；[Multi-timescale CA](https://arxiv.org/pdf/2608.08255)；[LLM Value Functions](https://www.alphaxiv.org/abs/2610.15840)
- 记忆、context 管理和 self-evolving 已成为一条独立主线。"A Survey of Agent Memory in the Second Half: Towards Self-Evolving and Long-Horizon Agents"（arXiv 2602.06052）把记忆管理看作一项可学习技能，包括"reinforcement-learned context curation and experience consolidation"，并认为核心问题是长视野下的 context explosion。同期相关论文：
  - MemRL（2601.03192）
  - Agentic Memory（2601.01885，用 RL 统一管理长短期记忆）
  - Recursive Experiential-Working Memory Evolution（2608.24876）
  - Evo-Harness（2608.15071）
  - 批评性论文 "Contextual Agentic Memory is a Memo, Not True Memory"（2604.27707）

  — [Agent Memory survey](https://arxiv.org/abs/2602.06052)；[Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)；[Memo not Memory](https://arxiv.org/pdf/2604.27707)；[Evo-Harness](https://arxiv.org/pdf/2608.15071)
- 第三方综述（Turing Post，2026）认为当前格局主要由以下几类方法塑造：GRPO、RLVR、critic-free 优化、DPO 变体、"agentic policy optimization"、test-time diversity 方法。 — [Turing Post: Reasoning RL in 2026](https://www.turingpost.com/p/reasoning-rl-in-2026)
- ICML 2026（Seoul，2026-07-06 至 07-11）的奖项：
  - Test of Time 奖给了 A3C（2016）。委员会特别提到它对 LLM 后训练 RL 的影响。
  - Outstanding Paper Honorable Mention 中有一篇 RLVR 分析论文 "The Obfuscation Atlas: Mapping Where Honesty Emerges in RLVR with Deception Probes"（Wang、Zavatone-Veth、Pehlevan）。
  - Outstanding Paper 本身由扩散模型相关论文获得。

  — [ICML 2026 Awards blog](https://blog.icml.cc/?p=1347)；[ICML 2026 CFP](https://icml.cc/Conferences/2026/CallForPapers)
- NeurIPS 2025：分析型论文 "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"（Yue 等，清华/上交）为 Oral。作者主页和 NeurIPS 幻灯片标注其获得 Best Paper Runner-Up，但 NeurIPS 的 poster/oral 页面只写 oral。论文核心结论：RLVR 主要提升采样效率，base model 在大 k 下 pass@k 更高。 — [NeurIPS slides](https://nips.cc/media/neurips-2025/Slides/119944.pdf)；[NeurIPS oral page](https://neurips.cc/virtual/2025/oral/119945)；[project page mirror](https://hn.jakechampion.name/reader/https://limit-of-rlvr.github.io)
- NeurIPS 2025 接收了 GiGPO（Group-in-Group Policy Optimization，用 anchor state 分组做 step-level credit）。作者报告它在 ALFWorld 上比 GRPO 高 12% 以上、在 WebShop 上高 9% 以上，且显存开销与 rollout 相同，额外耗时不到 0.002%。 — [NeurIPS 2025 poster](https://neurips.cc/virtual/2025/poster/118123)；[verl-agent](https://github.com/langfengQ/verl-agent)

### Inferences
- （推断）credit assignment 已从冷门变成热门，2026 年至少有一篇专门综述和 5 篇以上方法论文，其中 GiGPO 已成为 agent RL 社区的标准 baseline。现在再提出一个 credit 方法，必须与 GiGPO、BranPO、Hindsight CA 等对比，竞争很激烈。
- （推断）2026 年下半年的新热点集中在三处：(a) memory / context 管理作为可学习的动作；(b) continual 或 test-time learning agent；(c) agent 评测与资源受限 agent。NeurIPS 2026 的 workshop 标题与此一致。
- （推断）"分析和证伪型"论文在顶会上有高可见度（Yue 等获 NeurIPS 2025 oral 或 runner-up，ICML 2026 的 RLVR honesty 分析获 honorable mention），这对拥有严格对照负结果的研究者是利好。

### Gaps
- 未能获取 ICLR 2026、ICML 2026 中 "agentic RL" 或 "credit assignment" 关键词的精确论文计数，因为 OpenReview 和 arXiv 在本会话中不可抓取。
- 没有找到 rubric/LLM-judge reward（非可验证任务）、test-time RL（TTRL 类）、off-policy/async RL、multi-agent LLM RL、RL scaling laws 在 2026 年的计数或代表综述。这些方向是否热门只能依据先验判断，本次没有证据。
- 未核实 Paper Digest 和 OpenAccept 上 NeurIPS 2026 的关键词热度数据。

## Q2. 哪些方向拥挤或饱和，哪些方向研究不足

### Takeaway
"在数学 RLVR 上再提一个 GRPO 变体"已高度拥挤：2026 年的综述把它列为一长串变体目录，单篇论文的增益常在 1–3 分，而且越来越多工作质疑 RLVR 本身的增益来源。综述明确列出的研究不足方向有：超长视野 credit、multi-agent credit、exploration 与 credit 的交互、可计算 ground truth 的 credit 评测基准，以及在等算力或等有效样本量下的公平比较。

### Cited Findings
- 一篇 2026 年 GRPO 改进论文的自报增益：在 Qwen2.5-Math-1.5B 上 Pass@1 最高 +2.6 分，在 7B 上 +1.1 分。这反映出单个变体的边际增益已经很小。 — [arXiv 2604.20659 / 2601.20614（ICLR 2026 论文，搜索摘要）](https://arxiv.org/pdf/2601.20614v1)
- 一篇 ICLR 2026 论文（DGPO）指出 GRPO 及其变体普遍存在优势估计不平衡（难题上的更新幅度更小）。它本身也提出了一个新变体，说明"变体之间互相修补"的格局。 — [arXiv 2601.20614](https://arxiv.org/pdf/2601.20614v1)
- 在已饱和的题目上，标准 GRPO 平均准确率 40.7%，与 base model 基本持平。 — [Failure-Prefix Conditioning, arXiv 2601.20829](https://arxiv.org/html/2601.20829v1)
- 2026 年有预印本发现，基于规则的 verifier 的假阴性率会随策略变强而上升，这可能混淆已报告的增益。 — [Are Verifier Errors Independent Within a GRPO Group?, arXiv 2609.06386](https://arxiv.org/pdf/2609.06386)
- RLVR 是否扩展推理能力存在争议：大 k 下 base model 的 pass@k 更高。 — [Yue et al. NeurIPS 2025](https://neurips.cc/virtual/2025/oral/119945)
- credit assignment 综述（2604.09459）的路线图列出的开放方向：multi-agent credit、超长视野、exploration 与 credit 的交互；并呼吁建立统一的 credit-assignment 测试套件，包含"controlled bifurcation points"，即正确 credit 可计算的合成环境。 — [arXiv 2604.09459](https://arxiv.org/html/2604.09459v1)（经搜索摘要获取，未读全文）
- "The Landscape of Agentic RL for LLMs: A Survey"（arXiv 2509.02547，v5）把长视野时间 credit assignment 列为核心挑战，并认为 multi-agent 下可扩展的 credit 分解仍未解决。 — [arXiv 2509.02547](https://arxiv.org/html/2509.02547v5)
- 一篇非同行评审综述（Zhang 2026，GitHub gist）列出的剩余开放问题：reward hacking、长视野稀疏 credit、工具使用探索。 — [gist](https://gist.github.com/zhanglpg/daa3382d8667e91878eb71f2a2347d87)（二手，权重低）
- 立场或概览论文 "Rethinking Agentic RL in LLMs"（Cui 等，arXiv 2604.27859，2026-04-30；v2 改名为 "A Brief Overview..."）。据第三方笔记，它主张单步 RLHF/RLVR 本质是 contextual bandit，真正的 agentic 任务需要多步 POMDP 建模。本笔记未能核对原文。 — [arXiv 2604.27859](https://arxiv.org/abs/2604.27859)；[第三方笔记](https://zhangshuibai.github.io/paper-notes/agents/notes/05_RethinkingAgenticRL_2604.27859.html)
- 与研究者自身发现直接相关的论文："Credit Without Ground Truth"（Haiyue Zhang，arXiv 2608.19760，2026-08）。
  - 该文在一个 7 臂预注册训练实验中发现，没有任何一臂稳定优于未训练策略。
  - checkpoint 之间的表面差异实际反映的是"training dose"：稀疏 credit 丢弃样本，导致优化器步数相差一个数量级。
  - 作者因此主张 credit 规则必须在匹配的有效样本量下比较。
  - replay 审计中，LLM-judge 分数、outcome-conditioned logprob ratio、策略自信度都不比 marginal-matched shuffle 对照更可靠。只有 30.5% 的决策点有可测的因果效应。
  - 实验设置为 ALFWorld + Qwen2.5-7B-Instruct，50 条轨迹，20,538 次 replay rollout。

  — [arXiv 2608.19760](https://arxiv.org/pdf/2608.19760)；[摘要转述](https://www.roboticscenter.ai/research/papers/credit-without-ground-truth-auditing-step-level-credit-assignment-in-llm-agents-against-ex-2608)；[aiunderstanding 报道](https://aiunderstanding.org/news/preprint-audit-finds-common-credit-signals-fail-to-identify-causally-important-ste)；[deeplearn 摘要](https://deeplearn.org/arxiv/812119/credit-without-ground-truth:-auditing-step-level-credit-assignment-in-llm-agents-against-executed-replay)

### Inferences
- （推断）研究者 E3–E7 的负结果有两点与 2608.19760 高度同构：turn-level credit 带来早期加速但终点无增益；按等 rollout 算力判读后增益消失。这一方面说明研究者的结论有外部旁证，不是孤例；另一方面意味着"credit 方法的增益其实是 dose 或算力效应"这个点已被他人抢先公开（2026-08），单独发表时新颖性会被削弱。研究者需要差异化，可选角度：(a) 工具使用和数学环境，而非 ALFWorld；(b) 机制层面的"工具过度调用"病理及零和修复；(c) 系统地随 horizon 长度扫描（研究者目前只有 ≤4 轮）。
- （推断）拥挤方向：数学 RLVR 上的 GRPO 优势或归一化变体、零方差过滤类采样技巧（DAPO 之后大量跟进）、短视野 ALFWorld/WebShop 上的又一个 step-level credit 方法。
- （推断）研究不足方向：在正确 credit 可计算的受控环境中系统测量 credit 质量；credit 与 exploration 或工具调用行为的交互（研究者观察到的 over-calling 正属此类）；在 horizon 增长时 credit 收益的 scaling 曲线；等算力、等有效样本量的比较协议本身。

### Gaps
- 未找到明确批评"GRPO 变体增量化"的立场论文原文，现有证据是间接的。
- 2604.09459 的路线图仅由搜索摘要转述，未读全文，具体开放问题条目可能更多。

## Q3. 算力可行性：单卡与 1.7B–8B 模型能做出被认可的工作吗

### Takeaway
agent RL 中最有影响的 credit 方法论文常用 1.5B–7B 的 Qwen 模型，GiGPO 的主实验就包含 Qwen2.5-1.5B-Instruct。从规模上看，单卡 GH200 跑 1.7B 多轮 RL 处在"可发表"区间的下沿。真正的门槛是 seed 数量、跨环境泛化和等算力对照，而不是模型大小。

### Cited Findings
- GiGPO（NeurIPS 2025）在 Qwen2.5-1.5B-Instruct 上的结果为 ALFWorld 86.7%、WebShop 67.4%（第三方转述的论文数字）。它的卖点之一是"same GPU memory overhead, identical LLM rollout, <0.002% time cost"。 — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/118123)；[emergentmind 摘要](https://www.emergentmind.com/papers/2505.10978)
- verl-agent 的 README 称现在支持"用 2 张 H100 训练 7B 模型"。这是框架能力，不是论文的实验配置。 — [verl-agent GitHub](https://github.com/langfengQ/verl-agent)
- "Credit Without Ground Truth"（2026-08）只用 Qwen2.5-7B-Instruct、50 条轨迹加约 2 万次 replay rollout，就做出了一篇完整的分析或审计论文，算力规模适中。 — [arXiv 2608.19760](https://arxiv.org/pdf/2608.19760)
- 小模型的数学 RLVR 论文常见设置：Qwen2.5-Math-1.5B 和 7B。 — [arXiv 2601.20614](https://arxiv.org/pdf/2601.20614v1)
- 合成或精确 oracle 的 credit 评测可以在极小 MDP 上进行。AgentCreditBench 在小型有限视野 MDP 上用精确的 policy-advantage oracle 测试 GRPO、RLOO、GAE、GiGPO 的 turn-level credit，算力需求可以忽略。 — [GitHub hectopascal/agent-credit-bench](https://github.com/hectopascal/agent-credit-bench)

### Inferences
- （推断）单卡 1.7B 适合三类工作：(a) 分析或审计型论文（受控环境、多 seed、等算力协议）；(b) 合成环境与 benchmark 类论文；(c) 机制性的小方法，每 run 低成本、多 seed 验证。不适合宣称 SOTA 的大规模 agent 方法论文（如需在 SWE、WebArena 上用 7B 以上训练数百 step）。
- （推断）研究者已有的预注册判读和 bootstrap CI 方法论（E3–E7）本身就是审稿人越来越看重的 rigor 卖点，2608.19760 同样采用了预注册。
- （推测）单卡跑 8B 多轮 GRPO 在 GH200 的 146GB 显存下可行，但 rollout 吞吐会成为 seed 数量的瓶颈。本次未找到具体数字。

### Gaps
- 未能获得 GiGPO、VinePPO、TRACE、Memory-R2 论文中的确切 GPU 数与 GPU-hours（原文均在 arXiv，无法抓取）。VinePPO 的会议与算力未核实；TRACE 和 Memory-R2 本次搜索无结果。
- 没有找到"单卡完成并被顶会接收的 agent RL 论文"的系统统计。

## Q4. 评测缺口：是否存在直接衡量 credit-assignment 质量的 benchmark

### Takeaway
截至 2026-11，没有被广泛采用的标准 credit-quality benchmark。有三类零散尝试：(1) 小 MDP 精确 oracle 的一致性测试（AgentCreditBench，GitHub 项目）；(2) 基于 executed replay 的因果审计（2608.19760，结论是常见信号都不可靠）；(3) 过程质量 benchmark（AgentProcessBench，测的是 PRM 而非 credit）。credit 综述本身也呼吁建立带"controlled bifurcation points"的统一套件。这是一个明确且可在小算力下推进的空缺。

### Cited Findings
- AgentCreditBench（GitHub）用 exact-oracle 一致性测试评估 turn-level credit。它在小型有限视野 MDP 上计算精确的 policy-advantage，覆盖 GRPO、RLOO、GAE、GiGPO 和 verl、TRL、OpenRLHF 的实现，并检查 credit 是否落在无关 turn 上。 — [agent-credit-bench](https://github.com/hectopascal/agent-credit-bench)（未核实其成熟度与 star 数）
- "Credit Without Ground Truth" 以重新执行替代动作得到的因果贡献为参照，审计 step-level credit，结论是常见信号均不显著优于 shuffled 对照。 — [arXiv 2608.19760](https://arxiv.org/html/2608.19760v1)
- AgentProcessBench（arXiv 2603.14465）为工具使用 agent 提供基于环境的密集 step 标签，用于诊断 step 级过程质量，评估的是 PRM。 — [arXiv 2603.14465](https://arxiv.org/html/2603.14465v1)
- 综述 2604.09459 建议建立统一的 credit-assignment 套件，包含正确 credit 可计算的合成 agentic 任务，即"controlled bifurcation points"。 — [arXiv 2604.09459 v2](https://arxiv.org/html/2604.09459v2)

### Inferences
- （推断）这是与研究者现有资产最契合的空缺。研究者已有 Python 解释器工具环境、严格的等算力评测协议，以及"over-calling"这类 credit 病理现象。可以构建一个 horizon 可调、关键决策点（bifurcation）位置可控、正确 credit 解析可得的多轮工具使用环境，并在其中系统比较 GRPO、turn-level、零和重分配、GiGPO 等方法的 credit 保真度与终点性能。这类工作在 NeurIPS Datasets & Benchmarks 轨道或分析型论文中都有位置。
- （推断）需要和 AgentCreditBench 明确区分：它是极小 MDP 上的单元或一致性测试，不训练 LLM。研究者可以定位为"真实 LLM 训练循环中的受控环境与 horizon scaling"。

### Gaps
- 未能确认 AgentCreditBench 是否有配套论文，以及其社区采纳程度。
- 搜索未找到专门用于 LLM credit 分析的公开合成长视野环境（如可调 horizon 的 tool-use gridworld）。不排除存在，但本次没有找到。

## Q5. 未来约 12 个月的投稿日历与各会偏好

### Takeaway
今天（2026-11-17）ICLR 2027 主会投稿已截止，2026-12-16 出决定。下一个顶会窗口是 ICML 2027，官方页面显示 abstract 截止 2027-01-14、全文截止 2027-01-22，距今约 9 周。然后是 ACL 2027 的 ARR（约 2027 年 1–2 月）、NeurIPS 2027（日期尚未公布，往年在 5 月中旬，属推断）、EMNLP 2027 的 ARR（暂定 2027-05-28）。

### Cited Findings
- ICLR 2027：
  - 官方 Dates 页：abstract 截止 2026-09-18，paper 截止 2026-09-25，最终决定 2026-12-16。
  - 另一个官方 Author Guidelines 页写的是 9-11 和 9-16，两页互相矛盾。
  - 作者与审稿人讨论到 11-18 为止。
  - 现在仍开放的只有 Blog Post 轨道，截止 2026-12-01 AoE。
  - ICLR 2027 规定每位作者最多 20 篇投稿，并要求 AI 使用声明。

  — [ICLR 2027 Dates](https://www.iclr.cc/Conferences/2027/Dates)；[ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)；[ICLR 2027 Blog Posts CFP](https://iclr.cc/Conferences/2027/CallForBlogPosts)；[aiweekly](https://aiweekly.co/alerts/iclr-2027-caps-authors-at-20-papers-mandates-ai-use-statement)
- ICML 2027：
  - 官方 reciprocal reviewing 页面：abstract 注册截止 2027-01-14，全文截止 2027-01-22。
  - 第三方 OpenCurious 页面写 abstract 截止 2027-01-16、地点在南美，并自称是按上一届推算。
  - 主办城市尚未确认。ICML 2026 在首尔举办。

  — [ICML 2027 Reciprocal Reviewing](https://icml.cc/Conferences/2027/ReciprocalReviewingAndSubmissionQuotas)；[OpenCurious](https://www.opencurious.com/ai-conference-deadlines/icml-2027)
- ACL 2027 于 2027-08-17 至 08-22 在日本举办（hybrid）。对应 ARR 周期的截止有两种说法：2027-01-04（较具体）和暂定的 2027-02-15，ARR 官方日期页只写"January 2027"。
  - EACL 2027 的 ARR 周期为 2026-08-03 投稿、2026-10-11 commit。
  - NAACL 和 COLING 2027 为 2026-10-12 投稿、2026-12-23 commit。
  - 自 2025-05 起 ARR 改为 10 周一个周期。

  — [ARR dates](https://aclrollingreview.org/dates)；[trybibby](https://trybibby.com/latex/template-comparison/acl-vs-emnlp-vs-naacl)
- EMNLP 2027：ARR 截止暂定 2027-05-28 AoE，会议暂定 2027 年 11 月初。 — [trybibby](https://trybibby.com/latex/template-comparison/acl-vs-emnlp-vs-naacl)（第三方，暂定）
- NeurIPS 2027：本次搜索没有找到任何官方日期。
- NeurIPS 2026 workshop 的投稿已截止，例如 AgenticOS 截止 2026-08-29，NEmo 截止 2026-09-10。 — [AgenticOS CFP](https://www.sigarch.org/call-contributions/agenticos-neurips-2026/)；[NEmo CFP](https://groups.google.com/g/rl-list/c/QqPMrCP4kcc)

### Inferences
- （推断）可行路径有三种：
  - 激进：赶 ICML 2027（1 月 22 日）。如果把现有 E3–E7 加上"等算力 / dose 混淆"分析整理成分析型论文，约 9 周可行，但最好补一个 horizon 扫描或第二个环境来回应 2608.19760 的竞争。
  - 稳妥：瞄准 NeurIPS 2027（推测 2027 年 5 月），主轨道或 Datasets & Benchmarks 轨道，用约 6 个月完成受控环境与 benchmark 工作。
  - 并行：ICLR 2027 workshop 通常在 2027 年 1–2 月征稿（推测，基于往年惯例），适合先发负结果或短文抢占时间点。
- （推断，依据往年惯例，非本次检索证据）各会偏好如下：
  - ICLR 与 NeurIPS 对"严谨的分析或负结果 + 新诊断工具"接受度较高，近两届的 RLVR 分析论文获奖或 oral 可作佐证。
  - ICML 偏好有理论或清晰机制解释的工作。
  - ACL 与 EMNLP 对"LLM agent 行为分析"类工作友好，但 RL 算法细节不是那里的核心卖点。
  - TMLR 适合扎实的负结果，没有截止日期。

### Gaps
- NeurIPS 2027、ICLR 2027 workshop、ICML 2027 workshop 的日期均未公布或未检索到。
- ICLR 2027 两个官方页面的截止日期互相矛盾（已截止，影响不大）。
- 未检索到各会议审稿指南中对分析型论文或负结果的明确表态。

## Q6. 工业界信号：1–3 年内 agentic RL 走向哪里

### Takeaway
本次检索到的一手工业信号很少，多为二手博客或 gist。可以确认的方向是：(a) 2026 年各家旗舰都把"agent"作为卖点，例如 Qwen3.7 "The Agent Frontier"、Kimi K3、DeepSeek-V4 的百万 token 上下文；(b) 长上下文与 context 管理是共同主题；(c) 开源 agent 训练框架与 deep-research 类 agent 持续推进，例如 Tongyi DeepResearch。没有找到 OpenAI、Anthropic、Google 关于 agentic RL 路线的一手声明。

### Cited Findings
- Kimi K3 据称于 2026 年 7 月发布权重与技术报告（arXiv 2607.24653），总参数 2.78T、激活参数 104B。两个二手来源给出的发布日期不同（7-16 与 7-27）。 — [Medium（二手）](https://buzzgrewal.medium.com/why-2026s-best-ai-models-keep-97-of-their-brains-asleep-the-trillion-parameter-recipe-behind-e1a867abb8a5)；[GitHub gist（二手）](https://gist.github.com/sing1ee/f824d5b191136d5d27a38f584dad98f2)
- DeepSeek-V4 的论文标题为 "Towards Highly Efficient Million-Token Context Intelligence"（arXiv 2606.19348，经二手来源引用），表明长上下文是旗舰路线的重点。 — [Medium（二手）](https://buzzgrewal.medium.com/why-2026s-best-ai-models-keep-97-of-their-brains-asleep-the-trillion-parameter-recipe-behind-e1a867abb8a5)
- Qwen 有题为 "Qwen3.7: The Agent Frontier" 的发布文（仅见于参考文献列表，未读原文）。 — [gist（二手）](https://gist.github.com/sing1ee/f824d5b191136d5d27a38f584dad98f2)
- 阿里 Tongyi DeepResearch 发布了技术报告（arXiv 2510.24701，2025-10），属于工业界 deep-research agent 的 RL 训练路线。 — [arXiv 2510.24701](https://arxiv.org/pdf/2510.24701)
- 2026 年的论文参考文献中频繁出现以下主题：context compaction for long-horizon agents、"OpenClaw-RL: Train any agent simply by talking"，以及 SkillRL 等技能增强的 RL。 — [Continual Learning in Transition, arXiv 2608.06216](https://arxiv.org/pdf/2608.06216)
- 学术界判断（agent memory 综述）：长视野 agent 面临超出上下文窗口的 context explosion，记忆管理会成为可训练能力。 — [arXiv 2602.06052](https://arxiv.org/abs/2602.06052)
- ICML 2026 的 Test of Time 奖颁给 A3C 时，委员会明确提及 asynchronous RL 对 LLM 后训练的作用，可视为 async/off-policy RL 受重视的旁证。 — [ICML 2026 Awards](https://blog.icml.cc/?p=1347)

### Inferences
- （推测）1 年内仍然可行的方向：credit assignment 的严谨评测与诊断、长视野工具使用中的行为病理（over-calling、过早停止），以及 context/memory 动作的 credit（"哪次压缩或记忆写入导致了成功"本质上就是 credit 问题）。
- （推测）3 年视角：工业界正推向百万 token、数百轮的自主 agent 和持续学习。届时短视野（≤4 轮）数学工具任务本身的价值会下降，但"credit 信号在 horizon 增长时如何退化"的 scaling 结论和受控评测工具仍有长期价值。对单人研究者来说，把现有管线扩展到可调长视野，并加入 memory 或 context 动作，是兼顾 1 年与 3 年的折中方案。
- （推测）研究者观察到的"turn-level credit 导致工具过度调用"与工业界关心的 agent 效率、成本（如 "Resource-Aware Agentic AI" workshop）直接相关，可以包装成"credit assignment 与工具使用效率"的问题。

### Gaps
- 未找到 OpenAI、Anthropic、Google DeepMind、ByteDance Seed 关于 agentic RL 方向的一手文本（技术报告、博客或演讲）。Kimi K3、DeepSeek-V4、Qwen3.7 的说法全部来自二手来源，没有读到原始技术报告中关于 RL 训练（如 credit、长视野 RL）的具体内容。
- 由于 arXiv 和 HF 不可访问，无法核实上述技术报告的编号与内容。若需要可靠的工业信号，需在网络放行 arxiv.org 后重新抓取原文。
