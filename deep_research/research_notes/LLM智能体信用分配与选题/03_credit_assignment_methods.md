# Credit-Assignment Methods for LLM RL (Reasoning and Agents), 2024–2026: Taxonomy, Mechanisms, Costs, Results

> Scope: LLM-specific credit-assignment (CA) methods, 2024–2026, with 2025–2026 prioritized. Classic RL CA is covered elsewhere. Memory-R2 is only placed in the taxonomy.
> Research date: 2026-11-17.
>
> **How things were checked (read this first).** In this session, `arxiv.org` and `export.arxiv.org` were blocked by the network proxy, so I could not open arXiv abstract pages directly. I checked IDs in two ways:
> (a) **[V-search]**: a web-search result returned the arXiv URL with the matching title, and for some, dates and authors too.
> (b) **[V-list]**: the ID appears with a matching title in one of two curated GitHub lists that I downloaded in full: the companion repo of the 2604.09459 survey ([xxzcc/Awesome-Credit-Assignment-in-LLM-RL](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)) and [Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL).
> (c) **[UNVERIFIED]**: the ID or number comes from model memory and was not confirmed this session. Recheck these before citing.
> Numbers marked "secondary" come from third-party summaries (emergentmind, themoonlight, alphaxiv and similar), not from the paper's own text.

---

## Q1. Process reward models (PRMs) and implicit/dense rewards: status, critiques, use in agent RL

### Takeaway
PRMs started as MC-labelled or human-labelled step verifiers (Math-Shepherd, PRM800K). They then moved to implicit or "free" PRMs learned from outcome labels (PRIME line) and to generative "thinking" PRMs. Used as **RL training rewards**, they are fragile. Sum-form PRM returns cause fast reward hacking (PURE: collapse around step 25), and later work reports that even min-form only delays the hacking. In 2025–2026 agent RL, PRMs do appear, but mostly in three forms: (i) implicit/online PRMs learned from outcomes (iStar/OPRL), (ii) **verifiable** turn-level process rewards from oracles (VPR), and (iii) TD/GAE-trained progress models (AgentPRM). Each is mixed with an outcome reward that must stay dominant.

### Cited Findings
**Foundations (2023–2024)**
- Math-Shepherd (arXiv 2312.08935, ACL 2024, Wang, Li, Shao et al.) pioneered automatic step-level labelling by MC sampling of continuations from each step [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- OmegaPRM (arXiv 2406.06592, Luo et al. 2024) scaled automated process supervision via divide-and-conquer (binary-search) MC exploration [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- "From r to Q*: Your Language Model is Secretly a Q-Function" (arXiv 2404.12358, Rafailov et al. 2024) shows that DPO implicitly learns token-level Q-values. It is the theoretical basis for implicit token/step credit [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

**PRM-as-RL-reward critiques and fixes**
- **PURE / "Stop Summation: Min-Form Credit Assignment Is All Process Reward Model Needs for Reasoning"** (NeurIPS 2025, Cheng et al.). It attributes PRM-induced reward hacking mainly to the standard summed / γ-discounted return, which pushes the policy toward steps that already score high. Its fix is a min-form value: a response is worth its weakest step, implemented as a temperature-controlled softmax over process rewards. In its experiments, sum-form variants **collapsed around training step 25** while min-form stayed stable. Adding a small amount of ground-truth verifiable reward reduced hacking further. — [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html); [arXiv PDF 2504.15275](https://arxiv.org/pdf/2504.15275)
  - **ID conflict:** a search result gives PURE as arXiv **2504.15275** [V-search], but the survey's companion list links **2504.08662**. Treat 2504.15275 as the likely correct ID. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- A 2026 follow-up ("Unsupervised Process Reward Models", arXiv 2605.10158) reports that PURE **delays but does not fully prevent** hacking. It also reports training Qwen2.5-Math models with an unsupervised PRM with no observed hacking. This is a single-paper claim (secondary summary). — [arXiv 2605.10158](https://arxiv.org/pdf/2605.10158)
- A broad PRM survey exists: "A Survey of Process Reward Models: From Outcome Signals to Process Supervisions for LLMs" (arXiv 2510.08049) [V-search]. — [arXiv 2510.08049](https://arxiv.org/html/2510.08049v3)

**Reasoning-side step-level methods that avoid an external PRM (survey companion list)**
- SPRO (arXiv 2507.01551): self-guided process reward with a leave-one-out masked step advantage. The list claims "3.4x efficiency over GRPO" [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- CAPO (arXiv 2508.02298): an LLM acts as a generative PRM that self-critiques each step [V-list]. — same
- PRL "Process Reward Learning" (arXiv 2601.10201, Yao, Wang, Zhang 2026): derives optimal step-level process rewards from entropy-regularized RL [V-list]. — same
- FinePO (arXiv 2601.05688): sub-step PRM for visual reasoning [V-list]. — same

**PRMs in agent RL (2025–2026)**
- "Process Reward Models for LLM Agents: Practical Framework and Directions" (arXiv 2502.10325) [V-search]. — [arXiv 2502.10325](https://arxiv.org/html/2502.10325v1)
- **AgentPRM** "Process Reward Models for LLM Agents via Step-Wise Promise and Progress" (arXiv 2511.08325, Xi et al.) uses TD+GAE for turn-level value estimation. The list claims "8x sample efficiency vs MC-based PRM" [V-list, V-search]. The paper notes that agent steps lack clear correctness labels, so step judgments must measure progress toward a distant goal. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [arXiv 2511.08325](https://arxiv.org/html/2511.08325v1)
- **OPRL / iStar** (arXiv 2509.19199, listed as ICLR 2026): trains an implicit PRM online from outcome-labelled rollouts (DPO-style) and fuses implicit step rewards with the outcome reward. It argues that token-level PRMs from outcome labels "tend to be overly fine-grained for agentic learning, amplifying variance and destabilizing training as trajectories grow." In its ablation, raw step penalties in VisualSokoban gave only limited gains over vanilla RL. — [arXiv 2509.19199 v1 (titled "Online Process Reward Learning for Agentic RL")](https://arxiv.org/html/2509.19199v1); [OpenReview iStar](https://openreview.net/forum?id=ooROvpmxMV)
  - The same ID appears under two titles (v1 "OPRL" in the search result, "iStar: Agentic RL with Implicit Step Rewards" in both lists). This is most likely a retitle across versions.
- **VPR** "Verifiable Process Rewards for Agentic Reasoning" (arXiv 2605.10325, 2026) turns intermediate symbolic/algorithmic oracles into dense turn-level supervision. It reports beating outcome-level RL and rollout-based process-reward baselines in controlled environments, with transfer to ALFWorld and WebShop (secondary summary of abstract). — [arXiv 2605.10325](https://arxiv.org/html/2605.10325v1)
- "Process vs. Outcome Reward: Which is Better for [agentic RAG]…" (arXiv 2505.14069, ReasonRAG) argues that outcome-supervised agentic RAG suffers from low exploration efficiency, gradient conflict and sparse reward, and that process-supervised RL helps. — [arXiv 2505.14069](https://arxiv.org/pdf/2505.14069)
- Turn-level PRMs/judges in tool agents: TARL (arXiv 2509.14480) uses an LLM-as-judge for turn-level evaluation of multimodal tool-use agents [V-list]. MAPPA (arXiv 2601.23228) uses per-action process rewards from AI feedback in multi-agent systems and claims +5.0–17.5pp on AIME [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- 2026 dense-signal evaluation: **QVal** (arXiv 2606.32034) is a training-free benchmark that tests whether dense supervision scores agree with reference-policy Q-values in long-horizon agent tasks [V-list; the 5-digit suffix "32034" is unusually high, so recheck]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

### Inferences
- The field is converging on "PRM signal = shaping on top of a dominant verifiable outcome reward." Pure PRM-as-reward is considered unsafe for RL. The evidence is PURE's collapse result, the 2605.10158 claim that min-form only delays hacking, OPRL's variance argument, and practitioner guidance (see Q8).
- For agents, the "verifier per step" assumption usually fails. Successful agent PRMs either (a) learn from outcomes (implicit, TD), or (b) exploit environment-specific oracles (VPR). For math + Python tool use (this project), (b) is partly available: execution success or error is a free, verifiable per-turn signal, but it measures validity, not progress.

### Gaps
- **[UNVERIFIED IDs and claims; do not cite without checking]:**
  - PRM800K / "Let's Verify Step by Step" (OpenAI, arXiv 2305.20050).
  - **PRIME** "Process Reinforcement through Implicit Rewards" (arXiv 2502.01456; Eurus-2-7B-PRIME). The search gave only a secondary statement that PRIME derives process rewards online from outcome labels; I could not retrieve its numbers.
  - "Free Process Rewards without Process Labels" (implicit PRM, arXiv 2412.01981).
  - **ThinkPRM** "Process Reward Models That Think" (arXiv 2504.16828).
  - GenPRM (arXiv 2504.00891).
  - Qwen "The Lessons of Developing Process Reward Models in Mathematical Reasoning" (arXiv 2501.07301). From memory: MC-estimated labels generalize worse than LLM-as-judge or human labels, and BoN-only evaluation biases PRM assessment. **Not confirmed this session.**
  - The DeepSeek-R1 report's statement that PRMs were not used at scale because of reward hacking and annotation cost (arXiv 2501.12948, [V-list for the ID]). The exact wording was not retrieved.
- I found no 2026 head-to-head study that isolates "learned PRM vs. outcome-only" in **multi-turn tool-use agent RL** under equal compute.

---

## Q2. Monte-Carlo / rollout-based value estimation and tree/branching methods

### Takeaway
This family estimates V(s_t) or step advantages by re-sampling continuations from intermediate states ("vines", segments, or trees). It is unbiased (MC) and critic-free, and it works well for single-turn reasoning, where the "state" is just a text prefix and is trivially resettable. For agents, tree/branching variants (Tree-GRPO, ARPO/AEPO, AT²PO, SIPO) present prefix-sharing as a **rollout-budget saving** rather than an added cost. Tree-GRPO claims parity with chain-based methods at **¼ of the rollout budget**. The cost is the need to re-enter a mid-trajectory state, which is easy for text and Python sandboxes but requires snapshotting the environment for stateful tools.

### Cited Findings
- **VinePPO** (arXiv 2410.01679, ICML 2025, Kazemnejad … Courville, Le Roux) replaces the learned value network with unbiased MC value estimates at intermediate positions via "vine" rollouts [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [PMLR v267](https://proceedings.mlr.press/v267/kazemnejad25a.html)
- **SPO: Segment Policy Optimization** (arXiv 2505.23564, NeurIPS 2025, Guo et al.) splits reasoning chains at cutpoints and computes MC advantages per segment. It has a chain variant and an "SPO-tree" variant [V-list both lists]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- **TEMPO** "Exploiting Tree Structure for Credit Assignment in RL Training of LLMs" (arXiv 2509.18314): branch-gated TD corrections on a tree of reasoning paths, critic-free [V-list]. — same lists
- **TreeRPO** (arXiv 2506.05183) [V-list]; **TreeAdv** "Tree-Structured Advantage Redistribution for Group-Based RL" (arXiv 2601.03703) [V-list]; **RTMC** "Step-Level Credit Assignment via Rollout Trees" (arXiv 2604.11037) [V-list]. — [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- **Tree-GRPO** "Tree Search for LLM Agent Reinforcement Learning" (arXiv 2509.21240, ICLR 2026, AMAP-ML). Each tree node is a full agent step (turn), and shared prefixes reduce token and tool-call cost. On Qwen2.5-3B it reports beating chain-based methods "while using only a **quarter of the rollout budget**." A secondary summary reports a **112% relative improvement** over GRPO in multi-hop QA when the budget is about 2 full trajectories per prompt. — [GitHub AMAP-ML/Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO); [arXiv 2509.21240 HTML v3](https://arxiv.org/html/2509.21240v3)
- **ARPO** "Agentic Reinforced Policy Optimization" (arXiv 2507.19849) and **AEPO** "Agentic Entropy-Balanced Policy Optimization" (arXiv 2510.14545), both RUC-NLPIR: entropy-guided adaptive branching at uncertain tool-use turns [V-list]. — [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL); described as "entropy-guided rollout to selectively branch at uncertain tool-use turns" in [arXiv 2607.06223](https://arxiv.org/html/2607.06223v1)
- **AT²PO** "Agentic Turn-based Policy Optimization via Tree Search" (arXiv 2601.04767, ACL 2026 Main per list): turn-level tree, entropy-guided expansion, turn-wise credit [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **SEEA-R1** (NeurIPS 2025): tree-structured RFT for embodied agents [V-list]. **ATPO** "Adaptive tree policy optimization for multi-turn medical dialogue" (arXiv 2603.02216, ICLR 2026 per list) [V-list]. **SIPO** "Selective-Inference Policy Optimization for Tree-Structured Agentic RL" (arXiv 2609.34805, by the list's maintainers) [V-list; unusually high suffix, recheck]. — [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- 2026 tree-rollout follow-ups: **IGRPO** "Information Gain-based Rollout Policy Optimization: An Adaptive Tree-Structured Rollout Approach for Multi-Turn LLM Agents" (arXiv 2607.06223) sets per-stage budget B_i=5 to match chain-based search budgets [V-search]. "Process Reward Informed Tree Rollout for Effective Multi-Turn RL" (arXiv 2607.15610) [V-search, title only]. — [arXiv 2607.06223](https://arxiv.org/html/2607.06223v1); [arXiv 2607.15610](https://arxiv.org/pdf/2607.15610)
- **Fork/retry from an intermediate state:** PivoARL "Agent RL via Pivotal-Aware Self-Feedback Retry" (arXiv 2607.03702) finds the pivotal erroneous turn via self-reflection and **retries locally from that state**, rewarding correct prefixes while isolating erroneous suffixes [V-list]. InT "Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning" (arXiv 2601.14209, Kumar group) has the model propose counterfactual interventions at steps [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

### Inferences
- Cost model: vine/segment MC costs about K extra partial rollouts per estimated point (O(K·#cutpoints)). Tree methods instead **reallocate** a fixed budget, reusing prefixes and branching on high-entropy points, so they can be cheaper than i.i.d. GRPO at equal sample count. The honest comparison is at equal tokens **and** equal tool calls, which Tree-GRPO reports. Many others report only equal number of trajectories.
- Bias/variance: MC estimates are unbiased but high-variance with small K. Tree sharing correlates sibling estimates. Entropy-triggered branching biases where credit is resolved, toward uncertain turns.
- Assumption: a **resettable/forkable state**. That is trivial for pure-text reasoning and stateless Python cells (replay the prefix), but nontrivial for stateful sandboxes (kernel variables, file system, web pages). This matters directly for multi-turn Python-interpreter RL.

### Gaps
- **[UNVERIFIED]** TreeRL (arXiv 2506.11902, "TreeRL: LLM RL with On-Policy Tree Search", from memory: EPTree, entropy-guided forking) and TreePO (arXiv 2508.17445). Neither ID was confirmed this session.
- I could not retrieve VinePPO's or SPO's quantitative results (accuracy gains, wall-clock overhead) from primary text.
- ARPO's claim of fewer tool calls than trajectory-level RL (from memory: about half the tool-call budget) is **unverified**.

---

## Q3. Anchor-state / grouping methods (GiGPO and successors)

### Takeaway
GiGPO (NeurIPS 2025) adds a second, step-level grouping on top of GRPO's trajectory groups. It finds **identical environment states that recur across the group's trajectories** ("anchor states") and compares the actions taken there, using discounted returns from each occurrence. It costs no extra rollouts and no critic, and it reported +12% (ALFWorld) and +9% (WebShop) over GRPO. Its key assumption, exact state matching, is also its stated limitation. Successors relax "identical state" to similar or semantic states, or build state graphs.

### Cited Findings
- **GiGPO** "Group-in-Group Policy Optimization for LLM Agent Training" (arXiv 2505.10978, NeurIPS 2025, Feng, Xue, Liu, An) [V-search]. It uses a two-level advantage: an outer trajectory-group advantage plus an inner anchor-state group advantage, critic-free. — [arXiv 2505.10978](https://arxiv.org/abs/2505.10978); [NeurIPS PDF](https://personal.ntu.edu.sg/boan/papers/NeurIPS25_GiGPO.pdf)
  - Abstract: performance gains of **>12% on ALFWorld and >9% on WebShop** over GRPO. Per a secondary reading of the paper's table, GiGPO (w/o std) beats GRPO by 13.3% (ALFWorld) and 10.6% (WebShop) at 1.5B, and by 12.6% and 9.1% at 7B. — [arXiv 2505.10978](https://arxiv.org/abs/2505.10978)
  - Authors' stated limitation: "reliance on state matching for anchor group construction." — [NeurIPS PDF](https://personal.ntu.edu.sg/boan/papers/NeurIPS25_GiGPO.pdf)
  - Secondary sources disagree on absolute WebShop numbers (75.2 vs 66.1 success in one; 86.2 vs 79.3 in another). These likely mix "score" and "success". Check Table 1 directly. — [emergentmind topic](https://www.emergentmind.com/topics/group-in-group-policy-optimization-gigpo)
  - Code: verl-agent (langfengQ/verl-agent). — [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- Same group, multi-agent extension: **Dr. MAS** (arXiv 2602.08847, Feng, Zheng, He, Zhang, An 2026) uses agent-wise advantage normalization to stabilize multi-agent GRPO, with +5.6% avg@16 [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- Successors and relatives that relax the exact-match requirement (2026; titles from search/lists, mechanisms only partly confirmed):
  - **SALT** "Step-level advantage assignment for long-horizon agents via trajectory graph" (Findings of EACL 2026) [V-list]. — [ACL Anthology](https://aclanthology.org/2026.findings-eacl.247.pdf)
  - **RewardFlow** "Topology-Aware Reward Propagation on State Graphs for Agentic RL" (arXiv 2603.18859) [V-search, title only]. — [arXiv 2603.18859](https://arxiv.org/pdf/2603.18859)
  - **Semantic Consistency Policy Optimization for RL of LLM Agents** (arXiv 2606.25852) [V-search, title only; cites HCAPO and GiGPO]. — [arXiv 2606.25852](https://arxiv.org/pdf/2606.25852)
  - **StepPO** "Step-Aligned Policy Optimization for Agentic RL" (arXiv 2604.18401) [V-list]. — [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)

### Inferences
- GiGPO's free lunch depends on **state recurrence**. In grid or room environments (ALFWorld, Sokoban, WebShop page states), identical observations recur often across the G rollouts of one prompt. In math + Python tool use, the "state" is the full text history plus interpreter outputs, and exact recurrence beyond the initial prompt is rare. Anchor groups then collapse to singletons and GiGPO reduces to GRPO. Normalizing tool outputs (for example, matching on interpreter-state hashes or the last tool result) would be a design choice, not an established method. Treat this as a hypothesis.
- Bias/variance: exact-match anchors give an unbiased within-state comparison, so variance falls when groups are large. Similarity-based anchors add bias in exchange for coverage.

### Gaps
- I found no paper that reports GiGPO on math + code-interpreter tool use. Its applicability there is untested in the sources I found.
- The exact HCAPO-vs-GiGPO numbers were not retrieved (see Q5).

---

## Q4. Turn-level / step-level reward and advantage for multi-turn tool use

### Takeaway
This family reformulates the agent episode as a **turn-level MDP**: one action is one turn or tool call. It assigns either (a) turn-level rewards (verifiable checks, LLM judges, information gain, reference-model likelihood of the gold answer) or (b) turn-level critics or importance ratios. The 2026 trend is critic-free dense signals computed from a frozen or reference model's belief about the answer (TRACE, ΔBelief-RL, IGPO, ITPO). Privileged-information critics (SWEET-RL, CriticSearch) are the main learned-critic alternative.

### Cited Findings
- **ArCHer** (arXiv 2402.19446, ICML 2024, Zhou, Zanette, Pan, Levine, Kumar): hierarchical CA with an off-policy turn-level critic plus an on-policy token-level actor [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **POAD** (arXiv 2405.15821, 2024): intra-action (token) plus inter-action (turn) credit decomposition [V-list]. — same
- **MT-GRPO / "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Reward Design"** (arXiv 2505.11821, NeurIPS 2025 per list; Wei, Zeng, Li, Brown et al.): hybrid turn rewards, with automated verification for verifiable turns (for example, correct tool-call format or execution) plus LLM-judge for subjective turns, and turn-level advantages [V-list]. Note that the earlier title of the same ID was "…via Turn-Level Credit Assignment". — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **SWEET-RL** (arXiv 2503.15478, Meta FAIR; Zhou, Jiang, Tian, Weston, Levine, Sukhbaatar, Li): an asymmetric step-wise critic that sees **privileged training-time information** (for example, the reference solution) and trains a turn-level advantage function. It introduces the ColBench benchmark [V-list both]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [code](https://github.com/facebookresearch/sweet_rl)
- **Turn-PPO** (arXiv 2512.17008, Findings of EACL 2026): reformulates multi-turn RL as a turn-level MDP with turn-level importance ratios and turn-level advantage estimation with PPO [V-list]. — [ACL Anthology](https://aclanthology.org/2026.findings-eacl.328.pdf)
- **SORL** (arXiv 2511.20718): turn-level importance sampling plus clipping-triggered normalization for off-policy long-horizon agents (SO-PPO/SO-GRPO) [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **IGPO** "Information Gain-based Policy Optimization" (arXiv 2510.14967, ICLR 2026): the turn-level reward is the increase in the policy's probability of the ground-truth answer after each turn (information gain), applied to multi-turn search agents [V-list]. Follow-up **A²TGPO** (arXiv 2605.06200) adds turn-group normalization, variance-rescaled accumulation and adaptive turn-level clipping [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [OpenReview IGPO](https://openreview.net/forum?id=qkWP6phrvZ)
- **TRACE** "Turn-level Reward Assignment via Credit Estimation for Long-Horizon Agents" (arXiv **2607.13988**, July 2026; UW–Madison + Microsoft Research) [V-search]. It is critic-free. States sit at tool-call boundaries, a **frozen reference model** scores the log-probability of the gold answer at each state, and each tool call receives the **TD change** in that value. The outcome reward stays in the objective to enforce final correctness. It trains search agents with pure RL (no cold-start SFT, no live-web training), using text tools browser.search/open/find over an offline FAISS corpus or the Serper API. A third-party summary reports **Qwen3-4B going from 7.2 to 35.6 on BrowseComp-Plus**; this number is not checked against the paper. — [arXiv 2607.13988](https://arxiv.org/abs/2607.13988); [emergentmind](https://www.emergentmind.com/papers/2607.13988); [Awesome-CUA scope issue](https://github.com/reacher-z/Awesome-CUA/issues/2606)
- **ΔBelief-RL** "Intrinsic Credit Assignment for Long Horizon Interaction" (arXiv 2602.12342): uses changes in the policy's log-probability of the known target as dense per-turn rewards, then computes turn-wise group-relative advantages without a critic or PRM [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **ITPO** (arXiv 2603.23550): implicit turn-wise rewards from the model's own log-probability changes, for proactive user-LLM interaction [V-list]. **InfoPO** (arXiv 2603.00656): information-gain rewards for user-centric agents [V-list]. — same lists
- "Self-Induced Outcome Potential: Turn-Level Credit Assignment for Agents without Verifiers" (arXiv 2605.04984) [V-search, title only]. **TCPO** "Turn-Level Credit Policy Optimization" (arXiv 2608.01667) [V-search, title only]. — [arXiv 2605.04984](https://arxiv.org/pdf/2605.04984); [arXiv 2608.01667](https://arxiv.org/pdf/2608.01667)
- Dynamic-rubric step credit: **DRACO** (arXiv 2609.04094, IBM). An LLM judge scores per-trajectory rubrics generated during training, and the group-relative advantage is redistributed in closed form over the steps the judge attributes to each criterion. It uses no ground-truth task-success signal [V-list]. **TRCA** "Transition-wise Rubric Credit Assignment for Long-horizon LLM Agents" (arXiv 2608.16156) [V-search, title only]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [arXiv 2608.16156](https://arxiv.org/pdf/2608.16156)
- Tool/CLI-specific action credit: **A³** "Learning CLI Agents with Structured Action Credit under Selective Observation" (arXiv 2605.08013) [V-list]. **GVPO** (OpenReview RY47Tq0VsV) uses outcome-verifiable plus process-verifiable execution feedback for interactive coding agents [V-list]. **SCRIBE** (arXiv 2601.03555) provides skill-prototype mid-level supervision for tool-using LMs [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- Stepwise progress estimators: **SPA-RL** (arXiv 2505.20732) trains a lightweight MLP progress estimator and decomposes the final reward into per-step progress contributions [V-list]. "Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents" (arXiv 2606.26080) derives a progress signal from the log-ratio between an RL-trained policy and its reference [V-list]. — same
- Entropy/uncertainty-modulated turn advantages (CA-adjacent): **AEM** (arXiv 2605.00425), **T²PO** (arXiv 2605.02178, ICML 2026 spotlight per list), **CARL** "Criticality-Aware Agentic RL" (arXiv 2512.04949), which focuses updates on high-entropy critical actions and claims 72% fewer gradient updates [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

### Inferences
- There are three sub-families with different assumptions:
  - (i) **Gold-answer-likelihood signals** (IGPO, TRACE, ΔBelief, ITPO) need a known reference answer at training time, which holds for math and QA. They cost one extra forward pass per turn boundary on a frozen or reference model. They can be **hacked** by turns that leak or restate the answer, or that make the answer more likely without making progress.
  - (ii) **Privileged critics** (SWEET-RL, CriticSearch) need a training-only information channel and a trained critic.
  - (iii) **Judge/rubric signals** (MT-GRPO's judge part, TARL, DRACO) cost LLM-judge calls and inherit judge bias.
- For this project (math + Python, gold answer known), TRACE/IGPO-style signals are directly applicable and cheap. MT-GRPO's verifiable turn checks (code executes, output parses) are free but mostly measure format and validity.

### Gaps
- I could not get primary-source numbers for MT-GRPO, SWEET-RL (from memory, ColBench gains over multi-turn DPO; **unverified**), IGPO, or ArCHer.
- **[UNVERIFIED]** StepTool (arXiv 2410.07745), RLVMR (arXiv 2507.22844), EMPG (arXiv 2509.09265), GTPO (arXiv 2511.14846). These are plausible related turn/step methods, but their IDs were not confirmed this session.

---

## Q5. Hindsight / counterfactual / Shapley / attribution / key-token methods

### Takeaway
This family attributes a sparse outcome to steps after the fact. It uses (a) an **LLM-as-hindsight-critic** (HCAPO, RICL/RICOL, CriticSearch, LaRe), (b) **counterfactual / leave-one-out** perturbations (C3, CCPO, CRAFT, OAR), (c) **Shapley** decompositions (SCAR, SSVPO, SHARP; and the general-RL Counterfactual Shapley Value), (d) **gradient/attention attribution** (ACPO, GRAIL, OAR's gradient proxy, RED), or (e) **key-token selection** (high-entropy "forking tokens", planning tokens in HICRA, discriminative tokens in DelTA). HCAPO reports large ALFWorld/WebShop gains over GRPO. The forking-token result shows that **restricting gradients to the top 20% highest-entropy tokens** can match or beat full-token RLVR.

### Cited Findings
**Hindsight / LLM-critic**
- **HCAPO** "Hindsight Credit Assignment for Long-Horizon LLM Agents" (arXiv **2603.08754**, 2026; Tan, Yang, Chen, … Guo, Y.-F. Li) [V-search + V-list both]. It is value-free and uses the LLM itself as a post-hoc critic to refine step-level Q-values through hindsight reasoning and counterfactual continuation analysis. Qwen2.5-7B-Instruct results: **ALFWorld 91.4% vs GRPO 77.6% (+13.8pp)**; **WebShop success 66.1% → 73.8%, score 79.3 → 85.1**. It also compares against GiGPO, but those numbers were not retrieved. One third-party table lists 90.8 for ALFWorld, a conflict. — [arXiv PDF 2603.08754](https://arxiv.org/pdf/2603.08754); [themoonlight review](https://www.themoonlight.io/en/review/hindsight-credit-assignment-for-long-horizon-llm-agents)
- **RICL / RICOL** "Retrospective In-Context Learning for Temporal Credit Assignment with Large Language Models" (arXiv **2602.17497**, v1 Feb 2026; Wentse Chen, Jiayu Chen, Fahim Tajwar, Hao Zhu, Xintong Duan, Ruslan Salakhutdinov, Jeff Schneider) [V-search]. A reflector LLM gives verbal feedback for individual states after an attempt. The shift in action probabilities with and without that feedback approximates the advantage. RICOL iterates this online. On **BabyAI** it significantly improves sample efficiency over online RL and is comparable to imitation from expert demos. **Venue conflict:** mlanthology lists it as NeurIPS 2025 although the arXiv ID is from 2026. A "~100x efficiency vs MC" claim is from a secondary aggregator and unverified. Note it is evaluated on a classic gridworld (BabyAI), not LLM-reasoning RL. — [arXiv PDF](https://arxiv.org/pdf/2602.17497); [mlanthology NeurIPS 2025](https://mlanthology.org/neurips/2025/chen2025neurips-retrospective/); [Lacuna (secondary)](https://lacuna.tiptreesystems.com/work/retrospective-in-context-learning-for-temporal-credit-assignment-with-large/wrk_3d99955f45d437e6b58b1b4b3fde0f15)
- **CriticSearch** (arXiv 2511.12159): a retrospective asymmetric critic for search-agent turn-level credit [V-list]. **LaRe** (arXiv 2412.16989, AAAI 2025): LLM-generated "latent reward" credit, turned into scalars [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **RELACE** "Retrospective likelihood-based action credit estimation for long-horizon language agents" (arXiv 2610.07349) [V-search, title only]. — [arXiv 2610.07349](https://arxiv.org/html/2610.07349)

**Counterfactual / leave-one-out**
- **C3** (arXiv 2603.06859): contextual counterfactual leave-one-out credit for multi-agent LLM collaboration [V-list]. **CCPO** (arXiv 2603.21563): an SCM view of trajectories with average treatment effect as credit [V-list]. **CRAFT** (arXiv 2606.29476): signed counterfactual token credit from "free" sibling rollouts for self-distilled agentic RL [V-list]. **OAR** (arXiv 2601.07408): redistributes GRPO advantages via outcome-sensitivity from counterfactual perturbations or gradient proxies (math) [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

**Shapley**
- **Counterfactual Shapley Credit Assignment** (arXiv **2607.16999**; Mingxuan Li, Kaizhan Lee, Elias Bareinboim; Reinforcement Learning Journal 2026 / RLC 2026) [V-search]. It defines a causal "Counterfactual Shapley Value" (φ-value) on an SCM. φ-values redistribute environment reward to improve temporal CA under sparse causality, high stochasticity and delayed rewards, **while preserving the optimal policy**. **Important:** it is a general-RL paper. The search did not confirm any LLM experiments, so classify it as a theoretical foundation, not an LLM method. — [arXiv 2607.16999](https://arxiv.org/abs/2607.16999); [emergentmind](https://www.emergentmind.com/papers/2607.16999)
- **SCAR** (arXiv 2505.20417; Cao, Zhang, Chang, Precup): Shapley decomposition of a sequence reward across segments [V-list]. **SSVPO** "Effective Step-Level Credit Assignment for RL Training of Language Models" (ICLR 2026, sequential Shapley value) [V-list, OpenReview g33DGvnHYd]. **SHARP** (arXiv 2602.08335): three-component Shapley decomposition for multi-agent systems, +23.7% over single-agent baselines [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL); [OpenReview SSVPO](https://openreview.net/forum?id=g33DGvnHYd)

**Gradient / attention attribution**
- **ACPO** (arXiv 2510.08899): gradient-based attribution to pinpoint crucial steps, plus a difficulty-aware curriculum [V-list]. **GRAIL** (arXiv 2606.04889): gradient-activation saliency reweights sequence advantages into token advantages [V-list]. **RED** (arXiv 2411.08302): token-level credit by linear-probing reward-model internals [V-list]. **T-REG** (arXiv 2412.02685): token-level reward regularization via contrastive self-prompting [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

**Key-token methods**
- **"Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RL for LLM Reasoning"** (arXiv 2506.01939, NeurIPS 2025, Shenzhi Wang et al.) [V-search]. It applies DAPO policy gradients only to the **top 20% highest-entropy "forking" tokens**. On Qwen3-32B: **+11.04 AIME'25 and +7.71 AIME'24** over full-token DAPO (secondary: 63.5 AIME'24 and 56.7 AIME'25). On Qwen3-8B it is comparable to full-gradient updates. Extending max length from 20k to 29k raises AIME'24 from 63.54 to 68.12. It also generalizes better to LiveCodeBench when trained on math. — [arXiv 2506.01939](https://arxiv.org/abs/2506.01939); [NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/a797c2d2e0c1fdabf4d1ab8cd0b465c6-Abstract-Conference.html)
- **HICRA** (arXiv 2509.03646): focuses credit on planning tokens over procedural tokens [V-list]. **DelTA** (arXiv 2605.21467): discriminative token coefficients [V-list]. **SC-GRPO** (arXiv 2606.18810): per-token KL credit from verified self-conditioned trajectories [V-list]. **GRPO-λ** (arXiv 2510.00194): critic-free λ-return / eligibility-trace weighting [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)

### Inferences
- Costs differ by sub-family:
  - LLM hindsight critics: one extra LLM pass over each trajectory, which can be the policy itself.
  - Counterfactual/LOO: extra rollouts per perturbed step, so the cost scales like MC.
  - Exact Shapley: exponential in the number of segments, so it needs approximation (sampling permutations).
  - Gradient attribution: about one extra backward pass.
  - Entropy key-token selection: essentially free.
- Key-token methods are mostly "which tokens get gradient" rather than "how much credit each step deserves." They are orthogonal to, and composable with, turn-level methods.
- LLM-as-critic hindsight methods inherit judge bias, and their critic is not grounded by a verifier. Their validity has to be audited (see the 2026 audit papers in Q8).

### Gaps
- **[UNVERIFIED]** "Critical Tokens Matter" / cDPO (arXiv 2411.19943) and KTAE "Key-token advantage estimation" (arXiv 2505.16826). IDs are from memory.
- No primary numbers were retrieved for SCAR, C3, CCPO or ACPO.

---

## Q6. Value-model revival: VAPO, VC-PPO, learned critics for long CoT; do critics work for agents?

### Takeaway
For **long-CoT math**, ByteDance Seed showed that PPO with a learned critic can beat critic-free GRPO/DAPO, provided the critic is fixed. VC-PPO's fixes are value pretraining against reward-model-init bias and decoupled GAE (λ=1 for the critic). VAPO adds length-adaptive GAE and other changes, and reaches **60.4 on AIME 2024** with Qwen2.5-32B. For **agents**, learned critics survive mainly as **turn-level** critics (ArCHer, Turn-PPO, AgentPRM's TD/GAE) or **privileged/asymmetric** critics (SWEET-RL, CriticSearch). Most 2025–2026 agent methods remain critic-free.

### Cited Findings
- **VC-PPO** "What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the Secret" (arXiv 2503.01491) [V-search]. It identifies value-initialization bias (critic initialized from the reward model) and reward-signal decay as the causes of PPO's collapse. Fixes are value pretraining and decoupled GAE. Its ablations show that removing Decoupled-GAE hurts. Its AIME numbers differ across tables (48.8 in the 16K-context comparison vs 41.9 in the ablation table), so do not compare them directly. — [arXiv 2503.01491](https://arxiv.org/html/2503.01491v1)
- **VAPO** (arXiv 2504.05118): "built on the Qwen 32B pre-trained model, attains a state-of-the-art score of **60.4**" on AIME 2024, beating DeepSeek-R1-Zero-Qwen-32B and DAPO-32B. It adopts VC-PPO's value pretraining and decoupled GAE and adds length-adaptive GAE, which adjusts λ by sequence length, among other changes. Built on veRL. — [arXiv 2504.05118](https://arxiv.org/abs/2504.05118); [H. Lin announcement](https://x.com/eric_haibin_lin/status/1909659617057288305); [alphaxiv overview](https://www.alphaxiv.org/overview/2504.05118v3)
- Agent-side learned critics: ArCHer (turn-level off-policy critic) [V-list]; Turn-PPO (turn-level advantage estimation with PPO) [V-list]; AgentPRM (TD+GAE turn values) [V-list]; SWEET-RL (asymmetric privileged critic) [V-list]. — [Awesome-CA list](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- The survey states that agentic CA specifically drives "privileged asymmetric critics" and "turn-level MDP reformulations," which have no direct precedent in reasoning RL. — [survey via search summary of arXiv 2604.09459](https://arxiv.org/html/2604.09459)

### Inferences
- Why critics are harder for agents: observations include tool outputs that the policy did not generate. A token-level critic must evaluate extremely long and heterogeneous contexts, and episodes are stochastic. A turn-level critic shrinks the decision count by about 100× and matches the decision granularity, which explains the shift.
- I found no 2025–2026 evidence of a token-level learned-critic PPO beating GRPO-family methods on multi-turn **tool-use** benchmarks under equal compute. This is open (see Gaps).

### Gaps
- I found no controlled study of VAPO/VC-PPO-style critics on multi-turn tool agents.
- The SORL paper (off-policy turn-level IS, arXiv 2511.20718) may contain PPO-vs-GRPO agent comparisons. Not retrieved.

---

## Q7. The survey arXiv 2604.09459 and the awesome-lists

### Takeaway
"From Reasoning to Agentic: Credit Assignment in Reinforcement Learning for Large Language Models" (arXiv 2604.09459, single author **Chenchen Zhang**, first posted **13 Apr 2026**, v2 exists) organizes LLM CA on two axes. **Granularity** is token / segment / step-turn / multi-agent. **Methodology** is MC / TD / LLM-as-critic or model-based / game-theoretic / information-theoretic, extended in the repo with uncertainty-control and verifiable-feedback shaping. The snapshot covers 47 methods (41 core + 6 adjacent enablers), and the living repo adds 29 more. Its main thesis: reasoning CA is maturing around PRMs and critic-free group comparison, while agentic CA is producing new mechanisms (hindsight counterfactuals, privileged critics, turn-level MDPs).

### Cited Findings
- Bibliographic: arXiv 2604.09459, author Chenchen Zhang. One blog calls it a "Stanford survey (Zhang et al.)", but the arXiv listing and alphaXiv give a single independent author, so prefer arXiv. First posted April 13, 2026. — [arXiv HTML](https://arxiv.org/html/2604.09459); [alphaXiv v2 overview](https://www.alphaxiv.org/overview/2604.09459v2); [HF papers](https://huggingface.co/papers/2604.09459)
- Counts differ by version: v1 HTML lists 23 reasoning + 40 agentic/mixed + 6 multi-agent = 69 papers. The later abstract reports 47 CA methods (41 core, 6 adjacent enablers). One summary states coverage Jan 2024 – Jul 31, 2026. — [arXiv HTML v1](https://arxiv.org/html/2604.09459v1); [companion repo](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- Regime framing: reasoning RL spreads credit over 500–30K+ tokens in one CoT. Agentic RL has stochastic transitions, partial observability and 100+ turn horizons (100K–1M tokens), which makes episode-level credit less informative. — [companion repo](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- Taxonomy buckets in the repo (with example members): reasoning token-level (VinePPO, RED, T-REG, r-to-Q*); segment-level (SPO, SCAR, TEMPO); step-level (PURE, SPRO, CAPO, ACPO, HICRA, PRL, InT, FinePO); agentic turn-level PRMs (AgentPRM, SWEET-RL, MT-GRPO, Turn-PPO, SORL, TARL, ITPO); hindsight & counterfactual (HCAPO, C3, CCPO, CriticSearch); critic-free step-level (GiGPO, POAD); hierarchical (ArCHer, PilotRL, CARL); information-theoretic (IGPO); implicit/DPO-based (iStar, StepAgent); infrastructure/practical (Agent Lightning, RAGEN, SPA-RL, LaRe, SCRIBE, PRS, adaptive segment reward); multi-agent (M-GRPO, LLM-MCA, QLLM, SHARP, MAPPA, Dr. MAS). — [companion repo](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- The repo observes that recent agentic papers move "toward turn/action-aware optimization: execution feedback for coding agents, response-level entropy or uncertainty control, information-gain normalization across turns, and structured action credit for CLI-style environments." — same
- The survey contains an appendix "Reporting Checklist for Future Credit Assignment Papers". Its content was not retrieved. The repo also ships a "Method Decision Guide" and a machine-readable catalog. — [search summary](https://arxiv.org/pdf/2604.09459); [companion repo](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- Benchmarks the survey tabulates. Reasoning: GSM8K, MATH, MATH-500, AIME, AMC, CodeContests. Agentic: ALFWorld, WebShop, WebArena, SWE-bench, AppWorld, ShellOps, TextCraft, HotpotQA, Search QA, ColBench (with typical turns and tokens). — [companion repo](https://github.com/xxzcc/Awesome-Credit-Assignment-in-LLM-RL)
- **Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL** (Zenghuang-Fu; created 2026-05-27; maintainers' own method SIPO, arXiv 2609.34805). Its 8-way taxonomy: (1) trajectory-level, (2) step/turn-level, (3) branch/tree-structured, (4) action/tool-use, (5) retrieval/evidence-level, (6) memory/skill-level, (7) long-horizon advantage estimation and policy optimization, (8) benchmarks/evaluation (including **AgentCreditBench**, an "exact-oracle turn-level credit conformance suite", and MemGym). — [GitHub list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- **Memory-R2** placement: "Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents" (arXiv 2605.21768, method LoGo-GRPO) sits under **memory-level / skill-level CA** in the Long-Horizon list, next to Memory-T1 (arXiv 2512.20092), SCRIBE and CODESKILL (arXiv 2605.25430). Not deep-dived per scope. — [GitHub list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)

### Inferences
- The survey's granularity × methodology grid is a good backbone for the chapter. Two axes it underweights are worth adding: **(a) what the method assumes about the environment** (resettable state, recurring states, gold answer available, per-step verifier, privileged info), and **(b) where the extra compute goes** (rollouts vs. critic training vs. judge calls vs. forward passes).

### Gaps
- The survey's open-problems section and checklist text could not be fetched (arxiv.org blocked). The open problems therefore are **not summarized from primary text**. The writer should not attribute specific open problems to the survey beyond the thesis quoted above.

---

## Q8. Cross-family comparison: assumptions, extra compute, bias/variance, where it helps vs not, negative results

### Takeaway
No family dominates. Methods that change **where gradient goes** at near-zero cost (entropy forking tokens, GiGPO anchors when states recur, gold-likelihood turn rewards) have the best cost/benefit when their assumption holds. Rollout-heavy MC/tree methods buy unbiasedness with compute, unless prefix sharing turns that into savings. Learned PRMs and judges carry hacking risk. Documented failure modes include PRM collapse (sum-form), **tool-call hacking** (rewarding calls rather than use), tool-call **suppression** under coarse shaping, and dense shaping that agents exploit by spamming tools. 2026 work has started to **audit** whether dense credit signals are correct at all.

### Cited Findings
**Assumption/cost table (synthesized from the sources above)**

| Family | Representative (verified ID) | Key assumption | Extra compute | Bias / variance | Reported where it helps |
|---|---|---|---|---|---|
| Learned / generative PRM | Math-Shepherd 2312.08935; AgentPRM 2511.08325 | Step labels (MC, human, or TD) are meaningful | PRM training + inference per step | Biased (model error), low variance; hackable | Reasoning BoN/RL; agents only with outcome mixing |
| Implicit PRM | iStar/OPRL 2509.19199 | Outcome labels suffice | Train implicit PRM online | Moderate bias | WebShop/VisualSokoban-type agents (OPRL) |
| Verifiable process reward | VPR 2605.10325 | Per-turn oracle exists | Oracle calls | Low bias where the oracle is correct | Controlled envs → ALFWorld/WebShop |
| MC vine/segment | VinePPO 2410.01679; SPO 2505.23564 | Resettable prefix state | K extra rollouts per cutpoint | Unbiased, high variance at small K | Math reasoning |
| Tree / branching | Tree-GRPO 2509.21240; ARPO 2507.19849 | Forkable env state | Re-allocated, can save budget (¼ claimed) | Correlated siblings | Multi-hop QA, web search agents |
| Anchor-state grouping | GiGPO 2505.10978 | Identical states recur | ~0 | Unbiased within anchor | ALFWorld, WebShop |
| Gold-likelihood turn reward | TRACE 2607.13988; IGPO 2510.14967 | Gold answer known at train time | 1 frozen-model forward per turn | Biased toward answer-revealing turns | Search agents (BrowseComp-Plus) |
| Privileged critic | SWEET-RL 2503.15478 | Training-time privileged info | Critic training | Learned bias | ColBench collaborative tasks |
| LLM hindsight critic | HCAPO 2603.08754; RICOL 2602.17497 | LLM can judge steps post hoc | LLM pass per trajectory | Judge bias | ALFWorld/WebShop; BabyAI |
| Shapley / counterfactual | SCAR 2505.20417; CSV 2607.16999 | Segment perturbations are meaningful | Exponential, approximated | Principled, costly | Mostly theory / small scale |
| Key-token selection | 80/20 forking 2506.01939 | Entropy marks decision points | ~0 | Ignores low-entropy errors | Qwen3-32B AIME |
| Learned token critic | VAPO 2504.05118; VC-PPO 2503.01491 | Critic can be pretrained | Critic model | Low variance, needs tricks | Long-CoT math (AIME 60.4) |

**Negative results, failure modes and caveats**
- PRM sum-form returns → reward hacking and training collapse around step 25 (PURE). Min-form only delays hacking, per a 2026 follow-up. — [NeurIPS 2025 PURE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/be91eb86eb74efc055cff83e953f86ce-Abstract-Conference.html); [arXiv 2605.10158](https://arxiv.org/pdf/2605.10158)
- Raw step penalties gave "limited improvement over vanilla RL" in VisualSokoban (OPRL ablation). Overly fine-grained token-level PRM rewards amplify variance in long agent trajectories. — [arXiv 2509.19199](https://arxiv.org/html/2509.19199v1)
- **Tool-call hacking:** "Proof-of-Use: Mitigating Tool-Call Hacking in Deep Research Agents" (arXiv 2510.10931) documents agents being rewarded for *invoking* tools rather than using their outputs, and proposes evidence-grounded RL. — [arXiv 2510.10931](https://arxiv.org/pdf/2510.10931)
- **Tool over-calling under dense shaping:** Fireworks' multi-turn RL guide warns that per-tool-call shaping can be exploited "by spamming retrieval calls," and recommends that the final outcome reward dominate (practitioner source, not peer-reviewed). — [Fireworks blog](https://fireworks.ai/blog/best-practices-for-multi-turn-RL)
- **The opposite failure, tool-call suppression:** a daily-papers summary (LongCat-Flash-Prover) notes coarse shaping can "encourage indiscriminate tool-call suppression". Another (ParaVT) describes a "skip-tool reward shortcut" under temperature sampling (secondary, aggregator page). — [HF daily papers](https://huggingface.co/papers?q=agentic+RL)
- Reward Hacking Benchmark (arXiv 2605.02964): RL post-training is associated with higher reward-hacking rates in tool-use agents. Simple environment hardening cut hacking by 5.7pp without hurting task success. — [arXiv 2605.02964](https://arxiv.org/pdf/2605.02964)
- GRPO baseline bias: "Your Group-Relative Advantage Is Biased" (arXiv 2601.08521, HA-DW) [V-list]. This is relevant because many CA methods are built on GRPO's group baseline. — [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- **Auditing whether dense credit is correct (2026):** "Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay" (arXiv 2608.19760) [V-search, title only]. VICT "Verifier-Instrumented Credit Tracing for Long-Horizon LLM Agent RL" (arXiv 2608.28128) [V-search, title only]. QVal (arXiv 2606.32034) [V-list]. AgentCreditBench, an exact-oracle conformance suite. — [arXiv 2608.19760](https://arxiv.org/pdf/2608.19760); [arXiv 2608.28128](https://arxiv.org/pdf/2608.28128); [Long-Horizon list](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL)
- Comparison caveat: HCAPO's improvements over GiGPO appear to be much smaller than its gains over GRPO. A citing paper (VICT) gives ambiguous incremental numbers of about 3–4pt on ALFWorld. Low confidence. — [arXiv 2608.28128](https://arxiv.org/pdf/2608.28128)

### Inferences
- **Recurring evaluation gap:** most CA papers report gains over GRPO at an **equal number of trajectories**, not equal tokens, tool calls or wall-clock. Many also omit comparisons with other CA methods (HCAPO vs GiGPO shrinks the margin). A survey chapter should mark which results are compute-matched. This mirrors this project's own finding in M10/E7: Δ_step +1.44pt with CI crossing 0, and Δ_compute −0.39pt.
- The tool-use-specific hazard: any **per-tool-call positive shaping** creates an over-calling incentive, and any **per-call cost** creates a suppression incentive. Gold-likelihood signals (TRACE/IGPO) avoid explicit call counts but can reward "answer-leaking" turns.
- Choosing a method by environment property:
  - Resettable + gold answer (math + Python): MC/tree branching, gold-likelihood turn rewards, entropy key tokens.
  - Recurring discrete states (ALFWorld/WebShop): GiGPO.
  - Privileged info but no repeats (collaboration, SWE): asymmetric critic.
  - No verifier at all: LLM hindsight critic or rubrics (DRACO), with audit.
- Placement of Memory-R2: memory-level CA (cross-session credit for memory writes and reads). It is orthogonal to the turn-level families above.

### Gaps
- **Not found:** a peer-reviewed paper reporting that a CA method **failed to improve endpoint accuracy** under compute-matched comparison in multi-turn tool use. The audit papers (2608.19760, 2608.28128) might contain such evidence, but I did not open their text.
- **[UNVERIFIED]** OTC "Acting Less is Reasoning More: Teaching Model to Act Efficiently" (from memory arXiv 2504.14870), on penalizing tool over-calls. The search did not surface it.
- **[UNVERIFIED]** "Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability" (from memory arXiv 2505.11227), a possible negative result for explicit PRMs in RL.
- Some IDs carry unusually high 5-digit suffixes: 2609.34805 (SIPO), 2606.32017 (TRIAGE), 2606.32034 (QVal). They appear only in curated lists and should be rechecked on arXiv.
- Network constraint: arxiv.org and export.arxiv.org were unreachable from this session. Allowing them in the session network settings would let every [V-list]/[UNVERIFIED] item be checked in batch.
