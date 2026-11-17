# Agentic RL for LLMs (2025–2026): Landscape, Task Families, Methods, Infrastructure, Difficulties

How these notes were checked (read this first):
- **arXiv access was blocked in this session.** `export.arxiv.org` and `arxiv.org` could not be reached from the sandbox or from WebFetch. IDs were checked in two ways: (a) cross-referencing curated GitHub lists fetched raw: thinkwee/AgentsMeetRL (last updated 2026-11), TsinghuaC3I/Awesome-RL-for-LRMs, xhyumiracle/Awesome-AgenticLLM-RL-Papers and 0russwest0/Awesome-Agent-RL; (b) web search results that return the arXiv page for that ID. Tags used below:
  - **[V]**: the ID and title matched on arXiv through web search, or a curated list printed the title next to that exact ID.
  - **[L]**: the ID appears only in the AgentsMeetRL community list. The title and content were *not* checked independently.
  - **[U]**: not verified.
- Where I give a method detail without having re-read the paper in this session, I tag it **(detail from prior knowledge, not re-fetched)**. The report writer should treat those as lower confidence.
- Scope: the credit-assignment algorithms themselves are left to another researcher. This file only points out *where* credit assignment is the bottleneck.

## 1. What agentic RL is, and the surveys that define it

### Takeaway
"Agentic RL" moves LLM RL from a degenerate single-step MDP (prompt → one completion → scalar reward) to a temporally extended POMDP. In that setting the policy interleaves reasoning, actions/tool calls and environment observations over many turns. The canonical survey is arXiv 2509.02547 (Sept 2025, revised through v4), which synthesizes 500+ works.

### Cited Findings
- **"The Landscape of Agentic Reinforcement Learning for LLMs: A Survey"**, arXiv **2509.02547** [V], by Guibin Zhang et al. v1 is dated 2 Sep 2025, v3 is dated 8 Nov 2025, and a v4 HTML exists. It contrasts "degenerate single-step MDPs of LLM RL with the temporally extended POMDPs that define Agentic RL". Its taxonomy has two halves: capabilities (planning, tool use, memory, reasoning, self-improvement, perception) and application domains. It covers 500+ works and compiles open-source environments, benchmarks and frameworks. — [arXiv 2509.02547](https://arxiv.org/abs/2509.02547); [HTML v4](https://arxiv.org/html/2509.02547v4)
- **"A Survey of Reinforcement Learning for Large Reasoning Models"**, arXiv **2509.08827** [V: TsinghuaC3I list header links this ID]. It is the companion reasoning-RL survey and its repo also tracks agentic/tool RL. — [TsinghuaC3I/Awesome-RL-for-LRMs](https://github.com/TsinghuaC3I/Awesome-RL-for-LRMs)
- Living community indexes. They are not peer-reviewed surveys, but they are the most current 2026 coverage:
  - AgentsMeetRL tags each repo with RL algorithm, outcome/process reward, single/multi-turn, task and RL framework. Its 2026-11 update added 45 repos, mostly from Oct–Nov 2026. — [thinkwee/AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
  - Two other lists. — [xhyumiracle/Awesome-AgenticLLM-RL-Papers](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers); [0russwest0/Awesome-Agent-RL](https://github.com/0russwest0/Awesome-Agent-RL)

### Inferences
- For a background chapter, 2509.02547 is the citation that defines the field: POMDP framing plus a capability taxonomy. 2509.08827 is the citation for the reasoning-RL side.
- In 2026, the field's center of gravity is visible mostly through repos and arXiv preprints indexed by community lists. There is no new consolidated survey I could verify.

### Gaps
- I could not verify these candidate surveys:
  - "Deep Research Agents: A Systematic Examination and Roadmap" (I believe 2506.18096) [U].
  - "Reinforcement Learning Foundations for Deep Research Systems: A Survey" (I believe 2509.06733) [U].
  - Any 2026 agentic-RL survey. None surfaced in the lists or searches I ran.
- I could not open OpenReview, so I did not check which of the papers below have ICLR 2026, ICML 2026 or NeurIPS 2026 acceptance. Two exceptions: SimpleTIR has NeurIPS 2025 and ICLR 2026 virtual pages ([NeurIPS](https://neurips.cc/virtual/2025/127990), [ICLR](https://iclr.cc/virtual/2026/poster/10010643)), and a NeurIPS 2026 paper on workspace branching exists (see §4).

## 2. Multi-turn tool-integrated reasoning (TIR) and search/deep-research RL: method-by-method

### Takeaway
Almost every TIR and search-agent RL paper of 2025 used **outcome-only, trajectory-level reward**: final-answer correctness, sometimes plus a format term. They were optimized with PPO or GRPO-family methods, with tool and retrieval tokens masked out of the loss. The "tricks" mostly target *stability* (masking, void-turn filtering, entropy control) and *data/horizon* (synthetic hard QA, async long rollouts). Per-turn or process rewards appear only in a minority: StepSearch, R1-Searcher (both), and by 2026 a growing set of turn-level attribution papers. That minority is exactly where credit assignment is named as the bottleneck.

### Cited Findings

#### Master table (2025 core)

| Method | arXiv | Date | Task | Key trick | Reward assignment |
|---|---|---|---|---|---|
| Search-R1 | 2503.09516 [V] | 2025-03 | multi-hop QA + search engine | Interleaved `<search>` calls in RL. **Retrieved-token loss masking** (detail from prior knowledge, not re-fetched). veRL | **Outcome** (PPO/GRPO), single-agent, multi-turn — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [arXiv](https://arxiv.org/abs/2503.09516) |
| R1-Searcher | 2503.05592 [V] | 2025-03 | search QA | Two-stage outcome-based RL: stage 1 learns *to invoke* retrieval, stage 2 learns answer quality (detail from prior knowledge) | Listed as **Both** (outcome + process-style stage reward), PPO/DPO — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [arXiv](https://arxiv.org/abs/2503.05592) |
| ReSearch (ReCall) | 2503.19470 [V] | 2025-03 | search QA | Search ops embedded in the reasoning chain. No supervised reasoning steps (detail from prior knowledge) | Outcome — [arXiv](https://arxiv.org/abs/2503.19470); code [Agent-RL/ReCall](https://github.com/Agent-RL/ReCall) |
| ReTool | 2504.11536 [V] | 2025-04 | competition math + code interpreter | Cold-start SFT on code-interleaved data, then PPO with interleaved real-time code execution (detail from prior knowledge) | **Outcome**, PPO, multi-turn, external (sandbox) reward — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [arXiv](https://arxiv.org/abs/2504.11536) |
| ToRL | 2503.23383 [V] | 2025-03 | math + code | "Scaling Tool-Integrated RL" directly from base models. Emergent code-usage behaviors (detail from prior knowledge) | Outcome — [arXiv](https://arxiv.org/abs/2503.23383); [GAIR-NLP/ToRL](https://github.com/GAIR-NLP/ToRL) |
| ZeroTIR ("Agent RL Scaling Law: Agent RL with Spontaneous Code Execution for Mathematical Problem Solving") | 2505.07773 [V] | 2025-05 | math | RL from base model with spontaneous code execution. Reports scaling relations between training steps, code-call frequency and accuracy (detail from prior knowledge). Built on an async OpenRLHF pipeline | Outcome — [arXiv](https://arxiv.org/abs/2505.07773); [code](https://github.com/yyht/openrlhf_async_pipline) |
| SimpleTIR | 2509.02479 [V] | 2025-09 | multi-turn TIR math/code | **Filters trajectories containing "void turns"** (no code block and no final answer) from the policy update but keeps them in advantage estimation. This fixes gradient-norm explosion caused by distributional drift from tool outputs. Qwen2.5-7B AIME24: 22.1 → 50.5 | Outcome — [arXiv](https://arxiv.org/abs/2509.02479); [NeurIPS 2025 page](https://neurips.cc/virtual/2025/127990); [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL) |
| ARPO (Agentic Reinforced Policy Optimization) | 2507.19849 [V] | 2025-07 | search/knowledge/math TIR | Entropy-based adaptive rollout that branches more after tool calls, where token entropy spikes. Advantage attribution across shared/branched segments (detail from prior knowledge) | Outcome (+ attribution over branches) — [arXiv](https://arxiv.org/abs/2507.19849). **Disambiguation:** AgentsMeetRL's "ARPO" row is a *different* GUI paper, 2505.16282 (CUHK) — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL) |
| AEPO (Agentic Entropy-Balanced PO) | 2510.14545 [V] | 2025-10-16 | multi-turn web agents | Dynamic entropy-balanced rollout plus entropy-balanced policy optimization. Beats 7 agentic RL baselines on 14 benchmarks. Qwen3-14B with 1K RL samples: GAIA 47.6, HLE 11.2, WebWalkerQA 43.0 | Outcome — [arXiv](https://arxiv.org/abs/2510.14545v1) |
| Tool-Star | 2505.16410 [V] | 2025-05 | multi-tool reasoning | Tool-integrated data synthesis plus two-stage training. Self-critic RL (detail from prior knowledge) | Outcome (listed); model/external reward — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [arXiv](https://arxiv.org/abs/2505.16410) |
| WebDancer | 2505.22648 [V] | 2025-05 | web info-seeking | Data construction → SFT cold start → RL (DAPO) (detail from prior knowledge) | Outcome — [arXiv](https://arxiv.org/abs/2505.22648) |
| WebSailor | 2507.02592 [V] | 2025-07 | super-hard web QA (BrowseComp) | High-uncertainty synthetic tasks plus the DUPO RL algorithm (detail from prior knowledge) | Outcome — [arXiv](https://arxiv.org/abs/2507.02592) |
| WebShaper | 2507.15061 [V] | 2025-07 | data synthesis for info-seeking | Formalization-driven task synthesis | (data paper) — [arXiv](https://arxiv.org/abs/2507.15061) |
| WebSailor-V2 | 2509.13305 [V] | 2025-09-16 | deep research | Synthetic data, SFT, and RL with **DUPO (Duplicating Sampling PO)**. Qwen3-30B-A3B: BrowseComp-EN 35.3, BrowseComp-ZH 44.1, HLE 30.6 | Outcome — [arXiv](https://arxiv.org/abs/2509.13305) |
| Tongyi DeepResearch tech report | 2510.24701 [V-partial: ID appears in a curated list next to the Alibaba-NLP/DeepResearch repo; title not printed] | 2025-10 | deep research | Umbrella report for the WebSailor/WebDancer/WebShaper line | Outcome — [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) |
| Kimi-Researcher | no arXiv (blog) | 2025-06 | deep research (search, browse, code) | **End-to-end agentic RL**, REINFORCE. **Avg 23 reasoning steps, 200+ URLs explored per task**. HLE Pass@1 26.9% | **Outcome** — [Kimi-Researcher blog](https://moonshotai.github.io/Kimi-Researcher); [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL) |
| ASearcher ("Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL") | 2508.07976 [V] | 2025-08 | long-horizon search | Fully **async RL (AReaL)** so that long trajectories do not stall the batch. Decoupled PPO | **Outcome**, PPO/GRPO + decoupled PPO — [arXiv](https://arxiv.org/abs/2508.07976); [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL) |
| MiroThinker v1.0 | 2511.11793 [V] | 2025-11-14 | open research agent | **"Interactive scaling"** is a third scaling axis after model size and context. With 256K context, **up to 600 tool calls/task**, versus <100 for prior open models. **Recency-based context retention**: only the most recent tool responses are kept, while the full thoughts and actions are preserved. BrowseComp 47.1% (72B). Three-stage training | Outcome RL — [arXiv](https://arxiv.org/abs/2511.11793) |
| StepSearch | 2505.15107 [V] | 2025-05 | search QA | Step-wise PPO with information-gain / redundancy step rewards (detail from prior knowledge) | **Process** (per-step) — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [arXiv](https://arxiv.org/abs/2505.15107) |
| ZeroSearch | 2505.04588 [V] | 2025-05 | search QA | Simulated search engine (an LLM) with curriculum degradation, which avoids real API cost | Outcome — [arXiv](https://arxiv.org/abs/2505.04588) |
| WebAgent-R1 | 2505.16421 [V] | 2025-05 | WebArena-Lite web nav | End-to-end multi-turn RL | Outcome — [arXiv](https://arxiv.org/abs/2505.16421) |
| ToolRL | 2504.13958 [V] | 2025-04 | function calling | Studies reward design for tool learning (fine-grained format/correctness) | Rule-based, per-call-structured — [arXiv](https://arxiv.org/abs/2504.13958) |
| Kimi K2 | 2507.20534 [V] | 2025-07 | open agentic foundation model | Large-scale agentic data plus RL | — [arXiv](https://arxiv.org/abs/2507.20534) |
| GLM-4.5 | 2508.06471 [V] | 2025-08 | ARC (agentic/reasoning/coding) | Agentic RL stage; trained with slime | — [arXiv](https://arxiv.org/abs/2508.06471) |
| WebExplorer | 2509.06501 [V] | 2025-09 | long-horizon web agents | Explore-and-evolve data synthesis plus RL | — [arXiv](https://arxiv.org/abs/2509.06501) |

#### 2026 successors (from AgentsMeetRL; mostly [L])
Format: name, arXiv, date, org, RL algorithm, reward. Source for all: [thinkwee/AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL).

Deep-research / search agents:
- **O-Researcher**, 2601.03743 [L], 2026-01, OPPO. GRPO + RLAIF, **process** reward.
- **ProRAG**, 2601.21912 [L], 2026-01, RUC. GRPO + dual-granularity advantage; both outcome and process.
- **MR-Search**, 2603.11327 [L], 2026-03. In-context meta-RL across episodes; outcome.
- **LiteResearcher**, 2604.17931 [L], 2026-04. Curriculum with a "lite virtual world" (local search/browse env); outcome. Self-reported GAIA 71.3%, Xbench-DS 78.0% at 4B.
- **DR-Venus**, 2604.19859 [L], 2026-04, Ant. GRPO + **IGPO information-gain turn-level reward**; both.
- **Harness-1**, 2606.02373 [L], 2026-06, UIUC. GRPO; outcome. Long-horizon search with a state-externalizing harness.
- **SlimSearcher**, 2606.07074 [L]. Efficiency-aware deep research, adaptive reward gating; outcome.
- **LAPO**, 2607.13501 [L], 2026-07. GRPO + **Leave-One-Turn attribution**; both.
- **Axion 1.0**, 2026-11, ByteDance/Duke. No arXiv in the list. 2×SFT then GRPO-style RL; outcome.

Tool-use / TIR:
- **Agent-STAR**, 2603.21972 [L]. Dense/curriculum reward.
- **AgenticQwen**, 2604.21590 [L], Alibaba PAI.
- **APPO**, 2606.12384 [L], Alibaba AMAP. Procedure-aware branching, process reward.
- **SPADER**, 2606.00593 [L]. Step-wise peer advantage.
- **ECHO**, 2610.16027 [L]. Responsibility-attributed rewards on τ²-bench.
- **ToolOPD**, 2610.25693 [L]. Credit-aware on-policy distillation.
- **StateAlign**, 2611.16304 [L], RPI/IBM.
- **Agent-R2**, 2611.08219 [L], ZJU/Baidu. Rubric-based reward.

### Inferences
- **Reward assignment pattern.** 2025 TIR and search RL is overwhelmingly outcome-only. Of the core 2025 rows in AgentsMeetRL, only StepSearch is labeled "Process", and R1-Searcher, RAGEN and verl-agent are labeled "Both". By 2026 a visible share of new search/tool-use entries carry "Both" or "Process" labels with explicit *turn-level* attribution: IGPO/DR-Venus, LAPO, ProRAG, SPADER, ECHO, ToolOPD. This shift suggests the community now treats turn-level credit as the next bottleneck once outcome-only RL saturates. The evidence is list labels, which are crude.
- **Two kinds of "tricks".** The 2025 tricks fall into stability fixes (masking tool tokens, SimpleTIR void-turn filtering, ARPO/AEPO entropy handling, StarPO-S) and horizon/data fixes (synthetic hard QA in WebSailor/WebShaper, async rollout in ASearcher/AReaL, context management in MiroThinker). Neither kind addresses *which turn deserves credit*. That gap is the opening a credit-assignment project would target.
- **Relevance to this project (math + Python interpreter, GRPO).** The closest prior art is ReTool, ToRL, ZeroTIR and SimpleTIR. All four are outcome-only, and SimpleTIR's void-turn analysis is a direct diagnostic of turn-level failure.

### Gaps
- The exact max-turn settings for Search-R1, ReTool and ToRL were not re-fetched. My recollection is single-digit turn caps for Search-R1-style QA, but this is unconfirmed.
- I did not verify "OTC: Optimal Tool Calls" (over-calling penalty paper; my recollection is 2504.14870) [U]. It is the canonical over-tool-calling citation.
- I did not verify "Demystifying RL in Agentic Reasoning" (my recollection is 2510.11701) [U].
- The 2026 [L] entries rest on one community list. Specific benchmark numbers in them, such as LiteResearcher's GAIA 71.3%, are self-reported and unchecked.

## 3. Long-horizon interactive environments, SWE/terminal agents, and typical horizons

### Takeaway
Horizons span roughly three orders of magnitude:
- Search QA and TIR math: a handful of turns.
- ALFWorld/WebShop/text games and Kimi-Researcher-style deep research: tens of turns, e.g. Kimi-Researcher averages 23 steps.
- SWE agents: tens to ~100 steps.
- 2025–26 frontier research agents: hundreds of tool calls (MiroThinker up to 600) and up to ~1,000 steps in some 2026 harness environments.

Credit assignment under a single terminal reward becomes the bottleneck as horizon grows. This is the explicit motivation for RAGEN's analysis and for GiGPO.

### Cited Findings
- **RAGEN / StarPO**, arXiv **2504.20073** [V].
  - It is a general trajectory-level RL framework for multi-turn agents, tested on symbolic environments (e.g., Sokoban, FrozenLake per the paper; detail from prior knowledge).
  - It identifies the **"Echo Trap"**: agents overfit to locally rewarded reasoning patterns. The signature is reward-variance collapse, entropy drop and gradient spikes.
  - The fix is **StarPO-S**: variance-based trajectory filtering, critic baselining and decoupled clipping. — [arXiv 2504.20073](https://arxiv.org/abs/2504.20073)
  - Secondary summaries report keeping the top 25–50% highest-reward-variance prompts. Treat that as secondary. — [EmergentMind](https://www.emergentmind.com/topics/starpo-s-stabilization)
- **verl-agent / GiGPO**, arXiv **2505.10978** [V] ("Group-in-Group Policy Optimization for LLM Agent Training").
  - verl-agent supports PPO, GRPO, GiGPO, DAPO, RLOO and REINFORCE++ across Phone-Use, Math, Code, Web and text games (ALFWorld, WebShop, Sokoban). It is listed with "Both" outcome/process reward.
  - GiGPO adds step-level groups by grouping identical anchor states across trajectories. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent)
  - Typical caps are ALFWorld ≈50 steps and WebShop ≈15 steps (detail from prior knowledge, not re-fetched).
- **AgentGym-RL**, **2509.08755** [V].
  - Fudan, Sept 2025. veRL-based; PPO/GRPO/RLOO/REINFORCE++.
  - Covers web, search, games, embodied and science environments.
  - Outcome reward. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [WooooDyy/AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL)
- **SWE-RL**, **2502.18449** [V] (Meta, Feb 2025). RL on open software-evolution data with a rule-based patch-similarity reward. — [arXiv](https://arxiv.org/abs/2502.18449)
- **SWE-Gym**, 2412.21139 [V]; **R2E-Gym**, 2504.07164 [V]. Environment and verifier sources for SWE RL. — [arXiv SWE-Gym](https://arxiv.org/abs/2412.21139); [arXiv R2E-Gym](https://arxiv.org/abs/2504.07164)
- **DeepSWE-Preview** (Agentica/Together, July 2025; blog, no arXiv).
  - Qwen3-32B trained **with RL only** using rLLM on **~4.5K R2E-Gym problems**. Repos that overlap SWE-Bench-Verified were filtered out.
  - Trained for 6 days on 64 H100s.
  - SWE-Bench-Verified Pass@1 42.2% (avg of 16), 59% with test-time scaling. Score went from 23% to 42% over the first 200 RL steps.
  - Outcome reward. — [Together blog](https://together.ai/blog/deepswe); [HF model card](https://huggingface.co/agentica-org/DeepSWE-Preview); [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
- **SkyRL** (NovaSky/Berkeley). Its own skyrl-train stack targets long-horizon agents (SWE-Bench, search, math, SQL); GRPO/PPO. AgentsMeetRL links paper **2511.16108** [L; title unchecked]. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [NovaSky-AI/SkyRL](https://github.com/NovaSky-AI/SkyRL)
- **UI-TARS-2**, 2509.02544 [V]. GUI agent with multi-turn RL. — [arXiv](https://arxiv.org/abs/2509.02544)
- **AgentRL** (THUDM), 2510.04206 [V-partial: in list next to the THUDM/AgentRL repo]. Multi-task, multi-turn agent RL. — [THUDM/AgentRL](https://github.com/THUDM/AgentRL)
- **Terminal / OS agents (2026, [L]).** Source for all: [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL).
  - 2601.16443: procedurally generated containerized terminal tasks with test verifiers, PPO on SkyRL.
  - 2607.10891: 4,500+ verifiable Docker environments in Terminal-Bench format.
  - 2608.09096: agent-harness environment with 608 tasks in an E2B sandbox, **up to 1,000 steps**.
  - 2607.28074: 722 *resettable* DB-grounded synthetic web tasks.
  - 2611.03634: 153 real websites as Docker images, 2,288 RL tasks.
  - **ARLArena** 2602.21534 (UCLA): SAMPO "stable agentic policy optimization" across web, embodied, math, game and search.
- **2026 SWE RL ([L]).** Source for all: [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL).
  - **SWE-World** 2602.03419 (RUC): Docker-free SWE RL using a *learned world model* of the environment.
  - **CodeScout** 2603.17829 (OpenHands): GSPO.
  - **SWE-Edit** 2604.26102 (MSR).
  - **SPROUT** 2610.14405: per-turn sandbox snapshots and checkpoint-branching exploration.
  - **SWE-Ladder** 2610.33667.
  - **Polar / ProRL Agent Server** 2605.24220 (NVIDIA): RL on real harnesses such as shell, Codex and Claude Code.
- **Horizons, collected:**
  - Kimi-Researcher averages 23 steps per task — [blog](https://moonshotai.github.io/Kimi-Researcher).
  - ASearcher's title claims to go "Beyond Ten Turns" — [arXiv](https://arxiv.org/abs/2508.07976).
  - MiroThinker reaches up to 600 tool calls, versus <100 for prior open models — [arXiv](https://arxiv.org/abs/2511.11793).
  - One 2026 harness environment allows up to 1,000 steps — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL).

### Inferences
- Rough horizon tiers:
  - TIR math (our setting): ~1–10 tool turns.
  - Search QA: ~2–10.
  - Embodied/web text environments: 15–50.
  - SWE: 50–100+.
  - Deep research: 20–600.
- The sparse single terminal reward is shared across all tiers. Credit dilution therefore scales with turn count, so multi-turn math TIR is the *easiest* tier in which to study turn-level credit cleanly. This is both a strength (clean experiments) and a threat to external validity for any credit-assignment claim.
- SWE and terminal tasks have the heaviest state: containers and filesystems. They are where snapshot/fork infrastructure (§4) matters most.

### Gaps
- I did not re-fetch DeepSWE's max-turn setting; the search could not confirm a 100-turn cap.
- I did not re-fetch exact GiGPO/ALFWorld/WebShop step limits.
- RAGEN's exact environments and turn counts were not re-fetched.
- I found no verified source on Kimi-Dev (believed 2509.23045) [U] or DeepSeek-V3.2's agentic RL (believed 2512.02556) [U].

## 4. Training frameworks, systems, and infrastructure problems (incl. resetting to intermediate states)

### Takeaway
veRL (HybridFlow) is the default substrate for academic agentic RL. Most rows in AgentsMeetRL list veRL, and 2026 adds verl-project's uni-agent with fully-async partial rollout. Async stacks (AReaL, slime, ROLL, SkyRL, rLLM, Agent Lightning, NeMo-RL/Polar) exist because multi-turn rollouts have long-tailed latency from tool calls and long contexts.

On resetting to an intermediate state: classic agent RL resets only to s₀. In 2026, a cluster of systems and methods appeared that **snapshot/fork sandboxes at intermediate states**: BPO, DeltaBox, Crab, AgentENV, ForkRL, SPROUT and FocalRL. This enables tree-structured rollouts, which double as a credit-assignment mechanism through sibling baselines. Pure-text TIR environments (a stateless Python call plus the prefix) can be "reset" trivially by replaying the token prefix.

### Cited Findings
Frameworks (source for all rows: [AgentsMeetRL Base Framework table](https://github.com/thinkwee/AgentsMeetRL) unless noted):

| Framework | Org / date | Paper | Notes |
|---|---|---|---|
| veRL (HybridFlow) | ByteDance, 2024-10 | 2409.19256 [V] | Base for Search-R1, ReTool, SimpleTIR, RAGEN, verl-agent, AgentGym-RL, rLLM, DR-Venus, LAPO |
| verl-project **uni-agent** | 2026-04 | — | GRPO/GSPO; **partial rollout, fully-async**; SWE-Bench/search/general agents; **1000+ concurrent**; unified model/tool/env abstractions |
| OpenRLHF | 2023-07 | 2405.11143 [V] | ZeroTIR used an async OpenRLHF pipeline ([code](https://github.com/yyht/openrlhf_async_pipline)) |
| AReaL | Ant/Tsinghua, 2025-06 | 2505.24298 [V] | Fully asynchronous RL; decoupled PPO; basis of ASearcher |
| AReno | Ant inclusionAI, 2026-06 | — | Single-node RL post-training with code-agent tools |
| slime | THUDM, 2025-06 | blog ([LMSYS](https://lmsys.org/blog/2025-07-09-slime/)) | GRPO/GSPO/REINFORCE++; used for GLM-4.5 |
| ROLL | Alibaba, 2025-06 | 2506.06122 [V] | Large-scale RL library |
| SkyRL (skyrl-train) | Berkeley NovaSky | 2511.16108 [L] | Long-horizon agents |
| rLLM | Agentica/Berkeley/Together | blog | veRL-based; DeepSWE |
| Agent Lightning | MSR, 2025-06 | 2508.03680 [V] | "Train ANY AI agents with RL" by decoupling agent execution from training |
| NeMo-RL | NVIDIA, 2026-01 | — | GRPO/DAPO/GDPO/DPO |
| Polar (ProRL-Agent-Server) | NVIDIA, 2026-05 | 2605.24220 [L] | RL on real agent harnesses |
| Molt | NVIDIA NeMo Labs, 2026-07 | 2607.21653 [L] | — |
| Orchard | Microsoft, 2026-07 | 2607.21557 [L] | — |
| RLinf | Tsinghua et al. | 2509.15965 [L] | — |
| siiRL | SII | 2507.13833 [L] | — |
| AgentJet | ModelScope, 2026-06 | 2606.04484 [L] | — |
| Libra | NetX Lab | 2606.03077 [L] | — |
| Canopy | HKUST/Tencent, 2026-10 | — | **Prefix-sharing-aware, tree-aware scheduling** of multi-turn rollouts on verl |
| ForkRL | UCSD WukLab, 2026-11 | 2611.10453 [L] | **Whole-workspace forking** for faster agentic rollouts (OSWorld/web); tool-call prefix caching |
| Others | — | — | prime-rl, verifiers, Tunix (Google), trl |

Infrastructure problems:
- **Tool/retrieval output tokens must be loss-masked.** Search-R1 introduced retrieved-token masking (detail from prior knowledge, not re-fetched) — [arXiv 2503.09516](https://arxiv.org/abs/2503.09516).
  - The masking is necessary but not sufficient. SimpleTIR shows that tool outputs still cause distributional drift: low-probability tokens fed back as input compound across turns and produce gradient-norm explosions. — [arXiv 2509.02479](https://arxiv.org/abs/2509.02479)
- **Async rollouts and long-tailed latency.**
  - ASearcher's motivation is that long-horizon search needs large-scale async RL ("Beyond Ten Turns"). — [arXiv 2508.07976](https://arxiv.org/abs/2508.07976)
  - uni-agent lists partial rollout and fully-async execution as core features. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
- **Context growth.**
  - MiroThinker keeps only the most recent tool responses so that 256K context can sustain 600 calls. — [arXiv 2511.11793](https://arxiv.org/abs/2511.11793)
  - Mosaic (2610.22174 [L]) studies search agents with context compaction and sub-agent forking. It uses *per-session advantages*, so compaction itself creates a credit-assignment problem across sessions. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
- **Environment reset to intermediate states (snapshot/fork).**
  - **Branching Policy Optimization: Sandbox-Native Language Agent RL**, arXiv **2607.14171** [V via search].
    - Snapshots the sandbox at chosen branch points and samples sibling continuations, building tree-structured rollouts per prompt.
    - The siblings act as baselines, which the paper argues significantly reduces variance.
    - It assumes "snapshot fidelity": restoring a snapshot yields an identical transition distribution. — [arXiv PDF](https://arxiv.org/pdf/2607.14171)
  - **DeltaBox**, 2605.22781 [V via search]. Millisecond-level checkpoint and rollback for stateful agent sandboxes. It argues that with k (branches) in the tens to low hundreds, fork latency directly bounds training throughput. — [arXiv PDF](https://arxiv.org/pdf/2605.22781)
  - **Crab**, 2604.28138 [V via search]. A semantics-aware checkpoint/restore runtime that can fork a sandbox from a checkpoint instead of re-executing a shared prefix. It cites Tree-GRPO-style methods as the motivating workload. — [alphaXiv](https://alphaxiv.org/abs/2604.28138)
  - **AgentENV** (Kimi / kvcache-ai; secondary report only). Forks a running sandbox into up to 16 children on the same node, with <50 ms resume and <100 ms pause. Source is a vendor/secondary page, unverified against the primary repo. — [MorphLLM RL env infra overview](https://www.morphllm.com/rl-environment-infrastructure)
  - **"Content-Addressed Workspace Branching for Agentic RL Rollouts"** (NeurIPS 2026). A copy-on-write workspace store integrated with SkyRL, reporting negative results on per-branch full-text indexing. — [NeurIPS 2026 virtual](https://neurips.cc/virtual/2026/166782)
  - Method side ([L]; source for all: [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)):
    - **SPROUT** 2610.14405: per-turn sandbox snapshots plus checkpoint-branching exploration for long-horizon SWE, on the Miles framework.
    - **FocalRL** (Microsoft, 2026-10): GRPO with **branching from pre-error states**, plus "focal credit assignment / local repair rewards", for BrowseComp and SWE.
    - **Tree-GRPO**, 2509.21240 [V]: "Tree Search for LLM Agent RL", which shares prefixes across agent-step tree nodes.
  - Daytona also publishes a sandbox-for-RL write-up (vendor). — [Daytona](https://www.daytona.io/dotfiles/sandbox-infrastructure-for-reinforcement-learning-agents)

### Inferences
- **Can an environment be reset to an intermediate state?**
  - In 2025 mainstream frameworks, generally only via *prefix replay*: regenerate from the stored token prefix, which re-executes tools. This is exact only for deterministic, stateless tools.
  - For stateful sandboxes (SWE, terminal, OS, web), true intermediate-state reset needed dedicated checkpoint/restore infrastructure. That infrastructure arrived as a 2026 cluster: DeltaBox, Crab, AgentENV, ForkRL, BPO and workspace branching.
  - Its correctness hinges on "snapshot fidelity", which BPO states explicitly as an assumption.
- **Implication for this project.** A Python-interpreter math environment is close to stateless: each code cell runs in a fresh or short-lived sandbox, or the state is reconstructible by replaying earlier cells. Branching from intermediate turns (Tree-GRPO/BPO-style sibling baselines, i.e. Monte-Carlo turn-value estimates) is therefore cheap here. It is much more expensive in SWE and web settings. This is a concrete feasibility advantage to note.
- The co-design of tree rollouts, snapshot infrastructure and sibling baselines shows that infrastructure and credit assignment are converging. Rollout branching *is* the credit-assignment estimator.

### Gaps
- I could not confirm whether veRL's mainline agent loop (2025–26) natively supports environment snapshot/fork. No primary veRL docs were fetched.
- The AgentENV numbers come only from a secondary vendor page.
- ForkRL, SPROUT and FocalRL are [L] only. The search engine did not find ForkRL or SPROUT by name, so treat them as unverified until checked.
- The details of slime, ROLL, Agent Lightning and NeMo-RL async designs were not fetched.

## 5. Distinctive difficulties vs single-turn reasoning RL (and where credit assignment is the bottleneck)

### Takeaway
Agentic RL inherits reasoning-RL's sparse outcome reward. It adds tool-output distribution drift, multi-turn entropy and variance collapse (the Echo Trap), long-tailed rollout latency, context growth, stateful and non-resettable environments, and behavioral reward hacking such as over- or under-calling tools. Credit assignment is the stated bottleneck where the reward is terminal but decisions are per-turn. This is why 2026 work concentrates on turn-level attribution and branching.

### Cited Findings
- **Sparse, delayed reward.** Nearly all core 2025 TIR, search, SWE and deep-research systems use outcome-only reward. AgentsMeetRL labels Search-R1, ReTool, SimpleTIR, ASearcher, Kimi-Researcher, AgentGym-RL, DeepSWE and rLLM as "Outcome". — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
- **Multi-turn instability and collapse.**
  - RAGEN's Echo Trap shows reward-variance collapse, entropy drop and gradient spikes. — [arXiv 2504.20073](https://arxiv.org/abs/2504.20073)
  - SimpleTIR shows gradient-norm explosion from tool-induced distribution drift. — [arXiv 2509.02479](https://arxiv.org/abs/2509.02479)
  - ARLArena (2602.21534 [L]) frames "stable agentic RL" as its central problem. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
- **Entropy dynamics around tool calls.**
  - ARPO (2507.19849) and AEPO (2510.14545) build their rollout and optimization around entropy behavior at tool-call boundaries.
  - AEPO balances entropy both in rollout and in the policy update. — [arXiv 2510.14545](https://arxiv.org/abs/2510.14545v1); [arXiv 2507.19849](https://arxiv.org/abs/2507.19849)
- **Error propagation and compounding.** SimpleTIR attributes harmful negative samples to "compounding errors induced by using external tool outputs during multi-turn rollout". — [arXiv 2509.02479](https://arxiv.org/abs/2509.02479)
- **Long horizons and context growth.**
  - MiroThinker needs recency-based truncation of tool outputs to reach 600 calls. — [arXiv 2511.11793](https://arxiv.org/abs/2511.11793)
  - Kimi-Researcher averages 23 steps and 200+ URLs. — [blog](https://moonshotai.github.io/Kimi-Researcher)
- **Stochastic and stateful environments.**
  - Real web search is non-stationary and costly. ZeroSearch replaces it with a simulated LLM search engine for this reason. — [arXiv 2505.04588](https://arxiv.org/abs/2505.04588)
  - SWE-World replaces Docker with a learned world model. — [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL)
  - Snapshot-fidelity concerns for stateful sandboxes. — [BPO 2607.14171](https://arxiv.org/pdf/2607.14171)
- **Throughput.** DeltaBox: fork latency bounds training throughput when branching is used. — [arXiv 2605.22781](https://arxiv.org/pdf/2605.22781)
- **Where credit assignment is named as the bottleneck.** These are pointers only; the algorithms are covered elsewhere.
  - MT-GRPO, "Reinforcing Multi-Turn Reasoning in LLM Agents via Turn-Level Credit Assignment", 2505.11821 [V].
  - GiGPO 2505.10978 [V]; SPA-RL "Stepwise Progress Attribution" 2505.20732 [V]; StepSearch 2505.15107 [V]; Tree-GRPO 2509.21240 [V].
  - 2026 [L]: LAPO 2607.13501 (leave-one-turn attribution), DR-Venus/IGPO 2604.19859 (info-gain turn reward), ProRAG 2601.21912, SPADER 2606.00593, ECHO 2610.16027, Mosaic 2610.22174 (multi-session credit), FocalRL (focal credit / pre-error branching), DARS 2610.01207 (potential-based step reward on ALFWorld/WebShop), ConCal 2610.37195 (GiGPO extension), DRACO 2609.04094 (rubric-attributed step reward on AppWorld/τ-Bench), SAPO 2608.19842, Agent-G² 2608.23318.
  - Process-reward models for agents: Agent-RRM 2601.22154 [L]; ToolPRMBench 2601.12294 [L].
  - Sources: [AgentsMeetRL](https://github.com/thinkwee/AgentsMeetRL); [Awesome-AgenticLLM-RL-Papers](https://github.com/xhyumiracle/Awesome-AgenticLLM-RL-Papers)

### Inferences
- These difficulties interact:
  - Outcome-only reward plus long horizons dilutes per-turn signal.
  - Tool-induced drift and the Echo Trap produce high-variance or degenerate gradients.
  - Stability fixes such as filtering (SimpleTIR, StarPO-S, and DAPO-style zero-variance filtering) improve *training stability*, not attribution.
- Turn-level credit methods (2025: MT-GRPO, GiGPO, SPA-RL; 2026: about a dozen [L] entries) are the response to the attribution problem. Their proliferation in Oct–Nov 2026 indicates a crowded but unconsolidated space with no standard benchmark or comparison protocol. I infer this from list labels, not from a meta-analysis.
- Over-calling and under-calling tools, and reward hacking via format, are frequently discussed. However, I did not fetch primary evidence in this session (e.g., OTC or ToRL's code-frequency analyses), so the writer should cite carefully.

### Gaps
- I have no primary-source quantitative evidence fetched for:
  - over-tool-calling / reward hacking (OTC [U]);
  - entropy collapse specifically in multi-turn TIR, beyond RAGEN, SimpleTIR and AEPO abstracts;
  - error-propagation statistics (e.g., the fraction of failures attributable to an early wrong tool call).
- I did not check OpenReview to see which 2026 [L] credit-assignment papers are peer-reviewed.
- The reference IDs I could not verify are 2506.18096, 2509.06733, 2510.11701, 2509.23045, 2512.02556 and OTC (2504.14870). Each is marked [U] above.
