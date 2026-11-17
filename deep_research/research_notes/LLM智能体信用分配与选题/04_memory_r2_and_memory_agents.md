# Memory-R2 (close reading) and the landscape of RL-trained memory-augmented LLM agents (2024–2026)

> **Method note / reliability caveat (read first).** In this session arxiv.org, huggingface.co and github.com
> were blocked by the sandbox network policy (CONNECT 403 / DNS not allowed). I could **not** open the
> Memory-R2 full text (HTML/PDF), its GitHub repo, or OpenReview. Everything below that is attributed to a
> source came from web-search result snippets/synthesis pointing at that source (mostly the arXiv HTML/PDF of
> 2605.21768 and its GitHub README), not from my own reading of the full paper. Exact equations, table
> layouts, group sizes and compute numbers for Memory-R2 are therefore **gaps**. Items marked
> "[prior knowledge]" come from my training data (cutoff mid-2026) and were not re-verified this session.
> To close the gaps, someone must allow `arxiv.org` / `github.com` in the session network settings and
> re-read §3–§5 and the appendix of arXiv 2605.21768v1.

## Q1. Memory-R2: identity, authors, venue

### Takeaway
Memory-R2 (arXiv 2605.21768, submitted 20 May 2026) is by Sikuan Yan, Ahmed Bahloul, Ercong Nie, Susanna Schwarzmann, Riccardo Trivisonno, Volker Tresp, Yunpu Ma; NeurIPS 2026 acceptance is supported by author self-reports and a NeurIPS-2026 paper digest, but I could not check the official proceedings. Huawei affiliation is plausible but unconfirmed.

### Cited Findings
- Title "Memory-R2: Fair Credit Assignment for Long-Horizon Memory-Augmented LLM Agents", arXiv 2605.21768, submitted 20 May 2026; abstract page lists "Sikuan Yan and 6 other authors"; code at github.com/ahmedehabb/Memory-R2; built on the ReMA framework (per search synthesis of abstract/README) — [arXiv abs](https://arxiv.org/abs/2605.21768); [GitHub](https://github.com/ahmedehabb/Memory-R2)
- Full author list as cited by later papers: Sikuan Yan, Ahmed Bahloul, Ercong Nie, Susanna Schwarzmann, Riccardo Trivisonno, Volker Tresp, Yunpu Ma — [SpeakerMem-R1 reference list, arXiv 2609.26780](https://arxiv.org/html/2609.26780v1); [arXiv abs](https://arxiv.org/abs/2605.21768)
- Ahmed Bahloul and Susanna Schwarzmann have alphaXiv profiles linked to the paper — [alphaXiv @ahmed-bahloul](https://www.alphaxiv.org/@ahmed-bahloul); [alphaXiv @susanna-schwarzmann](https://www.alphaxiv.org/@susanna-schwarzmann)
- Venue: an author (Ercong Nie, now listed at Shanghai Jiao Tong University) announced acceptance at #NeurIPS2026 on LinkedIn; the SJTU LaSAI lab page lists it among NeurIPS 2026 accepts (Sept 2026); Paper Digest's NeurIPS 2026 catalogue includes it — [LinkedIn (Ercong Nie)](https://www.linkedin.com/in/ercong-nie-6375a5104/); [LaSAI lab](https://lasai-sjtu.github.io/); [Paper Digest NeurIPS 2026](https://www.paperdigest.org/2026/10/neurips-2026-papers-highlights/)
- Conflicting citation: one paper's reference list cites Memory-R2 with DOI `10.18653/v1/2026.acl-long.583`; but that ACL Anthology PDF is returned as the **Memory-R1** ACL 2026 paper — [ACL Anthology 2026.acl-long.583](https://aclanthology.org/2026.acl-long.583.pdf)
- A GitHub awesome-list tags it simply "arXiv 2026" — [RUC-NLPIR/Awesome-Long-Horizon-Agents](https://github.com/RUC-NLPIR/Awesome-Long-Horizon-Agents)

### Inferences
- The ACL-DOI citation is most likely a bibliographic error (conflating Memory-R2 with Memory-R1, whose ACL 2026 long paper is 2026.acl-long.583). Treat venue as "NeurIPS 2026 (per authors and a digest; proceedings not checked)".
- Affiliations: Yan/Tresp/Ma are LMU Munich / MCML [prior knowledge]; Schwarzmann and Trivisonno are, to my knowledge, Huawei Munich Research Center researchers (networking/6G) [prior knowledge] — consistent with the claimed Huawei involvement, but no fetched source confirmed it. A related Tresp/Ma-group paper on "unbiased collective memory for agentic 6G cross-domain management" (arXiv 2509.26200) appeared in search, hinting at a Huawei-6G collaboration line — [arXiv 2509.26200](https://arxiv.org/pdf/2509.26200).

### Gaps
- Official NeurIPS 2026 proceedings / OpenReview page not checked (network blocked). Exact affiliation footnotes not read.

## Q2. Memory-R2: problem formulation, architecture, memory operations, reward, horizon

### Takeaway
Memory-R2 keeps Memory-R1's "build memory from a multi-session dialogue, then answer questions with a fixed answer agent" setup, but (i) unifies a fact **extractor** and a **memory manager** in one shared policy (Qwen2.5-7B-Instruct, role prompts), (ii) trains over long horizons (up to 32 sessions), and (iii) uses the downstream QA outcome as the trajectory reward.

### Cited Findings
- "A fact extractor and a memory manager are instantiated from the same LLM backbone through role-specific prompts"; the extractor turns dialogue chunks into atomic facts and the manager issues memory operations (README wording: INSERT / UPDATE / DELETE) — [arXiv HTML v1](https://arxiv.org/html/2605.21768v1); [GitHub README](https://github.com/ahmedehabb/Memory-R2)
- The README frames the single model as both "memory extractor and memory manager" (paper: jointly learning memory *formation* and *evolution*) — [GitHub](https://github.com/ahmedehabb/Memory-R2); [arXiv abs](https://arxiv.org/abs/2605.21768)
- Extractor and manager share a Qwen2.5-7B-Instruct backbone and are jointly trained; **the answer agent is held fixed during training**; GPT-OSS-120B is used as a fixed answer agent; results averaged over three runs — [arXiv HTML v1](https://arxiv.org/html/2605.21768v1)
- Curriculum over session horizon 8 → 16 → 32 sessions "to stabilize multi-step RL"; the 32-session checkpoints are the ones evaluated zero-shot OOD — [arXiv abs](https://arxiv.org/abs/2605.21768); [GitHub](https://github.com/ahmedehabb/Memory-R2)
- Training data: the authors state the gains come from training on only **two LoCoMo conversations** — [arXiv 2605.21768](https://arxiv.org/html/2605.21768)

### Inferences
- Reward: the snippets say "trajectory-level global reward" from QA performance; given lineage from Memory-R1 (EM-based outcome reward on the answer agent's output) it is very likely answer correctness (EM/F1-style) of the fixed answer agent on questions about the conversation. Not verified.
- Operation set: Memory-R1 used ADD/UPDATE/DELETE/NOOP [prior knowledge]; the Memory-R2 README says INSERT/UPDATE/DELETE. Whether NOOP is an explicit action in R2 is unconfirmed.
- Freezing the answer agent turns the problem into "write-side" credit assignment only: every bit of reward variance must be attributed to extraction/management decisions made sessions earlier.

### Gaps
- Exact reward formula (EM vs F1 vs LLM-judge; per-question averaging; any format/length penalties), session chunking, memory-bank size limits, retrieval mechanism for the answer agent (top-k?), and whether NOOP exists — not readable this session.

## Q3. Why trajectory-level reward is "unfair": the paper's argument

### Takeaway
The paper's fairness argument is about **comparison-group validity in GRPO**: once rollouts take different write/update/delete actions, their intermediate memory states diverge, so later decisions are effectively made in different "environments", yet GRPO normalizes all trajectory rewards in one group. Fair credit = comparing alternative memory operations from an **identical** intermediate memory state.

### Cited Findings
- Paper wording (via snippet): "Once different rollouts write, update, or delete different memories, they no longer share the same intermediate memory state, making trajectory-level comparisons fundamentally unfair. Consequently, trajectory-level rewards provide noisy or biased credit signals for long-horizon memory operations." — [arXiv HTML v1](https://arxiv.org/html/2605.21768v1)
- Also: "different rollouts induce different intermediate memory states as their effective environments, yet GRPO normalizes their rewards within the same comparison group" — [arXiv PDF](https://arxiv.org/pdf/2605.21768)

### Inferences
- This is essentially a state-conditioned baseline argument: GRPO's group mean is a baseline for V(s_0) only; for a decision at session k the appropriate baseline is V(m_k), which differs across rollouts. Local rerollouts estimate a per-state baseline Monte-Carlo style (cf. VinePPO, tree/branching-rollout credit methods like TreeRL / SPO / Tree-GRPO — [prior knowledge]).
- "Noisy" = high variance from attributing to session-k action the luck of all other sessions; "biased" = rollout-dependent state shift (a good write later in a rollout that was already ruined earlier gets negative advantage).

### Gaps
- Whether the paper gives a formal definition (e.g., a theorem on variance/bias of advantage estimates) is unknown.

## Q4. LoGo-GRPO: global + local advantages, rerollouts, group sizes, overhead

### Takeaway
LoGo-GRPO = standard trajectory-level GRPO advantages over n global rollouts **plus** session-wise local advantages obtained by snapshotting the memory state at sampled sessions and re-rolling that session (or suffix) multiple times from the identical snapshot; both signals feed one GRPO-style objective, with the per-step advantage broadcast to all tokens of that step. Ablation: LoGo-GRPO 49.67 F1 vs plain GRPO 46.62 F1 on LoCoMo.

### Cited Findings
- "preserves a trajectory-level global reward for end-to-end long-horizon optimization, while additionally introducing session-wise attribution signals and local rerollouts that compare trajectories starting from identical intermediate memory states" — [arXiv HTML](https://arxiv.org/html/2605.21768)
- The global branch "computes group-relative advantages across the n global rollouts"; the local branch "performs rerollouts from shared intermediate memory states"; "each step's advantage is then used as the token-level advantage for all tokens in step u" — [arXiv PDF](https://arxiv.org/pdf/2605.21768)
- README: snapshots from sampled sessions are reused for local rerollouts; both signals feed a single GRPO-style objective — [GitHub](https://github.com/ahmedehabb/Memory-R2)
- Ablation (LoCoMo): replacing LoGo-GRPO with standard GRPO lowers F1 49.67 → 46.62 and BLEU-1 43.77 → 40.97 — [arXiv HTML](https://arxiv.org/html/2605.21768)
- Third-party framing: "Local re-rollouts from identical memory states to enable fair credit assignment" — [Paper Digest NeurIPS 2026](https://www.paperdigest.org/2026/10/neurips-2026-papers-highlights/)

### Inferences
- Likely combination: A_total(step u) = A_global(trajectory) + λ·A_local(session u), with A_local computed by group-normalizing rewards of K rerollouts from the same snapshot. The weighting λ, whether local reward is the final QA reward (requires completing the remaining sessions) or a session-local proxy ("session-wise attribution signals"), is unknown. If rerollouts are run to the end of the horizon, compute scales ≈ n + S_sampled·K·(remaining sessions) episodes per prompt — the overhead question is central and unanswered.
- Core assumption: **resettable/serializable environment state** — the memory bank is a plain data structure that can be snapshotted and restored exactly. This holds for text-memory banks but not for real-world environments with side effects.

### Gaps
- Exact equations, λ/weighting, global group size n, local group size K, number of sampled sessions per trajectory, whether local rollouts continue to horizon end, wall-clock/GPU-hours overhead vs GRPO — all unread (network-blocked). Hardware/GPU count not found.

## Q5. Memory-R2: experimental setup, numbers, generalization

### Takeaway
On LoCoMo, Memory-R2 (Qwen2.5-7B extractor/manager) reports the best overall F1/BLEU-1 in a backbone-controlled comparison against training-free (Mem0, A-Mem, etc.) and trained (MEM1, MemAgent, Memory-R1) baselines; headline vs Memory-R1: F1 43.14 → 50.60, B1 36.44 → 44.01, judge 80.99. Most of the gain comes from training the memory module, not the answer agent. Zero-shot OOD: LongMemEval-oracle, LongMemEval-s, MSC-Self-Instruct, MemBench.

### Cited Findings
- vs Memory-R1 on LoCoMo: overall F1 43.14 → 50.60, BLEU-1 36.44 → 44.01, LLM-judge 80.99 — [arXiv HTML](https://arxiv.org/html/2605.21768)
- "In the backbone-controlled setting, Memory-R2 achieves the best overall F1 and BLEU-1 among all training-free and trained baselines, including MEM1, MemAgent, and Memory-R1" — [arXiv HTML v1](https://arxiv.org/html/2605.21768v1)
- Gain decomposition: training the memory module raises F1 26.4 → 45.2 with a 7B-Base answer agent, and 30.6 → 49.7 with a GPT-OSS(-120B) answer agent — [arXiv HTML v1](https://arxiv.org/html/2605.21768v1)
- An RL-tuned 7B answer-agent variant beats the GPT-OSS-120B answer-agent variant on F1 and B1 — [arXiv 2605.21768](https://arxiv.org/html/2605.21768)
- Zero-shot OOD on LongMemEval-oracle, LongMemEval-s, MSC-Self-Instruct, MemBench with consistent improvement over the base model on all metrics; on LongMemEval-oracle F1 reportedly 27.88 → 50.60 vs base — [arXiv 2605.21768](https://arxiv.org/html/2605.21768)
- Generalizes 3B → 7B and across answer agents, with the largest gains on the smaller model (README) — [GitHub](https://github.com/ahmedehabb/Memory-R2)
- Independent re-evaluation: Uni-Mem (arXiv 2610.29671, Oct 2026, Huawei Noah's Ark-credited) re-evaluates RL memory baselines from "officially released checkpoints using our Mem0-aligned judge prompt" and claims to beat "the strongest RL-based memory" on LoCoMo and LongMemEval — [Uni-Mem arXiv HTML](https://arxiv.org/html/2610.29671); [arXiv abs](https://arxiv.org/abs/2610.29671)

### Inferences
- Number consistency check: 49.67 (ablation, full LoGo) ≈ 49.7 (GPT-OSS answer agent) suggests the ablation was run with the GPT-OSS-120B answer agent; 50.60 (headline vs Memory-R1) may be the RL-tuned-7B-answer-agent configuration. The LongMemEval-oracle "→ 50.60" figure coincides exactly with the LoCoMo headline and may be a search-synthesis conflation — **verify before quoting**.
- Training on two LoCoMo conversations (LoCoMo has only 10 conversations total [prior knowledge]) means the held-out test is tiny in number of distinct conversations; variance across conversations matters (they average 3 runs, but seeds ≠ conversation diversity).
- Fairness of comparisons: "backbone-controlled" suggests baselines were re-run with the same backbone; Uni-Mem's judge-prompt re-alignment shows LLM-judge (J) numbers are prompt-sensitive, so J comparisons across papers are fragile. F1/BLEU-1 on LoCoMo are lexical and penalize verbose correct answers.
- Not tested (as far as snippets show): non-dialogue memory tasks (web/agentic tool use), real resets-impossible environments, compute-matched comparison vs. GRPO with more global rollouts (cf. our own E7 lesson: extra rollouts alone can explain gains), per-category LoCoMo breakdown.

### Gaps
- Full Table 1 rows (Mem0, A-Mem, LangMem, Zep, MemGPT/Letta, MemoryOS, MEM1, MemAgent numbers), per-category scores, judge model, Table numbers/section references, training compute, stated limitations section — none retrievable.

## Q6. Memory-R1 (arXiv 2508.19828) and what Memory-R2 changes

### Takeaway
Memory-R1 trains two separate agents with outcome RL (PPO or GRPO): a Memory Manager choosing ADD/UPDATE/DELETE/NOOP per extracted fact and an Answer Agent that filters retrieved memories ("memory distillation") and answers; it uses only 152 LoCoMo QA pairs and reports large gains over Mem0/MemoryOS. Memory-R2 changes: shared extractor+manager policy, frozen answer agent, long horizons with curriculum, and LoGo-GRPO local rerollouts for per-session credit.

### Cited Findings
- Two RL-fine-tuned agents (PPO or GRPO); Memory Manager does structured ops ADD/UPDATE/DELETE/NOOP; Answer Agent selects relevant entries and reasons over them; trained with as few as 152 QA pairs; GRPO generally better than PPO — [arXiv 2508.19828](https://arxiv.org/pdf/2508.19828); [arXiv v5](https://arxiv.org/html/2508.19828v5)
- Published as ACL 2026 long paper (2026.acl-long.583) — [ACL Anthology](https://aclanthology.org/2026.acl-long.583.pdf)
- Headline relative gains differ across versions: v1/ACL text: +28.5% F1, +34.0% BLEU-1, +30.2% judge (LLaMA-3.1-8B, vs MemoryOS); v2: +68.9% F1 / +48.3% B1 / +37.1% J vs Mem0 (LLaMA) and +57.3% / +41.5% / +33.8% vs Mem0 (Qwen-2.5-7B, GRPO); GitHub README: +48% F1, +69% BLEU-1, +37% J vs prior best — [arXiv v1](https://arxiv.org/html/2508.19828v1); [arXiv v2](https://arxiv.org/html/2508.19828v2); [GitHub yansikuan/memory-r1](https://github.com/yansikuan/memory-r1)
- v1 table overall F1: LLaMA-3.1-8B Mem0 31.41 → Memory-R1 45.20; Qwen-2.5-7B Mem0 38.44 → 46.70 (search-synthesis reading of v1 table) — [arXiv v1](https://arxiv.org/html/2508.19828v1)
- Secondary summary: Memory-R1 45.02 F1 / 37.51 BLEU-1 vs MemoryOS 35.04 / 27.99 (unverified secondary) — [themoonlight review](https://www.themoonlight.io/en/review/memory-r1-enhancing-large-language-model-agents-to-manage-and-utilize-memories-via-reinforcement-learning)
- Memory-R1's LongMemEval evaluation is zero-shot — [arXiv 2605.21768 related-work snippet / Memory-R1](https://arxiv.org/pdf/2508.19828)
- Memory-R2 reports Memory-R1 at 43.14 F1 / 36.44 B1 on LoCoMo in its own setting — [arXiv HTML](https://arxiv.org/html/2605.21768)

### Inferences
- Memory-R1's Memory Manager reward = whether the (frozen) answer agent answers correctly (EM) after the operation, i.e., an outcome reward with effectively short horizon per update [prior knowledge, consistent with "outcome-driven" framing]. Memory-R2 identifies exactly this as the weakness at long horizons: a single terminal reward over 32 sessions of writes.
- The headline-percentage inconsistency across Memory-R1 versions (different baselines) is a caution for citing either paper's relative gains; quote absolute F1/B1/J with version.

### Gaps
- Exact Memory-R1 reward definition and data split (I recall 1:1:8 train/val/test over LoCoMo conversations [prior knowledge]) not re-verified.

## Q7. Non-RL memory systems (brief)

### Takeaway
The pre-RL generation hand-designs memory pipelines (LLM-prompted extraction + vector/graph store + heuristic update); RL work trains the write/read policy instead. Benchmarks (LoCoMo) numbers for these systems are contested.

### Cited Findings
- A-Mem: "Agentic Memory for LLM Agents", arXiv 2502.12110 (Zettelkasten-style linked notes) — [arXiv 2502.12110](https://arxiv.org/pdf/2502.12110)
- Mem0: arXiv 2504.19413 (LLM extracts facts, ADD/UPDATE/DELETE/NOOP against vector store; graph variant) — [MemoryAgentBench/ICLR refs](https://proceedings.iclr.cc/paper_files/paper/2026/file/fd1eff9dd295df50a41f2521942fa31d-Paper-Conference.pdf)
- Zep disputes Mem0's LoCoMo reporting: Zep reran itself at 75.14% vs 65.99% reported by Mem0, attributing the gap to implementation errors (vendor, interested party) — [Zep blog](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- Mem-T reports beating A-Mem and Mem0 by up to 14.92% — [arXiv 2601.23014](https://arxiv.org/abs/2601.23014)

### Inferences ([prior knowledge], IDs not re-verified)
- MemGPT / Letta (arXiv 2310.08560): OS-style virtual context with paging between main context and archival/recall storage via function calls.
- MemoryBank (arXiv 2305.10250): long-term memory with Ebbinghaus-curve forgetting.
- Zep (arXiv 2501.13956): temporal knowledge graph (Graphiti) memory.
- MemoryOS (arXiv 2506.06326): hierarchical short/mid/long-term storage inspired by OS memory management.
- MIRIX (arXiv 2507.07957): multi-agent, six memory types (core, episodic, semantic, procedural, resource, knowledge vault).
- LangMem: LangChain library for extraction/procedural memory (no paper).

### Gaps
- No fetched primary sources for MemGPT/Zep/MemoryOS/MIRIX details this session.

## Q8. RL for memory / context management (2025–2026)

### Takeaway
Two families: (a) **context management for a single long task** (MEM1, MemAgent, ReSum, Context-Folding/AgentFold, MemSearcher, MemPO) — the agent rewrites its own working memory; (b) **persistent memory construction across sessions** (Memory-R1/R2, Mem-α, Mem-T, Uni-Mem, AgeMem, Memex(RL)). Credit assignment for memory writes is now an explicit 2026 theme (Mem-T's MoT-GRPO, tree-based credit, Hindsight Memory-PRM, LoGo-GRPO).

### Cited Findings
- Mem-α (arXiv 2509.25911): RL for memory construction with hierarchical (core/semantic/episodic) memory; trained on ≤30k-token instances, generalizes to >400k tokens; contrasts with MEM1/MemAgent's single-paragraph memory — [arXiv 2509.25911](https://arxiv.org/pdf/2509.25911)
- MemAgent ("Reshaping long-context LLM with multi-conv RL-based memory agent"), listed as ICLR 2026; MEM1 ("Learning to synergize memory and reasoning for efficient long-horizon agents") — [Mem-α refs](https://arxiv.org/pdf/2509.25911)
- ReSum (arXiv 2509.13313): "Unlocking Long-Horizon Search Intelligence via Context Summarization" — [Mem-T / search refs](https://arxiv.org/pdf/2601.23014)
- AgentFold: "Long-horizon web agents with proactive context management" — [Mem-α/Mem-T refs](https://arxiv.org/pdf/2509.25911)
- MemSearcher (arXiv 2511.02805): end-to-end RL to reason, search and manage memory — [Mem-T](https://arxiv.org/pdf/2601.23014)
- Mem-T (arXiv 2601.23014, Jan 2026): MoT-GRPO, "tree-guided RL framework that turns sparse terminal feedback into dense, step-wise supervision through memory operation tree backpropagation and hindsight credit assignment"; jointly optimizes construction and retrieval; −24.45% inference tokens vs GAM — [arXiv 2601.23014](https://arxiv.org/abs/2601.23014)
- MemRL (arXiv 2601.03192): self-evolving agents via runtime RL on episodic memory; Memex(RL) (arXiv 2603.04257): indexed experience memory for long-horizon agents — [search refs](https://arxiv.org/pdf/2605.13486)
- MemPO (arXiv 2603.00680): Self-Memory Policy Optimization for long-horizon agents — [arXiv 2603.00680](https://arxiv.org/pdf/2603.00680)
- Tree-based Credit Assignment for Multi-Agent Memory System (arXiv 2605.04811) — [arXiv 2605.04811](https://arxiv.org/pdf/2605.04811)
- Hindsight Memory-PRM: supervising memory (arXiv 2608.29605) — [arXiv 2608.29605](https://arxiv.org/pdf/2608.29605)
- Learning to Remember: end-to-end training of memory agents for long-context reasoning (arXiv 2602.18493) — [arXiv 2602.18493](https://arxiv.org/pdf/2602.18493)
- Uni-Mem (arXiv 2610.29671): unified construction + retrieval memory manager with online RL; "Stage-Decoupled GRPO plus ECA"; argues prior work optimizes construction alone or mixes online retrieval RL with offline construction supervision — [arXiv 2610.29671](https://arxiv.org/abs/2610.29671)
- Meta-Cognitive Memory Policy Optimization for Long-Horizon LLM Agents (arXiv 2605.30159) — [HF papers listing](https://huggingface.co/papers/2605.30159)
- A later survey line describes AgeMem, ReMemR1 and Mem-T as moving toward unified memory actions, revisitable memory, and denser credit assignment — [Mem-T search synthesis](https://arxiv.org/pdf/2601.23014)
- Curated lists for this niche: [Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL](https://github.com/Zenghuang-Fu/Awesome-Long-Horizon-Credit-Assignment-for-Agentic-RL); [tfatykhov/awesome-agent-memory](https://github.com/tfatykhov/awesome-agent-memory)

### Inferences ([prior knowledge], IDs not re-verified)
- MEM1 (arXiv 2506.15841): constant-size internal state rewritten each turn, RL with outcome reward (multi-objective QA / WebShop).
- MemAgent (arXiv 2507.02259): reads long docs chunk-by-chunk, overwrites a fixed-length memory; Multi-Conv DAPO; extrapolates to multi-million-token RULER-HotpotQA.
- Context-Folding (arXiv 2510.11967, "FoldGRPO"): branch/return sub-trajectories folded into summaries, with process rewards.
- Agentic Context Engineering, ACE (arXiv 2510.04618): evolving "playbook" context via generator/reflector/curator — not RL weight training but context optimization.
- AgentFold (arXiv 2510.24699).
- Structural similarity: Mem-T (tree backprop), tree-based credit (2605.04811) and LoGo-GRPO (shared-state rerollouts) all use **branching from shared prefixes/states** to get per-step baselines — the same idea as TreeRL/Tree-GRPO/SPO in reasoning RL. Memory-R2's novelty is mainly in applying it at the session/memory-state level with a global term retained.

### Gaps
- Mechanism-level details for ReSum/AgentFold/Context-Folding/ACE/AgeMem not fetched this session.

## Q9. Benchmarks and their known issues

### Takeaway
LoCoMo (the training/eval set for Memory-R1/R2) is small, has answer-key errors (~6.4%), unscorable adversarial items, judge permissiveness, and fits in modern context windows; LongMemEval and MemoryAgentBench are the stronger 2025–26 alternatives; many 2026 papers question single-score evaluation.

### Cited Findings
- LoCoMo audit (non-peer-reviewed): 99 score-corrupting errors in 1,540 questions (6.4%); LLM judge accepts up to 63% of intentionally wrong answers — [Penfield Labs on dev.to](https://dev.to/penfieldlabs/we-audited-locomo-64-of-the-answer-key-is-wrong-and-the-judge-accepts-up-to-63-of-intentionally-33lg)
- 444 of 446 refusal/adversarial-category LoCoMo questions have empty gold answers (~23% of benchmark unscorable as written); repeated failure to reproduce published LoCoMo numbers with Mem0's runner — [ATANT v1.1, arXiv 2604.10981](https://arxiv.org/pdf/2604.10981)
- LoCoMo/LongMemEval conflate memory with multi-step reasoning; some category labels ambiguous — [arXiv 2604.27906](https://arxiv.org/pdf/2604.27906)
- Zep: LoCoMo lacks knowledge-update questions (vendor critique) — [Zep blog](https://www.getzep.com/blog/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- LongMemEval: 500 questions; categories temporal reasoning, multi-session, knowledge update, single-session user/assistant/preference; LongMemEval-V2 (2026) with 451 questions over multimodal web-agent trajectories — [cognee benchmark guide](https://www.cognee.ai/agent-memory-evaluation-datasets-benchmarks)
- MemoryAgentBench (arXiv 2507.05257, ICLR 2026, UCSD): incremental multi-turn format; four competencies: accurate retrieval, test-time learning, long-range understanding, selective forgetting (conflict resolution) — [ICLR 2026 paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/fd1eff9dd295df50a41f2521942fa31d-Paper-Conference.pdf); [alphaXiv](https://www.alphaxiv.org/abs/2507.05257)
- MemBench (arXiv 2506.21605, ACL Findings 2025): factual and reflective memory, participation/observation scenarios; distinct from "MemoryBench" — [cognee guide](https://www.cognee.ai/benchmark-ai-agent-memory-on-your-own-data)
- Evaluation critiques 2026: "Is One Score Enough? Rethinking the Evaluation of Sequentially Evolving LLM Memory" (arXiv 2605.15384); "Memory Tax: Dissecting the Net Value of Agent Memory" (arXiv 2610.34073); Entity-Collision attribution protocol (arXiv 2605.29630) — [2605.15384](https://arxiv.org/html/2605.15384); [2610.34073](https://arxiv.org/html/2610.34073v1); [2605.29630](https://arxiv.org/pdf/2605.29630)
- Transfer limits: PSD (arXiv 2609.23449) finds LoCoMo-trained gains transfer less fully to multi-session and temporal LongMemEval questions — [arXiv 2609.23449](https://arxiv.org/pdf/2609.23449)

### Inferences
- LoCoMo (10 conversations, ~300 turns each [prior knowledge]) fits in a 128k context window; a full-context baseline is a must-have comparator for any memory-RL paper — whether Memory-R2 includes one is unconfirmed.

### Gaps
- No formal audit of LongMemEval found.

## Q10. Why memory is a hard credit-assignment case

### Takeaway
Memory writes are actions whose payoff is realized only when a later question (possibly many sessions later) retrieves them; the effect is mediated by a persistent, policy-dependent state, so different rollouts face different future environments — making trajectory-level group baselines biased, and per-step rewards unavailable without counterfactual rollouts.

### Cited Findings
- Memory-R2's statement that divergent memory states make trajectory comparisons "fundamentally unfair" and credit "noisy or biased" — [arXiv HTML v1](https://arxiv.org/html/2605.21768v1)
- Mem-T: RL memory approaches "remain limited by sparse rewards and temporal credit assignment in long-horizon settings" — [arXiv 2601.23014](https://arxiv.org/abs/2601.23014)
- Uni-Mem: construction and retrieval must co-adapt; optimizing one side alone limits it — [arXiv 2610.29671](https://arxiv.org/abs/2610.29671)

### Inferences
- Specific difficulties: (1) long delay (write at session 3, query at session 30); (2) counterfactual nature — the value of a DELETE/NOOP is only visible if a future query would have hit the deleted/missing fact; (3) interaction effects (an UPDATE's value depends on later UPDATEs); (4) the answer agent/retriever is part of the "environment", so reward also reflects retrieval noise; (5) sparse question coverage — most stored facts are never queried, so most writes get zero-signal credit.
- Relevance to our ToolCredit project: our turn-level / step-level credit work on math+Python is the short-horizon, stateless-tool analogue; Memory-R2's "rerollout from identical intermediate state" is the memory-domain version of state-anchored baselines. Our E7 finding (gains largely explainable by 2.5× rollouts under equal-step comparison) suggests asking whether LoGo-GRPO's gain over GRPO survives a **rollout-compute-matched** GRPO baseline — a key open critique.

### Gaps
- Whether Memory-R2 reports a compute-matched GRPO baseline is unknown.

## Q11. Sikuan Yan's other 2026 papers (research-direction signal)

### Takeaway
The LMU/MCML (Tresp, Ma) group's 2026 output points to RL/distillation for efficient and self-improving agents: memory RL (Memory-R1 → R2), reasoning compression via mixed-policy distillation, and recursive self-improvement via meta-skill evolution.

### Cited Findings
- "Reasoning Compression with Mixed-Policy Distillation" (arXiv 2605.08776, 9 May 2026): authors Han Yang, Mingyan Wu, Bailan He, Zeyu Cao, Kevin Qinghong Lin, Sikuan Yan, Zifeng Ding; teacher rewrites student-sampled trajectories into concise traces (combining on- and off-policy distillation); on Qwen3-1.7B, up to 27.1% fewer tokens with improved accuracy — [arXiv abs](https://arxiv.org/abs/2605.08776); [alphaXiv overview](https://www.alphaxiv.org/overview/2605.08776); [OPD survey 2604.00626](https://arxiv.org/html/2604.00626v3)
- "MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale Meta-Skill Evolution" (arXiv 2607.05297, 6 Jul 2026): Zefeng Wang, Minxi Yan, Jinhe Bi, Sikuan Yan, Volker Tresp, Yunpu Ma; evolves both task skill and a branch-local meta-skill (Retriever, Allocator, Analyzer, Proposer, Evolver) on a slower outer loop; frozen backbone; +23.54 / +16.09 / +1.92 pts on OfficeQA / SealQA / ALFWorld over No-Skill, +6.38 / +8.05 / +1.92 over single-level evolution — [arXiv PDF](https://arxiv.org/pdf/2607.05297); [alphaXiv](https://www.alphaxiv.org/abs/2607.05297)
- Google Scholar profile exists for Sikuan Yan — [Google Scholar](https://scholar.google.com/citations?user=2DRWdnIAAAAJ&hl=en)

### Inferences
- Sikuan Yan is first author on the memory line and a middle author on the distillation/skill-evolution lines, suggesting his own agenda is memory-RL, with the group broadly on agent self-improvement (weights via RL, or context/skills with frozen backbones).

### Gaps
- MetaSkill-Evolve backbone (one secondary summary says Gemma-4 31B) unverified.
