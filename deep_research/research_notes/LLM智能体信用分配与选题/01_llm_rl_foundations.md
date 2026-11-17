# Foundations: LLM RL from RLHF to RLVR/GRPO, and the classic credit-assignment toolbox

> Research date: 2026-11-17. Verification method: `arxiv.org` and `export.arxiv.org` were **not reachable** from this session (proxy/DNS refusal), so arXiv IDs were verified by web search results that returned the arXiv URL together with the exact title (marked **[V]**). IDs marked **[M]** come from prior knowledge and were *not* re-verified in this session; they are canonical, widely cited IDs but the report writer should spot-check them before final publication.
>
> Notation used below: prompt q, G sampled responses o_1..o_G, scalar outcome reward r_i ∈ {0,1} (or similar), token index t, ratio ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_old(o_{i,t}|q,o_{i,<t}).

---

## Q1. RLHF era: InstructGPT/PPO (reward model + critic) and DPO-style offline alternatives; costs and why critic-free methods took over

### Takeaway
Classic RLHF (InstructGPT) runs PPO with four models in memory (policy, reference, learned reward model, learned value/critic) and uses GAE for per-token credit; DPO removes the RM and the RL loop entirely but becomes offline/off-policy. For reasoning with *checkable* answers, the field moved to online, critic-free policy-gradient methods (GRPO/RLOO/REINFORCE++) because the critic was expensive and—specifically for long chain-of-thought—poorly calibrated, so its per-token credit was not worth its cost.

### Cited Findings
- **Deep RL from human preferences** (Christiano et al., 2017) introduced learning a reward model from pairwise human preferences and optimizing a policy against it. arXiv 1706.03741 **[M]** — https://arxiv.org/abs/1706.03741
- **InstructGPT** — "Training language models to follow instructions with human feedback" (Ouyang et al., Mar 2022), arXiv 2203.02155 **[M]** — https://arxiv.org/abs/2203.02155. Pipeline: SFT → reward model trained on human rankings → PPO against RM with a per-token KL penalty to the SFT model (PPO-ptx mixes in pretraining gradients). The RM gives a single scalar at the end of the response; the PPO critic + GAE is what spreads that scalar across tokens.
- **PPO** — "Proximal Policy Optimization Algorithms" (Schulman et al., 2017), arXiv 1707.06347 **[M]** — https://arxiv.org/abs/1707.06347. Clipped surrogate min(ρA, clip(ρ,1−ε,1+ε)A); in LLM RLHF, A comes from GAE over a learned value head.
- **DPO** — "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., May 2023), arXiv 2305.18290 **[M]** — https://arxiv.org/abs/2305.18290. Closed-form reparameterization of the KL-regularized RLHF optimum lets the policy be trained with a classification-style loss on preference pairs; no RM, no critic, no sampling during training. Offline descendants include IPO (2310.12036 **[M]**), KTO (2402.01306 **[M]**), SimPO (2405.14734 **[M]**).
- **Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study** (Xu et al., 2024), arXiv 2404.10719 **[M]** — https://arxiv.org/abs/2404.10719 — reports that a well-tuned PPO beats DPO across tasks, especially code (competitive programming), and attributes DPO weaknesses partly to distribution shift between the preference data and the policy. (Content from prior knowledge.)
- VC-PPO authors explicitly note that PPO's failure on long-CoT "has led some to abandon PPO in favor of GRPO," even though PPO retains the theoretical advantage of token-level feedback via the value model — [VC-PPO, arXiv 2503.01491 [V]](https://arxiv.org/abs/2503.01491)
- VAPO (ByteDance Seed) names three key challenges for value-based methods on long-CoT: **value model bias, heterogeneous sequence lengths, and reward sparsity**; it reports vanilla PPO scoring ~5 on AIME24 vs. 60.4 for VAPO (Qwen2.5-32B base) — [VAPO, arXiv 2504.05118 [V]](https://arxiv.org/abs/2504.05118)
- REINFORCE++ is formally PPO with the critic removed, GAE λ=γ=1, and a global-batch normalization used as the baseline — [REINFORCE++, arXiv 2501.03262 [V]](https://arxiv.org/abs/2501.03262)
- **Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs** (Ahmadian et al., Cohere, Feb 2024), arXiv 2402.14740 **[M]** — https://arxiv.org/abs/2402.14740. Argues PPO's machinery (critic, clipping) is unnecessary for RLHF because the LLM policy starts from a strong SFT init and the effective "environment" is a bandit (whole response = one action); proposes **RLOO** (REINFORCE Leave-One-Out: sample k responses, baseline for each is the mean reward of the other k−1). (Mechanism from prior knowledge; search did not return this paper.)

### Inferences
- Cost picture (memory/compute): PPO-RLHF = 4 large models (actor, ref, RM, critic) + critic training; GRPO/RLOO = actor + ref (+ rule-based verifier, no neural RM in RLVR) — roughly halving the large-model footprint. DPO = actor + ref, no generation, but offline.
- The decisive argument against the critic for *reasoning* was not only cost; it was that under sparse terminal reward and very long responses, the critic's per-token estimates were biased (init bias, signal decay; see VC-PPO), so GAE's credit was *wrong*, not merely noisy. Critic-free methods give up token-level credit entirely in exchange for an unbiased (but high-variance, uniform) sequence-level signal. This trade is exactly the gap later credit-assignment work tries to close.

### Gaps
- Direct quantitative cost comparison (GPU-hours / memory) PPO vs GRPO from a primary source was not retrieved in this session.
- RLOO paper content (2402.14740) not re-fetched; mechanism stated from prior knowledge.

---

## Q2. RLVR: DeepSeekMath/DeepSeek-R1 GRPO; what "verifiable reward" means; group-normalized advantages and why every token gets the same advantage

### Takeaway
RLVR replaces the learned reward model with a programmatic checker (exact-match math answer, unit tests), giving a sparse terminal reward that is hard to hack. GRPO computes, per prompt, A_i = (r_i − mean(r_1..r_G)) / std(r_1..r_G) and assigns that **same scalar to every token t of response o_i** — the critic's per-token baseline is replaced by a per-prompt group baseline, so the token dimension carries no credit information at all.

### Cited Findings
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** (Shao et al., Feb 2024), arXiv 2402.03300 **[M]** — https://arxiv.org/abs/2402.03300. Introduces **GRPO**: drop the value model; for each q sample G outputs; objective (1/G) Σ_i (1/|o_i|) Σ_t [min(ρ_{i,t} Â_{i,t}, clip(ρ_{i,t},1−ε,1+ε) Â_{i,t}) − β D_KL(π_θ‖π_ref)]. Under **outcome supervision**, Â_{i,t} = r̃_i = (r_i − mean(r))/std(r) for *all* t. DeepSeekMath also defines a **process-supervision** variant where Â_{i,t} = sum of normalized step rewards from steps ending at or after t — i.e., GRPO was designed to accept denser credit if a PRM is available, but the R1 line uses only outcome reward. (Formulas from prior knowledge of the paper.)
- **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** (DeepSeek-AI, Jan 2025), arXiv 2501.12948 **[M]** — https://arxiv.org/abs/2501.12948 (later published in *Nature*, Sept 2025 **[M]**). R1-Zero applies GRPO directly to DeepSeek-V3-Base with **rule-based accuracy reward + format reward**; DeepSeek explicitly avoided neural outcome/process reward models citing reward hacking and pipeline complexity. Reported emergent long CoT, "aha moment" self-reflection.
- Term "RLVR" popularized by **Tülu 3: Pushing Frontiers in Open Language Model Post-Training** (Lambert et al., AI2, Nov 2024), arXiv 2411.15124 **[M]** — https://arxiv.org/abs/2411.15124: "Reinforcement Learning with Verifiable Rewards" = reward is 1 if a deterministic verifier accepts the completion (math answer match, instruction-constraint check), else 0.
- GSPO's authors frame the core mismatch: reward is assigned to the whole sequence, so the off-policy correction/optimization unit should also be the sequence — [GSPO, arXiv 2507.18071 [V]](https://arxiv.org/abs/2507.18071); [Qwen blog](https://qwenlm.github.io/blog/gspo)
- Dr. GRPO shows GRPO's two normalizers (1/|o_i| per-response length and 1/std per-group) introduce optimization biases — [Understanding R1-Zero-Like Training, arXiv 2503.20783 [V]](https://arxiv.org/abs/2503.20783)
- Edge case: when all G responses get the same reward (all-correct or all-wrong), every Â is 0 and the prompt contributes no gradient; DAPO's dynamic sampling and Scaf-GRPO both target this ("all sampled answers fail … collapses the advantage calculation to zero") — [DAPO, 2503.14476 [V]](https://arxiv.org/abs/2503.14476); [Scaf-GRPO, 2510.19807 [V]](https://arxiv.org/abs/2510.19807)

### Inferences
- Why uniform: GRPO's advantage is a function of (q, i) only — it is a *bandit* advantage of the whole response vs. siblings. Each token's gradient ∇log π(o_{i,t}) is scaled by the same Â_i, so a correct-answer trajectory with a wasted/erroneous tool call gets that call reinforced, and a failed trajectory with an excellent early step gets that step penalized. Only the statistics across many samples (tokens that co-occur with success more often) slowly disentangle credit.
- In multi-turn tool use, the broadcast spans *interleaved* model tokens and (masked) tool-observation tokens across turns: every assistant turn inherits the final answer's advantage regardless of which turn was decisive. This is the "trajectory-level advantage broadcast to every token/turn" limitation targeted by later turn-level/step-level methods.

### Gaps
- DeepSeekMath and R1 PDFs were not re-fetched (arxiv blocked); the GRPO/process-supervision formula is from prior knowledge of the paper.

---

## Q3. 2025–2026 GRPO variants: what each fixes and what it says about credit assignment

### Takeaway
Almost all 2025 variants fix **optimization pathologies** (clipping asymmetry, length/std normalization bias, importance-ratio noise, zero-variance groups, off-policy/precision mismatch) while **keeping a single sequence-level advantage per response**. Only the value-based revival (VC-PPO/VAPO) and a minority of 2026 segment-level methods actually change *token/step credit*. GSPO goes further in the opposite direction—making the ratio sequence-level too.

### Cited Findings (chronological-ish table)

| Method | arXiv / date | One-line mechanism | Credit-assignment stance |
|---|---|---|---|
| RLOO (Back to Basics) | 2402.14740 **[M]**, Feb 2024 | k samples per prompt; baseline for sample i = mean reward of the other k−1 (unbiased leave-one-out) | Treats whole response as one action (bandit); explicitly argues token-level modeling unnecessary for RLHF |
| REINFORCE++ | 2501.03262 **[V]**, Jan 2025 (v9 Nov 2025) | Critic-free; advantage normalized over the **global batch** instead of per-prompt group; proves GRPO's prompt-level normalization is a biased estimator whose bias vanishes with batch size; variant with group-mean subtraction then global norm — [arXiv](https://arxiv.org/abs/2501.03262) | Equivalent to PPO with γ=λ=1 → per-token advantage = return-to-go = final reward; uniform credit |
| VC-PPO | 2503.01491 **[V]**, 3 Mar 2025 | Value pretraining (fix init bias) + **decoupled GAE** (λ_critic=1 so reward propagates back; separate λ for actor) — [arXiv](https://arxiv.org/abs/2503.01491) | Diagnoses *value init bias* and *reward signal decay* as why PPO's token-level credit fails on long CoT |
| DAPO | 2503.14476 **[V]**, 18 Mar 2025, NeurIPS 2025 | **Clip-Higher** (decoupled ε_low/ε_high, avoids entropy collapse); **Dynamic Sampling** (drop prompts with all-0/all-1 group accuracy, oversample until batch full); **Token-level PG loss** (average over all tokens in batch, not per-sample then per-group); **Overlong reward shaping** (soft length penalty / overlong filtering). 50 AIME24 on Qwen2.5-32B with half R1-Zero-Qwen-32B steps — [arXiv](https://arxiv.org/abs/2503.14476), [project](https://dapo-sia.github.io), [NeurIPS](https://papers.nips.cc/paper_files/paper/2025/hash/a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html) | Advantage still uniform per response; token-level loss changes *weighting* (long responses' tokens no longer down-weighted), not credit |
| Dr. GRPO | 2503.20783 **[V]**, 26 Mar 2025 (v2 Oct 2025) | Remove 1/|o_i| (length) and 1/std (difficulty) normalization; GRPO otherwise inflates length esp. of incorrect outputs; 43.3% AIME24 with a 7B base — [arXiv](https://arxiv.org/abs/2503.20783), [code](https://github.com/sail-sg/understand-r1-zero) | Shows normalization choices bias *which sequences* get more gradient per token; credit still uniform |
| VAPO | 2504.05118 **[V]**, 7 Apr 2025 | Value-based PPO revival: value pretraining, decoupled GAE, **length-adaptive GAE**, plus DAPO tricks; 60.4 AIME24 on Qwen2.5-32B, +10 over DAPO — [arXiv](https://arxiv.org/abs/2504.05118) | The main 2025 argument *for* learned token-level credit; addresses value bias, heterogeneous lengths, sparse reward |
| MiniMax-M1 / **CISPO** | 2506.13585 **[V]**, Jun 2025 | Clip the (stop-gradient) **IS weight** rather than the update, so every token keeps a gradient; motivated by low-probability "reflective" tokens ("Recheck", "Aha") being clipped out; claims 2× speedup vs DAPO; full RL 3 weeks on 512 H800 (~$534,700) — [arXiv](https://arxiv.org/abs/2506.13585) | Uniform advantage; argues clipping silently removes rare-but-important tokens from learning (a token-*inclusion*, not credit, fix) |
| GSPO (Qwen) | 2507.18071 **[V]**, Jul 2025 | Sequence-level importance ratio (length-normalized sequence likelihood ratio) with sequence-level clipping; more tolerant to precision mismatch, removes need for MoE Routing Replay; used for Qwen3 — [arXiv](https://arxiv.org/abs/2507.18071), [blog](https://qwenlm.github.io/blog/gspo) | Explicitly aligns *off-policy correction* with the sequence-level reward — doubles down on sequence-level unit; no token credit |
| ScaleRL | 2510.13786 **[V]**, Oct 2025, ICLR 2026 (oral) | >400k GPU-hours study; sigmoidal compute–performance fits separating asymptote A from efficiency B; recipe = PipelineRL async, forced length interruptions, CISPO (truncated IS) loss, prompt-level loss averaging, **batch-level advantage normalization**, FP32 logits; CISPO & GSPO beat DAPO on asymptote, CISPO most robust — [arXiv](https://arxiv.org/abs/2510.13786), [ICLR](https://iclr.cc/virtual/2026/oral/10010608) | Most algorithmic knobs (normalization, aggregation, curriculum) shift **efficiency not asymptote**; no credit-assignment component |
| SAPO (Qwen) | 2511.20347 **[V]**, Nov 2025 | Replace hard clip with a temperature-controlled **soft gate**; down-weights only offending tokens; asymmetric temperatures (negative tokens decay faster); used for Qwen3-VL — [arXiv](https://arxiv.org/abs/2511.20347), [docs](https://docs.qwenlm.ai/sapo/index.html) | Token-adaptive *trust region*, not token-adaptive *credit* |
| Trust Region Masking | 2512.23075 **[V]**, Dec 2025 | For long-horizon LLM RL, token-level clip/mask cannot bound worst-case error; mask whole sequences by max token divergence; flags length bias — [arXiv](https://arxiv.org/abs/2512.23075v2) | Off-policy stability for long horizons; credit untouched |
| Scaf-GRPO | 2510.19807 **[V]**, ICLR 2026 (per search summary) | When all samples fail (zero advantage), inject hierarchical off-policy hints ("scaffolds") while keeping on-policy GRPO — [arXiv](https://arxiv.org/abs/2510.19807) | Attacks *signal absence* on hard prompts, not distribution of credit |
| GRPO-VPS | 2604.20659 **[V-ID; venue unverified]**, 2026 | Verifier-free, model-free **segment-level** targeted policy optimization on top of GRPO for intermediate reasoning steps — [arXiv](https://arxiv.org/abs/2604.20659) | One of the 2026 variants that directly targets intermediate-step credit (details only from search snippet; search summary said "ICLR 2026", which is doubtful for an April-2026 arXiv ID — treat as unverified) |
| μ-GRPO | 2605.17570 **[V-ID only]**, May 2026 | "How Off-Policy Can GRPO Be?" — off-policy efficiency — [arXiv](https://arxiv.org/html/2605.17570v1) | Not about credit (title only seen) |
| First-principles survey | 2606.16733 **[V-ID]**, Jun 2026 | Derives REINFORCE→PPO→GRPO→post-GRPO; organizes variants on "trajectory side" vs "reward side"; lists e.g. HT-GRPO (phase-differentiated advantage weighting) — [arXiv](https://arxiv.org/abs/2606.16733) | Useful taxonomy source for the survey |

Additional 2026 items surfaced (title/ID only, not read): "Probing RLVR Training Instability through the Lens of Objective-Level Hacking" [2602.01103](https://arxiv.org/html/2602.01103v2); "FP8-RL" [2601.18150](https://arxiv.org/pdf/2601.18150); "Missing Old Logits in Asynchronous Agentic RL" [2605.12070](https://arxiv.org/pdf/2605.12070); iGRPO [2602.09000](https://arxiv.org/html/2602.09000v1). Training–inference mismatch (vLLM/SGLang vs Megatron/FSDP numerics; MoE routing flips) and fixes (off-policy sequence masking, Masked IS, IcePop, routing replay R2/R3) are documented in [ms-swift docs](https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/training_inference_mismatch.html) and the [slime mismatch blog](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/mismatch/blog-en.md).

### Inferences
- A useful framing for the survey: variants act on three different axes — (1) **advantage estimator** (group mean/std vs leave-one-out vs global batch norm vs learned value) — still one number per response except VAPO/VC-PPO; (2) **loss aggregation/weighting** (per-sample mean vs token-level vs prompt-level) — changes how much each token's *identical* advantage counts; (3) **trust region / off-policy correction** (clip-higher, CISPO, GSPO, SAPO, TRM) — stability. Credit assignment (which tokens/turns deserve the advantage) is mostly orthogonal and largely unaddressed in the mainstream GRPO family.
- ScaleRL's finding that normalization/aggregation/curriculum mainly move efficiency, not asymptote, is consistent with this project's own M10/E7 result (DAPO-style zero-variance filtering: Δ at equal compute −0.39pt, CI spans 0; see `CLAUDE.md` status line) — filtering changes sample efficiency per step, not the information content of the per-response signal.
- DeepSeek-V3.2 (rumored "unbiased KL estimate" + "off-policy sequence masking") could not be confirmed from primary sources in this session.

### Gaps
- DeepSeek-V3.2 tech report ID and its RL changes: not verified (search returned no primary source).
- Kimi k1.5 (2501.12599 **[M]**) and Kimi K2 RL details not checked.
- GMPO (geometric-mean policy optimization, 2507.20673 **[M]**) not verified.
- No authoritative 2026 survey of RLVR algorithms was read beyond the 2606.16733 abstract-level summary.

---

## Q4. Known critiques: capability expansion vs sharpening; entropy collapse; spurious/random rewards on Qwen

### Takeaway
Strong evidence that standard RLVR mainly **reweights** solutions already in the base model's support (better pass@1, worse/equal pass@large-k), that it trades entropy for reward predictably, and that on Qwen2.5-Math even random rewards produce large gains via clipping bias — so "reward went up" is weak evidence that credit was assigned to the right behavior.

### Cited Findings
- **Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?** (Yue et al., 18 Apr 2025; NeurIPS 2025 Oral; ICML'25 AI4MATH best paper), arXiv 2504.13837 **[V]**: RLVR models win at small k but base models win at large k pass@k; reasoning boundary often *narrows* during RLVR; coverage/perplexity suggest abilities originate from and are bounded by the base model; distillation, unlike RLVR, can add new patterns — [arXiv](https://arxiv.org/abs/2504.13837), [NeurIPS](https://neurips.cc/virtual/2025/oral/119945)
- **The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models** (Cui et al., May 2025), arXiv 2505.22617 **[V]**: without entropy/KL intervention, entropy is traded for reward following R = −a·e^H + b; entropy change is driven by the covariance between an action's log-prob and its advantage (high-prob high-advantage actions reduce entropy); proposes Clip-Cov / KL-Cov restricting updates on high-covariance tokens — [HF/arXiv](https://huggingface.co/papers/2505.22617)
- DAPO motivates Clip-Higher by entropy collapse under naive GRPO — [DAPO](https://arxiv.org/abs/2503.14476)
- **Spurious Rewards: Rethinking Training Signals in RLVR** (Shao et al.; arXiv 2506.10947 **[V]**; ICML 2026, PMLR 306): random reward gives +21.4 pts MATH-500 on Qwen2.5-Math-7B vs +29.1 with ground truth (format 13.8, incorrect label 24.1, 1-shot RL 26.0, majority vote 27.1); fails to transfer to Llama3/OLMo2; mechanism attributed to GRPO clipping amplifying high-prior behaviors ("code reasoning" frequency 65% → >90%); disabling clipping removes the gain — [PMLR](https://proceedings.mlr.press/v306/shao26d.html), [ICML](https://icml.cc/virtual/2026/poster/61082)
- Dr. GRPO: base models (e.g., DeepSeek-V3-Base) already exhibit "aha moment" before RL — [arXiv 2503.20783](https://arxiv.org/abs/2503.20783)
- Counterpoint (prior knowledge, **[M]**): ProRL (NVIDIA, arXiv 2505.24864) claims prolonged RL with KL control and reference resets *does* expand pass@k boundaries on some tasks; "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective RLVR" (arXiv 2506.01939 **[M]**) reports that restricting gradients to ~20% high-entropy "forking" tokens matches/exceeds full updates.

### Inferences
- For credit assignment, the "high-entropy forking tokens" result and the Clip-Cov covariance analysis are both implicit token-credit arguments: a minority of tokens carry most of the useful learning signal, yet uniform broadcast spends gradient equally on all.
- Spurious-reward results mean any credit-assignment paper evaluated *only* on Qwen2.5-Math is weak evidence; random-reward and multi-family baselines should be standard controls.

### Gaps
- ProRL and 80/20 IDs/claims not re-verified this session.
- Did not survey 2026 follow-ups to the pass@k debate (e.g., pass@k-as-reward training).

---

## Q5. Classic credit-assignment toolbox and its mapping to LLM settings

### Takeaway
Classic RL has two families of fixes for sparse/delayed reward: **bootstrapping** (TD, eligibility traces, GAE) which needs a good value function, and **hindsight/redistribution** (RUDDER, HCA) which reshapes or reweights the return using information after the fact; multi-agent RL adds **counterfactual** (COMA) and **Shapley** credit across agents. In LLM RL the critic needed by family 1 is exactly what failed on long CoT, while families 2–3 map naturally onto steps/turns (turn = "agent" or "action chunk") and underlie much 2025–2026 turn-level credit work.

### Cited Findings
- **TD learning / eligibility traces** (Sutton 1988; Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. 2018 **[M]**, http://incompleteideas.net/book/the-book-2nd.html): TD(0) updates V(s_t) toward r_t + γV(s_{t+1}), propagating reward back one step per update using a learned estimate (bootstrapping: lower variance, biased by V errors); TD(λ)/eligibility traces blend n-step returns with geometric weights λ, sending a reward back to recently visited states in proportion to a decaying trace. LLM mapping: states = prefixes, actions = tokens; T ~ 10^3–10^4 with reward only at the end, so with γ=1 and λ<1 the signal decays geometrically before reaching early tokens — this is VC-PPO's "reward signal decay" [arXiv 2503.01491](https://arxiv.org/abs/2503.01491).
- **GAE** — "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (Schulman et al., 2015), arXiv 1506.02438 **[M]** — https://arxiv.org/abs/1506.02438. Â_t = Σ_l (γλ)^l δ_{t+l}, δ_t = r_t + γV(s_{t+1}) − V(s_t); λ trades bias (λ→0, trust V) vs variance (λ→1, Monte-Carlo return). LLM mapping: λ=1,γ=1 with no critic collapses to "final reward for every token" (REINFORCE++/GRPO); VC-PPO/VAPO re-tune λ (decoupled actor/critic λ, length-adaptive λ) precisely because standard λ=0.95 starves early tokens — [VAPO](https://arxiv.org/abs/2504.05118), [VC-PPO](https://arxiv.org/abs/2503.01491).
- **RUDDER: Return Decomposition for Delayed Rewards** (Arjona-Medina et al., 2018; NeurIPS 2019), arXiv 1806.07857 **[M]** — https://arxiv.org/abs/1806.07857. Train a sequence model (LSTM) to predict the final return; decompose the prediction into per-step contributions (contribution analysis / differences of successive predictions) and **redistribute** reward so the expected future reward is ~0, making Q-values easy to learn. LLM mapping: a learned return predictor over prefixes ≈ an implicit PRM; redistributing an outcome reward to steps/turns by prediction differences is conceptually what "implicit process reward" methods do (e.g., PRIME, arXiv 2502.01456 **[M]**, derives token rewards as log-ratio differences of an outcome-trained implicit RM).
- **Hindsight Credit Assignment** (Harutyunyan et al., DeepMind, NeurIPS 2019), arXiv 1912.02503 **[V]**: assign credit to past actions by the (learned) likelihood that they led to the observed outcome — a hindsight distribution h(a|s, outcome) compared to π(a|s); value functions can be rewritten this way, giving new policy-gradient estimators — [arXiv](https://arxiv.org/abs/1912.02503), [NeurIPS](https://proceedings.neurips.cc/paper/2019/hash/195f15384c2a79cedf293e4a847ce85c-Abstract.html). LLM mapping: "which steps were informative given that the answer turned out correct" ≈ ratio π(step | prefix, correct outcome)/π(step | prefix); in LLMs this can be approximated by conditioning the model itself on the final answer, but the action space (tokens) is huge, so it is usually applied at step/turn granularity.
- **COMA — Counterfactual Multi-Agent Policy Gradients** (Foerster et al., AAAI 2018), arXiv 1705.08926 **[M]** — https://arxiv.org/abs/1705.08926. Centralized critic; each agent's advantage = Q(s, joint action) − Σ_{a'} π(a'|·) Q(s, (a', other agents' actions)), i.e., counterfactual baseline that marginalizes only that agent's action. LLM mapping: treat each turn/step as an "agent"; counterfactual = resample that turn while keeping others, compare outcomes (Monte-Carlo rollouts from the prefix). VinePPO (arXiv 2410.01679 **[M]**) is the single-agent analogue: replace the critic by MC rollouts from intermediate states to get step values.
- **Shapley-based credit**: e.g., "Shapley Q-value: A Local Reward Approach to Solve Global Reward Games" (Wang et al., AAAI 2020), arXiv 1907.05707 **[M]** — https://arxiv.org/abs/1907.05707: assign each agent its average marginal contribution over coalitions (approximated by sampling). LLM mapping: Shapley over steps/turns/tool calls is principled (efficiency axiom: credits sum to the outcome) but exponential; needs sampling + many rollouts, costly with long generations.
- **Pignatelli et al., "A Survey of Temporal Credit Assignment in Deep Reinforcement Learning"** (TMLR 04/2024), arXiv 2312.01072 **[V]**: formalizes credit as the influence of an action on an outcome learned from limited experience; organizes challenges as **delayed effects**, **transpositions** (many paths reach the same outcome), and **weak action influence** (low signal-to-noise); reviews methods and evaluation/diagnosis protocols — [arXiv](https://arxiv.org/abs/2312.01072), [TMLR via mlanthology](https://mlanthology.org/tmlr/2024/pignatelli2024tmlr-survey). Authors: Pignatelli, Ferret, Geist, Mesnard, van Hasselt, Toni (one indexer lists Pietquin instead of Ferret; arXiv/TMLR list Ferret).

### Inferences
- Mapping table for the survey:
  - Bootstrapping (TD/GAE) → needs critic → failed on long CoT due to init bias + decay (VC-PPO) → revived only with value pretraining/decoupled λ (VAPO).
  - MC-from-intermediate-states (VinePPO, COMA-style counterfactual rollouts) → unbiased step values, cost = extra rollouts per step; multi-turn tool use has natural cut points (turn boundaries), which makes this cheaper than token-level.
  - Return decomposition (RUDDER) / implicit PRM → learned redistribution; risk = reward hacking of the learned decomposer (the reason R1 avoided PRMs).
  - Hindsight (HCA) → conditioning on outcome; natural for LLMs since the policy can be prompted with the answer.
  - Shapley/counterfactual multi-agent → turns-as-agents; principled but rollout-expensive.
- All three Pignatelli challenges are severe in LLM agents: delayed effects (reward after many turns), transpositions (many different tool-call sequences yield the same correct number), weak action influence (most tokens are near-deterministic/formatting; a few "forking" tokens decide outcome).

### Gaps
- TD/eligibility-trace, GAE, RUDDER, COMA, Shapley Q IDs not re-verified this session (arxiv blocked); all are canonical IDs from prior knowledge.
- Did not retrieve the counterfactual-credit LLM papers of 2025–2026 (likely covered by sibling researchers).

---

## Q6. Precise statistical problems of outcome-only rewards on long sequences

### Takeaway
With a single terminal reward broadcast to T tokens, the per-token gradient is (i) **high-variance** because the reward's noise multiplies a sum of T score-function terms, (ii) **diluted** because the few decisive tokens share one scalar with thousands of irrelevant ones, and (iii) prone to **biases** introduced by the normalizations used to tame (i) — length normalization, per-group std, zero-variance groups, and critic bias if one bootstraps instead.

### Cited Findings
- Bias–variance trade-off of return estimators: Monte-Carlo (λ=1) is unbiased but high-variance; bootstrapping (λ<1) lowers variance but is biased by value error — [GAE, arXiv 1506.02438 [M]](https://arxiv.org/abs/1506.02438)
- With a learned critic on long CoT: **value initialization bias** and **reward signal decay** (terminal reward attenuated by γλ per token) — [VC-PPO, 2503.01491](https://arxiv.org/abs/2503.01491); value bias, heterogeneous lengths, reward sparsity — [VAPO, 2504.05118](https://arxiv.org/abs/2504.05118)
- **Length bias**: dividing by |o_i| gives short correct answers larger per-token gradients and penalizes long wrong answers less per token, inflating incorrect-response length — [Dr. GRPO, 2503.20783](https://arxiv.org/abs/2503.20783); DAPO's token-level loss is a partial fix — [DAPO](https://arxiv.org/abs/2503.14476)
- **Difficulty (std) bias**: dividing by group std up-weights near-all-correct/near-all-wrong prompts — [Dr. GRPO](https://arxiv.org/abs/2503.20783); prompt-level normalization is a biased estimator (bias → 0 with global batch normalization) — [REINFORCE++](https://arxiv.org/abs/2501.03262)
- **Zero-signal groups**: all-same rewards give zero advantage → wasted compute (DAPO dynamic sampling) and no learning on hard prompts (Scaf-GRPO) — [DAPO](https://arxiv.org/abs/2503.14476); [Scaf-GRPO](https://arxiv.org/abs/2510.19807)
- **Weak action influence / transpositions / delayed effects** as the canonical credit-assignment difficulties — [Pignatelli et al., 2312.01072](https://arxiv.org/abs/2312.01072)
- **Clipping-induced bias** that can make even uninformative rewards move the policy toward high-prior behaviors — [Spurious Rewards, PMLR v306](https://proceedings.mlr.press/v306/shao26d.html); clipping drops rare reflective tokens from learning — [MiniMax-M1/CISPO, 2506.13585](https://arxiv.org/abs/2506.13585)
- Long-horizon off-policy error: token-level clip/mask can't bound worst-case error over long sequences — [Trust Region Masking, 2512.23075](https://arxiv.org/abs/2512.23075v2)

### Inferences (stated plainly; derivations are standard, not from a single cited source)
- **Variance**: REINFORCE gradient g = Â · Σ_{t=1}^{T} ∇log π(a_t|s_t). Score terms have zero mean but nonzero variance, so Var(g) grows roughly linearly in T (more if terms are correlated), while the *signal* — the expected contribution of the k decisive tokens — does not grow with T. Signal-to-noise ≈ O(k/√T) per sample; more samples per prompt (G) or more prompts are needed to compensate, roughly ∝ T/k².
- **Signal dilution**: every token gets the same Â, so in a correct trajectory with one bad tool call that call is pushed *up*; in a failed trajectory with a good early step that step is pushed *down*. Correct attribution only emerges statistically across many trajectories in which the decisive step varies independently of the others — slow when steps are correlated (e.g., a good plan usually precedes good execution).
- **Binary rewards and group size**: with r ∈ {0,1} and success prob p, P(zero-variance group) = p^G + (1−p)^G; for G=8 and p=0.9 that is ~43% of prompts giving no gradient — a structural, not tunable, information loss.
- **Bias sources** are introduced by the variance fixes themselves: length normalization, std normalization, clipping, and learned critics/PRMs each trade variance for a specific bias (length, difficulty, prior-amplification, value error / reward hacking).
- Multi-turn tool use compounds all of these: T includes many turns, tool observations are off-policy context (masked from loss) that still affect later tokens, and the decisive event is often one tool call or one interpretation of a tool result — the textbook case of "weak action influence + delayed effect".

### Gaps
- No single primary 2025–2026 paper retrieved that formally quantifies variance growth vs T for GRPO in LLMs; the O(T) argument above is a standard derivation, not a cited empirical result.
