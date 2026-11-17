# Deep Research 进度与交接：LLM 智能体信用分配与选题

> 本文件是续写入口。任何会话（含早上 7:05 的定时续写）先读这里，再按"下一步"继续。
> 分支：`claude/sleepy-heisenberg-t5qlfh`（每完成一个单元就 commit + push）。

## 用户需求（原话要点，2026-11-17）
- 以 Sikuan Yan（LMU Munich / MCML / Huawei）的 **Memory-R2**（arXiv 2605.21768，LoGo-GRPO）为主线，
  扩展到 LLM RL、Agentic RL、credit assignment 方法谱系、记忆增强 agent。
- 目的：**找到能发 paper 的 research idea**。
- 篇幅：精炼、通俗易懂、涵盖要点（篇幅由 Claude 定，目标正文约 12–16 页 NeurIPS 版式）。
- 经典工作要覆盖，其余尽量 2025–2026 的工作。
- 格式：**NeurIPS 模板 LaTeX**。先写**学术英文版**，写完再翻译**中文版**。
- 第 8 章（与用户 ToolCredit 项目对照）由 Claude 决定 → 决定：**保留，作为第 8 章 "Lessons from a
  controlled study (ToolCredit)"**，用其负结果（短 horizon 下轮级信用无终点增益、非守恒信用导致
  over-calling、零方差过滤无免费午餐）论证"选题应走向长 horizon / 可重置状态 / 信用质量评测"。
- 最后一章（第 9 章）：热门方向；现在 / 1 年 / 3 年可行性；可做且值得做的问题；
  如果现在开始，给出可执行计划（面向投稿）。

## 用户资源（写第 8、9 章时用）
- 单卡 NVIDIA GH200（146 GB），veRL 0.8.0 多轮工具调用 GRPO 流水线已跑通，基座 Qwen3-1.7B。
- 已完成实验（见仓库 README 结果总表）：E3 GRPO baseline、E4 shaping、E5 turn-level credit（v1 非守恒 → over-calling）、
  E5-v2 守恒轮级信用（情形 A：无显著增益）、E6 no-mask、E7 DAPO 零方差过滤（情形 C：等算力持平）、E3-NoTool。

## 章节规划（英文版标题）
0. Abstract / Executive summary
1. Introduction
2. Background: from RLHF to RLVR and the GRPO family
3. Agentic RL: multi-turn, tool-using, long-horizon agents
4. The credit assignment problem (classic RL view + LLM granularity)
5. A taxonomy of credit assignment methods for LLM agents (2024–2026)
6. Memory-augmented agents and Memory-R2: a close reading
7. Cross-family comparison and selection guide
8. Lessons from a controlled study (ToolCredit)
9. Research directions and an actionable plan (now / 1 yr / 3 yr; concrete ideas; start-now plan)
Appendix: paper list, glossary

## 文件布局
- 调研笔记：`deep_research/research_notes/LLM智能体信用分配与选题/0{1..5}_*.md`
  - 01 LLM RL 基础与经典信用分配；02 Agentic RL；03 信用分配方法；04 Memory-R2 与记忆 agent；05 趋势与可行性
- 英文 LaTeX：`deep_research/latex/en/main.tex` + `sections/*.tex` + `refs.bib`
- 中文 LaTeX：`deep_research/latex/zh/`（英文完成后翻译）
- NeurIPS 官方 sty：网络下载被拒（media.neurips.cc 被权限检查拦截）。`main.tex` 用 `\IfFileExists`
  兼容：放入 `neurips_2025.sty`（或 2026）即用官方模板，否则降级为同版式近似。
- 编译：`cd deep_research/latex/en && latexmk -xelatex main.tex`（xelatex、xeCJK、Noto CJK 已在本容器安装；
  新容器需 `apt-get install texlive-xetex texlive-latex-extra texlive-lang-chinese fonts-noto-cjk latexmk`）。

## 状态（每完成一步就更新）
| 步骤 | 状态 |
|---|---|
| 章节规划 | ✅ |
| 调研 01 LLM RL 基础 | ✅（约半数 arXiv ID 未直接核实，标 [M]） |
| 调研 02 Agentic RL | ✅（arXiv 未直连，ID 经 awesome-list 交叉核对） |
| 调研 03 信用分配方法 | ✅（部分 ID 仅 awesome-list 核对） |
| 调研 04 Memory-R2 | ✅（未读到全文，只有检索摘要；LoGo-GRPO 公式/组大小/算力未知） |
| 调研 05 趋势与可行性 | ✅ |
| LaTeX 骨架（en） | ✅ 可编译 |
| 英文各章 §1–§9 | ⏳ §2–3、§6、§8 写作中；§4、§5、§7 写作中；§1、§9、摘要、附录待写（最后写） |
| 英文编译 PDF | ☐ |
| 中文翻译 | ☐ |
| 中文编译 PDF | ☐ |

## 关键发现（写作时必须体现）
- **近邻工作** "Credit Without Ground Truth"（Haiyue Zhang, arXiv 2608.19760, 2026-08）：ALFWorld + Qwen2.5-7B，
  预注册 7 臂，信用规则间差异主要来自"训练剂量"（有效样本数），主张按匹配有效样本数比较；replay 审计中
  LLM-judge/logprob/置信度不比 shuffled 对照好。→ 与 ToolCredit E3–E7 结论高度一致，降低其新颖性；差异化方向：
  数学+工具、over-calling 与守恒修正、horizon 扫描。
- 没有被广泛采用的"直接衡量信用质量"的 benchmark（AgentCreditBench 只有小决策问题）——最契合用户资产的空白。
- Memory-R2：Qwen2.5-7B 共享 extractor+manager，冻结 answer agent，课程 8→16→32 session，仅 2 个 LoCoMo 对话训练；
  LoCoMo F1 Memory-R1 43.14 → R2 50.60；消融 GRPO 46.62 → LoGo 49.67。缺同 rollout 预算的 GRPO 对照（E7 教训）。
- 2026 记忆信用方法（Mem-T、tree-based credit、LoGo-GRPO）都靠"从共享前缀/状态分叉"拿 per-step baseline。
- 调研时 arxiv.org / github.com / huggingface.co 被网络策略挡住，很多数字来自检索摘要，正文须标注"待核实"。

## 下一步（续写者照做）
1. 检查 research_notes 下 01–05 是否齐全；缺的话只补缺的那一份（不要重做已有的）。
2. 按章节顺序写 `latex/en/sections/NN_*.tex`，每章写完：编译通过 → 更新本表 → commit + push。
3. 英文全部完成后，逐章翻译到 `latex/zh/sections/`，同样逐章提交。
4. 全部完成后，删除 7:05 的定时续写任务（trigger_id `trig_01AfysA5nFvCS5oMR3NUdcYx`，北京时间 2026-11-18 07:05 触发；如果还没触发）。

## 写作约定（续写者沿用）
- 每章一个文件 `latex/en/sections/NN_*.tex`；每章的 BibTeX 放 `latex/en/bib/sNN.bib`；`bash latex/en/build.sh` 合并 bib 并编译。
- 引用 key：小写 一作姓+年份+标题首个实词（如 `shao2024deepseekmath`、`yan2026memory`=Memory-R2、`yan2025memory`=Memory-R1、`zhang2026credit`=2608.19760）。
- 未能核实的数字/ID 后加 `\unverified`（渲染为橙色 †）。
- 章节 label：sec:background, sec:agentic, sec:problem, sec:taxonomy, sec:memory, sec:comparison, sec:toolcredit, sec:agenda。
