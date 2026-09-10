# CLAUDE.md — ToolCredit 项目约定

多轮工具调用 RL 中的信用分配研究（数学推理 + Python 解释器，veRL + GRPO）。
**唯一权威文档是 `PLAN.md`**：研究问题、实验矩阵 E3–E7、Milestone 划分、验收标准、降级方案
全部以它为准。本文件只记约定与状态，不复制 PLAN 内容。

## 当前状态

**当前 Milestone: M0–M10 全部完成（tag `m10`，2026-09-10）。M10 / E7 DAPO 式零方差过滤：训练 `rl/runs/e7_dynamic_filtering_20260909_200152`（200 有效 step，259,072 条轨迹 = E3 2.53×，等算力 checkpoint step 95），评测 `eval/runs/m10_e7_eval_20260910_140124`（协议 v6，7,600/7,600），预注册判读**情形 C**（Δ_step +1.44pt CI [−1.06,+4.13]、Δ_compute −0.39pt CI [−2.89,+2.11]；附报 2.53× rollout 的终点 +5.13pt [+2.63,+7.63]，缺 E3 同预算臂）。匿名内存泄漏定位到 `ray::WorkerDict`（只诊断不修）。E7 run 的 185 GiB checkpoints 待用户裁定删除。没有获批的新实验：E3 同预算对照臂、过滤降本变体、额外 seed、β sweep、§3.4 变体、CoT-SFT 对称起点均需另行授权 | 平台: GH200/aarch64 JupyterHub pod（无 docker，conda env `toolcredit`）**

（每个 Milestone 完成后更新本行；会话结束更新 `LOG.md`；交接状态写 `HANDOFF.md`。）

## 项目约定

1. **依赖 pin 死**：所有版本写入 `requirements.txt` 与 `environment.md`（含 GH200/aarch64 踩坑记录）。
2. **每个训练 run 必须落盘五件套**：resolved config、checkpoint/adapter、预测 JSONL、
   metrics JSON、一段 Markdown 摘要。每个 run 有唯一 name 与 config 快照（W&B 或 TensorBoard）。
3. **先原样跑通，再改**：veRL 官方 multi-turn tool 示例原样跑通之后才允许任何定制；
   多轮交互的标记格式直接采用官方示例，不自己发明。
4. **代码风格**：函数短、有 type hints；关键张量 shape 在 debug 模式下打印，shape 不符立即 fail。
5. **节奏**：每个 Milestone 完成即 commit + tag + 更新 README 结果总表一行 +
   更新 `reports/technical_report.md` 对应章节（动机/方法/结果/面试叙事，living document）。
6. **数据/评测纪律**：训练与评测严格分 split；训练子集必须先过 `data/dedup_check.py` 污染检查。
7. **计划留痕**：每个 Milestone 一份 `plans/M<N>.md` —— 计划批准时写入原文，
   执行中记录偏差与决策，验收时写结果表。不得事后追改计划原文。
8. **问答留痕**：用户（陈冀超）提出的问题及解答，凡有沉淀价值（概念澄清、设计决策、
   面试素材）即追加到 `reports/qa_log.md`（日期 + 场景 + Q + 浓缩版 A + 证据指针）。

## 禁止事项

1. **不 fork veRL**。对框架的一切改动用最小 patch / 子类 / 配置注入实现，集中放在
   `rl/custom/`，并同步更新 `rl/custom/CHANGES.md`（改了哪个函数、为什么）。
2. **不静默跳过 reward 错误**。任何 reward/答案提取失败都要记日志并计入 `invalid_rate` 指标。
3. **修改框架内部逻辑前先询问用户**（陈冀超），说明改动点与理由，得到确认再动手。
4. **长任务必须 tmux 后台运行**（训练、大批量生成、全量评测），日志落盘到项目内，
   不允许阻塞交互会话；会话内只做提交任务与轮询状态。
5. **不原地死磕**：任何一步卡住超过其时间预算的 50%，触发 PLAN §12.2 对应降级并报告用户。
6. **只在 `/home/jovyan/toolcredit/` 下工作**，绝不删除项目目录之外的任何文件
   （HF 缓存、conda 环境等家目录常规写入除外，且只增不删）。
