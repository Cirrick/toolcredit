# M10 步骤 5–6 执行会话 prompt（2026-09-10 生成，复制整段给新 session）

---

你在 `/home/jovyan/toolcredit/` 继续 **M10 / E7：DAPO 式零方差 group 动态过滤与补采样**。
**步骤 0–4 已全部完成**（边界批准、实现+测试、preflight、smoke+resume-check、200 有效 step formal 训练，门禁全绿）。
你只需执行 `plans/M10.md` §8 的 **步骤 5（评测 + 配对分析 + 判读）与步骤 6（文档 + commit + tag `m10`）**。
**不要重跑训练，不要重做步骤 0–4 的任何部分。**

先读 `CLAUDE.md`，再按顺序读：
`plans/M10.md`（v2，**已批准，原文不得追改**；偏差写 §12、结果写 §13；§13 第 0/1/2/3/4/8 行已回填，你要填 5/6/7/9），
`plans/M10_STEP0_BOUNDARY.md`（训练侧边界，已完成，仅供理解证据文件），
`HANDOFF.md` 顶部 2026-09-10 一条（formal 结果全文），
`plans/M10_KICKOFF_PROMPT.md`（上一段的 prompt，硬约束仍然有效），
`reports/qa_log.md` Q17（cgroup OOM 根因）与 Q22，
`eval/build_diagnostic_freeze_v5.py` + `eval/checkpoints_v5.py`（v6 的父冻结与先例），
`rl/custom/CHANGES.md` 的 M10 条目（含"评测侧对框架零改动"一节）。

## 步骤 4（formal 训练）的结果——你要在这些产物上做评测，不要重跑

训练 run：**`rl/runs/e7_dynamic_filtering_20260909_200152`**，2026-09-09 20:01 → 2026-09-10 12:37，
200/200 有效 step，4 段 `--stop-at-step` 50/100/150/200，`analysis/segment_gate.json` 与
`analysis/formal_completion_gate.json` 均 `status=completed`。两个 pin SHA 逐段复验一致，
trainer/boundary activation 各 4 次，恢复归档 `recovery/resume_from_{50,100,150}_*`。

| 项 | 数值 |
|---|---|
| 每有效 step 的 generation 批数 | 直方图 `{2: 97, 3: 100, 4: 3}`，均值 **2.53** |
| 用满 4 批的步 | 175 / 181 / 200 三步，**最大连续 1 步**（硬停线 10） |
| `underfilled` | **0**，`selected_groups` 恒 64 |
| 累计轨迹 | **259,072**（E3 是 102,400，即 **2.53×**） |
| 累计题数 / rollout token / generation 墙钟 | 32,384 / 2.579 亿 / 6.64 h |
| 零方差比例（逐段均值） | 34.5% → 46.3% → 54.9% → **61.4%**（step 200 当步 63.7%） |
| 等算力 checkpoint | `checkpoints/equal_compute_step_95`，累计**恰好 102,400** 条，不写 tracker、不参与轮换 |
| checkpoint | 常规 25/50/…/200 共 8 个 + 等算力 1 个，tracker=200，目录合计 **185 GiB** |
| cgroup anon | 峰值 66.41 GiB，`oom_kill` 0，全程未触及 110 GiB 提前分段线 |

**fixed-100 greedy 曲线（step: E7 / E3）**：
0: .61/.60、25: .65/.67、50: .73/.67、75: .73/.70、100: .72/.73、125: .76/.73、150: .78/.74、175: .75/.77、200: **.76/.76**。
E7 中段领先、终点打平。**面板只有 100 题，1pt = 1 题，不要据此下结论**；
Δ_step 必须按 `analysis/m10_compute_axes.py` 里冻结的 AUC 规则算，Δ_compute 按 M7-panel FULL760 配对算。

**一个必须写进结论的独立发现**：零方差比例随训练单调上升（34.5%→61.4%），
即**策略越准，DAPO 式过滤越贵**——平均批数从 2.0 涨到 3.0 以上，总算力是 E3 的 2.53×。
所以等 step 的比较会系统性高估 E7，**双 x 轴不是附报而是判读的一半**。
失败模式（本 run 未触发）：零方差涨到约 75% 时 4 批凑不满 64 组，会显式 underfilled 并硬停。

**匿名内存泄漏诊断（已定案，写进 qa_log 是步骤 6 的硬要求）**：
`analysis/memwatch_report.{json,md}`，828 采样 / 16.63 h / steps 1–200。cgroup anon 5.23→66.41 GiB。
按驱动 PID 分 4 段拟合，逐段 **0.5047 / 0.5382 / 0.5431 / 0.5580 GiB/有效 step**
（headline 取 post-startup 采样最多的第 4 段；跨段混拟会给出 0.038 的假数字）。
按 `Pss_Anon` 增量排名，**每段的单个 `ray::WorkerDict`（actor 与 ref 共置的 FSDP worker）稳居第一**：
第 4 段 0.5213 GiB/step = cgroup 斜率的 **93.4%**（9.70→29.60 GiB，RSS 终值 48.6 GiB）；
驱动 actor 0.0024、`sglang::schedul` 0.0019、`gcs_server` 0.0027、`ray::AgentLoopW` ≤ 0.0011 GiB/step。
这**否定**了 `HANDOFF.md` 2026-09-09 条目的假设（AgentLoop/sglang 轨迹缓存或 reward 沙箱残留）。
关闭 CPU offload 只把常驻基线从 49 GiB 降到 25 GiB、**不改斜率**。按 `plans/M10.md` §1 只诊断不修。

## 已经写好但**尚未运行过**的评测侧代码（单测全绿，真实数据未跑）

| 文件 | 作用 |
|---|---|
| `eval/checkpoints_v6.py` | 两个 role 的发现、校验、物化（`equal-compute-step` / `validate-source` / `materialize` / `validate-materialization` / `resolve-roles`）；run 路径与 `formal_completion_gate.json` 已硬编码绑定 |
| `eval/build_diagnostic_freeze_v6.py` | 以 v5 的 107 文件闭包为父，生成 v6 protocol + manifest；`--verify` 复验 |
| `eval/verify_diagnostic_freeze_v6_semantics.py` | 语义差异校验器；v6 起 `generation_mode` 进入冻结字段（共 13 个） |
| `eval/m10_e7_eval.py` | `preflight` / `generate-shard` / `score` / `downstream` / `closure`；配对 question-level bootstrap、双 x 轴、§9 四格判读 |
| `scripts/m10/run_m10_eval.sh` | tmux 入口，串起 SGLang 起服务 → 2 role × 2 setting × 4 source → score → downstream → closure |
| `analysis/m10_compute_axes.py` | 冻结的 AUC 规则（线性插值、共同预算 `[0, 102400]`、归一化） |

单测：`eval/test_checkpoints_v6.py` 18 项、`eval/test_m10_e7_eval.py` 38 项、`analysis/test_m10_compute_axes.py` 4 项，全绿。
**评测侧对 veRL 零改动**（只读 checkpoint + SGLang 推理）。

## 步骤 5 的执行顺序

1. **回归**：`/opt/conda/bin/conda run --no-capture-output -n toolcredit python -m pytest -q --ignore=third_party`
   （期望 ≥ 287 passed；**用 `python -m pytest`，裸 `pytest` 会 `ModuleNotFoundError: analysis`**）。
2. **两个 role 的来源校验与物化**：
   `python -m eval.checkpoints_v6 equal-compute-step` → 应打印训练时记录的 N；
   对每个 role 依次 `validate-source` → `materialize` → `validate-materialization`。
   物化是 FSDP 分片 → HF 格式，两个 role 各约 3.3 GB，落在 `MATERIALIZED_ROOT` 下。
3. **建冻结 v6 + 语义验证**：`python -m eval.build_diagnostic_freeze_v6`，再 `--verify`，
   再 `python -m eval.verify_diagnostic_freeze_v6_semantics --output <路径>`。
   语义校验器必须证明 panel/generation/tool/verifier/taxonomy/pairing/bootstrap 语义相对 v5 未变，
   只多出两个 TIR role。**任何一条不过就停下来报告用户，不要改校验器去迁就产物。**
4. **preflight**：`python -m eval.m10_e7_eval preflight --run-dir <eval run dir>`（磁盘 + GPU + 冻结 hash）。
5. **smoke 评测**（每 source 一题，不打分不下游）：
   `tmux new-session -d -s m10-eval-smoke 'M10_EVAL_SMOKE=1 bash scripts/m10/run_m10_eval.sh'`，
   看轨迹结构、工具调用、终止原因是否正常。**报告用户后**再进入全量。
6. **全量评测**：`tmux new-session -d -s m10-eval 'bash scripts/m10/run_m10_eval.sh'`。
   2 role × 2 setting（greedy/sampled）× 4 source（MATH500/AIME2024/AIME2025/GSM8K）= **7,600 条**，
   其余角色复用 M7/M8/M9 的 canonical 产物并重验 hash。脚本按 stage_log 断点续跑，中断后原样重启即可。
   预计数小时；期间只轮询状态，不阻塞会话。
7. **downstream + closure**：脚本末尾自动跑；确认 `stage_logs/eval.completed.json` 存在、
   `expected_keys_v6()` 的 7,600 键全部命中、closure 的 hash ledger 完整。
8. **配对分析**：两组 primary 配对（E3 → E7-step-200 等 step；E3 → E7-equal-compute 等算力）
   + question-level 配对 bootstrap CI；两条 x 轴的归一化 AUC（有效 step 轴 + 累计轨迹轴，
   共同区间 `[0, 102400]`）；附报 rollout token 与 wall-clock 两条次级算力轴、step-200 终点配对 Δ、
   每有效 step 平均批数与总轨迹数（对照 E3 的 102,400）、informative 集合的 level/mixed 比例漂移（描述性）。
9. **判读**：按 `plans/M10.md` §9 四格（Δ_step × Δ_compute，阈值 ±2pt 且 95% CI 不跨 0）落格，
   用预注册写法成文。**落格结果向用户报告，这是门禁**，得到确认后再进入步骤 6。

## 步骤 6

`reports/02_main_results.md` 新增 M10 节、`reports/technical_report.md` 对应章节、README 结果总表一行、
`LOG.md`、`HANDOFF.md`、`reports/resume_bullets_autumn.md`、`reports/qa_log.md`（**必须含匿名内存泄漏进程诊断结论**，
见上面"步骤 4 的结果"末节）；回填 `plans/M10.md` §13 第 5/6/7/9 行与 §12 的新偏差；
最后 commit + tag `m10`。

## 硬约束（沿用 `plans/M10_KICKOFF_PROMPT.md`，仍然全部有效）

1. **不 fork veRL、不改 site-packages**；评测侧本来就零框架改动，若发现需要改，**先停下来问用户**。
2. **长任务一律 tmux 后台**，日志落盘项目内，会话内只提交任务与轮询状态。
3. `conda` 在交互 shell 里可能是坏函数：用 `/opt/conda/bin/conda run --no-capture-output -n toolcredit ...`；
   `scripts/m10/run_m10_eval.sh` 内部直接用 `/home/jovyan/.conda/envs/toolcredit/bin/python`，是同一个环境。
4. **每个门禁向用户报告后再进入下一步**（回归全绿、物化+冻结、smoke 评测、全量完成、判读）。
5. **不静默跳过 reward/答案提取错误**，计入 `invalid_rate`。
6. **不原地死磕**：任何一步超时间预算 50% 触发 `PLAN.md` §12.2 降级并报告用户。
7. **只在 `/home/jovyan/toolcredit/` 下工作**，绝不删除项目目录外的任何文件。
8. 计划原文不得追改；偏差写 `plans/M10.md` §12，结果写 §13。

## 已知事实

1. **E3 对照 run**：`rl/runs/e3_grpo_baseline_20260819_224555`，200 step，总轨迹 **102,400** = 200×64×8。
2. **协议父冻结 v5**：107 文件闭包，manifest `5238d08e…b2e4d`（M9）。v6 只新增两个 TIR role，其余复用。
3. **fixed-100 validation panel** 是训练侧的 greedy 曲线（`metrics.json` 里的 `val/...`），
   与评测侧的 M7-panel FULL760 是两套东西，别混用：Δ_step 用 fixed-100 曲线的 AUC，Δ_compute 用 FULL760 配对。
4. **GPU**：GH200 单卡 142 GB；容器 cgroup `memory.max` = 128 GiB 且 `oom.group=1`（超限整 pod 被杀，无 Traceback）。
   评测阶段 SGLang 单进程，内存压力远小于训练，但仍要看着。
5. **磁盘**：`/home/jovyan` 可用约 3.8 TB。E7 run 的 `checkpoints/` 目录共 **185 GiB**（8 个常规 + 1 个等算力），**步骤 6 完成并确认前不要删**。
6. **匿名内存泄漏诊断**：见上面"步骤 4 的结果"末节，数字以那里为准
   （逐段 0.5047/0.5382/0.5431/0.5580 GiB/有效 step，`ray::WorkerDict` 占 93.4%）。写进 qa_log 是步骤 6 的硬要求。
7. **M10 代码目前尚未 commit**（HEAD = `83bbd62`，只有文档）。工作区里的 M10 文件是完整可用的，
   新会话直接接着用即可；步骤 6 的 commit 要包含训练侧 + 评测侧 + 文档全部改动。
   **不要**提交 `PLAN.md`、`Untitled.ipynb`、`.vscode/`、`analysis/tool_use_conditional_accuracy.*`、
   `reports/05_data_qa.md`、`today_research_workflow.md`、`sft/experiments/.../tensorboard/`（用户的其他改动）。

## 第一件事

`git status` 确认工作区与上表一致，然后跑步骤 5 第 1 条的全仓回归，把结果报告给用户。
