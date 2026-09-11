# 交接状态

## 当前状态

- **2026-09-11：M10 收尾裁定执行完毕。** 用户授权删除 E7 run 的 7 个轮换残留 checkpoint（`global_step_{25,50,75,100,125,150,175}`，
  各约 21.9 GB，合计 153.4 GB），保留 `global_step_200` 与 `equal_compute_step_95`（M10 评测物化的两个来源）。
  checkpoints 目录 185 GiB → **41 GiB**，挂载可用 3.7 TiB → 3.9 TiB。删除前把 7 份 per-checkpoint ledger 复制到
  `analysis/deleted_checkpoint_ledgers/`；记录写 `rl/runs/e7_dynamic_filtering_20260909_200152/checkpoint_cleanup.json`
  （schema `toolcredit_raw_checkpoint_cleanup_v1`，含逐步字节数、写入时间、fixed-100 准确率）。
  删除后重跑 `python -m rl.validate_e7_run formal`，`status=completed` 仍成立，结果存
  `analysis/formal_completion_gate_post_cleanup.json`；该次重跑会就地改写 `analysis/formal_completion_gate.json`，
  已按重跑前副本恢复验收时原件（两份只差 `checkpoints.checkpoint_steps`）。**未做任何新训练/评测/判读改动。**
  用户同时指示：**E3 同预算对照臂暂不执行**，转为长期待办，见"下一步"第 6 条。

- **2026-09-10（晚）：M10 / E7 全部完成（步骤 0–6），预注册判读情形 C，tag `m10`。** 评测 `eval/runs/m10_e7_eval_20260910_140124`
  （协议 v6，manifest `26d89b19…ccc63` 138 文件；7,600/7,600，exact-key digest `6be60168…3d73`；49 产物 `hashes.sha256` `10cd3e26…e0a5`；
  复用 M7 157 + M8 30 + M9 93 个 canonical 产物三次重验）。物化 `eval/checkpoints/m10_e7_{dynamic_filter_step_200,equal_compute}_hf`
  （各 3.3 GB，manifest `eval/materialization/e7_*.json`）。
  **主判据**：Δ_step（fixed-100 greedy 0–200 归一化 AUC，E7 .7256 vs E3 .7113）**+1.44pt CI [−1.06, +4.13]**（题级配对 bootstrap 10k）；
  Δ_compute（FULL760 greedy，E7 等算力 step-95 .729 vs E3 step-200 .733）**−0.39pt CI [−2.89, +2.11]**（45 fixed / 48 new）→ 两轴持平 → **情形 C**。
  **附报**：E3→E7 step-200 **+5.13pt [+2.63, +7.63]**（69/30；.784 vs .733；2.53× rollout）；E7-eq→E7-200 +5.53pt [+3.16, +8.03]；
  累计轨迹轴 AUC [0, 102,400] E7 .6941 vs E3 .7113 = −1.72pt；sampled matched-index E3→E7-200 +3.68 [+2.27, +5.13]、E3→E7-eq −2.11 [−3.59, −0.62]；
  level：E7-200 增益集中 L5 +7.46 / L3 +4.76 / L4 +3.91 / AIME +13.3/+16.7（n=30），E7-eq 在 L2/L4/L5 各 −3pt。
  行为（greedy FULL760）：mean calls E3 .888 / E7-eq .630 / E7-200 .668；截断 9.6% / 10.1% / 6.8%；per-call 成功 88.2% / 90.0% / 93.3%；
  重复代码 2.9% / 1.4% / 0.8%；4-call 1.3% / 0.9% / 0.7%；无 over-calling。
  **读法**（写进 02 节与 technical report §13）：等算力无免费午餐；step-200 的 +5pt 是"用 2.53× rollout 换终点"，缺 E3 同预算臂，不得归因于过滤本身。
  **训练侧独立发现**：零方差 34.5→61.4%、全对组 24→50%、每步批数 2.0→3.06——过滤成本随策略变准单调上升，约 75% 时会 underfilled。
  **执行记录**：回归 350 passed；smoke 模式补 `--limit-questions`（M9 同款）后 v6 冻结在生成前重建一次（两次语义校验都过）；
  全量生成保持 M7/M8 的逐题串行；评测墙钟 9 h 32 min，其中约 4 h 是 pod 上用户另一项目（`~/wireless_AI/beam_anymodal_kd`，8 个训练进程 ×12 worker）
  的 CPU 争用把 SGLang 解码从 435 拖到 72 tok/s，按禁止事项 #6 未触碰；修正了上一会话把 §13 第 3/4 行误替换进 `plans/M10.md` §0 的问题
  （按 HEAD 逐字恢复，记 §12）。`plans/M10.md` §12 新增 8 行、§13 第 5/6/7/9 行已填。qa_log Q26（泄漏进程诊断）、Q27（判读与面试口径）。
  **E7 run 的 185 GiB checkpoint 目录（8 常规 + 1 等算力）尚未删除**：物化只引用 `global_step_200` 与 `equal_compute_step_95`，其余 7 个（约 150 GiB）可删，等用户裁定。
  **下一步需另行授权**：(a) E3 同预算对照臂——从 E3 step-200 续训到累计 259,072 条轨迹（约再 306 步，按 E3 单步 161 s 约 14 h）或在 E7 内按 E3 预算截断的多点比较；
  (b) 过滤降本变体（按需缩批、跨 snapshot 回收 surplus）；(c) 泄漏根治仍不在范围内（`ray::WorkerDict` 内 `tracemalloc`/`MALLOC_ARENA_MAX` 对照）。

- **2026-09-10（午）：M10 / E7 步骤 1–4 全部完成，200 有效 step 的 formal 训练收工，两个门禁全绿。【步骤 5–6 已于当晚完成，见上一条；本条保留为训练侧记录】**
  **训练 run**：`rl/runs/e7_dynamic_filtering_20260909_200152`，2026-09-09 20:01 → 2026-09-10 12:37，总墙钟 16 h 35 min（其中第 3→4 段之间因本地会话重启终止 tmux server 空转 2 h 48 min，训练进程未在步中被杀，见 `plans/M10.md` §12）。
  4 段 `--stop-at-step` 50/100/150/200，trainer/boundary activation 各 4 次，两个 pin SHA 逐段复验一致，恢复归档 `recovery/resume_from_{50,100,150}_*`。
  `analysis/segment_gate.json` 与 `analysis/formal_completion_gate.json` 均 `status=completed`（后者是 v6 冻结按文件名绑定的那份，由 launcher 在最后一段自动追加）。
  **ledger（200 行）**：批数直方图 `{2: 97, 3: 100, 4: 3}`，平均 **2.53 批/有效 step**；`selected_groups` 恒 64；`underfilled` 全 0；用满 4 批的只有 step 175/181/200 三步，**最大连续满批 1 步**（硬停线是 10 步）；累计 **259,072 条轨迹（E3 的 2.53×）**、32,384 题、2.579 亿 rollout token、generation 累计 6.64 h。
  **零方差比例逐段上升 34.5% → 46.3% → 54.9% → 61.4%**（step 200 当步 63.7%）。这是 M10 的一个独立发现：策略越准，DAPO 式过滤越贵，每步平均批数从 2.0 涨到 3.0 以上。**必须报双 x 轴**，等 step 的比较会系统性高估 E7。若零方差再涨到约 75%，4 批将凑不满 64 组而显式 underfilled——本 run 没到。
  **等算力 checkpoint**：`checkpoints/equal_compute_step_95`，累计**恰好 102,400** 条（= E3 总量），不写 tracker、不参与轮换，21.9 GB。常规 checkpoint 25/50/…/200 共 8 个（分段导致每段轮换独立，实际保留 8 个而非 3 个，见 §12），tracker=200，checkpoints 目录共 185 GiB，**步骤 6 收尾确认前不要删**。
  **fixed-100 greedy 曲线（step: E7 / E3）**：0: .61/.60、25: .65/.67、50: .73/.67、75: .73/.70、100: .72/.73、125: .76/.73、150: .78/.74、175: .75/.77、200: **.76/.76**。E7 中段领先、终点打平；面板只有 100 题，1pt = 1 题，**不要据此下结论**，Δ_step 要按 `analysis/m10_compute_axes.py` 的冻结 AUC 规则算。
  **匿名内存泄漏定案**（`analysis/memwatch_report.{json,md}`，828 采样 / 16.63 h / steps 1–200）：cgroup anon 5.23→66.41 GiB，峰值 66.41，`oom_kill` 0，全程未触及 110 GiB 提前分段线。按驱动 PID 分 4 段拟合，逐段 **0.5047 / 0.5382 / 0.5431 / 0.5580 GiB/有效 step**（headline 取 post-startup 采样最多的第 4 段；跨段混拟会给 0.038 的假数字）。按 `Pss_Anon` 增量排名，**每段的单个 `ray::WorkerDict`（actor 与 ref 共置的 FSDP worker）稳居第一**，第 4 段 0.5213 GiB/step = cgroup 斜率的 **93.4%**（9.70→29.60 GiB，RSS 终值 48.6 GiB）；驱动 actor 0.0024、`sglang::schedul` 0.0019、`gcs_server` 0.0027、`ray::AgentLoopW` ≤ 0.0011 GiB/step。**这否定了本文件 2026-09-09 条目里的假设**（怀疑 AgentLoop/sglang 轨迹缓存或 reward 沙箱残留）：泄漏在 FSDP worker 进程内；关闭 CPU offload 只把常驻基线从 49 GiB 降到 25 GiB、不改斜率。按 `plans/M10.md` §1 只诊断不修，结论写 qa_log 是步骤 6 的硬要求。
  **步骤 5 的代码已经写好但一次都没在真实数据上跑过**：`eval/checkpoints_v6.py`、`eval/build_diagnostic_freeze_v6.py`、`eval/verify_diagnostic_freeze_v6_semantics.py`、`eval/m10_e7_eval.py`、`scripts/m10/run_m10_eval.sh`、`analysis/m10_compute_axes.py`；单测 `eval/test_checkpoints_v6.py` 18 + `eval/test_m10_e7_eval.py` 38 + `analysis/test_m10_compute_axes.py` 4 全绿；评测侧对 veRL 零改动。E7 run 路径与门禁路径已硬编码在 `eval/checkpoints_v6.py`，新会话不需要传参。

- **2026-09-09：M10 草案 v2 已批准；步骤 0 完成并批准（`plans/M10_STEP0_BOUNDARY.md` §8，用户授权助手决定：8 处白名单、驱动边界方案 (b)、resume 显式拒绝启发式）。下一会话按 `plans/M10_KICKOFF_PROMPT.md` 执行步骤 1–6；M10 相关文件尚未 commit。** 21.9 GB offload-check checkpoint 已按用户指令删除（`checkpoint_cleanup.json`）。

- **2026-09-09：offload 关闭验证（下方"下一步"第 5 条）完成，结论为"部分通过"。** run `rl/runs/e3_offload_check_20260909_150849`（formal 形状 5 步 + step-5 checkpoint，resolved config 相对 E3 只差 6 路径：`actor.fsdp_config.{param,optimizer}_offload`、`ref.fsdp_config.param_offload` 置 false 与 `trainer.{total_training_steps,save_freq,test_freq}=5`）。对照表见 `analysis/offload_check_compare.md`（同目录 `.json` 与 `container_memory.jsonl`）。四项验证：(a) 数值：5 步 prompt 与 E3 完全相同，score .508/.588/.533/.591/.556 vs E3 .531/.570/.522/.610/.552，pg_loss/entropy/KL/grad_norm 同量级，**非逐位一致**（sglang 采样非确定，E3 自身重跑也不会逐位一致）；(b) GPU：max_allocated 43.5 GB 与 E3 相同，reserved 63.1 vs 50.2 GB，无 OOM，142 GB 卡上余量充足；(c) CPU：WorkerDict RSS 25 GB（E3/M9 49–50 GB），psutil used 159–163 vs E3 167–188 GB，pinned 池消失、基线降约 25 GB；**但 cgroup anon 在训练段仍以约 0.5 GB/step 线性增长**（39.0→41.7 GB / 5 步，WorkerDict 22.4→24.9 GB），与 E3 每 150 步 +51–63 GB 的斜率同量级，即泄漏源不是 offload 池本身；线性外推 200 步 anon 约 140 GB，**仍可能撞 128 GiB**。checkpoint 写入时 page cache 冲到 107 GB（可回收，anon 不变）；(d) 速度：单步 165–189 s vs E3 151–167 s，差异全在 gen（54–77 vs 48–57 s），old_log_prob/ref/update_actor 相同；5 步样本太少，不定论。**建议**：M10+ 关闭 offload（三条 diff 进允许集合，标为纯基础设施差异）**并且保留 step-100 计划性分段恢复**；如需根治，下一步应在更长 run 上按 `smaps_rollup` 定位 anon 增长进程（候选：AgentLoop/sglang HTTP 侧的轨迹缓存、reward 沙箱残留），不是再调 FSDP。memwatch 的 `step` 字段在本 run 恒为 0（读 log 的 grep 未命中，时间戳可用），待修。21 GB smoke checkpoint 未删。新增文件：`rl/launch/e3_offload_check.py`、`scripts/infra/run_offload_check.sh`、`analysis/offload_check_compare.py`；qa_log 新增 Q22（DAPO vs Dr.GRPO，用户决定先做 M10）。

- **2026-09-09：repo 状态与 M10 适用性审查完成，未授权或启动 M10。** HEAD `8e69542` / tag `m9`；
  M9 step-200 checkpoint 文件大小与 completion gate 相符，v5 的 5 个 metrics 文件 hash 重验通过，
  `conda run --no-capture-output -n toolcredit pytest -q -p no:cacheprovider eval/test_m9_notool_eval.py`
  **14 passed**（CUDA/NVML 初始化警告）。本会话 `nvidia-smi` 无法连接驱动，不能确认 GPU 可用；
  `df -h .` 显示挂载可用约 4.0 TiB，不代替训练 pod/quota preflight。veRL trainer/main 源码 SHA
  与 E7 旧审查完全相同；没有 E7 trainer/config/launcher/run。`AGENTS.md` 的 M4/M5 状态摘要已过期。
  **建议继续 M10 方向，但先修订草案再进入 exact-boundary 专项审批**：
  (1) 保持 E3 TIR 对照，M9 的 NoTool 零方差统计只能作旁证；协议确定为 v6。
  (2) 整批 64 题补采样且 surplus 丢弃使 1.8× rollout 预算偏低：以 E3 informative 比例
  .5593/.4725 作独立同分布二项近似，最多 4 批的期望分别约 2.08/2.70 批，仅为预算推算，非 E7 实测。
  (3) 冻结 rollout-AUC 的共同预算区间、插值及归一化规则，同时报告 token/wall-clock；200-step 终点
  不是等算力比较。CI 跨 0 只表示未检出，需补齐 §9 判读空档，不强制把提升归因于 L5/mixed 漂移。
  (4) 零优势仅使 GRPO policy-gradient 项为零，E3 仍有 KL loss；过滤还改变训练分布和 PG/KL 权重关系。
  (5) 将下方 offload 验证/分段恢复待办纳入 M10：5-step smoke 不能证明 200-step 无内存增长，
  也未覆盖 checkpoint 峰值；若关闭 offload，要更新三项 config diff 并记录数值与速度可比性限制。
  原有未提交改动保留，未改 `PLAN.md`、M10 草案或冻结产物；本次仅新增交接/日志记录。

- **2026-09-07：M9 / E3-NoTool 完成（步骤 0–5），预注册判读情形 B。**
  训练 `rl/runs/e3notool_grpo_20260906_220907`（200/200；step 150 遭容器组级 OOM，用户批准后从 step-125 恢复，
  `recovery/resume_from_125_20260907_140752/`；`analysis/formal_completion_gate.json` 全绿，102,400 条、tool 计数与
  prompt schema 逐条为 0、标签吐出率 1/102,400）。评测 `eval/runs/m9_notool_eval_20260907_175726`（协议 v5，
  manifest `5238d08e…b2e4d` 107 文件；15,200/15,200；复用 M7 157 + M8 30 个 canonical 产物并重验 hash）。
  **primary Δ = E3-NoTool − E3 = −0.92pt，CI [−3.55, +1.71] → 情形 B**。
  分解：SFT→E3 的 +12.76pt = 6.05（不再被工具拖累）+ 4.74（推理提升，[+2.11,+7.50]）+ 1.97（工具边际价值，跨 0）。
  工具增益 raw −13.95 → SFT −6.05 → E3 −1.97pt；优势只在 MATH L4/L5（+7.03/+2.99pt）。
  四个 notool 角色 `tool_tag_emitted` 全 0/760。训练侧零方差组 44.6%→56.3%（E3 26.6%）→ M10 动机。
  文档：`reports/02_main_results.md` M9 节、`01_tool_gain.md` M9 附录、technical report §12、README 总表、qa_log Q21。
  用户另行授权的 step-125 敏感性分析 `eval/runs/m9_step125_sensitivity_20260907_191332`（exploratory，不入协议）：
  step-125 .721 vs step-200 .724 vs E3 .733，step200→step125 Δ −0.26pt CI [−2.50,+1.97]——fixed-100 上的峰不复现，判读不敏感。
  **重要**：所有 200-step run 的"pod 中断"根因已查明（qa_log Q17），待办见"下一步"第 5 条。


- **2026-09-06：HER 可行性讨论完成，仅文档。** `reports/qa_log.md` Q16 记录优先方向：从 train 失败轨迹提取独立验证的成功子任务，重标注后辅助 SFT，再考虑与 E3 配方结合。建议先固定抽样审计 100–200 条；本次仅检查 E3 首行日志 schema 与 M8 指标，未做候选挖掘、训练、生成或评测。E3 日志是拼接文本和汇总指标，完整片段恢复尚未验证；独立数学验证、衍生题去重和等 token 对照是后续设计要点。未批准 HER 实验或改变 M9/M10 状态；M9 formal 仍待授权。复核入口：`reports/qa_log.md` Q16、`rl/custom/turn_advantage_v2.py`、`rl/runs/e3_grpo_baseline_20260819_224555/predictions/train/1.jsonl`。

- **2026-09-06：M9 步骤 0–2 完成，步骤 3（200-step formal）未启动，等用户检查后再授权。**
  `plans/M9.md` 已批准；状态、偏差（§11 四条）、验收（§12 第 1/2 行）已回填。训练侧实现全部是新文件且**没有对 veRL
  做任何 patch/子类/包裹**：`rl/configs/e3notool_grpo.yaml`、`rl/launch/e3notool_grpo.py`、
  `rl/validate_e3notool_run.py`、`scripts/m9/{run_e3notool.sh,watch_e3notool.py,watch_e3notool.sh}`，
  测试 `rl/launch/test_e3notool_config.py`（Fixture J/L）、`rl/custom/test_m9_notool_reward.py`（K）、
  `rl/test_validate_e3notool_run.py`（N）。treatment 就是把工具拿掉：`multi_turn.enable=false` +
  veRL 原生 `single_turn_agent` + `agent_loop_config_path=null`；E3→E3-NoTool 的 resolved-config diff
  **精确等于**预注册的 5 条路径。全仓 first-party 回归 `pytest -q --ignore=third_party` **237 passed**。
  preflight `analysis/m9_preflight.json`：磁盘 227.91 GiB 可用（下面 66 GiB 一条已确认是过期快照，
  未删除任何历史产物），formal 门槛 61.63+30=91.63 GiB，GPU 单卡 142.2 GiB 空闲。
  smoke `rl/runs/e3notool_grpo_smoke_20260906_213322/`（5/5，`save_freq=-1` 无 checkpoint，
  `analysis/smoke_gate.json` 全绿）：2,560 条 = 5 步 × 64 组 × 8；tool 计数/截断/parse error 逐条恒 0；
  prompt 无 schema marker（10 条解码 prompt 落盘 `analysis/prompt_audit.{jsonl,md}`）；`<tool_call>` 标签
  吐出率训练 5 步与验证 step 0/5 **全为 0**（起点根本不吐标签，§4 门禁与 §2.2 的起点债务担忧都比预期轻）；
  固定 panel 0.62→0.67；response length 931→857 token、3072 截断率 3.5%→1.4%、entropy 0.211→0.205、
  KL 2.7e-4→6.2e-4；组内零方差 44.7%。同 seed step-1 描述性对照（**非结论**）：E3 acc .447 / format .842 /
  截断 10.6% / mean calls 1.59 / 零方差 26.6%，E3-NoTool .488 / .963 / 0 / 0 / 46.9%。
  单步约 134 s，200 步预计约 8 h（计划 §9 估 6–10 h）。**未做**：评测侧 `eval/checkpoints_v5.py`、
  `eval/build_diagnostic_freeze_v5.py`、`eval/verify_diagnostic_freeze_v5_semantics.py`、
  `eval/m9_notool_eval.py`、`scripts/m9/run_m9_eval.sh` 与 Fixture M —— 顺延到步骤 4 前（v5 freeze 必须
  绑定尚不存在的 step-200 materialization），已记入 §11。

- **2026-09-06：完成秋招简历 bullets 提炼。** 新稿 `reports/resume_bullets_autumn.md` 含四条可投递表述与证据口径，核对 M7/M8 metrics、SFT 数据和轮级信用实现；区分 MATH500 +17.2pt 与 FULL760 +12.8pt，避免将 v2 多项联合改动解释为单一因素因果证明。仅文档整理，未启动训练或评测；M9/M10 仍为待批准草案。

- **2026-09-06：M8 后复盘完成；`plans/M9.md` 已批准并开工（见上条），`plans/M10.md`（E7 零方差过滤）仍为草案待批准。**
  当时未创建任何代码、config、launcher、测试或 run。qa_log Q14 记录了 M8 判读、"工具是否提高准确率"的因果空洞（无 CoT-RL 对照）、
  E5-v2 缩小自身作用面的新观察（零调用轨迹 48.6% vs E3 26.6%）与下一步排序（M9 先行，M10 需 §8 步骤 0 专项批准）。
  新增只读脚本 `analysis/tool_use_conditional_accuracy.py` 及 JSON 输出（相关性证据，有选择偏置）。df 于 2026-09-06 显示
  约 229 GiB 可用，与下一条的 66 GiB 不一致，启动任何 run 前需现场复核。
- **2026-09-05：M8 / E5-v2（守恒轮级信用）完成，tag `m8`；预注册判读 §9.2 情形 A。** 训练
  `rl/runs/e5v2_conserved_credit_20260905_003041/`（200/200，step 147 pod 中断后从 125 恢复，双段 proof + recovery ledger，
  `analysis/formal_completion_gate.json`）；评测 `eval/runs/m8_e5v2_eval_20260905_134737/`（协议 v4，E5-v2 3,800/3,800，
  复用 M7 15,200 条，`metrics/m8_headline.json` 含机械判读）。FULL760 greedy E3/E5-v1/E5-v2 .733/.722/.741，E3→v2 Δ +0.79pt
  CI [−1.84,+3.42]；4-call 92.9%→0.8%、截断 100%→8.7%、mean calls 0.60；early AUC +1.87pt < 2pt。结论：v1 的行为漂移源于
  非守恒隐式奖励；守恒再分配在 ≤4 轮 horizon、exposure 10.4%→4.2% 下未检出增益。代码全在新模块（v1 冻结文件未动）：
  `rl/custom/turn_advantage_v2.py`、`rl/custom/turn_credit_v2_agent_loop.py`、`rl/launch/e5v2_conserved_credit.py`、
  `rl/validate_e5v2_run.py`、`eval/checkpoints_v4.py`、`eval/build_diagnostic_freeze_v4.py`、
  `eval/verify_diagnostic_freeze_v4_semantics.py`、`eval/m8_e5v2_eval.py`、`scripts/m8/`。计划/偏差/验收：`plans/M8.md` §12/§13。
  **待用户决定**：磁盘 66 GiB 可用，历史产物删除（M8 run 目录 63 GiB 含 3 个 checkpoint，只有 step-200 被物化引用）留待用户裁定；
  §3.4 `|A_traj|` 缩放变体、E7、额外 seed 均未授权。
- **2026-09-04：M8 / E5-v2 计划草案已写入 `plans/M8.md`，待用户批准；未创建任何代码、config、launcher、测试或 run。**
  动机是 E5 v1 修正项在轨迹内不守恒（隐式稠密奖励），定量证据与推导见 `reports/qa_log.md` Q12；
  离线套用 v2 公式已验证逐轨迹修正和精确为 0、SFT 起点附近 treatment exposure 约 14.6%（`plans/M8.md` §4，
  即席脚本未落盘，步骤 0 需正式实现）。简历草稿在 `reports/resume_bullets.md`。
  批准后执行顺序与逐步授权门禁见 `plans/M8.md` §8；范围外事项见 §1。
- **2026-08-28完成E3 70条budget/truncation post-hoc诊断，不训练。** 原70条按互斥机制拆为
  response-token 58、assistant/tool-turn 2、repeated normalized code 10（正交停止信号仍为63 token、7 turn）。
  唯一长预算run `eval/runs/e3_budget_diagnostic_20260828_095658/`仅把3072/4/5提高到6144/6/7，其他
  prompt/model/greedy/seed/tool/verifier保持不变；16/70转正确，54仍错。54中只有14条正常结束后答错，38条
  再次撞token上限、2条再次撞turn上限；38条中34条零工具、30条有至少三次完全相同的非短输出行，主模式是
  verbal loop而非预算略短。诊断题永久禁止回灌训练；未来训练数据只能来自train split并重新污染检查。详细方法、
  边界和证据见`plans/M7_BUDGET_DIAGNOSTIC.md`；diagnostic与canonical两套hash ledger均通过。
  该轮关于成功率、失败中间步骤、解决方向和GRPO零方差group的完整会话总结见
  `reports/docs/grpo_failure_diagnostic_session_summary.md`。
- **M7 canonical generation、strict scoring、downstream analysis与documentation synthesis已完成并由用户
  最终验收（2026-08-25）；completion commit与tag `m7`已获授权。** 唯一run
  `eval/runs/m7_unified_eval_20260824_201032/`在frozen v3下完成greedy 3,040、sampled 12,160，raw/scored
  均为exact 15,200/15,200；missing/unexpected/duplicate/conflict/infra failure均为0。full exact-key digest为
  `c43ce719…fa8b8`，3,800个跨role pairing group与15,200个UID完整。canonical `hashes.sha256`及17项
  downstream `analysis_hashes.sha256`均通过。
- M7 FULL760 greedy raw/SFT/E3/E5为426/460/557/549 correct，即`.561/.605/.733/.722`。paired
  Raw→SFT为+4.47pt、95% CI `[+1.05,+7.89]`；SFT→E3为+12.76pt `[+9.74,+15.92]`；E3→E5为
  −1.05pt `[−3.68,+1.58]`。sampled matched-index结论同向。**E3是最强endpoint；E5无final accuracy
  benefit。** M7 endpoint-only不能验证M6 faster-early-learning。
- E5有强烈跨source behavioral treatment effect：E3→E5 mean calls greedy `.888→3.755`、sampled
  `.958→3.935`；4-call `1.3%→92.9%`与`1.2%→97.2%`，四source均出现，per-call success约升至98%。
  定性为credit-signal/proxy gaming，不自动等同classic reward hacking；E4是更干净的reward-shaping
  exploitation案例。
- frozen taxonomy覆盖15,200行（10,085 correct、5,115 wrong、0 unclassified）；192条stable-hash
  checkpoint-blinded manual Codex audit不是独立human audit，总agreement 151/192。E5 agreement仅greedy
  6/24、sampled 3/24，因此**不得科学解释E5 coarse taxonomy migration**；exact outcome/behavior/four-cell仍有效。
  三段×两setting共11,400 paired rows与10,000次source-stratified question-cluster bootstrap齐全。
- required docs已更新：`reports/02_main_results.md`、`03_badcase_taxonomy.md`、新建
  `04_reward_hacking.md`与`interview_outline.md`，以及README、technical report、M7 plan、LOG/HANDOFF。
  canonical `summary.md`因受`hashes.sha256`保护保持不变，新增`final_summary.md`记录下游结论。
- behavioral-identity fixture通过真实veRL state machine证明M7 metadata-only loop与`ToolCreditAgentLoop`的
  prompt/response IDs、mask、turn、metrics和baseline extras完全相同，唯一新增`turn_credit_ledger` audit
  metadata。greedy paired transitions为`primary_diagnostic`；sampled matched-index为
  `secondary_exploratory`并禁止强因果解释。
- E3/E5 step-200已用pin veRL 0.8.0官方FSDP merger物化到`eval/checkpoints/`下两个唯一HF inference路径；
  每个export为11文件、310 tensors、1,720,574,976参数。外部identity manifests分别为
  `eval/materialization/e3_grpo_baseline_step_200.json`（`b543647a…95db1`）和
  `eval/materialization/e5_turn_credit_step_200.json`（`78441f51…65ae3`）。
- v3 runtime freeze状态为`FROZEN_V3_SMOKE_PASSED_AWAITING_CANONICAL_REVIEW`：66-file manifest
  `b765ec2c…5b05a4`、protocol `c461d22b…36212`。semantic verifier证明parent v2 37-file manifest
  `6fa185e1…07f3f0`及panel/generation/tool/verifier/taxonomy/pairing/bootstrap/data-isolation语义未变，
  `gpu_generation_performed=false`（freeze时事实）。smoke完整保留resolved bindings、materialization manifests、32+32 shards、
  strict scores、server/driver logs、exact closure、descriptive-only metrics与hash ledger。E5在四题smoke中20/20均到达
  assistant-turn budget，与M6 over-calling风险一致，但不得从四题外推prevalence或performance effect。
- **M6 / E5 Tier A 已完整验收（2026-08-24）。** formal run
  `e5_turn_credit_20260823_082012`完成200/200；status/tracker=completed/200，step-200完整checkpoint可读。
  102,400条canonical train trajectory、363,350个turn、两个checkpoint-bound wrapper proof segment全部
  通过formula/span/mask/UID和restore检查；旧126–138及受控停止的旧126均留在recovery archive，没有混入。
- E5 fixed-100 curve为`0.61/0.69/0.73/0.73/0.74/0.75/0.71/0.70/0.76`；E3为
  `0.60/0.67/0.67/0.70/0.73/0.73/0.74/0.77/0.76`。E5 0–100 AUC `0.70625`比E3
  `0.67625`高3pt，但final同为0.76，0–200 AUC只高0.005625。
- step-200 paired结果为5 fixed、5 new、71 unchanged-correct、19 unchanged-failed；accuracy delta 0，
  10,000次paired bootstrap 95% CI=`[-0.06,0.06]`。conditional fix 5/24，new failure 5/76。
- mechanism audit覆盖全部rollout并固定抽30条mixed-quality trajectory。native `A_traj`同轨迹恒定；
  213,403个useful turn中99,996个获正correction，47,806个非no-tool问题turn中41,444个获负correction，
  方向违约为0。treatment确实生效，不是接线空开关。
- 结论为**mixed/negative**：early更快但final无增益，并出现显著over-calling。训练mean calls
  `0.963→2.551`、4-call fraction `2.71%→44.05%`、repeated code `2.02%→39.71%`、truncation
  `1.70%→3.42%`；error recovery `41.17%→40.98%`基本不变。fixed panel E5有93/100条4-call tag，
  E3仅2/100。
- canonical smoke、人类adoption gate（14 TP/1 FP，precision 0.9333）、freeze v2 closure
  `6fa185e1…b07f3f0`、formal completion、30条mechanism audit、checkpoint-blinded 48 failure裁决、
  paired matrix/secondary transitions/10k bootstrap和M6文档全部齐全。Codex taxonomy裁决没有冒充human audit。
- M6收尾时first-party全量回归为**150 passed, 1 skipped**；freeze v2语义验证和37-file
  `sha256sum -c`再次通过，analysis JSON可解析且30/48/100行审计产物数量吻合。
- 按冻结协议没有调β、改heuristic、追加seed、实施Tier B/GiGPO、启动E7或运行M7 generation/evaluation。
- **M5 已完成 E6、E4-A、E4-B，处于阶段验收状态**（2026-08-22）。E7 exact implementation
  boundary 设计审查已经完成；用户明确将 E7 有意 defer 到 M6/E5 完成之后。E7 trainer/config/launcher
  未创建，smoke/full run 未启动；这不是实现失败、blocked 或取消。M5 不 tag。
- E6 `e6_nomask_20260820_235621` 完成 80/80：mask 操纵实际覆盖 4.662% loss token，固定 panel
  `0.61→0.71→0.74→0.68→0.70`（step 0/25/50/75/80），未触发 early stop，也未观察到灾难性退化。
- E4-A `e4a_exec_only_20260821_040632` 与 E4-B `e4b_joint_shaping_20260821_153100` 均完成
  200/200。A/B final pass@1 为 0.77/0.76，0–100 AUC 为 0.66625/0.68500；E3 为
  0.76/0.67625。A 明显增加工具调用和 hacking candidates；B 相对 A 小幅缓解这些启发式信号，
  但 penalty 暴露很弱，不能解释为稳定能力提升。
- A/B 各经历一次 JupyterHub 中断，分别从最后完整 checkpoint 125/150 恢复；旧重算区间和半写
  checkpoint 已归档，未混入正式曲线。E4-B 通过 smoke 后，其 21 GiB smoke checkpoint 已按用户
  指令删除，resolved config、轨迹审计、metrics、summary 和 cleanup ledger 保留。
- 最终分组回归为 **105 passed, 1 skipped**；pod 对 `/etc/hostname` 的拒绝由 EACCES 变成 EROFS，
  sandbox test 现接受两种内核错误，同时仍断言删除失败且文件存在。
- **M4 / E3 已完成并验收通过**（2026-08-20，tag `m4`）：正式 run
  `e3_grpo_baseline_20260819_224555` 完成 200/200 step，固定 MATH500-100 greedy pass@1
  **0.60→0.76**（峰值 0.77）。KL、entropy 与响应长度无病态；训练前/后 25-step 的工具
  错误率 12.84%→7.66%、格式无效率 10.30%→3.14%、截断率 6.30%→0.55%。
- 最终 checkpoint `global_step_200` 含 model/optimizer/scheduler/RNG/DataLoader 状态；
  200 份训练预测、9×100 条验证预测、resolved config、metrics、summary 与 TensorBoard 齐全。
  数据 schema/哈希/split/污染检查通过，真实 pod 权限下 **74 tests passed**。
- JupyterHub 中断恢复证据保留在 run 的
  `recovery/resume_from_150_20260820_100834/`：中断前 step 168 完成、169 未完成，从最新完整
  checkpoint 150 恢复并重算 151–168；旧预测已归档，未与恢复后正式曲线混用。
- **M3 验收通过**（2026-07-14，tag `m3`）；RL 统一起点继续是
  `sft/checkpoints/qwen3-1.7b-sft`。2026-08-19 的 30B teacher/full SFT 最小复核均未支持
  切换起点，产物与结论见 `plans/M3.md`。

## 下一步（需用户另行授权）

1. M7 canonical仍冻结；70题budget diagnostic只作post-hoc诊断，不改变M7 denominator，也不进入训练数据。
2. E7、Tier B、额外seed、β sweep、post-hoc tuning、新训练和继续提高诊断预算仍禁止。
3. 若用户未来选择恢复E7，先重新review `plans/M5_E7_IMPLEMENTATION_REVIEW.md` 的pin源码、exact boundary、
   compute/storage和专项授权；M6完成不自动批准E7。
4. 保持E3/E4/E5/E6正式产物、M6 freeze与recovery archives不变；~~不删除checkpoint腾空间~~
   **（2026-09-11 更新：该条写于 M7 期，此后用户已多次逐条授权删除已物化或纯残留的 raw checkpoint——E4-A/E4-B/E6、
   M9 三个轮换残留、offload-check、E7 resume-check、2026-09-11 的 E7 七个轮换残留。现行口径是：**只删有 cleanup 记录、
   无 HF 物化引用、无 freeze closure 引用、且 run 已 completed 的 checkpoint，每次删除前须用户点名授权并落
   `checkpoint_cleanup.json`**。已物化来源与 tracker 指向的 checkpoint 一律保留；特别地，
   `rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200` 是第 6 条同预算臂的 resume 起点，**不得删除**。）**
5. **【2026-09-09 已执行，结果见"当前状态"首条：GPU/数值通过，CPU 泄漏斜率未消除，建议关 offload + 保留分段恢复】待办（用户 2026-09-07 指定，M9 全部跑完后执行）：验证"关掉 offload"。** 背景见 `reports/qa_log.md` Q17 与
   `plans/M9.md` §11（2026-09-07 行）：所有 200-step formal run 都因容器 `memory.max`=128 GiB + `memory.oom.group=1`
   在 step 138–172 被整 pod 杀死；CPU 内存主要来自 veRL FSDP offload 的 pinned 池（actor worker 内约 45 GB 的
   `/dev/zero` 共享映射，只增不还）与 actor worker 匿名内存逐步增长，checkpoint 保存再叠加瞬时峰值。
   验证内容：把 actor `fsdp_config.param_offload`/`optimizer_offload` 与 ref `fsdp_config.param_offload` 置 false
   （GH200 142 GB 显存，sglang 占 0.5 后剩 71 GB；1.7B 的 fp32 参数+梯度+Adam+bf16 ref 常驻约 31 GB），跑一次
   5-step smoke：(a) 同 seed 对照 E3 前 5 步 reward/loss 曲线在浮点噪声内一致；(b) 显存峰值不 OOM（GPU OOM 会在
   日志抛异常，不像 CPU 组级 OOM 无声消失）；(c) `scripts/m9/memwatch.sh` 记录的 cgroup 曲线变平；(d) 单步时间对比。
   通过后它成为 resolved config 相对 E3 的三条新 diff：加进 M10+ 的 diff gate 允许集合，并在对应计划 §11 记为
   "纯基础设施差异、不影响数值"。**M9 这条 run 不改**。备选：200 步在 step 100 计划性分段，用现有 resume 机制续跑。

6. **【待办，用户 2026-09-11 明确"暂时不做，下次做的时候再说"——未授权启动，等用户点名】E3 同预算（equal-budget）对照臂。**
   **为什么需要**：M10 判读是情形 C（等 step、等算力两轴都持平），但附报里 E7 step-200 比 E3 高 **+5.13pt [+2.63, +7.63]**。
   这条增益目前**无法归因**：E7 在 200 有效 step 里烧掉 259,072 条轨迹（E3 的 2.53×），而 E3 只跑了 102,400 条就停在 step 200。
   缺的是"E3 也花 259,072 条轨迹会到哪"。没有它，+5.13pt 只能写成"多花算力换终点"，不能写成"过滤更强"；
   这是 M10 唯一的科学缺口，也是 `reports/02_main_results.md` M10 节与 technical report §13 明确声明的限制。
   **怎么做**（两个方案，二选一，执行前需新计划 + 新授权）：
   - (a) **首选，续训 E3**：从 `rl/runs/e3_grpo_baseline_20260819_224555/checkpoints/global_step_200` 继续跑到累计 259,072 条轨迹，
     即再 **306 步**（64 prompt × 8 rollout = 512 条/步，(259072−102400)/512 = 306）。按 E3 单步 161 s 估 **约 14 h**；
     加关 offload 后的观测斜率（gen 稍慢）宜按 15–16 h 预留，并沿用 M10 的分段恢复（`--stop-at-step`）与 memwatch。
     **resume 前提已于 2026-09-11 核实**：该 checkpoint 完整在盘（21 GB），`actor/{model,optim,extra_state}_world_size_1_rank_0.pt`
     + `actor/fsdp_config.json` + `actor/huggingface/` + 根目录 `data.pt`（dataloader 状态）齐全，tracker = 200，
     即优化器与数据顺序状态足以真正续训，不必退化为从 step-200 权重冷启动。
   - (b) **退化方案，E7 侧截断多点比较**：在 E7 已有 ledger 上按 E3 预算的整数倍取多个累计轨迹点比较，不新训练。
     便宜但只动 E7 一条曲线，回答不了"E3 多花算力会怎样"，只能作旁证。
   **前置条件**：(i) `global_step_200` 与 `equal_compute_step_95` 两个 E7 checkpoint 必须还在（2026-09-11 清理已明确保留）；
   (ii) E3 原 run 的 step-200 checkpoint 必须还在，**清理磁盘时不得删**；(iii) 评测要沿用协议 v6 的 FULL760 口径，
   新臂作为新 role 接进去，不得改动已冻结的 13 个语义字段；(iv) 判读措辞在计划里预注册，不得看到结果再定。
   **预算**：训练约 14–16 h + 一个 role 的评测（M10 单 role 约 4–5 h，视 pod CPU 争用）+ 一个 21.9 GB checkpoint。

## M7 复现与证据指针

- canonical run：`eval/runs/m7_unified_eval_20260824_201032/`；计划、执行偏差与验收：`plans/M7.md`。
- canonical closure：`metrics/completeness.json`、`metrics/full_scored_completeness.json`、`hashes.sha256`。
- performance/behavior：`metrics/pass_at_k.json`、`metrics/tool_behavior.json`。
- taxonomy/audit：`taxonomy/coarse_metrics.json`、`taxonomy/audit_metrics.json`、`taxonomy/adjudications.jsonl`。
- paired/bootstrap：`transitions/{raw_to_sft,sft_to_e3,e3_to_e5}.json`、`paired_rows.jsonl`、
  `bootstrap.json`；downstream integrity：`analysis_hashes.sha256`。
- synthesis：`final_summary.md`、`reports/02_main_results.md`、`reports/03_badcase_taxonomy.md`、
  `reports/04_reward_hacking.md`与`reports/interview_outline.md`。
- 只读重验：run目录执行`sha256sum --quiet -c hashes.sha256`及
  `sha256sum --quiet -c analysis_hashes.sha256`；分析单测使用conda `toolcredit`环境运行
  `python -m pytest -q -p no:cacheprovider analysis/test_m7_analysis.py`。

## M6 复现与证据指针

- formal run：`rl/runs/e5_turn_credit_20260823_082012/`；计划、偏差与验收：`plans/M6.md`。
- 只读重验：`python -m rl.validate_e5_formal_completion <run_dir>`；分析：`python -m rl.finalize_m6`。
- 主结论：`reports/02_main_results.md`；paired taxonomy：`reports/03_badcase_taxonomy.md`；技术叙事：
  `reports/technical_report.md` §8。
- 机器证据：run内`analysis/formal_completion_gate.json`、`m6_e5_completion_manifest.json`、
  `e3_e5_performance_behavior.json`、`mechanism_analysis.json`、`mixed_quality_mechanism_audit.jsonl`、
  `paired_failure_transition_analysis.json`与`paired_failure_transitions.jsonl`。
- failure labels：`rl/m6_failure_adjudications.jsonl`；这是checkpoint-blinded Codex manual adjudication，
  不算独立human audit。smoke adoption gate才是用户确认的人类审计。

## M4 复现与证据指针

- 配置：`rl/configs/e3_grpo_baseline.yaml`；launcher：`rl/launch/e3_grpo_baseline.py`；
  后台入口：`scripts/m4/run_e3.sh`；监控：`rl/monitor_run.py`。
- 正式 run：`rl/runs/e3_grpo_baseline_20260819_224555/`；详细验收与执行偏差：`plans/M4.md`；
  技术叙事：`reports/technical_report.md` §6。
- M4 数据：源训练池 5203 条、有效训练池 5195 条、固定验证 100 条；manifest 位于
  `rl/data/manifest.json`，污染报告位于 `data/contamination_report_m4.md`。
- 核心验收命令：

  ```bash
  conda run --no-capture-output -n toolcredit pytest -q \
    env/test_sandbox.py rewards/test_rewards.py sft/test_trace_tokenizer.py \
    rl/custom/test_masking.py rl/custom/test_m4_adapters.py \
    rl/launch/test_e3_config.py rl/test_monitor_run.py rl/test_prepare_data.py
  ```

## M5 证据指针

- 权威计划与偏差：`plans/M5.md`；E7 边界审查：`plans/M5_E7_IMPLEMENTATION_REVIEW.md`。
- 正式 runs：`rl/runs/e6_nomask_20260820_235621/`、
  `rl/runs/e4a_exec_only_20260821_040632/`、`rl/runs/e4b_joint_shaping_20260821_153100/`。
- 三方比较：`rl/runs/m5_e3_e4_comparison/e3_e4a_e4b_comparison.json` 与 `summary.md`。
- 汇总叙事：`README.md` 结果总表；`reports/technical_report.md` §7（动机、接线、结果、限制、
  面试叙事和证据索引）。
- 当前回归命令见各 test 文件；涉及真实 tokenizer/veRL import 的测试需在 pod 权限下使用
  `conda run --no-capture-output -n toolcredit pytest ...`。

## 风险与边界

- M7已把M6的over-calling/no-endpoint-gain扩展到完整MATH500/AIME/GSM8K与两setting，但仍是单seed、
  fixed β和endpoint-only；不能用M7声称faster-early-learning泛化。E5 coarse taxonomy因blind agreement低不得
  作category migration科学解释，任何小分母category也不能外推为某类已解决。
- E5 over-calling仍是主要风险：position-aligned final/no-tool比较和visible-text adoption可能奖励重复调用/复述。
  未来方案若修改此语义必须独立version和全套对照，不能回写M6/M7 frozen result。
- M4 已验证 NFS 约 21 GB checkpoint 写入；JupyterHub 仍可能重启，长跑继续使用 conda
  `toolcredit` 环境中的 tmux、唯一 run name、完整 checkpoint 与低频 watcher。
- M6 completion gate时磁盘约189 GiB free；未来任务仍须按`remaining_peak_write + 30 GiB`现场重算，
  不能依赖该快照，也不能自行删除历史正式证据。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项），不阻塞服务器侧后续 Milestone。
- 不提交或改动 `reports/qa_log.md`、`.vscode/` 和 M3 未跟踪 checkpoint/tensorboard 产物。
