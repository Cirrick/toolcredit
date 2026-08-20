# AGENTS.md — ToolCredit 的 Codex 项目约定

本文件是 Codex 在本仓库中的唯一 agent 指令入口，适用于整个仓库。`CLAUDE.md` 是迁移前的
历史文件；若两者在 agent 工作方式上冲突，以本文件为准。研究目标、实验设计与验收标准仍以
`PLAN.md` 为唯一权威来源，不在本文件重复维护。

## 项目目标与当前状态

ToolCredit 研究多轮工具调用 RL 中的信用分配：比较标准 GRPO 将轨迹级优势广播到全部轮次，
与 turn-level 信用分配在训练效率、最终性能和模型行为上的差异。技术栈为 Python、veRL、
GRPO 和 Python 解释器工具。

- 当前进度：M4 / E3 GRPO baseline 已完成；下一阶段是 M5 / E6 no-mask 与 E4 shaping。
- RL 统一起点：`sft/checkpoints/qwen3-1.7b-sft`。
- 运行平台：GH200/aarch64 JupyterHub pod，无 Docker；conda 环境为 `toolcredit`。
- M5 尚无获批计划。没有用户新的明确开工指令时，不启动 M5、训练、批量生成或全量评测。

## 文档优先级

开始工作前按任务范围读取相关文档，不要只依赖本文件中的状态摘要。

1. 用户当前指令：决定本次工作的具体目标与边界。
2. `PLAN.md`：研究问题、实验矩阵 E3–E7、Milestone、验收标准和降级方案的唯一权威来源。
3. `plans/M<N>.md`：对应 Milestone 已批准的计划原文、执行偏差、决策与验收记录。
4. `HANDOFF.md`：最近一次交接状态、下一步和已知风险。
5. `environment.md`：实际平台、固定依赖版本和 GH200/aarch64 踩坑记录。
6. `README.md`、`reports/technical_report.md`、`LOG.md`：结果摘要、技术叙事和会话历史。
7. `CLAUDE.md`：仅作历史参考；其中仍有效的项目约束已迁移到本文件。

若文档之间出现状态冲突，先用 Git 历史、产物和测试做只读核对；仍无法确定且会影响实验设计、
训练成本或结果解释时，向用户说明冲突并等待确认，不得自行选择有利版本。

## 每次开始工作的检查

1. 阅读本文件、`HANDOFF.md`、当前 `plans/M<N>.md`，以及 `PLAN.md` 中与任务直接相关的章节。
2. 执行 `git status --short`，识别用户已有改动；不得覆盖、回退或顺手整理无关变更。
3. 核对实际文件、配置和 checkpoint 是否与文档状态一致。文档摘要不能替代产物检查。
4. 明确本次任务属于诊断、实现、实验执行还是验收；只做用户授权范围内的动作。
5. 修改前先搜索现有实现与测试，优先复用已有入口，避免平行实现同一逻辑。

## 实现与实验约定

1. **依赖完全固定**：新增或变更依赖时同步更新 `requirements.txt` 与 `environment.md`，记录
   GH200/aarch64 特有问题。不得只修改当前 conda 环境而不留痕。
2. **先复现再定制**：veRL 官方 multi-turn tool 示例已在 M0 跑通。后续实现继续沿用 pin 版本
   的官方交互格式和配置字段，不自创 tool-call 标记或凭记忆猜 schema。
3. **最小侵入 veRL**：不 fork veRL。框架适配使用最小 patch、子类或配置注入，统一放入
   `rl/custom/`；每次变更同步更新 `rl/custom/CHANGES.md`，说明修改位置、原因和行为差异。
4. **修改框架内部逻辑前必须询问用户**：先给出修改点、必要性、替代方案和验证方法，得到确认
   后再实施。配置注入或仓库内已有扩展点的正常使用不视为修改框架内部逻辑。
5. **单一沙箱实现**：训练工具复用 `env/sandbox.py` 及其中的 `prepare_tool_code`，不要复制官方
   示例的行级启发式形成第二套行为。
6. **reward 错误必须可见**：答案提取、格式解析、verifier 或 reward 计算失败不得静默跳过；
   必须记录原因并计入 `invalid_rate`。截断和工具错误也应进入轨迹元数据与指标。
7. **数据隔离**：训练与评测严格按 split 隔离。任何训练子集投入训练前必须通过
   `data/dedup_check.py` 污染检查；不得为了改善指标移动或泄漏评测样本。
8. **代码质量**：函数保持短小并添加 type hints；边界条件显式失败。关键张量 shape 在 debug
   模式打印，shape 不符立即报错，不用隐式 reshape 或广播掩盖问题。
9. **保持实验可比性**：除当前实验明确操纵的变量外，数据、起点 checkpoint、生成参数、评测
   入口和指标口径应保持一致。任何偏差写入对应 `plans/M<N>.md`，不得事后美化。

## 工具与编辑习惯

- 文本和文件检索优先使用 `rg` / `rg --files`。
- 修改文件使用小而集中的 patch；不要无关重排、批量格式化或覆盖用户改动。
- 新增实现前查找仓库内是否已有 helper、schema、配置或测试夹具。
- 不使用破坏性 Git 命令，不删除未确认来源的文件，不清理用户缓存或环境。
- 只在 `/home/jovyan/toolcredit/` 内修改项目文件。HF 缓存和 conda 环境只允许任务必需的新增，
  不得删除；需要其他外部写入时先征得用户同意。
- 不把 token、API key、W&B key、私有路径或数据写入仓库、日志和命令输出。

## 验证要求

修改后先运行与改动直接相关的最小测试，再按风险扩展。当前核心测试入口为：

```bash
pytest -q \
  env/test_sandbox.py \
  rewards/test_rewards.py \
  sft/test_trace_tokenizer.py \
  rl/custom/test_masking.py
```

- 测试以及训练应在 conda 环境 `toolcredit` 中执行；若当前环境不是该环境，使用
  `conda run -n toolcredit ...`，不要擅自重建环境。
- 数据或配置变更还需做 schema、样本量、split、去重和 resolved config 检查。
- 训练接线变更必须先跑既定 smoke test；正式长跑前核对工具、reward、mask、保存路径和监控指标。
- 无法运行某项验证时，明确说明未验证内容和原因，不得用“应该可用”代替结果。
- 不为通过测试而降低断言、吞掉异常或改变既定指标口径。

## 长任务与资源安全

训练、大批量生成和全量评测必须在 `tmux` 后台运行，日志落到项目目录；交互会话只负责提交、
检查和轮询，不得用前台长进程阻塞。启动前必须：

1. 获得用户对该实验或长任务的明确授权。
2. 检查 GPU、磁盘、输出目录、run name、resolved config 和恢复点。
3. 为每个 run 使用唯一名称，避免覆盖 checkpoint 或指标。
4. 按 `PLAN.md` 的时间盒监控；单步卡住超过预算 50% 时触发 §12.2 对应降级并报告用户，
   不原地死磕。
5. M4 长跑重点监控 DataLoader worker 收尾被杀和 NFS checkpoint I/O 风险。

## Run 产物与可追溯性

每个训练 run 必须落盘五件套：

1. resolved config；
2. checkpoint 或 adapter；
3. 预测 JSONL；
4. metrics JSON；
5. 一段 Markdown 摘要。

同时保留唯一 run name 及 W&B 或 TensorBoard 的配置快照。失败或中止的 run 也要保留足以诊断
的配置、日志与状态说明，不得只留下成功结果。

## Milestone 执行与交接

- 每个 Milestone 使用一份 `plans/M<N>.md`。获批的“计划原文”不得事后修改；执行变化只追加到
  “执行偏差与决策记录”，验收结果写入验收区。
- 完成验收时同步更新 `README.md` 结果总表和 `reports/technical_report.md` 对应章节，覆盖动机、
  方法、结果、限制和面试叙事。
- 用户提出且有长期价值的概念澄清、设计决策或面试素材，追加到 `reports/qa_log.md`，包含日期、
  场景、问题、浓缩答案和证据指针。
- 会话结束或形成可交接状态时更新 `LOG.md` 与 `HANDOFF.md`，写清已完成、未完成、下一步、
  复现命令、关键产物和风险。
- Milestone 完成后按项目约定 commit 并 tag；执行前再次核对 diff，提交中不得混入用户的无关
  修改。若当前任务未授权完成整个 Milestone，则不擅自提交或打 tag。

## 明确禁止

- 不 fork veRL，不在未获批准时修改其内部逻辑。
- 不静默忽略 reward、verifier、解析、工具或数据错误。
- 不启动未经用户明确授权的训练、批量生成或全量评测。
- 不修改已批准的计划原文，不伪造、补写或挑选实验结果。
- 不跨训练/评测 split，不跳过污染检查。
- 不覆盖已有 run 产物，不复用会造成混淆的 run name。
- 不删除项目目录之外的文件，也不回退用户现有改动。
- 不扩大到 `PLAN.md` 的非目标：SOTA 对齐、novelty/related work、多机分布式、深度 fork、VLM、
  web agent 或 SWE 任务，除非用户明确变更项目范围。
