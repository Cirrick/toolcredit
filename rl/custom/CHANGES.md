# 框架改动记录

原则（PLAN §1.3.4）：不 fork veRL；一切改动为最小 patch / 子类 / 配置注入，全部记录在此。

## [已应用] verl 0.8.0 sglang kernel 检查的一行修复

- **状态**：用户 2026-07-12 批准方案 A，已应用（sglang_rollout.py:87）。
- **文件**：`verl/workers/rollout/sglang_rollout/sglang_rollout.py`（site-packages 内，约第 87 行）。
- **问题**：sglang server 启动时 verl 先检查新命名包 `sglang_kernel`，不存在时上游
  `assert_pkg_version` 抛的是通用 `Exception`，而 verl 的 fallback 只捕获 `AssertionError`，
  导致永远不会检查旧命名 `sgl_kernel`——但 sglang==0.5.8（verl 0.8.0 官方 pin）配套的
  正是旧命名 sgl-kernel==0.3.21（已正确安装）。
- **修复**：`except AssertionError:` → `except Exception:`（一行）。
- **应用**：`patches/apply_verl_patches.sh`（幂等，环境重装后重跑即可）。
- **回滚**：`pip install --force-reinstall --no-deps verl==0.8.0`。
- **否决方案**：改用 vllm rollout（教程原生支持，`vllm<=0.12.0` 有 aarch64 wheel），
  零框架改动，代价是另装一套引擎且引入 vllm 自身的 aarch64 未知数。

## [M4 配置注入] 训练沙箱与复合奖励

- **状态**：2026-08-19 按 M4 已批准计划实现；没有修改 veRL site-packages。
- **工具扩展点**：`sandbox_tool.ToolCreditSandboxTool` 继承 veRL `BaseTool`，由
  `tool_config.yaml` 注入。它只负责适配 agent loop，代码预处理和执行统一调用
  `env/sandbox.py:prepare_tool_code/run_python`，并把调用数、成功数和错误数写入轨迹
  `extra_fields`。
- **reward 扩展点**：`reward.py:compute_score` 由 veRL `custom_reward_function` 注入，标量奖励
  直接委托 M2 的 `rewards/composite_reward.py`（E3 中 shaping 系数保持 0），同时返回 acc、
  格式、invalid、工具错误和截断分量供 rollout JSONL/验证指标审计。
- **行为差异**：替换 M0 官方教程的 HTTP `SandboxTool`、行级 auto-print 和无限制子进程；
  不改变 ToolAgentLoop、GRPO 优势、trainer 或 loss masking 实现。
- **AgentLoop 子类**：首次 M4 smoke 的 step-0 验证发现，未调用工具的轨迹没有
  `tool_*_count` 字段，而 veRL 0.8.0 会把不同 worker 的非张量列直接拼接，造成列长与 batch
  不一致。`ToolCreditAgentLoop` 仅在官方 `ToolAgentLoop.run` 返回后为三项计数补零，并通过
  `agent_loop_config_path` 扩展点注入；生成、状态机和 masking 均保持官方实现。第二次 smoke 又发现
  Hermes parser 会把畸形 JSON 只写日志后丢弃、dispatch 失败也不会进入工具计数；该子类在不改变
  状态转移的前提下记录 `tool_parse_error_count`，并补记未进入 BaseTool 的 dispatch/执行错误。

## [M5 / E6 配置注入] tool-return no-mask 对照

- **状态**：2026-08-20 按获批 M5 计划实现；没有修改 veRL site-packages。
- **AgentLoop 子类**：`ToolCreditNoMaskAgentLoop` 继承 M4 的 `ToolCreditAgentLoop`，完整复用生成、
  工具执行、parser 与终止逻辑，仅在 `run()` 返回后把实际 response 中原本为 0 的 tool/environment
  return mask 改为 1。prompt 仍不在 response tensor 中；padding 仍由 veRL postprocess 的 attention
  mask 保持为 0。
- **审计字段**：逐轨迹记录原始 policy token、原始 tool-return token 和 E6 loss token 数；reward
  入口只在这些字段存在时校验守恒并透传到 raw JSONL，E3 缺省返回 schema/数值保持不变。
- **注入与回退**：E6 通过独立 agent-loop registration 与 config 选择该子类；E3 继续使用
  `ToolCreditAgentLoop`。回退只需使用 E3 config，不涉及框架文件。

## [M5 / E4 配置注入] 可审计 reward shaping

- **状态**：2026-08-21 按获批 M5 计划实现；没有修改 veRL site-packages。
- **单一 reward 入口**：`rl/custom/reward.py:compute_score` 通过 veRL 0.8.0 原生
  `custom_reward_function.reward_kwargs` 接收 `lambda_exec`、`lambda_budget` 和 `budget`，继续委托
  `rewards/composite_reward.py`；E4-A/E4-B 没有复制 verifier 或 reward 主逻辑。
- **E3 回归边界**：E3 未传 kwargs 时仍构造全零 shaping 配置，返回字段集合与数值由 golden test
  锁定不变。只有显式传入 E4 kwargs 时才附加 `base_score`、`exec_success_fraction`、`exec_bonus`、
  `budget_penalty` 和 `n_tool_success`。
- **行为差异**：E4-A 仅设置 `lambda_exec=0.2`；E4-B 在相同配置上仅增加
  `lambda_budget=0.1, budget=3`。truncated 轨迹继续总分为 0，不被 shaping 救活；负计数、非有限
  系数和 breakdown 不守恒显式失败。
## 2026-08-21 — interrupted-checkpoint recovery audit

- `rl.launch.e3_grpo_baseline.archive_interrupted_attempt` now moves any
  `global_step_<N>` directory newer than veRL's atomic
  `latest_checkpointed_iteration.txt` tracker into the run's recovery archive.
  This preserves partial checkpoint-write evidence and prevents a resume from
  silently overwriting it. Complete tracked checkpoints and training behavior
  are unchanged.

## [M6 / E5 Tier A] turn-level advantage correction

- **状态**：2026-08-22 按获批 M6 Tier-A计划和 exact runtime boundary实现；没有修改 veRL
  site-packages，也没有复制 `TaskRunner.run()` 或 `RayPPOTrainer.fit()`。
- **AgentLoop 子类**：`TurnCreditAgentLoop` 继承 M4 loop，只在原生状态机 hook中记录 assistant policy
  token span、masked tool span、execution status和 model-visible post-truncation tool text。adoption candidate
  只来自实际编码的 visible text；raw stdout为 audit-only且不进入 `s_t`。
- **优势扩展**：`turn_advantage.compute_turn_credit_data` 先调用 pin veRL原生 GRPO，验证同轨迹 native
  `A_traj`广播、UID/group/span/mask守恒，再仅对 assistant policy span增加冻结的
  `0.5 * (s_t - s_bar_t)`；tool return和padding保持 0。
- **trainer boundary**：E5专用 Ray `TaskRunner` 在 driver生命周期内临时包裹模块级
  `ray_trainer.compute_advantage` 和 `RayPPOTrainer._log_rollout_data`，用 source hash、proof ID和
  `try/finally` fail-closed；E3 launcher/config不导入该 wrapper。回退只需使用 E3入口。

## [M8 / E5-v2] conserved within-trajectory turn credit

- **状态**：2026-09-04/05 按获批 `plans/M8.md` 实现；没有修改 veRL site-packages。**v1 文件一字未动**：
  `turn_advantage.py`、`turn_credit_agent_loop.py`、`rl/launch/e5_turn_credit.py`、`rl/analyze_e5.py` 均被
  M6/M7 freeze v2/v3 清单 hash 锁定（§11 只读），因此 v2 全部放在新模块，只 import 不改写。
- **优势扩展**：`turn_advantage_v2.compute_conserved_turn_credit_data` 先调用 pin veRL 原生 GRPO，再对
  `credit_eligible`（恰好一次成功解析的调用）turn 加 `0.5·(s_t − s̃)`，`s̃` 为轨迹内 token 加权均值；
  `|T|<2`、组内 `score` 零方差或 `s_t` 全相同时逐元素等于原生 GRPO。每条轨迹断言
  `Σ c_t·n_t = 0`（1e-6），并断言 gated 轨迹张量与原生完全相等。复用 v1 的 span/mask/UID 校验函数。
- **AgentLoop 子类**：`turn_credit_v2_agent_loop.TurnCreditV2AgentLoop` 继承 v1 loop，仅覆写 `_ledger`
  的版本字符串并在 `_finalize_ledger` 末尾追加 §3.2 条件 3（AST 归一化哈希重复，复用
  `rl/analyze_e4.py:normalize_code`）和 4（此前 assistant 文本含 `\boxed{`），写入 `s_t_v1`、
  `repeat_of_turn`、`after_boxed`、`credit_eligible`、`adoption_status_v2`；ledger schema/adoption 版本为
  `toolcredit_turn_ledger_v2` / `toolcredit_adoption_v2`。生成、工具执行、masking、v1 heuristic 不变；
  独立注册名 `toolcredit_turn_credit_v2_agent`。
- **trainer boundary**：`rl/launch/e5v2_conserved_credit.py:ConservedTurnCreditTaskRunner` 与 v1
  `TurnCreditTaskRunner.run` 逐 hook 相同（临时包裹 `ray_trainer.compute_advantage` 与
  `RayPPOTrainer._log_rollout_data`，`try/finally` 恢复），仅 proof 前缀 `e5v2-wrapper-v1:`、log marker
  `TOOLCREDIT_E5V2_WRAPPER_ACTIVE` 与 payload（`boundary_version = e5v2_runtime_wrapper_v1`）不同。
  preflight/freeze bundle/pin veRL hash 复用 v1 函数。
- **配置门禁**：`config_diff_gate` 要求 E5-v1→E5-v2 只在 `turn_credit.*`、agent loop 注册与
  project_name 上不同，E3→E5-v2 与 v1 的允许集合完全一致。smoke 用 `save_freq=-1` 不留 checkpoint。
- **验证器（非冻结文件）**：新增 `rl/validate_e5v2_run.py` 重算每条轨迹修正、守恒残差、零方差 identity 与
  §3.2 不变量；`rl/e5_recovery.py` 按 `boundary_version` 选择 proof 前缀并识别两种 log marker；
  `rl/validate_e5_canonical_smoke.validate_rollouts` 增加可插拔 `audit_validator`（默认 v1 不变）。

## [M9 / E3-NoTool] 无工具 GRPO 对照臂

- **状态**：2026-09-06 按获批 `plans/M9.md` 实现步骤 0–2；**没有对 veRL 做任何 patch、子类或包裹**。
  这是本项目第一个"零框架改动"的实验臂：treatment 就是把工具从 prompt 与 rollout 里拿掉。
- **agent loop 切换**：`rollout.agent.default_agent_loop` 由自定义 `toolcredit_agent` 换成 veRL 原生
  `single_turn_agent`（`verl/experimental/agent_loop/single_turn_agent_loop.py`），并把
  `agent_loop_config_path` 置空，因此 `rl/custom/tool_agent_loop.py` 与整套 turn-credit 模块在本臂中
  完全不被加载。原生 loop 调用 `AgentLoopBase.apply_chat_template(messages)` 时 `tools` 取默认值 `None`，
  prompt 里不会出现 `code_interpreter` schema；`response_mask` 恒为 1（无 tool return token 需要 mask）。
  该文件的 SHA256 已加入 `rl/launch/e3notool_grpo.py:M9_PINNED_VERL_HASHES`，启动前重验。
- **配置**：`rl/configs/e3notool_grpo.yaml` 由 `e3_grpo_baseline.yaml` 复制，只改
  `multi_turn.enable`、`agent.default_agent_loop`、`agent.agent_loop_config_path`、`trainer.project_name`。
  `multi_turn.tool_config_path` 有意保留 E3 原值——在 `enable: false` + 原生单轮 loop 下它不产生任何
  prompt 或 rollout 效果，保留它可以把 E3→E3-NoTool 的 resolved-config diff 压到预注册的最小集合。
- **launcher（新文件，E3 launcher 未动）**：`rl/launch/e3notool_grpo.py` 复用冻结 E3 launcher 的
  `compose_config` / `validate_run_target` / `launch` / recovery archive，自带校验是因为 E3 的
  `validate_m4_config` 硬断言 `max_user_turns == 4` 且要求 agent-loop 配置文件存在。结构 diff 工具
  （`diff_paths` / `resolved`）与磁盘 resume 估计直接从 M8 launcher import，保证跨 milestone 门禁语义一致。
- **reward 不变**：仍是 `rl/custom/reward.py:compute_score`。无 agent-loop tool metadata 时四个计数字段
  缺省为 0，四次调用的 `truncated` 规则永不触发，`format_ok` 退化为"存在可解析的 `\boxed{}`"。
- **验证器（新文件）**：`rl/validate_e3notool_run.py`。本臂没有 turn-credit ledger，因而没有
  trajectory UID，完整性改由 rollout JSONL 证明：每步 512 条 = 64 组 × 8，逐条断言 prompt 无 tool schema
  且 tool 计数恒 0，并逐 step 记录 `<tool_call>` 标签吐出率曲线（§4 门禁：step 50 之后 > 5% 暂停）。
- **评测侧（2026-09-07 补记）**：协议 v5 与四个 notool 角色同样**没有**任何框架改动。
  `eval/m9_notool_eval.py` 的 notool 生成不走 agent loop，直接用
  `verl.utils.chat_template.apply_chat_template(tokenizer, [user], tools=None, enable_thinking=False)`
  产出 prompt token（与训练时原生 `single_turn_agent` 逐 token 一致，见 `eval/test_m9_notool_eval.py`
  的 `test_notool_prompt_reproduces_the_training_rollout_input`），再调 SGLang 原生 `/generate`；
  采样参数、seed 派生、verifier、配对与 bootstrap 全部 import 冻结代码。

## [M10 / E7] DAPO 式零方差 group 动态过滤与补采样

- **状态**：2026-09-09 按获批 `plans/M10.md` v2 与 `plans/M10_STEP0_BOUNDARY.md`（§3 八处白名单、§4 方案 (b)、§5 resume 语义）实现；
  **没有修改 veRL site-packages，不 fork**。pin 复核：`trainer/ppo/ray_trainer.py` SHA256
  `de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66`、`trainer/main_ppo.py`
  `e3fe6e73b18d63367402a2570c5ba054a2b05443ac40a168524dbf545b1da392`；另 pin `checkpoint_engine/base.py`
  （`3640690f…2ae382`）、`utils/checkpoint/checkpoint_manager.py`（`97ad2cf3…cb38a`）与
  `utils/checkpoint/fsdp_checkpoint_manager.py`（`87036a65…b37d10`），因为等算力 checkpoint 的"改名脱离轮换"
  与"chunk 间 replicas 保持 awake"依赖它们的语义。启动时 launcher 与 trainer 各自重验。
- **trainer 子类** `rl/custom/dynamic_filter_trainer.py:DynamicFilterRayPPOTrainer(RayPPOTrainer)`：
  - `fit()` 由脚本从 pin `ray_trainer.py:1362–1770`（409 行）生成，只含 STEP0 §3 的白名单差异，每处带 `# E7-W<n>`：
    W1 `:1389` epoch 改按 ledger 已消费批数整除；W2 `:1422–1423` 双层 `for` 改为显式 `iter()` + `while`（`:1716` 的
    `training/epoch` 随之读 `self._e7_epoch`）；W3 `:1435–1463, 1466–1525` 整段替换为一次 `_e7_collect_informative_batch()`
    调用；W3a `:1464` `is_last_step` 取 `min(total_training_steps, _e7_stop_step())`；W4 `:1671` 之后追加等算力保存；
    W5 `:1683` 之后追加 `_e7_log_candidates()`；W6 `:1718` 之后追加 `e7/*` 指标；W7 `on_batch_end` 不变。
    Fixture R（`rl/custom/test_dynamic_filter.py`）逐行 diff：被删/改的上游行只允许落在
    `{1389}, {1422–1423}, {1435–1464}, {1466–1525}, {1716}`，新增行必须带标记，其余 `1362–1388, 1390–1421, 1424–1434,
    1526–1670, 1672–1682, 1684–1715, 1717–1770` 逐字节相同；helper 内每个 chunk 逐字复用
    `:1435–1449, 1461–1462, 1466–1470, 1472–1480, 1495–1501, 1518–1525`，选定后对 64×8 batch 逐字执行一次 `:1502–1517`
    与 `:1518–1525`。REMAX 分支去掉（`assert adv_estimator == GRPO`）。
  - `_e7_collect_informative_batch()`：每 chunk 从同一冻结 actor 生成 64×8，按 `rm_scores.sum(-1)` 判零方差
    （校验它与 reward extra `score` 逐元素相等、有限、长度 512），最多 4 chunk，取采样顺序前 64 个 informative group；
    `:1471` 的 `sleep_replicas()` 移到 `finally` 恰执行一次；补采样期间 `_update_actor` 被守卫（`_e7_generation_phase`）
    拒绝、断言 `global_steps` 不变；chunk 间张量键与 response 填充形状不一致即失败；4 chunk 仍不足 64 → 落盘
    `predictions/candidates/<step>.underfilled.jsonl` 并抛 `E7UnderfilledError`（在 old-log-prob 之前，不做 update）。
  - `_save_checkpoint()`：先原子写 `global_step_N/e7_ledger.json`（ledger step 必须等于 `global_steps`），再 `super()`；
    `_load_checkpoint()`：`super()` 后读同目录 ledger，step 不符/缺失即失败；对上游 `:1095–1103` 的 epoch-boundary 启发式
    （`global_steps % len(dataloader) == 0` 时静默跳过 dataloader 恢复）**显式失败**；用磁盘上的 `equal_compute_step_*`
    目录校正/校验 ledger 的等算力标记；重写 run 根目录 `e7_ledger.jsonl` 去掉 checkpoint 之后的陈旧行。
  - 等算力 checkpoint（W4）：worker `save_checkpoint(..., max_ckpt_to_keep=None)` 写到 `equal_compute_step_N.saving/actor`，
    加 `data.pt` 与 ledger 后 `os.replace` 改名为 `equal_compute_step_N/`，不写 tracker。**行为差异**：worker 侧
    `previous_saved_paths` 仍登记了 `.saving` 路径（`remove_previous_save_local_path` 对不存在路径直接跳过，所以永不删除
    真实目录），但该幻影条目占用一个轮换槽位约 3 次常规保存——等算力保存后的两个常规周期内实际保留的常规 checkpoint
    为 2 个而非 3 个（更早的一个提前一个周期被删）。不影响数值、不影响 resume（只用最新 tracker checkpoint）。
  - 分段（§3.5）：`E7_STOP_AT_STEP` 类属性由 launcher 在驱动 actor 进程内设置（不进 resolved config，resume 的逐字
    config 比对不受影响）；watcher 写 `<run_dir>/e7_stop_request.json` 时在下一个 `save_freq` 倍数处停止。
  - 证据：`e7_ledger.jsonl`（每有效 step 一行）、`predictions/candidates/<step>.jsonl`（所有候选 group 行，含
    `role ∈ {selected, surplus, zero_std}`，比 STEP0 W5 写的"zero-std + surplus 行"多出 selected 行，用于守恒与
    "selected 多重集 == train JSONL"证明；不含文本）、`e7_trainer_active.json` + `e7_trainer_activations.jsonl`
    （启动打印 `TOOLCREDIT_E7_TRAINER_ACTIVE`）。
- **驱动边界**（STEP0 §4 方案 (b)，`rl/launch/e7_dynamic_filtering.py:DynamicFilteringTaskRunner`）：mixin `run()`
  在驱动 actor 内临时把模块全局名 `verl.trainer.main_ppo.RayPPOTrainer` 绑定到 `DynamicFilterRayPPOTrainer`
  （`main_ppo.py:299` 按名解析），`try/finally` 恢复；已被替换时拒绝叠加；proof `e7_boundary_installation.json`
  （installed/restored）+ `e7_boundary_activations.jsonl`，启动打印 `TOOLCREDIT_E7_BOUNDARY_ACTIVE`。0 行复制。
- **配置门禁**：`rl/configs/e7_dynamic_filtering.yaml` 由 E3 复制；`config_diff_gate` 要求 E3→E7 的 resolved diff **精确等于**
  `{algorithm.filter_groups, trainer.toolcredit_trainer_class, trainer.project_name, rollout.trace.project_name,
  actor.fsdp_config.{param_offload,optimizer_offload}, ref.fsdp_config.param_offload}` 并断言 `filter_groups` 恰三键
  `{enable: true, metric: score, max_num_gen_batches: 4}`；smoke / resume_check 只在
  `trainer.{total_training_steps,save_freq,test_freq}` 上不同。
- **验证器与脚本（非冻结文件）**：`rl/validate_e7_run.py`（smoke / resume-check / segment / formal / streak 门禁）、
  `scripts/m10/run_e7.sh`（tmux 入口，单段单次调用）、`scripts/m10/memwatch_v2.sh`（60 s，按进程 `Pss_Anon`，
  `step` 字段改用 `/usr/bin/grep -a`）、`scripts/m10/watch_e7.py`（anon > 110 GiB / 连续 10 步满 4 批 → stop request）、
  `analysis/m10_memwatch_report.py`、`analysis/m10_compute_axes.py`（§9 AUC 规则）。
- **回退路径**：使用 E3 launcher/config 即回到原生 `RayPPOTrainer`；E7 文件均为新增，删除即可。
- **评测侧（2026-09-10 补记）**：协议 v6 与两个 E7 角色同样**没有**任何框架改动。两个角色都是 TIR，
  因此 `eval/m10_e7_eval.py` 的生成路径直接复用冻结的 M7 metadata agent loop（`eval.generate.build_m7_loop`）
  与 SGLang HTTP 适配器，与 E3/E5/E5-v2 逐调用相同；v6 不新增任何 generation mode，整个 `generation_mode`
  块（含 M9 的 notool 语义）由 `eval/verify_diagnostic_freeze_v6_semantics.py` 作为**冻结字段**逐字节校验。
  等算力角色的定位 `checkpoints/equal_compute_step_<N>/` 中的 `N` 由 run ledger 判定后从磁盘发现并写入协议，
  `verify()` 每次重新比对；计划只冻结规则不冻结数值。
  2026-09-10 步骤 5 实跑补记：`eval/m10_e7_eval.py generate-shard` 增加 `--limit-questions`（仅 smoke 用，M9 `generate-role` 同款），
  v6 冻结因此在生成任何轨迹前重建一次（manifest `7ed92f8c…` → `26d89b19…ccc63`）；等算力 step 实测为 95。全量评测 7,600/7,600，
  判读情形 C（`eval/runs/m10_e7_eval_20260910_140124/metrics/m10_headline.json`）。仍然零框架改动。
