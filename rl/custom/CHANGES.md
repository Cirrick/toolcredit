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
