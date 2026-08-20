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
