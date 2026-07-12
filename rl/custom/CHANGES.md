# 框架改动记录

原则（PLAN §1.3.4）：不 fork veRL；一切改动为最小 patch / 子类 / 配置注入，全部记录在此。

## [待用户批准] verl 0.8.0 sglang kernel 检查的一行修复

- **状态**：已准备，未应用（2026-07-12 提交用户决策中）。
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
