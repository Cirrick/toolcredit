# 交接状态

## 当前状态

- 仓库骨架已建立（PLAN §2），CLAUDE.md / LOG.md / environment.md 就绪，首次 commit 完成。
- Milestone: **M0 未开始**，实施计划已提交用户，等确认。

## 下一步

1. 用户确认 M0 计划后：conda 建 py3.12 环境 `toolcredit` → torch aarch64 冒烟 →
   选定并 pin veRL 版本 → 原样跑官方 GSM8K multiturn tool 示例 → `scripts/run_smoke_test.sh`。

## 已知坑

- 本机是 GH200/aarch64 pod 且无 docker：NGC 容器路线不可用；sglang/vllm/flash-attn 的
  aarch64 wheel 可用性是 M0 最大风险（详见 environment.md 与计划文件）。
- home 是 NFS：HF_HOME 与 checkpoint 目录的 IO 待实测，可能需找本地 scratch。
