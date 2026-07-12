# 交接状态

## 当前状态

- M0 进行中。环境全栈已装通（conda `toolcredit`：torch 2.9.1+cu129、verl 0.8.0、
  sglang 0.5.8，版本与坑见 environment.md）；官方 agent-loop 示例已移植到 `scripts/m0/`；
  Qwen3-1.7B 与 MATH 数据已在 `~/Qwen`、`~/verl-team`。
- **阻塞在用户决策**：verl 0.8.0 的 sglang kernel 检查 bug（详见 rl/custom/CHANGES.md），
  方案 A = 一行补丁（`rl/custom/patches/apply_verl_patches.sh`，已备好未应用），
  方案 B = 改 vllm rollout。

## 下一步

1. 用户拍板后：应用补丁（或装 vllm）→ tmux 重跑 `scripts/m0/run.sh` → 5 步训练验收
   → 缩成 20 条 <10 分钟 smoke test → environment.md 收尾 → commit + tag m0。

## 已知坑

- 本机是 GH200/aarch64 pod 且无 docker：NGC 容器路线不可用；sglang/vllm/flash-attn 的
  aarch64 wheel 可用性是 M0 最大风险（详见 environment.md 与计划文件）。
- home 是 NFS：HF_HOME 与 checkpoint 目录的 IO 待实测，可能需找本地 scratch。
