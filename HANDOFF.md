# 交接状态

## 当前状态

- M0 接近完成。环境全栈装通（conda `toolcredit`，版本与全部坑见 environment.md +
  requirements.txt）；用户批准的一行 verl 补丁已应用（rl/custom/CHANGES.md）。
- **官方示例验收通过**（run #5，scripts/m0/logs/run_20260712_215334.log）：
  Qwen3-1.7B + MATH + 沙箱工具，step0 val acc 0.76，5/5 GRPO 步完成，指标健康
  （entropy ~0.2，KL ~1e-4，~40–52s/步，显存峰值 45GB）。
- smoke test（scripts/run_smoke_test.sh，20 条 + 1 梯度步 <10min）已启动待验收。

## 已知待观察

- run #5 第 4–5 步之间出现一次 `DataLoader worker killed by signal: Killed`，训练自行恢复
  跑完。M4 长跑时留意是否复现（怀疑与宿主内存或 dataloader worker 有关，未定位）。

## 下一步

1. smoke test 验收（<10 分钟）→ environment.md/README 收尾 → commit + tag m0 →
   更新 CLAUDE.md 状态行 → M0 完成，向用户汇报并给 M1 计划。
2. 5090 本地侧 smoke test 由用户自己跑（HANDOFF：复现步骤 = requirements.txt 注释顺序
   + scripts/run_smoke_test.sh；5090 是 x86+CUDA 12/13 均可，torch 直接 PyPI 装即带 CUDA）。

## 已知坑

- 本机是 GH200/aarch64 pod 且无 docker：NGC 容器路线不可用；sglang/vllm/flash-attn 的
  aarch64 wheel 可用性是 M0 最大风险（详见 environment.md 与计划文件）。
- home 是 NFS：HF_HOME 与 checkpoint 目录的 IO 待实测，可能需找本地 scratch。
