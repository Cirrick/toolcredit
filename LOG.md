# 工作日志

| 日期 | 会话 | 做了什么 | 结论 / 阻塞 |
|---|---|---|---|
| 2026-07-12 | 初始化 | 通读 PLAN.md；探明平台（GH200/aarch64 pod、无 docker、系统 py3.13）；建仓库骨架、CLAUDE.md、git init 首次 commit | M0 实施计划已提交用户待确认；NGC 容器路线不可用，改 conda + aarch64 wheel |
| 2026-07-12 | M0 | conda env + torch cu129 + verl 0.8.0[sglang,math] 全栈装通；GPU 冒烟过；官方 agent-loop 示例移植为 scripts/m0/；模型/数据下好；三次试跑排掉 cachetools、flash-attn 两坑 | verl 0.8.0 sglang kernel 检查 bug（CHANGES.md），一行补丁经用户批准后应用 |
| 2026-07-12 | M0 | flash_attn 用上游官方 aarch64 wheel 解决（零编译）；run #5 官方示例验收通过：val acc 0.76、5/5 GRPO 步、指标健康；smoke test 约 2 分钟通过 | **M0 服务器侧完成**（tag m0）；DataLoader worker 两次被杀均自愈（训练收尾阶段，模式一致），M4 留意；5090 侧由用户复跑 |
