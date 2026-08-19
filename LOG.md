# 工作日志

| 日期 | 会话 | 做了什么 | 结论 / 阻塞 |
|---|---|---|---|
| 2026-07-12 | 初始化 | 通读 PLAN.md；探明平台（GH200/aarch64 pod、无 docker、系统 py3.13）；建仓库骨架、CLAUDE.md、git init 首次 commit | M0 实施计划已提交用户待确认；NGC 容器路线不可用，改 conda + aarch64 wheel |
| 2026-07-12 | M0 | conda env + torch cu129 + verl 0.8.0[sglang,math] 全栈装通；GPU 冒烟过；官方 agent-loop 示例移植为 scripts/m0/；模型/数据下好；三次试跑排掉 cachetools、flash-attn 两坑 | verl 0.8.0 sglang kernel 检查 bug（CHANGES.md），一行补丁经用户批准后应用 |
| 2026-07-12 | M0 | flash_attn 用上游官方 aarch64 wheel 解决（零编译）；run #5 官方示例验收通过：val acc 0.76、5/5 GRPO 步、指标健康；smoke test 约 2 分钟通过 | **M0 服务器侧完成**（tag m0）；DataLoader worker 两次被杀均自愈（训练收尾阶段，模式一致），M4 留意；5090 侧由用户复跑 |
| 2026-07-13 | M1 | 数据五件套转换（MATH 7496/500/AIME 60/GSM8K 200）+ 13-gram 污染检查（剔 216）；probe 基础设施（sglang server + 异步两臂 + AST auto-print 修复）；全量 500 题 probe | **全 level 零样本工具增益为负** → 触发降级条款报告用户；机制分解证明潜在增益在 L3–5（工具成功时 L5 翻倍）→ 用户确认按 L3–5 选集（5403 条，复检 0 命中）；**M1 验收通过**（tag m1），SFT 后复测记入 M3 |
| 2026-07-14 | 文档 | 技术报告 living document（M0/M1 章节）；问答日志 reports/qa_log.md + CLAUDE.md 约定 #8 | M2 计划经两轮问答（hacking、reward 设计）后批准 |
| 2026-07-14 | M2 | 沙箱加固（真禁网 unshare netns + 资源限制 + 恶意 payload 测试）；rewards 三件套（严格/宽松双口径 verifier）；200 例审计（数学等价判定 FP/语义 FN 均 0 检出，严格 boxed 口径另有 4/200 格式漏分；修复 M1 判分 2 处假阴性并重算）；masking 测试驱动真实 ToolAgentLoop | **M2 验收通过**（tag m2）：56 tests 全绿；verl mask 构建验证无误；M1 指标重算 ≤1pt 结论不变 |
| 2026-07-14 | M3 | Qwen3-8B 本地蒸馏（10.4k 轨迹→五条件拒绝采样 6056 条，教师 L3-5 acc 0.85/0.76/0.57）；trace 经真实 ToolAgentLoop 重放转 SFT 样本（与 rollout 逐 token 同构）；LoRA SFT 2.5k vs 6k 敏感性对比后取 6k | **M3 验收通过**（tag m3，两项边缘指标经用户确认）：报错率 34%→19%、增益 −0.101→−0.032（L5 转正/L4 平价）、GRPO 方差 L4 0.54；SFT-6k 定为 RL 统一起点；教师覆盖率缺口（L5 41%）交 RL |
| 2026-08-19 | M3 最小复核 | 固定 200 题 probe 比较 8B/30B teacher；同 6056 条数据完成一次 full-parameter SFT，并在 held-out 200 上按 LoRA 同协议完成 CoT/TIR 各 800 条评测 | 30B yield 52% vs 53%、L5 覆盖 63% vs 65%，拒绝全量生成；full SFT TIR/CoT 0.5188/0.5900，但工具错误/弃用 0.2654/0.1150，拒绝替换 LoRA；M4 起点不变 |
