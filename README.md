# ToolCredit：多轮工具调用 RL 中的信用分配

**研究问题**：在多轮工具调用（TIR）的 RL 训练中，把 outcome 奖励以轨迹级优势广播到全部轮次
（标准 GRPO），与引入轮级（turn-level）信用分配相比，对训练效率、最终性能和模型行为的影响。

完整实施文档见 [PLAN.md](PLAN.md)；项目约定见 [CLAUDE.md](CLAUDE.md)；环境记录见
[environment.md](environment.md)；**逐 Milestone 的动机/方法/结果叙事见
[reports/technical_report.md](reports/technical_report.md)**（living document，面试复盘入口）。

## 非目标（防 scope creep）

- 不追求 SOTA 分数，不与论文数字对齐。
- 不声明 novelty，不写 related work 综述。
- 不做分布式多机训练，不深度 fork veRL。
- 不做 VLM、不做 web agent、不做 SWE 任务。

## 结果总表

| Milestone | 状态 | 结果 |
|---|---|---|
| M0 环境与 smoke test | ✅ 服务器侧完成（2026-07-12） | veRL 0.8.0 官方 agent-loop 示例原样跑通（Qwen3-1.7B+MATH+沙箱，val acc 0.76，5/5 GRPO 步）；smoke test（20 条+1 梯度步）约 2 分钟 < 10 分钟预算。5090 本地侧待用户复跑 |
| M1 数据与工具增益预实验 | ✅ 完成（2026-07-13） | 训练/评测数据五件套 + 污染检查（剔 216 条）；probe（Qwen3-1.7B，500 题分层，CoT vs TIR）：**零样本工具增益全 level 为负（总 −10pt），机制分解显示工具成功执行时 L3–5 增益 +8~+23pt**——SFT 冷启动必要性的定量证据；训练子集 = L3–5 共 5403 条（详见 [reports/01_tool_gain.md](reports/01_tool_gain.md)） |
| M2 沙箱/verifier/masking | ✅ 完成（2026-07-14） | **56 tests 全绿**：沙箱加固（真禁网 netns + 资源限制 + 恶意 payload 全过）；verifier 双口径 + 200 例人工审计（**数学等价判定假阳性/语义假阴性均 0 检出；严格 boxed 口径另有 4/200 格式漏分**，顺带修复 M1 判分 2 处假阴性）；verl ToolAgentLoop mask 构建验证无误（详见 [reports/02_appendix_verifier_audit.md](reports/02_appendix_verifier_audit.md)） |
| M3 SFT 冷启动 | ✅ 完成（2026-07-14；2026-08-19 最小复核） | Qwen3-8B 本地蒸馏 + 五条件拒绝采样 **6056 条**（含 797 条报错恢复）；工具报错率 34%→19%、弃用率 27%→7%、**增益 −0.101→−0.032**。30B teacher 未提高 yield/L5 覆盖；full SFT 仅小幅提高 held-out pass@1 且工具错误率恶化 6.4pt，均拒绝切换；LoRA SFT-6k 仍为全部 RL 实验统一起点（详见 [plans/M3.md](plans/M3.md)） |

## 复现

（M0 完成后补充：环境安装、`scripts/run_smoke_test.sh`。）
