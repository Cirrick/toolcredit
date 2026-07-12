# ToolCredit：多轮工具调用 RL 中的信用分配

**研究问题**：在多轮工具调用（TIR）的 RL 训练中，把 outcome 奖励以轨迹级优势广播到全部轮次
（标准 GRPO），与引入轮级（turn-level）信用分配相比，对训练效率、最终性能和模型行为的影响。

完整实施文档见 [PLAN.md](PLAN.md)；项目约定见 [CLAUDE.md](CLAUDE.md)；环境记录见
[environment.md](environment.md)。

## 非目标（防 scope creep）

- 不追求 SOTA 分数，不与论文数字对齐。
- 不声明 novelty，不写 related work 综述。
- 不做分布式多机训练，不深度 fork veRL。
- 不做 VLM、不做 web agent、不做 SWE 任务。

## 结果总表

| Milestone | 状态 | 结果 |
|---|---|---|
| M0 环境与 smoke test | 未开始 | — |

## 复现

（M0 完成后补充：环境安装、`scripts/run_smoke_test.sh`。）
