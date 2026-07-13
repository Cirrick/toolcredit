# 交接状态

## 当前状态

- **M1 验收通过**（2026-07-13，tag `m1`）：数据五件套 + 污染检查 + 工具增益预实验全部完成。
- **基座模型已定：Qwen3-1.7B**（与用户讨论确认，理由见 plans/M1.md Context）。
- 核心发现（reports/01_tool_gain.md）：零样本工具增益**全 level 为负**（总 −10pt），
  机制分解显示工具成功执行时 L3–5 增益 +8~+23pt（L5 翻倍）——负增益源于工具执行不可靠
  （报错率 17%→46%、截断循环），是 M3 SFT 冷启动必要性的定量证据。
- 训练子集：`data/processed/train_subset.jsonl`（L3–5 共 5403 条，污染复检 0 命中）。
- 评测集就绪：math500 / aime24 / aime25 / gsm8k200（`data/processed/`，统一 schema）。

## 下一步（M2，PLAN §6）

1. `env/sandbox.py` 加固（禁网、内存限制、禁写）+ `test_sandbox.py` 全套测试
   （现为 M1 最小版：子进程 + 超时 + 2k 截断；接口 `run_python` 保持不变）。
2. `rewards/verifier.py`（math-verify 封装 + fallback 链）+ 200 例人工审计。
3. loss masking 单元测试：目标是 verl 0.8.0 agent-loop 的 delta-based tokenization
   （docs/sglang_multiturn/multiturn.rst）。
4. M2 注意沿用 M1 的 AST 版 auto-print（`data/tool_gain_probe.py:preprocess_code`），
   官方 SandboxTool 的行版启发式有已知缺陷（见 plans/M1.md 偏差表）。

## 记账

- M3 验收新增一项：SFT 后复测 CoT vs TIR（验证"SFT 解锁工具增益"判读），
  probe 脚本可直接复用（`--tir-style bare` 指向 SFT 后模型的 server 即可）。
- probe 基础设施可复用：`scripts/m1/serve_sglang.sh` + `data/tool_gain_probe.py`
  （断点续跑、--score-only 重判分）。
- M0 遗留观察项不变：DataLoader worker 收尾阶段被杀（自愈），M4 长跑监控。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项）。
