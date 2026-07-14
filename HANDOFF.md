# 交接状态

## 当前状态

- **M2 验收通过**（2026-07-14，tag `m2`）：沙箱/verifier/masking 三地基完成，56 tests 全绿
  （`pytest env/ rewards/ rl/custom/`）。
- 沙箱 `env/sandbox.py`：真禁网（unshare netns，import 时探测）+ 资源限制；已知差距
  （同 uid 绝对路径删改）有测试钉住。`prepare_tool_code`（AST auto-print）在此处，
  是所有工具包装的单一来源。
- verifier `rewards/verifier.py`：**训练严格 boxed / 评测宽松**双口径；审计假阳/假阴
  0/200 检出（reports/02_appendix_verifier_audit.md）；M1 probe 判分已统一到此并重算
  （结论不变）。复合奖励默认 sparse（λ 全 0，E4 才开）。
- masking：verl 0.8.0 `ToolAgentLoop` mask 构建验证无误；测试基建
  （FakeServer + `__new__` 组装真实状态机）可复用于 M4 前的任何 rollout 行为验证。

## 下一步（M3，PLAN §7）——计划待用户批准

1. 蒸馏轨迹 `sft/gen_trajectories.py`：教师优先 API（DeepSeek/Qwen API），备选本地
   Qwen2.5-Math-7B TIR；工具格式与我们完全一致（hermes + code_interpreter），沙箱真实执行；
   rejection sampling 三条件（verifier 判对 + ≥1 次成功工具调用 + 未截断），目标 1.5k–2.5k 条。
2. LoRA SFT（r=32, alpha=64，1–2 epoch），工具返回 token 不计 loss。
3. 验收：held-out 200 题工具格式成功率 >90%、pass@1 显著高于零样本；
   **加测：SFT 后 CoT vs TIR 复测**（M1 判读的验证，probe 脚本 `--tir-style bare` 直接复用）。
4. 教师 API key 需要用户提供（或确认走本地教师路线）。

## 记账

- 用户问答持续记录到 `reports/qa_log.md`（CLAUDE.md 约定 #8）；技术报告
  `reports/technical_report.md` 每 Milestone 更新章节。
- M0 遗留：DataLoader worker 收尾阶段被杀（自愈），M4 长跑监控；NFS checkpoint IO 未实测。
- 5090 本地侧 smoke test 仍待用户复跑（M0 项）。
- 存疑项待用户抽检：verifier 对单位省略等价（`75^\circ`≡`75`）按判对处理
  （reports/02_appendix_verifier_audit.md 存疑节）。
