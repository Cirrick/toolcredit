# ToolCredit：多轮工具调用 RL 中的信用分配 — 实施文档

> 交给 coding agent（Claude Code）的执行文档。人类负责人：陈冀超。
> 总工期：3 周（约 21 天）。硬件：本地 RTX 5090（32GB，调试）+ 服务器单卡 ~144GB 显存（正式训练）。
> 本文档同时是面试准备材料：每个 Milestone 附带「面试官可能问什么、怎么答、证据在项目哪里」。

---

## 0. 项目定位

### 0.1 研究问题（整个项目只回答这一个问题）

**在多轮工具调用（tool-integrated reasoning, TIR）的 RL 训练中，把 outcome 奖励以轨迹级优势广播到全部轮次（标准 GRPO 做法），与引入轮级（turn-level）信用分配相比，对训练效率、最终性能和模型行为分别有什么影响？**

### 0.2 一句话动机

一条 8 轮的工具调用轨迹只在末尾拿到一个对/错奖励；标准 GRPO 把同一个组相对优势广播给所有轮次的所有 token——第 2 轮的关键正确调用和第 5 轮的报错调用拿到完全相同的梯度。这与多智能体 RL 中"团队奖励如何分配给各个体"（QMIX 所解决的问题）在结构上同构。本项目在 LLM 工具调用场景下系统验证：更细粒度的信用分配到底值不值。

### 0.3 明确的非目标（写进 README，防止 scope creep）

- 不追求 SOTA 分数，不与论文数字对齐。
- 不声明 novelty，不写 related work 综述。
- 不做分布式多机训练，不深度 fork veRL。
- 不做 VLM、不做 web agent、不做 SWE 任务。

### 0.4 交付物总览

1. 可一键复现的 GitHub 仓库（smoke test < 10 分钟跑通）。
2. `reports/` 下 4 份报告：工具增益预实验、主对比实验、badcase 分类、reward hacking 复盘。
3. 一份 `reports/interview_outline.md`（面试叙事 + 数据支撑索引）。
4. 简历 3–4 条 bullet（模板见 §14）。

---

## 1. 环境与硬件

### 1.1 软件栈

| 组件 | 选择 | 说明 |
|---|---|---|
| 训练框架 | **veRL**（pin 到一个具体 release，写进 README） | 原生支持 multi-turn rollout + tool calling；工业界认可度高 |
| Rollout 引擎 | SGLang（veRL multi-turn 官方路径）或 vLLM | 以 pin 版本的官方文档为准 |
| 基座模型 | Qwen2.5-1.5B-Instruct（调试）/ **Qwen2.5-3B-Instruct（主力）** / Qwen2.5-7B-Instruct（余力时） | Qwen2.5 系列是 RLVR 社区文档最全的基座 |
| 答案验证 | `math-verify`（HuggingFace）+ sympy fallback | 数学答案等价性判定 |
| 沙箱 | 子进程 + 超时 + 禁网络 + 受限 builtins；或 veRL 自带 sandbox 方案 | 见 §5 |
| 实验追踪 | Weights & Biases（或 TensorBoard 兜底） | 每个 run 必须有唯一 name 与 config 快照 |

### 1.2 硬件分工

- **5090（32GB）**：全部开发调试、单元测试、1.5B smoke test、数据处理、评测脚本。
- **144GB 服务器**：3B/7B 正式训练与蒸馏数据生成。
- ⚠️ **注意服务器平台**：若为 GH200（Grace-Hopper，ARM/aarch64 平台），PyTorch / vLLM / flash-attn 的预编译 wheel 支持不全，**优先使用 NVIDIA NGC PyTorch 容器**，并在第 1 天就验证 veRL 能在该平台跑通官方示例；若为 H200（x86，141GB），无此问题。这一步失败是全项目最大的日程风险，必须最先排除。

### 1.3 Coding agent 全局约束

1. 所有依赖版本 pin 死，写入 `requirements.txt` / `environment.md`。
2. 每个训练脚本必须保存：resolved config、checkpoint/adapter、预测 JSONL、metrics JSON、一段 Markdown 摘要。
3. 先跑通 veRL 官方的 multi-turn + tool 示例（examples 目录下有 GSM8K multiturn tool 示例），**原样跑通后再改**，不要一上来就写自己的。
4. 修改 veRL 内部逻辑时：不 fork，用最小 patch / 子类 / 配置注入的方式，把改动集中在 `rl/custom/` 目录，并写一份 `rl/custom/CHANGES.md` 说明改了框架的哪个函数、为什么。
5. 任何 reward 提取失败都不允许静默跳过——记日志并计入 `invalid_rate`。
6. 函数短、有 type hints；关键张量 shape 在 debug 模式下打印；shape 不符立即 fail。

---

## 2. 目录结构

```text
toolcredit/
├── README.md                  # 项目定位、复现步骤、结果总表
├── environment.md             # 环境与版本、GH200/H200 注意事项
├── env/
│   ├── sandbox.py             # Python 代码沙箱执行器
│   └── test_sandbox.py
├── data/
│   ├── download_convert.py    # MATH 等数据下载与统一 JSONL 转换
│   ├── dedup_check.py         # 训练/评测集去重与污染检查
│   ├── tool_gain_probe.py     # M1 工具增益预实验
│   └── README.md
├── rewards/
│   ├── verifier.py            # math-verify 封装 + 等价性判定
│   ├── format_reward.py
│   ├── composite_reward.py    # outcome + 可选 shaping 项
│   └── test_rewards.py
├── sft/
│   ├── gen_trajectories.py    # 教师模型蒸馏轨迹（含 rejection sampling）
│   ├── train_sft.py
│   └── README.md
├── rl/
│   ├── configs/               # 每个实验一个 yaml，命名 = 实验矩阵编号
│   │   ├── e3_grpo_baseline.yaml
│   │   ├── e4_shaping.yaml
│   │   ├── e5_turn_credit.yaml
│   │   ├── e6_nomask.yaml
│   │   └── e7_dapo_filter.yaml   # 可选
│   ├── custom/
│   │   ├── turn_advantage.py  # E5 轮级信用分配实现
│   │   └── CHANGES.md
│   └── launch/                # 每个实验一个启动脚本
├── eval/
│   ├── generate.py            # 统一评测生成（所有方法共用）
│   ├── metrics.py             # pass@k、错误恢复率、工具指标
│   └── evaluate.py
├── analysis/
│   ├── badcase_taxonomy.py    # 失败轨迹自动分类 + 人工抽检表
│   ├── hacking_cases/         # reward hacking 案例原始轨迹
│   └── plots.py
├── reports/
│   ├── 01_tool_gain.md
│   ├── 02_main_results.md
│   ├── 03_badcase_taxonomy.md
│   ├── 04_reward_hacking.md
│   └── interview_outline.md
└── scripts/
    ├── run_smoke_test.sh      # 20 条数据全流程 < 10 分钟
    └── run_all_eval.sh
```

---

## 3. 统一数据与输出格式

### 3.1 数据 JSONL schema

```json
{
  "id": "math_train_000001",
  "question": "...",
  "answer": "\\frac{3}{4}",
  "level": 4,
  "subject": "algebra",
  "source": "MATH",
  "split": "train"
}
```

### 3.2 多轮交互格式（与 veRL tool calling 约定对齐）

模型输出遵循：思考文本中可发起工具调用（以 pin 版本 veRL 的 tool call 格式为准，通常是特定 XML/JSON 标记包裹的 `python` 代码块）；环境返回执行结果作为下一轮输入；最终答案要求 `\boxed{}` 包裹。**具体标记格式不要自己发明，直接采用 veRL 官方 multiturn 示例的格式**，减少解析歧义。

约束参数（写入所有 config）：

- `max_turns`: 4（即最多 4 次工具调用）
- `max_prompt_length`: 1024；`max_response_length_total`: 3072（多轮累计）
- 超过 max_turns 未给出最终答案 → 强制截断，outcome reward = 0，`truncated=True` 记入轨迹元数据

---

## 4. Milestone 0 — 环境打通与 smoke test（第 1–2 天）

**任务**：
1. 服务器上用 NGC 容器（如为 GH200）或裸环境装好 veRL（pin 版本），**原样跑通官方 multi-turn tool 示例**（GSM8K + code interpreter 类示例），确认 rollout、工具调用、训练 loop 全部工作。
2. 建立目录骨架，写 `scripts/run_smoke_test.sh`：20 条 toy 数据 → 生成 → 验证 → 一次梯度更新，全程 < 10 分钟。
3. 5090 上用 1.5B 复跑同一 smoke test（验证本地调试链路）。

**验收标准**：两台机器 smoke test 均通过；`environment.md` 完成。

**失败即降级**：若 veRL multi-turn 在服务器平台跑不通且 2 天内无法解决 → 降级方案见 §12。

---

## 5. Milestone 1 — 数据与工具增益预实验（第 3–4 天）

### 5.1 数据准备

- 训练池：MATH train split（约 7.5k），转换为统一 JSONL；按 level 分层。
- 评测集：**MATH500**（社区标准 500 题子集）+ **AIME 2024/2025**（时间上晚于基座模型知识截止，污染风险最低）+ GSM8K test 200 题（仅作 sanity check）。
- `dedup_check.py`：训练集与全部评测集做精确匹配 + n-gram（n=13）重叠检查，输出污染报告；发现重叠即从训练集剔除。

### 5.2 工具增益预实验（`tool_gain_probe.py`）

用 Qwen2.5-3B-Instruct，在 MATH 各 level 分层抽样（每层 100 题），两种设置各测一次：(a) 纯 CoT prompting；(b) 带 Python 解释器的 TIR prompting（少样本示例引导工具格式）。温度 0.6，n=4，报告 pass@1。

**产出 `reports/01_tool_gain.md`**：工具增益 vs 难度曲线；据此选定训练子集（预期为 level 3–5 中计算密集题，目标 3–5k 条）。若全难度层增益都 < 3 个点 → 换更难数据（AMC/AIME 风格的开源集或 MATH level 4–5 全量），本步骤重跑。

**验收标准**：曲线图 + 选集理由成文；训练子集 JSONL 落盘并通过污染检查。

### 5.3 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| 为什么不用 GSM8K 训练？ | GSM8K 心算即可解，工具增益趋近零，任何信用分配差异都会被噪声淹没；我先做了工具增益预实验再选数据 | `reports/01_tool_gain.md` 的增益曲线 | ✅ 实验直接回答 |
| 怎么防止评测集污染？ | 训练/评测严格分 split + 精确与 n-gram 去重；评测用时间上晚于基座截止的 AIME 24/25 兜底 | `data/dedup_check.py` 与污染报告 | ✅ |
| 数据难度怎么配比？考虑过课程学习吗？ | 选工具增益显著且组内奖励有方差的难度段（全对/全错的题对 GRPO 无梯度）；课程学习是合理扩展但超出 3 周范围，作为 future work | 选集理由段落 | ⚠️ 部分覆盖，课程学习是知识延伸 |
| 训练数据要多少？ | RLVR 是 on-policy 采样，prompt 池 3–5k 在 100–300 step 的训练量下不会耗尽；关键是质量（有方差）不是数量 | config 中 step 数与数据量的对应关系 | ✅ |

---

## 6. Milestone 2 — 沙箱、verifier 与 loss masking 单元测试（第 4–6 天）

### 6.1 沙箱（`env/sandbox.py`）

- 子进程执行，超时 5s，内存上限 1GB，禁网络，禁文件写（白名单 tmp 除外），捕获 stdout/stderr。
- 返回结构：`{status: ok|error|timeout, stdout, stderr, wall_time}`。
- 测试用例必须包含：死循环、fork 炸弹式 payload、`import os` 删文件企图、超长输出（截断到 2k 字符）。

### 6.2 Verifier（`rewards/verifier.py`）

- 主路径 `math-verify` 等价判定；失败 fallback 到 sympy 规范化；再 fallback 到字符串精确匹配。
- **抽样审计**：随机抽 200 个（模型输出, 标准答案）对人工核对，记录假阳性/假阴性率，写入 `reports/02` 附录。

### 6.3 复合奖励（`rewards/composite_reward.py`）

```
r = 1.0 * answer_correct
  + 0.1 * format_ok                    # boxed 格式 + 工具调用可解析
  + λ_exec * frac_successful_tool_calls  # E4 专用，默认 λ_exec = 0
  - λ_budget * max(0, n_calls - budget)  # E4 专用，默认 λ_budget = 0
```

### 6.4 Loss masking 测试（本项目最重要的单元测试）

构造一条含 2 次工具调用的假轨迹，断言：prompt token、工具返回 token、padding 的 loss_mask 全为 0；模型生成 token 全为 1。**定位 pin 版本 veRL 中 multi-turn 轨迹的 mask 构建位置，测试直接针对该函数**。

**验收标准**：`test_sandbox.py`、`test_rewards.py`、masking 测试全绿；verifier 审计误差率成文。

### 6.5 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| 数学答案等价性怎么判？边界 case？ | math-verify + sympy 规范化处理分数/根式/区间等价；我抽了 200 例人工审计误判率 | `verifier.py` + 审计附录 | ✅ |
| verifier 假阳性和假阴性哪个对 RL 危害更大？ | 假阳性更糟——模型直接朝错误行为优化（即 hacking 入口）；假阴性只是稀释信号、增加全错组比例 | 审计数据 + `04_reward_hacking.md` 中因 verifier 漏洞产生的案例 | ✅ |
| 为什么工具返回的 token 要从 loss 里 mask 掉？ | 它们不是策略生成的；不 mask 等于让模型学复读环境输出，重要性比率也失去意义 | **E6 对照实验直接展示不 mask 的训练崩溃过程** | ✅ 全项目最硬的证据 |
| 沙箱怎么保证安全？ | 子进程隔离 + 超时 + 资源限制 + 禁网禁写；工业级会用 container/gVisor，我说明了差距 | `sandbox.py` 与其测试 | ✅（工业方案为知识延伸） |
| 格式奖励会被 hack 吗？ | 会——只给格式分模型可能学会输出空 boxed；所以格式分权重压到 0.1 且与 outcome 联合 | 复合奖励设计 + 训练早期 format-only 分数曲线 | ✅ |

---

## 7. Milestone 3 — SFT 冷启动（第 6–8 天）

### 7.1 蒸馏轨迹生成（`sft/gen_trajectories.py`）

- 教师：优先用 API（DeepSeek / Qwen 系列 API，成本可忽略）；备选在 144GB 服务器上部署量化的 Qwen2.5-72B-Instruct 或直接用 Qwen2.5-Math-7B-Instruct 的 TIR 模式。
- 对训练池每题采样 2 条完整多轮轨迹（教师被 prompt 成与我们完全一致的工具调用格式，工具由我们的沙箱真实执行）。
- **Rejection sampling 过滤**：只保留 (a) 最终答案 verifier 判对，且 (b) 至少一次成功的工具调用，且 (c) 未触发 max_turns 截断的轨迹。目标 1.5k–2.5k 条。

### 7.2 训练

LoRA（r=32, alpha=64）SFT，1–2 epoch，5090 上即可完成（3B）。工具返回 token 同样不计 loss。产出 SFT checkpoint 作为所有 RL 实验的统一起点。

**验收标准**：SFT 后模型在训练池 held-out 200 题上：工具调用格式成功率 > 90%，pass@1 显著高于零样本；两项数字记录在案（它们是面试素材）。

### 7.3 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| 为什么要 SFT 冷启动？直接 RL 不行吗？ | 3B 模型零样本工具格式成功率低 → 组内全错 → GRPO 无梯度、训练空转。我记录了 SFT 前后格式成功率对比 | 7.2 的验收数字；（可选）一组无 SFT 直接 RL 的短对照 run | ✅ |
| 蒸馏数据怎么过滤？为什么？ | rejection sampling 三条件；保留错误轨迹会把教师的失败模式蒸进去 | `gen_trajectories.py` 过滤统计日志 | ✅ |
| SFT 会不会压熵、限制 RL 探索？ | 会有此效应；所以只做 1–2 epoch 轻量 SFT，且 RL 阶段监控熵曲线（见 M4），R1 论文的 cold start 一节讨论过同一权衡 | RL 训练的 entropy 曲线起点 | ✅（R1 论文细节为知识延伸） |
| LoRA 做 SFT/RL 够吗？ | 3B 规模 + 任务内分布迁移，LoRA 经验上足够；我在 E3 上跑过 LoRA vs 全参的一次对比（若时间允许），差距在噪声内/有差距则如实报告 | 若做了对比则有曲线；没做则如实说"预算内选择，是已知局限" | ⚠️ 视时间，诚实处理 |

---

## 8. Milestone 4 — E3 主 baseline：multi-turn GRPO（第 8–12 天）

### 8.1 veRL 配置骨架（`rl/configs/e3_grpo_baseline.yaml` 的关键字段）

```yaml
algorithm:
  adv_estimator: grpo
data:
  train_batch_size: 64          # prompt 数 / step
  max_prompt_length: 1024
  max_response_length: 3072
actor_rollout_ref:
  model.path: <SFT checkpoint>
  actor:
    optim.lr: 1.0e-6
    use_kl_loss: true
    kl_loss_coef: 0.001          # 低 KL 档；消融另设 0.01
    clip_ratio_low: 0.2
    clip_ratio_high: 0.28        # DAPO clip-higher，防熵坍缩
    ppo_mini_batch_size: 16
  rollout:
    name: sglang                 # 以 pin 版本 multi-turn 支持为准
    n: 8                         # 每 prompt 采样 8 条轨迹
    temperature: 1.0
    multi_turn: {enable: true, max_turns: 4, tool_config: <python sandbox>}
trainer:
  total_training_steps: 200
  save_freq: 25
  test_freq: 25                  # 每 25 step 在 MATH500 子集上评测
```

（字段名以 pin 版本为准，agent 需对照官方示例校正。）

### 8.2 训练稳定性检查清单（每个 run 都执行）

监控并在 W&B 上建固定面板：`reward/mean`、`pass@1(eval)`、`actor/kl`、`actor/entropy`、`response_length/mean`、`n_tool_calls/mean`、`tool_error_rate`、`invalid_format_rate`、`group_all_correct_frac`、`group_all_wrong_frac`。

异常处置手册（写进 README）：
- KL 快速上飙 + eval 下降 → 升 kl_loss_coef 或降 lr；
- entropy 单调坍缩到 < 0.3 → 检查 clip_ratio_high、升温度；
- reward 升但 pass@1 不动 → 第一嫌疑 reward hacking，立即抽轨迹人检（进 M7 素材库）；
- 长度爆炸 → 检查截断惩罚与 max_turns 逻辑。

**验收标准**：E3 在 200 step 内 eval pass@1 相对 SFT 起点有稳定提升（哪怕几个点），全部曲线无病态形状；产出该 run 的一页训练日志摘要。**这是全项目最花时间的一步，预算 4 天，包含 2–3 次失败重调。**

### 8.3 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| GRPO 和 PPO 的核心区别？为什么选 GRPO？ | critic-free：组内相对优势替代 value model，省一半显存与 critic 训练不稳定性；代价是需要每 prompt 多次采样、优势是稀疏 outcome 场景的自然选择 | config 选择 + 显存预算表（§11） | ✅ 选择有据；PPO 细节（GAE 等）为知识题 |
| GRPO 优势公式？除以组内 std 有什么问题？ | A_i=(r_i−mean)/std；Dr.GRPO 指出除 std 引入难度偏置（简单/极难题组 std 小、优势被放大）与长度偏置 | 我在 E5 实现中触碰过优势计算代码，可讲源码位置 | ⚠️ 项目触及代码但未做 Dr.GRPO 消融——如实说"读过论文与 veRL 实现，未在预算内实验" |
| KL 项放 reward 里还是 loss 里？用哪个估计器？ | GRPO 常规做法是 KL 进 loss（veRL use_kl_loss），k3 低方差估计；DAPO 干脆去 KL——我用低系数折中并监控 | config + KL 曲线 | ✅ 配置层面覆盖；k1/k2/k3 推导为知识题 |
| 熵坍缩怎么发现、怎么处理？ | 固定面板监控 entropy；clip-higher 给低概率 token 更大上调空间是 DAPO 的对策，我默认启用并有曲线对照 | entropy 曲线 + clip 配置 | ✅ |
| 为什么用 veRL 不用 TRL？ | multi-turn + tool 原生支持、rollout 走 SGLang/vLLM 吞吐高、与工业训练栈对齐；TRL 适合单轮轻量原型 | 框架选型段落 | ✅ |
| 组内全对/全错的 prompt 怎么办？ | 优势为零、白耗 rollout 算力；DAPO 动态过滤解决——我把它做成了可选消融 E7，并监控两个 frac 指标 | `group_all_*_frac` 曲线 + E7（若跑） | ✅ |

---

## 9. Milestone 5 — 消融组：E6 no-mask、E4 shaping、E7 过滤（第 12–15 天）

### 9.1 E6：故意关闭工具 token 的 loss mask（必做，成本最低收益最高）

复制 E3 config，仅关闭工具返回段 mask，跑 50–80 step（预期崩，不必跑满）。记录：loss 曲线、生成样例中模型开始复读工具输出格式的现象、eval 崩溃点。**产出物是"经典 bug 的对照演示"**，写入 `02_main_results.md` 专门一节。

### 9.2 E4：奖励塑形（必做）

λ_exec=0.2（执行成功分）与 λ_budget=0.1（预算惩罚，budget=3）两个 run 或一个联合 run（预算内二选一优先 λ_exec）。核心观察：shaping 是否加速早期收敛、是否诱发"空刷执行成功"的 hacking（收集进 M7 素材库）。

### 9.3 E7：DAPO 式动态过滤（可选，时间富余才做）

丢弃全对/全错组、动态补采样。veRL 若有现成开关直接用；没有则跳过并在报告中说明。

**验收标准**：E6 完成且现象清晰；E4 至少一个 run 完成并与 E3 同图对比。

### 9.4 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| 不 mask 工具 token，训练具体怎么坏的？ | 展示 E6：loss 表面下降但模型学会生成假"工具输出"文本、eval 崩；机制是把环境 token 当策略行为优化 | E6 曲线 + 生成样例 | ✅ 最强演示 |
| 过程性 shaping 的风险？ | 奖励可分解性被利用——我观察到（或未观察到）模型刷执行成功分的行为，如实报告 | E4 曲线 + hacking 案例 | ✅ |
| shaping 和 PRM 什么关系？ | shaping 是规则化的过程信号，PRM 是学出来的过程信号；两者都改变最优策略的风险面（potential-based shaping 才保证不变），PRM 训练本身是另一条技术线 | 概念对比写在报告 discussion | ⚠️ PRM 训练为知识题 |

---

## 10. Milestone 6 — E5 轮级信用分配（第 15–18 天，核心贡献）

### 10.1 设计（两档实现，先做档 A）

**档 A（必做，最小改动）— 轮级基线修正**：保持组相对优势 A_traj 不变，为每一轮定义轮级修正项：
`A_turn(t) = A_traj + β · (s_t − s̄_t)`，其中 s_t ∈ {0,1} 表示该轮工具调用是否成功执行且其结果被后续推理采纳（"采纳"用启发式判定：工具输出的数值出现在后续文本或最终答案推导中），s̄_t 为组内同轮位置的均值。β=0.5 起步。该轮的所有生成 token 使用 A_turn(t)。实现位置：继承/包裹 veRL 的 GRPO 优势计算函数，改动集中在 `rl/custom/turn_advantage.py`。

**档 B（若档 A 顺利且有 2 天余量）— GiGPO 式两级分组**：episode 级组优势 + 以"相同前缀状态"锚定的 step 级组优势加权和。若 veRL 生态已有可借用实现则借用，否则不强做。

### 10.2 实验与判读

E5 与 E3 严格同配置（数据、SFT 起点、seed、step 数、评测协议），对比：收敛速度（达到同一 pass@1 所需 step）、最终 pass@1、工具行为指标（调用次数、错误率、**错误恢复率**）。

**两种结果都写成结论**：
- 若 E5 更好 → 细粒度信用在多轮工具场景有效，分析增益来自哪类轨迹（长轨迹？含报错恢复的轨迹？）。
- 若无差异/更差 → 诚实的否定性结论：候选解释包括轨迹太短（≤4 轮）组内方差已足够、启发式 s_t 信号噪声大、β 未调优；给出"什么条件下值得再试"（更长 horizon、真实 agent 任务）。**否定性结论 + 机制分析在面试中的价值不低于正向结果。**

**验收标准**：E5 至少一个 β 值完整 run；对比图表 + 判读写入 `02_main_results.md`。

### 10.3 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| 轨迹级优势广播具体丢了什么信息？ | 同轨迹内好/坏中间动作获得相同梯度；等价于把多步 MDP 折叠成 bandit | motivation 段 + E5 设计 | ✅ |
| 你的轮级方案怎么设计的？为什么这么设计？ | 档 A 公式与"采纳"启发式；设计原则是零额外模型（不训 PRM/critic）、零额外采样，只利用组内已有信息 | `turn_advantage.py` + CHANGES.md | ✅ |
| 和你硕士的 QMIX 有什么联系？ | 同构问题：团队奖励→个体信用 vs 轨迹奖励→轮次信用；QMIX 用值分解网络学分配，我这里用组内统计做免学习分配——可以对比两条路线的适用条件 | 简历叙事线 + 报告 discussion | ✅ 独家叙事 |
| 和 PRM / MCTS 估计步级价值的路线比？ | PRM 要训模型、有标注/合成成本与 hacking 面；MCTS 采样贵；我的方案是最便宜档位，天花板也更低——三档取舍能讲清 | discussion 段 | ⚠️ PRM/MCTS 细节为知识题 |
| 如果结果是没用呢？（面试官压力测试） | 直接给否定性结论 + 三个候选机制解释 + 何种条件下该重试；展示的是实验判断力而非刷分 | `02_main_results.md` 判读节 | ✅ 预先写好 |
| GiGPO 是怎么做的？你和它的区别？ | 两级分组优势（episode + 锚定状态的 step 级）；我的档 A 更简化（轮位置对齐而非状态锚定），档 B 即其复现 | 读论文笔记 + （若做）档 B 实现 | ⚠️ 至少精读论文 |

---

## 11. Milestone 7 — 评测、分析与报告（第 18–21 天）

### 11.1 统一评测协议（所有方法共用 `eval/generate.py`）

- pass@1：温度 0（greedy）与温度 0.6/n=4 两套都报；pass@k 用无偏估计公式。
- 每个 checkpoint 在 MATH500 全量 + AIME24/25 上评测；GSM8K 200 题 sanity。
- 工具行为指标：平均调用次数、工具错误率、**错误恢复率**（定义：首次工具执行报错的轨迹中，最终答案仍正确的比例——直接度量"会不会从失败恢复"）。

### 11.2 分析产出

1. **`02_main_results.md`**：主表（各方法 × 各指标）+ 训练曲线合图 + E5 判读 + E6 演示节。
2. **`03_badcase_taxonomy.md`**：对 E3 与 E5 最终模型各抽 100 条失败轨迹，按类型分类（不会恢复 / 工具滥用 / 调用解析失败 / 推理错误但工具正确 / 答案对格式错 / 截断），给出占比与训练前后变化；`badcase_taxonomy.py` 自动粗分 + 人工抽检修正。
3. **`04_reward_hacking.md`**：整个项目期间收集的 hacking 案例（目标 ≥3 个，如：代码暴力枚举候选答案、刷执行成功分、boxed 空答案骗格式分），每例附原始轨迹、成因、verifier/奖励修复方式、修复后验证。
4. **`interview_outline.md`**：60 秒版与 5 分钟版叙事 + 每句话对应的数据索引。

### 11.3 本 Milestone 的面试 Q&A

| 问题 | 回答要点 | 项目内证据 | 覆盖？ |
|---|---|---|---|
| pass@k 怎么算才无偏？eval 用什么温度？ | 组合无偏估计式；greedy 报确定性性能、采样报分布性能，两套都给避免 cherry-pick | `metrics.py` 实现 | ✅ |
| 错误恢复率为什么重要？ | agent 与单轮推理的本质差异就在失败后的行为；现成指标没有它，我定义并全程跟踪 | 指标定义 + 各方法对比 | ✅ 独家指标 |
| 你遇到的最典型 reward hacking？怎么修的？ | 讲 `04` 里最好的一例：现象→成因→修 verifier/奖励→复验 | `04_reward_hacking.md` | ✅ |
| badcase 归因怎么做的？ | 自动分类器粗分 + 人工抽检校准分类器；报告训练前后分布迁移 | `03_badcase_taxonomy.md` | ✅ |
| 训练指标好但线上/真实指标差，你怎么排查？ | 顺序：评测集污染→verifier 漏洞→分布偏移→长度/格式套利；我的项目里 E4 就出现过 reward 升 pass 不动的时刻，讲当时的排查过程 | 训练日志 + 异常处置记录 | ✅ 有真实经历可讲 |

---

## 12. 算力预算与风险降级

### 12.1 预算估计（按 3B + LoRA，144GB 单卡）

| 项 | 估计 |
|---|---|
| 工具增益预实验 | ~0.5 GPU 日（5090 可分担） |
| 蒸馏轨迹（API 教师） | GPU 忽略；API 成本低 |
| SFT | < 0.5 GPU 日（5090 可跑） |
| E3 主 run（200 step, bs64×n8, resp≤3072） | 1.5–2.5 GPU 日（rollout 占 70%+ 时间；multi-turn 有工具等待，吞吐低于单轮） |
| E4 / E5 / E6 / (E7) | 各 1–2 GPU 日（E6 只跑 80 step，~0.5 日） |
| 评测全量 | ~0.5 GPU 日 |
| **合计** | **约 8–11 GPU 日**，含重跑余量后贴合 3 周单卡日程；7B 复跑仅在提前完成时做 E3/E5 两个 run |

### 12.2 风险与降级方案

| 风险 | 触发条件 | 降级动作 |
|---|---|---|
| veRL multi-turn 在服务器平台（尤其 GH200/aarch64）跑不通 | M0 两天内未通 | 换 NGC 容器重试半天 → 仍不通则改用 x86 云租卡（A100/H100 单卡按小时）跑正式实验，5090 继续承担开发；框架层面备选：改用社区 multi-turn 方案（如 Search-R1 代码基）承载同一实验设计 |
| E3 训不稳（4 天未达验收） | M4 超期 | 降到 1.5B、max_turns 3、resp 2048 重试；仍不稳则 E3 以"已诊断的不稳定 + 处置记录"形态入报告，E5/E6 在 1.5B 上完成 |
| 时间不够 | 第 15 天 E4 未启动 | 砍 E7 → 砍 E4 → **E5 与 E6 永不砍**（它们是差异化核心） |
| E5 无收益 | 结果层面 | 不是风险——按 §10.2 写成否定性结论 |
| 蒸馏教师 API 不可用 | M3 | 本地量化 72B 或 Qwen2.5-Math-7B TIR 兜底 |

---

## 13. 项目未覆盖的高频知识题清单（需单独准备）

以下问题项目不提供证据，靠读论文/源码准备（每项 0.5–2 小时，AAAI 提交后的晚间碎片时间即可）：

1. **DPO**：从 RLHF 目标到 DPO 损失的推导；IPO/KTO 与 DPO 的区别（高频手推题）。
2. **RLHF 全流程**：SFT→RM→PPO 三阶段；Bradley-Terry 奖励模型训练与长度偏置校准。
3. **PPO 细节**：GAE 推导、λ 与偏差-方差、value clip；（你有经典 RL 底子，重点是 LLM 语境下的差异）。
4. **GSPO**：序列级重要性比率 vs token 级的动机（长序列 token 级比率方差爆炸、MoE 稳定性）。
5. **Dr.GRPO**：std 归一化与长度归一化的偏置分析（你在 E5 触碰过优势代码，衔接自然）。
6. **On-policy distillation（OPD/OPSD）**：稠密 token 级教师信号 vs 稀疏 outcome 奖励的取舍。
7. **推理系统**：vLLM PagedAttention、continuous batching 原理（rollout 为什么快）；你项目里用了 SGLang，能讲一层原理更好。
8. **LLM 基础手撕**：multi-head attention、RoPE、KV cache 显存估算；GRPO loss 的 numpy/torch 手写（把你项目里的实现默写一遍即是准备）。
9. **DeepSeek-R1 训练流程**：R1-Zero vs R1 的 cold start、多阶段流水线（与你的 M3 冷启动决策直接呼应）。
10. **Agent 评测基准**：SWE-bench、tau-bench、WebArena 大致设定（应用岗常问视野题）。

必读论文（精读方法与消融即可，每篇 ≤1 小时）：DeepSeek-R1、DeepSeekMath（GRPO 原文）、DAPO、Dr.GRPO、GSPO、ToRL、ReTool、Search-R1、GiGPO。

---

## 14. 简历 bullet 模板（项目完成后按实际数字填充）

> **多轮工具调用 RL 的信用分配研究（个人项目，代码开源）** 2026.08
> - 基于 veRL 构建"数学推理 + Python 解释器"多轮工具调用 GRPO 训练管线（Qwen2.5-3B，SGLang 多轮 rollout + 沙箱执行），SFT 冷启动后 RL 使 MATH500 pass@1 提升 x.x pt；
> - 系统对比轨迹级优势广播与自研轮级信用分配方案，[量化结论一句话]；设计"错误恢复率"指标度量 agent 失败恢复能力；
> - 通过对照实验演示工具 token loss masking 缺失导致的训练崩溃机制；收集并修复 x 类 reward hacking 案例，完成 badcase 归因与 verifier 加固；
> - 完整实验报告与复现脚本开源：github.com/Cirrick/toolcredit。

---

## 15. 面试叙事脚本

### 60 秒版

"单轮 RLVR 已经很成熟了，但真实 agent 是多轮的：模型调工具、环境返回、继续推理。这里有个训练层面的核心问题——一条轨迹只在末尾拿到一个对错奖励，标准 GRPO 把同一个优势广播给所有轮次，好的调用和坏的调用拿到一样的梯度。这和我硕士做 QMIX 时面对的'团队奖励怎么分给个体'是同构问题。所以我搭了一个数学加 Python 解释器的多轮 GRPO 管线，严格对照地验证了轮级信用分配的价值，[一句话结论]。过程中我还做了两件工程上有意思的事：故意关掉工具 token 的 loss mask 演示训练怎么崩，以及收集修复了 x 个 reward hacking 案例。"

### 5 分钟版结构（每节 ≤1 分钟，均有图表索引）

1. 研究问题与 QMIX 同构性（motivation）
2. 数据决策：工具增益预实验为什么否决了 GSM8K
3. 管线与关键实现：masking、沙箱、verifier 审计
4. 主结果：E3 vs E5 对比 + 判读（正向或否定性结论都按事实讲）
5. 两个"事故现场"：E6 崩溃演示 + 最典型的 hacking 案例
6. 局限与下一步：更长 horizon 的真实 agent 任务、PRM 路线对比

---

## 16. 给 coding agent 的执行顺序摘要

```text
Day 1-2   M0 环境打通（服务器平台验证优先级最高）+ smoke test
Day 3-4   M1 数据转换、去重、工具增益预实验 → 报告 01
Day 4-6   M2 沙箱 / verifier / masking 全部单元测试
Day 6-8   M3 蒸馏 + SFT 冷启动（5090 可承担）
Day 8-12  M4 E3 主 baseline 训稳（预算含 2-3 次重调）
Day 12-15 M5 E6（先做，最便宜）→ E4 →（E7）
Day 15-18 M6 E5 轮级信用分配实现与对比
Day 18-21 M7 全量评测、四份报告、README、面试 outline
```

每个 Milestone 完成后：commit + tag + 在 README 结果总表中更新一行。任何一步卡住超过其预算的 50%，触发 §12.2 对应降级，不允许原地死磕。
