# data/ — 数据与预实验

## 统一 JSONL schema（PLAN §3.1）

```json
{"id": "math_train_000001", "question": "...", "answer": "\\frac{3}{4}",
 "level": 4, "subject": "Algebra", "source": "MATH", "split": "train"}
```

`level`/`subject` 在源数据没有的评测集上为 `null`。

## processed/ 文件清单（由 `download_convert.py` 生成，2026-07-13）

| 文件 | 条数 | 来源 | 用途 |
|---|---|---|---|
| `math_train.jsonl` | 7496 | 本地 `~/verl-team/lighteval-MATH-preprocessed/train.parquet`（源 DigitalLearningGmbH/MATH-lighteval，M0 下载） | 训练池原始 |
| `math_train_clean.jsonl` | 7280 | 上行经 `dedup_check.py` 剔除 216 条命中 | 训练池（净） |
| `math500.jsonl` | 500 | HF `HuggingFaceH4/MATH-500` | 主评测集 |
| `aime24.jsonl` | 30 | HF `Maxwell-Jia/AIME_2024` | 低污染评测 |
| `aime25.jsonl` | 30 | HF `math-ai/aime25` | 低污染评测 |
| `gsm8k_test200.jsonl` | 200 | HF `openai/gsm8k` test 前 200 | sanity check |
| `train_subset.jsonl` | 5403 | `math_train_clean.jsonl` 的全部 L3–5（`select_train_subset.py`，选集理由见 reports/01） | RL 训练集 |

转换细节：MATH 训练池剔除 4 条（2 条 `Level ?`、2 条空 ground_truth）；question 统一剥离
verl-team 预处理追加的指令后缀（训练/评测时按需重新拼接）；GSM8K 答案取 `####` 之后并去千分位逗号。

## 污染检查（`dedup_check.py`）

归一化（小写、去标点、压空白）精确匹配 + 词级 13-gram 重叠；报告见
`data/contamination_report.md`。命中 216 条（1 条精确 + 215 条 13-gram），其中绝大多数是
答案格式模板句（如 "where m and n are relatively prime positive integers find m n"）而非
题目重复——按 PLAN §5.1 保守处理全部剔除。自测：`python data/dedup_check.py --self-test`。

## 工具增益预实验（`tool_gain_probe.py`，PLAN §5.2）

- 从净训练池按 level 分层抽样（100/level，seed 42），CoT 与 TIR 两臂，温度 0.6、n=4、pass@1。
- TIR 臂：hermes 工具调用格式 + `code_interpreter` schema（与 M0 官方示例/M4 训练环境一致），
  沙箱 `env/sandbox.py`，max_turns=4，截断判 0。`--tir-style bare`（默认）不加 system prompt，
  与 verl tool_agent 训练时设置严格一致；`system`/`fewshot` 风格仅用于诊断对照。
- 产出：`data/probe/trajectories_{cot,tir}.jsonl`、`data/probe/metrics.json`；
  报告 `reports/01_tool_gain.md`。
- 复现：先 `tmux new -d -s sgl 'bash scripts/m1/serve_sglang.sh'`，
  再 `tmux new -d -s probe 'bash scripts/m1/run_probe.sh'`。
