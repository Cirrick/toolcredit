# 污染检查报告

训练池: `sft/data/sft_pool.jsonl`（5203 条）｜方法: 归一化精确匹配 + 词级 13-gram 重叠

| 评测集 | 条数 | 命中数 |
|---|---|---|
| math500.jsonl | 500 | 0 |
| aime24.jsonl | 30 | 0 |
| aime25.jsonl | 30 | 0 |
| gsm8k_test200.jsonl | 200 | 0 |

**共剔除训练池条目: 0**（精确匹配 0，其余为 13-gram 命中）

注：13-gram 命中大多是答案格式模板句（如 'where m and n are relatively prime positive integers find m n'）而非题目本身重复——按 PLAN §5.1 保守处理，命中即剔除。
