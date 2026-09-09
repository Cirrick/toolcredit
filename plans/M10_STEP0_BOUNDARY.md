# M10 / E7 步骤 0 — exact boundary 专项审批材料

日期：2026-09-09。状态：**已批准（用户于 2026-09-09 授权助手自行审查并决定 §7 三项；决定见 §8）。无任何 E7 代码。**
依据：`plans/M10.md` v2 §6/§8 步骤 0；`plans/M5_E7_IMPLEMENTATION_REVIEW.md`（2026-08-22）。

## 1. pin 源码复核（2026-09-09 现场）

| 文件 | 审查记录 SHA256 | 现场 SHA256 | 结果 |
|---|---|---|---|
| `verl/trainer/ppo/ray_trainer.py` | `de58d295…6d66` | `de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66` | 相同 |
| `verl/trainer/main_ppo.py` | `e3fe6e73…1392` | `e3fe6e73b18d63367402a2570c5ba054a2b05443ac40a168524dbf545b1da392` | 相同 |

`fit()` 仍在 `ray_trainer.py:1362–1770`（409 行）；`TaskRunner.run()` 仍在 `main_ppo.py:223–315`，`:299` 硬编码
`trainer = RayPPOTrainer(...)`；`run_ppo(config, task_runner_class=None)` 在 `:52`，是 E5/E5-v2 已用的扩展点。
`algorithm.filter_groups.*` 在 pin `fit()` 内无读取点（YAML-only 不会过滤）。

## 2. 拟新增文件（全部 repo-local，不改 site-packages）

| 文件 | 内容 |
|---|---|
| `rl/custom/dynamic_filter_trainer.py` | `DynamicFilterRayPPOTrainer(RayPPOTrainer)`：override `fit()`（§3）、`_save_checkpoint`/`_load_checkpoint`（§5）、新增私有方法 `_e7_collect_informative_batch`、`_e7_save_equal_compute_checkpoint`、`_e7_log_candidates`；纯函数 `select_informative_groups`；`DynamicSamplingLedger` |
| `rl/launch/e7_dynamic_filtering.py` | 驱动边界（§4）、config diff gate（含 offload 三条）、preflight、`--stop-at-step`/`--resume`、proof 文件 |
| `rl/configs/e7_dynamic_filtering.yaml` | E3 复制 + `algorithm.filter_groups.{enable: true, metric: score, max_num_gen_batches: 4}` + offload 三条 false |
| `rl/custom/test_dynamic_filter.py` | Fixture N–T |
| `rl/validate_e7_run.py`、`scripts/m10/*.sh` | completion gate、tmux 分段循环、memwatch v2 |

## 3. `fit()` 复制段与白名单 diff 行

复制 pin `:1362–1770` 全部 409 行为 `DynamicFilterRayPPOTrainer.fit()`，**只允许**下列 8 处不同；每处在代码里以
`# E7-W<n>` 标记，Fixture R 逐行 diff 时只放行带标记的行与下列被替换的上游行段。

| # | 上游行 | 上游内容 | E7 改动 | 理由 |
|---|---|---|---|---|
| W1 | `:1389` | `current_epoch = self.global_steps // len(self.train_dataloader)` | `current_epoch = ledger.dataloader_batches_consumed // len(self.train_dataloader)` | E7 一个有效 step 消费 2–4 个 dataloader batch，`global_steps` 不再等于已消费批数 |
| W2 | `:1422–1423` | `for epoch in range(...): for batch_dict in self.train_dataloader:` | `while epoch < total_epochs:` + 显式 `self._e7_iter = iter(self.train_dataloader)`；`StopIteration` 时 `epoch += 1` 并重建 iterator | 补采样需要在同一有效 step 内多次 `next()`；上游双层 `for` 不暴露 iterator |
| W3 | `:1435–1525` | prompt/UID → gen → `sleep_replicas()` → merge → mask → balance/meta → reward → `extract_reward` | 整段替换为 `batch, reward_tensor, reward_extra_infos_dict, step_row = self._e7_collect_informative_batch(metrics, timing_raw, curr_step_profile)`。helper 内部：每个 chunk **逐字**复用 `:1435–1449, 1460–1462, 1466–1470, 1475–1480, 1495–1501`（去掉 REMAX 分支并 `assert adv_estimator == GRPO`），随后 `extract_reward` + `score` 校验（存在、长度 64×8、finite、`== rm_scores.sum(-1)`）+ `select_informative_groups`；最多 4 chunk；选定后对 selected 64×8 batch **逐字**执行一次 `:1502–1517`（balance/global_token_num/images）并再 `extract_reward` 一次；`:1471` 的 `sleep_replicas()` 移到 helper 的 `finally`，最后一 chunk 后执行恰一次 | 计划 §3.1 冻结语义；review 2026-08-22 插入点 |
| W3a | `:1464–1465` | `is_last_step = ...`；`with marked_timer("step")` | `is_last_step = self.global_steps >= min(self.total_training_steps, self._e7_stop_at_step)`；`"step"` 计时器位置不变，helper 在其内调用（上游 `:1435–1463` 的 prompt 准备在计时器外，E7 移到计时器内，差异为微秒级） | 分段停止（§3.5）；计时覆盖全部 chunk |
| W4 | `:1670–1671` 之后 | `self._save_checkpoint()` | 追加：`if step_row.cumulative_trajectories >= 102_400 and not ledger.equal_compute_saved: self._e7_save_equal_compute_checkpoint()`（写 `checkpoints/equal_compute_step_<N>/{actor,data.pt,e7_ledger.json}`，`max_ckpt_to_keep=None`，不写 tracker） | 计划 §3.4 |
| W5 | `:1680–1683` 之后 | `_log_rollout_data(batch, ...)` | 保留不变（selected batch 进训练 JSONL）；追加 `self._e7_log_candidates(step_row, rollout_data_dir)` 写 `predictions/candidates/<step>.jsonl`（zero-std + surplus 行：uid、chunk、score、acc、length） | 计划 §3.3 证据 |
| W6 | `:1713–1718` | training metrics | 追加 `e7/*` 指标：`generation_batches_used`、`informative/zero_std/surplus_groups`、`prompts/trajectories_sampled`、`rollout_tokens`、`cumulative_trajectories`、`all_correct/all_wrong/mixed_frac`、`generation/filter_wall_time` | 计划 §4 |
| W7 | `:1765–1767` | `on_batch_end(batch=batch)` | 不变（`batch` 即 selected batch，每有效 step 恰一次） | 冻结语义 |

不动的上游行：`:1362–1388, 1390–1421, 1424–1434, 1526–1663, 1672–1679, 1684–1712, 1719–1764, 1768–1770`，
含 old-log-prob、ref、advantage、`_update_actor`、`update_weights`、validation、metrics。

**补采样期间的不变量**（helper 内断言）：不调用 `_update_actor`/`update_weights`；replicas 在最多 4 chunk 间保持 awake，
`sleep_replicas()` 只在 `finally` 执行一次（`update_weights()` 在 actor update 后重新 wake，见 `checkpoint_engine/base.py:470–515`）；
所有 chunk 的 `global_steps` meta 相同；候选 UID 全局唯一；三类 group 之和等于输入组数。

## 4. 驱动边界：两种方案，需用户选择

| 方案 | 做法 | 复制行数 | 先例 |
|---|---|---|---|
| (a) 计划 §6 原文 | 复制 `main_ppo.py:223–315` 93 行为 `DynamicFilteringTaskRunner.run()`，只改 `:299` 的类名 | 93 | 无 |
| (b) **推荐** | mixin `run()`：在驱动进程内把 `verl.trainer.main_ppo.RayPPOTrainer` 名字绑定临时换成 `DynamicFilterRayPPOTrainer`，`try: TaskRunner.run(self, config) finally: 恢复`；`:299` 的查找按模块全局名解析到子类 | 0 | E5/E5-v2 的 `TurnCreditTaskRunner` 用同一机制换 `compute_advantage` 与 `_log_rollout_data`（`rl/launch/e5_turn_credit.py:334–400`），已在 3 条 formal run 上验证 |

(b) 的证明链：子类 `__init__` 打印 `TOOLCREDIT_E7_TRAINER_ACTIVE` 并写 proof 文件（含 pin hash、filter 参数、offload 三标志）；
`finally` 写 `restored=true`；completion gate 要求 proof 存在且训练 JSONL 每 step 的 `e7/*` 指标齐全。
选 (b) 属对计划 §6 原文的偏差，记入 `plans/M10.md` §12。

## 5. checkpoint / resume 边界

- `_save_checkpoint` override：先以 temp+rename 原子写 `global_step_N/e7_ledger.json`，再 `super()._save_checkpoint()`（pin `:974–1041`：actor → `data.pt` → tracker）。ledger 含 `effective_step`、`dataloader_batches_consumed`、`epoch`、累计量、`equal_compute_saved`。
- `_load_checkpoint` override：`super()._load_checkpoint()` 后读取同目录 ledger；`ledger.effective_step != global_steps` 即失败；恢复 `epoch` 与 iterator（`StatefulDataLoader.load_state_dict` 已由 super 完成）。
- **上游启发式的显式拒绝**：pin `:1095–1103` 在 `global_steps % len(train_dataloader) == 0`（81 批/epoch）时跳过 dataloader 状态恢复。该判断对 E7 无意义（有效 step ≠ 批数）。E7 override 在此条件成立时**显式失败**而不是静默跳过。实际上 E7 的常规 checkpoint 都在 25 的倍数，均非 81 的倍数，故只是防御。
- 恢复后重算的候选按 `(effective_step, generation_batch, uid)` 去重；`recovery/` 归档沿用 E3 launcher 的 `archive_interrupted_attempt`。

## 6. 测试映射

Fixture N（纯函数）、O（多 chunk 拼接 + balance 只做一次）、P（underfilled 失败且候选已落盘）、Q（ledger/resume + `--stop-at-step`）、
R（同源 diff：E7 `fit()` 源码与 pin `:1362–1770` 逐行比较，差异行必须带 `# E7-W<n>` 标记或落在 W3 被替换段）、
S（等算力触发恰一次）、T（AUC 插值规则）；config diff gate（`filter_groups` 三键 + trainer 标记 + 名称 + offload 三条）。

## 7. 请求批准的具体判断

1. 接受 §3 的 fit() 复制范围与 8 处白名单；
2. 驱动边界选 (a) 或 (b)；
3. 接受 §5 的 resume 语义（含对上游 epoch-boundary 启发式的显式拒绝）。

## 8. 决定（2026-09-09，助手受用户授权自行审查）

1. **接受 §3 的 fit() 复制范围与 8 处白名单。** 审查要点：W3 是唯一改变数据流的段，其内部每个 chunk 逐字复用上游生成/合并行，
   old-log-prob 之后的全部路径不动；W1/W2/W3a 只处理 iterator 与分段，不触碰数值；W4/W5/W6 只增加保存与记录。
2. **驱动边界选 (b)。** 理由：零复制、E5/E5-v2 三条 formal run 的同机制先例、`finally` 恢复、proof 文件可审计；
   (a) 的 93 行复制没有额外收益且会与 pin 漂移双重耦合。偏差记入 `plans/M10.md` §12。
3. **接受 §5 的 resume 语义**，含对上游 epoch-boundary 启发式的显式拒绝。理由：静默跳过 dataloader 状态恢复会让
   E7 的 prompt 顺序不可复现，宁可失败。
