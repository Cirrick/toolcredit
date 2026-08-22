# M5 / E7 trainer implementation boundary review

状态：**设计审查已完成；用户有意 defer 到 M6/E5 完成之后**  
核对环境：veRL 0.8.0，2026-08-22  
对应权威实验定义：`plans/M5.md` §3

本文件只记录拟议边界，不包含 trainer、launcher、config 或 filtering 实现。E3、E6、E4 均继续使用
veRL 原生 `RayPPOTrainer`。用户于 2026-08-22 明确暂不批准 E7 实现；这是主动调整执行时序，
不是实现失败、blocked 或取消。M6/E5 完成后如需恢复 E7，仍须基于当时 pin 源码重新核对并获得专项批准。

## Pin 源码证据

- `verl/trainer/ppo/ray_trainer.py` SHA256：
  `de58d295cf86656a28196b0718168d4a11666f3e30957b7e166914496c2a6d66`
- `verl/trainer/main_ppo.py` SHA256：
  `e3fe6e73b18d63367402a2570c5ba054a2b05443ac40a168524dbf545b1da392`
- 标准 `TaskRunner.run()` 在 `main_ppo.py:223–315` 完成 worker、dataset、sampler 初始化，并在
  `:299–310` **硬编码构造 `RayPPOTrainer`**。
- 标准 `RayPPOTrainer.fit()` 在 `ray_trainer.py:1362–1770`。其单步关键数据流是：
  epoch/dataloader 双层 `for` `:1422–1423` → prompt/UID `:1435–1448` → generation +
  `sleep_replicas()` `:1464–1476` → rollout 与 prompt 合并、mask/balance/meta `:1478–1517` → reward
  `:1518–1525` → old/ref log prob `:1527–1580` → advantage
  `:1588–1633` → actor update/checkpoint/weight sync `:1642–1678` → raw rollout dump
  `:1680–1683` → validation `:1685–1693` → metrics/global step `:1709–1761`。
- `algorithm.filter_groups` 字段在该 `fit()` 中没有读取点，YAML-only 不会过滤或补采样。

## 拟新增的 repo-local 边界

只有在用户批准后才新增：

1. `rl/custom/dynamic_filter_trainer.py`
   - `DynamicFilterRayPPOTrainer(RayPPOTrainer)`；
   - 纯函数 `select_informative_groups(...)`：按 UID 验证每组 8 条，使用实际 scalar `score` 的严格
     零方差，返回 selected / zero-std / surplus 和守恒 ledger；
   - 小型状态对象 `DynamicSamplingLedger`：累计 sampled prompts/trajectories/tokens/time，并支持
     checkpoint/resume 序列化；
   - override `fit()`：仅用于 E7，把动态补采样插在 reward 已得到、old log prob/advantage 尚未计算的
     位置；将隐藏在双层 `for` 中的 dataloader iterator 显式化，使一个 effective update 能安全消费
     1–4 个 prompt batch。
2. `rl/launch/e7_dynamic_filtering.py`
   - repo-local `DynamicFilteringTaskRunner(TaskRunner)`；
   - `run()` 复制标准 TaskRunner 的 `:223–315`（约 93 行）worker、dataset、sampler 建立流程，仅把
     `:299` 的 trainer class 从 `RayPPOTrainer` 换为 `DynamicFilterRayPPOTrainer`；上游没有 trainer
     factory hook，不能调用 `super().run()` 后再替换；
   - launcher 调用已有 `main_ppo.run_ppo(config, task_runner_class=...)` 扩展点，不修改 upstream main；
   - 启动时断言并打印实际 trainer class、`metric=score`、64 informative prompts、`n=8`、最多 4 批。
3. `rl/configs/e7_dynamic_filtering.yaml`、CPU tests、M5 shell/watcher 入口。

不会新增或修改任何 veRL site-packages 文件，不 fork、不升级；不会让 E3 launcher 自动选择 E7 trainer。

## Dynamic sampling 的精确插入点

标准 `fit()` 在第一批 reward 产生后立即进入 old-log-prob。E7 必须把 `:1422–1525` 的
“取 64 prompts → 生成 8 rollouts → 合并 → reward”变为同一 effective update 内的有界循环：

1. 从 dataloader 取 64 个新 prompt，创建 UID，生成 8 条 rollout；
2. 调用与 E3 相同的 reward 路径，从 `extract_reward()` 的 extra info 读取实际 scalar `score`；强制
   `score` 存在、长度为 `64×8`、全部 finite，并逐轨迹核对它与 `rm_scores.sum(-1)` 一致；
3. 按 UID 校验 8 条并分类；累积 informative groups；
4. 不足 64 且 generation batch 少于 4 时，再取下一批 prompt 并重复；
5. 取采样顺序前 64 个 informative groups组成 actor batch；zero-std 与 surplus 只存证据，不跨 actor
   snapshot 回收；四批仍不足则在 actor update 前显式失败；
6. 候选 chunk 先完成 prompt/rollout merge 和 response mask；为避免把 DP 重排指标重复计入每个候选
   chunk，标准 `:1502–1517` 的 balance/global-token/image meta 只在最终 `64×8` selected batch 上执行
   一次，然后重新 `extract_reward()` 并进入标准 `:1527` 的 old log prob、ref log prob、advantage 和
   actor update。CPU fixture 必须证明筛选前后的 UID/score/tensor 对齐和守恒。

同一 effective update 的补采样期间不调用 `_update_actor()` 或 `checkpoint_manager.update_weights()`，
因此所有候选由同一冻结 actor snapshot 生成。E7 固定使用 E3 的 rule-based streaming reward，并断言
colocated reward model 关闭；rollout replicas 在最多四个 generation chunk 期间保持 awake，在最终选择
结束后才执行一次标准 `sleep_replicas()`，而不是每批 sleep/wake。异常/underfilled 路径以 `finally`
保证 sleep 后再显式失败。这样不会在补采样中途改变权重，也避免无必要的 replica 状态抖动；actor
update 后仍由标准 `update_weights()` 恢复 rollout 服务。

## 是否必须复制 `fit()`

**当前 pin 版本必须 override `fit()`，且需要复制较大但有明确 hash 的 upstream 控制流。** 原因是：

- generation、reward、dataloader consumption、sleep/wake 和 global-step 生命周期都内联在 `fit()`；
- 当前双层 `for batch_dict in self.train_dataloader` 不暴露 iterator；补采样不能在 helper 中可靠调用
  `next(self.train_dataloader)`，必须由 override 持有显式 iterator，并在 epoch exhausted 时按原边界重建；
- 没有“生成并计分一个 prompt batch 后、old log prob 前”的可 override hook；
- agent loop/reward/callback 无权在同一次 actor update 前继续消费 dataloader 并驱动 rollout；
- 只 override `_get_gen_batch()` 或把 zero-std mask 清零无法补齐 64 informative groups。

拟复制范围是 `fit()` 的完整方法主体 `ray_trainer.py:1362–1770`，约 409 行，而不是复制整个 trainer；
另复制 TaskRunner `run()` 约 93 行，因为 upstream 同样没有 trainer factory method。
修改集中在 `:1422–1525` 的 iterator、候选生成/计分段，以及 ledger/resume 的少量调用点；每个
effective update 仍只调用一次现有 `train_dataset.on_batch_end(batch=selected_batch)`。其余行保持逐行同源，
并用 upstream 文件 SHA 和差分测试防止无声漂移。若未来仍要求实施、但不接受约 409 行 `fit()` 复制，
当前 pin 下没有忠实且更小的等价扩展点；届时才应标记 blocked，不能以 YAML-only 或 mask-only 近似
替代。当前状态是用户主动 defer，并非 blocked。

## 保持与 E3 不变的路径

| 路径 | E7 处理 | 不变量 |
|---|---|---|
| checkpoint/resume | pin `_save_checkpoint():974–1041` 依次保存 actor、`data.pt`，最后写 tracker。小 override 先在同一 `global_step_N/` 以 temp+rename 原子写 ledger，再调用 `super()`；若后续保存中断，旧 tracker 不会引用该目录。`_load_checkpoint()` 调用 `super()` 后按其已解析的 `global_steps` 读取同目录 ledger；E7 的非零 checkpoint 缺失/step 不符即失败 | model/optimizer/scheduler/RNG/StatefulDataLoader 恢复语义不变；tracker 指向的完整 E7 checkpoint 必有同 step ledger；候选 raw 以 `(effective_step, generation_batch, uid)` 去重，恢复后重算部分 step 不混入正式统计 |
| rollout sleep/wake | 最多四个 generation chunk 期间保持 awake；选定或异常后在 `finally` sleep 一次；actor update 后沿用 `update_weights()` | 不改变 SGLang replica 类或 checkpoint engine；无 colocated RM |
| reward | candidate 已由 E3 agent-loop reward 路径写入 `rm_scores`/extra info；每个 chunk 调用原生 `extract_reward()`，校验 `score == rm_scores.sum(-1)` | unshaped `score`、strict boxed、错误显式；不以 `acc`、答案 reward 或 advantage 代替 score |
| old log prob | 只对最终 selected 64×8 batch 调用继承的 `_compute_old_log_prob()` | estimator、entropy 与 rollout-correction 路径不变 |
| reference log prob | 继承 `_compute_ref_log_prob()` | KL reference 与 E3 不变 |
| advantage | 继承模块级 `compute_advantage()`，仍为 GRPO、`n=8`、norm-by-std | filtering 前不算 advantage；zero-std/surplus 不进入 advantage |
| actor update | 继承 `_update_actor()`，每 effective step 恰好一次 | optimizer、PPO minibatch、clip、KL loss 不变 |
| validation | 原样复用 `_validate()`，cadence 以 effective step 0/25/.../200 | fixed panel、greedy、reward/verifier 不变 |
| raw evidence | selected batch 继续走 `_log_rollout_data()`；另写 candidate ledger/raw JSONL，并记录 sampled trajectories、rollout tokens 和 generation/filter wall time | selected/zero-std/surplus 全部可追溯；可同时报告 per-effective-update、per-sampled-trajectory、per-rollout-token、per-wall-clock，且不混淆训练 JSONL |

## 需要专项批准的具体判断

未来批准意味着接受以下 exact boundary：只新增上述 repo-local 文件；E7 TaskRunner 显式构造子类；复制并
锁定约 409 行 pin `fit()`，只在 dataloader/generation/reward 后、old-log-prob 前加入最多四批的动态
筛选循环和 ledger/resume 接线；其余 worker、reward、mask、log prob、advantage、update、checkpoint
与 validation 路径保持标准 E3 实现。用户已决定先完成 M6/E5；在未来重新专项批准前，不创建这些
实现文件、不运行 E7 smoke/full run。
