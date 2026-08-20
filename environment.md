# 环境记录

## 服务器（正式训练）

- **平台：NVIDIA GH200（Grace-Hopper，aarch64）**，GPU 146 GB HBM3e —— 即 PLAN §1.2
  标注的高风险平台，所有 wheel 安装需确认 aarch64 支持。
- 运行环境：JupyterHub pod（用户 jovyan），Ubuntu 24.04，内核 6.8.0-1025-nvidia-64k（64k page size）。
- **pod 内无 docker** → PLAN 建议的 NGC 容器路线不可用，改走 conda 环境 + aarch64 wheel。
- home 为 NFS 挂载；M4 正式 run 已验证约 21 GB 的完整 checkpoint 可稳定写入，step-200
  checkpoint 写入约 23.3 秒，未出现 NFS I/O 失败。
- 系统 Python 3.13（过新，不用）；有 conda 与 uv，**约定用 conda 新建 Python 3.12 环境 `toolcredit`**。
- JupyterHub pod 重建后系统层 `tmux` 可能消失。2026-08-20 为恢复 M4 长跑，在现有
  `toolcredit` 环境安装并验证 **tmux 3.7**（conda-forge）；后台任务统一用
  `conda run -n toolcredit tmux ...`，不依赖 pod 镜像的系统包。

## 本地 5090（调试）

（由用户本人维护，M0 时补充。）

## 版本 pin 表（2026-07-12 选定，安装验证中）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12.13 | conda env `toolcredit`（~/.conda/envs/toolcredit） |
| veRL | **0.8.0**（PyPI 最新 stable） | sglang extra 官方 pin 如下两行 |
| torch | **2.9.1+cu129** | 必须从 download.pytorch.org/whl/cu129 装（见踩坑，cu130 与 sgl-kernel 冲突） |
| sglang | **0.5.8**（veRL 0.8.0 pin） | 纯 py wheel；编译件 sgl-kernel==0.3.21 有 aarch64 wheel |
| vllm | 0.12.0（备选引擎） | 有 aarch64 abi3 wheel；veRL 0.8.0 允许 0.8.5–0.12.0 |
| 驱动 | 580.95.05 / CUDA 13.2 | 向下兼容 cu12x/cu13x wheel |

选型理由：PLAN 首选 SGLang（veRL multi-turn 官方路径）；sglang 0.5.8 依赖链显式适配
aarch64（torchcodec 在 aarch64 上被官方排除，sgl-kernel 全系列有 ARM wheel），无需源码编译。

## 踩坑记录

- **conda 26.3 建的新环境不带 pip**（`conda create -n xx python=3.12` 后无 pip 二进制），
  需 `python -m ensurepip --upgrade` 补装。后台跑安装命令时勿用 `| tail` 吞掉退出码。
- **PyPI 的 torch aarch64 wheel 是 CPU-only**（装出来 `2.9.1+cpu`，x86 上则默认带 CUDA）。
  aarch64 必须显式走 `--index-url https://download.pytorch.org/whl/cu130`（cu126–cu130 均有
  aarch64 wheel）。已装 CPU 版后 `torch==2.9.1` 会被视为已满足，要先 uninstall 再装。
- ~~选 cu130~~ → **sgl-kernel 0.3.21 的 aarch64 wheel 是 CUDA 12 构建**（import 时报
  `libnvrtc.so.12: cannot open shared object file`），与 cu130 torch（带 CUDA 13 运行时库）
  不兼容。已切 **cu129** torch 栈解决。残留的 nvidia-* cu13 包无害，暂不清理。
- **veRL 0.8.0 漏声明运行时依赖 `cachetools`**（`workers/rollout/llm_server.py` import 失败），
  已手动补装并写入 requirements.txt。
- **verl FSDP worker 默认 attn_implementation=flash_attention_2**
  （`verl/workers/config/model.py:185`），已用
  `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` 覆盖模型注意力实现。
- **flash_attn 包本身仍是 verl 0.8.0 训练路径的硬依赖**（`_compute_old_log_prob` →
  `left_right_2_no_padding` → `flash_attn.bert_padding.unpad_input`，与注意力实现和
  rollout 引擎选择无关）。PyPI 无 aarch64 wheel，但上游 GitHub release v2.8.3.post1 起
  提供官方 aarch64 wheel（仅 cu13+torch2.9 变体）。已验证它与 cu129 torch 共存：
  其 libcudart.so.13 由 pod 系统的 /usr/local/cuda 提供；模型注意力仍走 sdpa，
  verl 只用到 bert_padding 的纯 torch 工具函数。
- **veRL 0.8.0 移除了旧版 `examples/sglang_multiturn/` 的 GSM8K 现成示例**（Gsm8kTool 已不存在，
  skypilot yaml 里的引用是陈旧的）。官方 multi-turn tool 路径现为 **agent loop**：
  `examples/tutorial/agent_loop_get_started/`（ReAct + 代码沙箱 + MATH 数据 + GRPO 5 步演示），
  M0 以它为"官方示例"。loss mask 采用 delta-based tokenization
  （docs/sglang_multiturn/multiturn.rst），M2 的 masking 单元测试以此为目标。

## 沙箱隔离能力（M2 实测，2026-07-14）

- **无特权 user namespace 可用**：`unshare --map-root-user -n` 成功 → 沙箱禁网走 fresh
  netns（`env/sandbox.py` import 时探测，不可用则自动降级为无网络隔离并在测试中 skip）。
- 无 mount namespace remount 权限 → 同 uid 绝对路径文件删改挡不住（已知差距，
  `test_user_file_delete_is_a_documented_gap` 钉住现状；工业方案 container/gVisor）。
