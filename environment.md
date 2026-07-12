# 环境记录

## 服务器（正式训练）

- **平台：NVIDIA GH200（Grace-Hopper，aarch64）**，GPU 146 GB HBM3e —— 即 PLAN §1.2
  标注的高风险平台，所有 wheel 安装需确认 aarch64 支持。
- 运行环境：JupyterHub pod（用户 jovyan），Ubuntu 24.04，内核 6.8.0-1025-nvidia-64k（64k page size）。
- **pod 内无 docker** → PLAN 建议的 NGC 容器路线不可用，改走 conda 环境 + aarch64 wheel。
- home 为 NFS 挂载（约 706 GB 剩余，2026-07-12）；大模型缓存/ckpt 的 IO 性能待 M0 实测。
- 系统 Python 3.13（过新，不用）；有 conda 与 uv，**约定用 conda 新建 Python 3.12 环境 `toolcredit`**。

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
  （`verl/workers/config/model.py:185`），flash-attn 无 aarch64 预编译 wheel。
  用 `+actor_rollout_ref.model.override_config.attn_implementation=sdpa` 覆盖，先跑通；
  flash-attn 源码编译（约 30–60 分钟）留作后续优化项。
- **veRL 0.8.0 移除了旧版 `examples/sglang_multiturn/` 的 GSM8K 现成示例**（Gsm8kTool 已不存在，
  skypilot yaml 里的引用是陈旧的）。官方 multi-turn tool 路径现为 **agent loop**：
  `examples/tutorial/agent_loop_get_started/`（ReAct + 代码沙箱 + MATH 数据 + GRPO 5 步演示），
  M0 以它为"官方示例"。loss mask 采用 delta-based tokenization
  （docs/sglang_multiturn/multiturn.rst），M2 的 masking 单元测试以此为目标。
