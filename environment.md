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

## 版本 pin 表

| 组件 | 版本 | 备注 |
|---|---|---|
| veRL | 待 M0 选定 | |
| torch | 待 M0 选定 | aarch64 CUDA wheel |
| rollout 引擎 | sglang 或 vllm，待 M0 验证 | aarch64 可用性决定 |

## 踩坑记录

（M0 起持续追加。）
