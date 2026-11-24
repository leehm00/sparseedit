# SparseEdit H200 可行性实验

本仓库提供 SparseEdit 在单张 NVIDIA H200 上的端到端吞吐实验。当前可运行的模型为 FLUX.1 Kontext Dev。运行器先验收两请求隔离，再用三例冻结校准输入测同步批次墙钟、images/s 和每请求 p50。原始收据、PNG 与 GPU 时间线可回传复核。

## 开始前

| 资源 | 本轮要求 |
|---|---|
| GPU | 完整 NVIDIA H200，一次运行使用一个物理 GPU UUID |
| 模型 | **FLUX.1 Kontext Dev**，本地 Diffusers 目录，29 个文件逐一核对 SHA-256 |
| 冻结环境 | Python **3.12.14**、PyTorch **2.11.0**、Triton **3.6.0**、vLLM **0.26.0**，PyTorch CUDA 13.0 构建 |
| 校准素材 | `sparseedit_h200_assets_v2.tar.gz` 和 `sparseedit_h200_assets_v2.tar.gz.sha256`，含三张输入、逐例轨迹、F0/F5 参考收据及冻结快照资源 |
| 结果空间 | 预留约 20 GiB 给日志、PNG、差异快照和 CUDA 时间线 |

校准素材包随实验资料传入内网。模型权重和冻结环境由执行节点准备。模型卡说明权重获取和使用条款，实验按现有模型目录与文件哈希识别版本。 [模型卡](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) · [模型 SHA-256 清单](config/model.sha256) · [环境和模型目录](docs/H200_RUNBOOK.md#冻结模型和环境)

## 克隆、预检和运行

下列命令使用 Bash。先把两个素材包文件放到 `asset_dir`，并把 `gpu` 改成 `nvidia-smi -L` 显示的 H200 UUID。

```bash
set -euo pipefail
exp=/mnt/sda1/experiments
repo="$exp/sparseedit-h200"
asset_dir=/path/to/transferred/assets

git clone https://github.com/leehm00/sparseedit.git "$repo"
cd "$repo"
(
  cd "$asset_dir"
  sha256sum -c sparseedit_h200_assets_v2.tar.gz.sha256
)
tar --keep-old-files -xzf "$asset_dir/sparseedit_h200_assets_v2.tar.gz" -C "$repo"

study="$repo/studies/concurrency_feasibility_v1"
flux_python="$exp/flux_a40_kontext_28step_20260912/envs/flux/bin/python"
export PYTHONPATH="$study/source" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

"$flux_python" "$study/scripts/package_h200.py" verify-extracted "$repo"
nvidia-smi -L
gpu=GPU-REPLACE-WITH-PHYSICAL-UUID

"$flux_python" -u "$study/scripts/run_h200_phase_a.py" \
  --gpu-uuid "$gpu" --output-root "$study/artifacts/h200_preflight_01" --preflight-only

run="$study/artifacts/h200_phase_a_01"
"$flux_python" -u "$study/scripts/run_h200_phase_a.py" \
  --gpu-uuid "$gpu" --output-root "$run" --throughput-only
```

吞吐模式保留 F0、单请求等价和 F1 数值隔离作为 F2 的验收门槛。F2 采集 full 串行、并发及 dense 对照，单独记录显存峰值。图像质量评分、独立容量扫描和模板探针在该模式下跳过。 [完整运行手册](docs/H200_RUNBOOK.md)

LongCat-Image-Edit 和 Qwen-Image-Edit 纳入同卡的跨模型吞吐计划。两者各自冻结协议，使用各自的原生或串行基线。多请求路径须完成模型适配与隔离验收后进入计时。 [跨模型吞吐计划](docs/CROSS_MODEL_THROUGHPUT.md)

## 查看结果及后续分析

先读 `$run/results/phase_a_report.md` 和 `$run/completion.json`。性能判断在 `$run/results/feasibility_summary.json`，包含运行状态、配对吞吐比、95% 区间与 `GO / WEAK / NO_GO` 判定。回传时带上整个 `$run` 目录和归档 SHA-256，汇总脚本会重查 F1/F2 原始 PNG 和收据，再重算统计。 [结果目录与分析命令](docs/RESULTS_AND_ANALYSIS.md)

## 代码入口

| 路径 | 用途 |
|---|---|
| [`run_h200_phase_a.py`](studies/concurrency_feasibility_v1/scripts/run_h200_phase_a.py) | H200 预检、Phase A 调度、阶段日志 |
| [`protocol.json`](studies/concurrency_feasibility_v1/protocol.json) | 固定工作负载、种子、校准图与轨迹的身份 |
| [`run_isolation.py`](studies/concurrency_feasibility_v1/scripts/run_isolation.py) | v4 与重构执行器的逐步等价、双请求 KV 隔离 |
| [`bench_concurrency.py`](studies/concurrency_feasibility_v1/scripts/bench_concurrency.py) | 同步端到端基准、dense 对照和 GPU 时间线 |
| [`analyze_concurrency.py`](studies/concurrency_feasibility_v1/scripts/analyze_concurrency.py) | 重算 F2 配对指标与 bootstrap |
| [`analyze_capacity.py`](studies/concurrency_feasibility_v1/scripts/analyze_capacity.py) | 采集和复核显存容量 |
| [`summarize_phase_a.py`](studies/concurrency_feasibility_v1/scripts/summarize_phase_a.py) | 生成整体 Markdown 和 JSON 汇总 |

[第三方软件与模型声明](THIRD_PARTY_NOTICES.md)
