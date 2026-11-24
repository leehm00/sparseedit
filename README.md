# SparseEdit H200 可行性实验

本仓库提供 SparseEdit Phase A 的独立运行入口。在单张 NVIDIA H200 上，用三例冻结校准输入检查 FLUX.1 Kontext Dev 的双请求隔离、完整请求吞吐、显存容量和 CUDA Graph 模板。输出包含原始收据、PNG、GPU 时间线与可重算的汇总报告。

## 开始前

| 资源 | 本轮要求 |
|---|---|
| GPU | 完整 NVIDIA H200，一次运行使用一个物理 GPU UUID |
| 模型 | **FLUX.1 Kontext Dev**，本地 Diffusers 目录，29 个文件逐一核对 SHA-256 |
| 冻结环境 | Python **3.12.14**、PyTorch **2.11.0**、Triton **3.6.0**、vLLM **0.26.0**，PyTorch CUDA 13.0 构建 |
| 校准素材 | `sparseedit_h200_assets.tar.gz` 和 `sparseedit_h200_assets.tar.gz.sha256`，含三张输入、逐例轨迹、F0/F5 参考收据及冻结快照资源 |
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
  sha256sum -c sparseedit_h200_assets.tar.gz.sha256
)
tar --keep-old-files -xzf "$asset_dir/sparseedit_h200_assets.tar.gz" -C "$repo"

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
  --gpu-uuid "$gpu" --output-root "$run" --capacity-max 16
```

运行器先复核 F0 和三例单请求等价，再在该 H200 上验证 F1。三对双请求在同一执行路径通过隔离后，运行 F4 模板探针和 F2 性能对照。F3 同时保留预测、实测峰值及准入证据。H200 实测结果由该节点的运行目录生成。 [完整运行手册](docs/H200_RUNBOOK.md)

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
