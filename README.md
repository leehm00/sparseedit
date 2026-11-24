# SparseEdit H200 可行性实验

本仓库提供 SparseEdit 在单张 NVIDIA H200 上的一次性实验入口。运行清单覆盖三模型原生吞吐、FLUX 数值隔离与配对吞吐、并发扩展、容量、模板、服务路径和正式单请求矩阵。各阶段独立留存原始收据、PNG、时间线与日志。FLUX F2 的判定用于解释该路径的并发证据。

## 开始前

| 资源 | 本轮要求 |
|---|---|
| GPU | 完整 NVIDIA H200，一次运行使用一个物理 GPU UUID |
| 模型 | 本地 Diffusers 格式的 **FLUX.1 Kontext Dev**、**LongCat-Image-Edit**、**Qwen-Image-Edit**。FLUX 逐文件核对冻结的 29 个 SHA-256 |
| 冻结环境 | Python **3.12.14**、PyTorch **2.11.0**、Triton **3.6.0**、vLLM **0.26.0**，PyTorch CUDA 13.0 构建 |
| 校准素材 | `sparseedit_h200_assets_v2.tar.gz` 和 `sparseedit_h200_assets_v2.tar.gz.sha256`，含三张输入、逐例轨迹、F0/F5 参考收据及冻结快照资源 |
| 正式素材 | 48 来源阶段另携 `sparseedit_h200_flux_formal_48_v1.tar.gz` 及其 SHA-256 文件 |
| 结果空间 | 预留约 20 GiB 给日志、PNG、差异快照和 CUDA 时间线 |

校准素材包随实验资料传入内网。素材和冻结运行时按配套包的清单验证，运行与分析脚本以 Git checkout 为准。模型权重和冻结环境由执行节点准备。模型卡说明权重获取和使用条款，实验按现有模型目录与文件哈希识别版本。 [模型卡](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) · [模型 SHA-256 清单](config/model.sha256) · [环境和模型目录](docs/H200_RUNBOOK.md#冻结模型和环境)

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

"$flux_python" "$study/scripts/package_h200.py" verify-extracted "$repo" --assets-and-runtime
nvidia-smi -L
gpu=GPU-REPLACE-WITH-PHYSICAL-UUID

# 有正式素材时解包，运行器会在此目录发现它
if [ -f "$asset_dir/sparseedit_h200_flux_formal_48_v1.tar.gz" ]; then
  (
    cd "$asset_dir"
    sha256sum -c sparseedit_h200_flux_formal_48_v1.tar.gz.sha256
  )
  mkdir -p "$repo/offline_formal"
  tar --keep-old-files -xzf "$asset_dir/sparseedit_h200_flux_formal_48_v1.tar.gz" -C "$repo/offline_formal"
fi

run="$study/artifacts/h200_window_01"
"$flux_python" -u "$study/scripts/run_h200_window.py" \
  --gpu-uuid "$gpu" --output-root "$run" \
  --longcat-model /mnt/sda1/models/meituan-longcat/LongCat-Image-Edit \
  --qwen-model /mnt/sda1/models/Qwen-Image-Edit --list

"$flux_python" -u "$study/scripts/run_h200_window.py" \
  --gpu-uuid "$gpu" --output-root "$run" \
  --longcat-model /mnt/sda1/models/meituan-longcat/LongCat-Image-Edit \
  --qwen-model /mnt/sda1/models/Qwen-Image-Edit
```

运行器先采集三模型原生吞吐，随后执行各项 FLUX 实验。某阶段的数值失败、超时、OOM 或缺少独立模型素材会写入该阶段收据，后续阶段继续。正式 48 来源是长时阶段，只有确定 GPU 归还时间时才传入 `--budget-minutes`。已完成的批次逐项落盘。 [H200 窗口清单](docs/H200_EXPERIMENT_MATRIX.md) · [完整运行手册](docs/H200_RUNBOOK.md)

## 查看结果及后续分析

先读 `$run/report.md` 和 `$run/completion.json`。FLUX 配对 `GO / WEAK / NO_GO` 位于 `$run/flux_f2/results/feasibility_summary.json`。扩展扫描和原生性能分别在各阶段目录汇总。回传整个窗口目录和归档 SHA-256，即可按阶段复核并重算统计。 [结果目录与分析命令](docs/RESULTS_AND_ANALYSIS.md) · [与论文指标对齐](docs/METRIC_ALIGNMENT.md)

## 代码入口

| 路径 | 用途 |
|---|---|
| [`run_h200_window.py`](studies/concurrency_feasibility_v1/scripts/run_h200_window.py) | 一次窗口内的独立作业编排、日志及汇总 |
| [`run_h200_phase_a.py`](studies/concurrency_feasibility_v1/scripts/run_h200_phase_a.py) | H200 预检、Phase A 调度、阶段日志 |
| [`protocol.json`](studies/concurrency_feasibility_v1/protocol.json) | 固定工作负载、种子、校准图与轨迹的身份 |
| [`run_isolation.py`](studies/concurrency_feasibility_v1/scripts/run_isolation.py) | v4 与重构执行器的逐步等价、双请求 KV 隔离 |
| [`bench_concurrency.py`](studies/concurrency_feasibility_v1/scripts/bench_concurrency.py) | 同步端到端基准、dense 对照和 GPU 时间线 |
| [`analyze_concurrency.py`](studies/concurrency_feasibility_v1/scripts/analyze_concurrency.py) | 重算 F2 配对指标与 bootstrap |
| [`analyze_capacity.py`](studies/concurrency_feasibility_v1/scripts/analyze_capacity.py) | 采集和复核显存容量 |
| [`summarize_phase_a.py`](studies/concurrency_feasibility_v1/scripts/summarize_phase_a.py) | 生成整体 Markdown 和 JSON 汇总 |

[第三方软件与模型声明](THIRD_PARTY_NOTICES.md)
