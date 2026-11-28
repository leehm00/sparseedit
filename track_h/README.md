# SparseEdit Track H 内网运行

本入口在一张完整 H20 或 H200 上按 H1、H2、H3、H4 的顺序运行。推理采用 `batch=1` 串行请求，按物理 GPU UUID 校验型号和进程，并在阶段执行期间持有 UUID flock。阶段收据记录同步请求墙钟、源码和素材标识、动作轨迹和输出 PNG。图像配对质量在回传端计算。

## 拉取和运行

```bash
git clone --branch track-h-serial-v2 https://github.com/leehm00/sparseedit.git /mnt/sda1/experiments/sparseedit-track-h
```

把另交的 `track_h_assets_v2.tar.gz` 和同名 `.sha256` 文件放在同一目录。运行器自动核对压缩包和包内每个素材哈希，再在临时目录解包。内网机需已备好冻结 Python 环境及本地模型权重。

```bash
/mnt/sda1/experiments/flux_a40_kontext_28step_20260912/envs/flux/bin/python -u /mnt/sda1/experiments/sparseedit-track-h/track_h/run_track_h.py \
  --gpu-uuid GPU-REPLACE-WITH-UUID-FROM-NVIDIA-SMI-L \
  --assets /path/to/track_h_assets_v2.tar.gz \
  --output /mnt/sda1/experiments/sparseedit-track-h-run-01
```

`--output` 使用新目录。整轮正式跑涉及数千个串行请求。短窗口可给同一条命令追加 `--run-mode cal3` 或 `--stages H1 H2`。H2 默认 24 来源一轮，`--representative-sources 48` 切换成 48 来源两轮。正式模式的 H1、H3 均按 48 来源两轮计时。加 `--dry-run` 可以在不初始化 CUDA 的情况下核验素材包并列出入口。

| 模型 | 默认权重目录 | 覆盖参数 |
|---|---|---|
| FLUX.1 Kontext Dev | `/mnt/sda1/experiments/flux_a40_kontext_28step_20260912/models/FLUX.1-Kontext-dev` | `--flux-model` |
| Qwen-Image-Edit 原版 | `/mnt/sda1/models/Qwen-Image-Edit` | `--qwen-model` |
| LongCat-Image-Edit | `/mnt/sda1/models/meituan-longcat/LongCat-Image-Edit` | `--longcat-model` |

沿用历史运行环境 Python 3.12.14、Torch 2.11.0、Triton 3.6.0、vLLM 0.26.0。Qwen 和 LongCat 运行还检查冻结 Diffusers 和 Transformers 接口及版本。Qwen 路径使用 Diffusers 0.38.0、Transformers 5.5.3。三个模型的权重在目标机按冻结资产清单验证。模型路径不同可加表中的参数。单独跑 H1 时可用 `--qwen-python` 指定兼容的 Qwen 解释器。选择 H3 时，启动解释器需要同时具备三个模型的依赖。

## 实际运行范围

| 阶段 | 当前入口覆盖 | 收据中的缺项 |
|---|---|---|
| H1 | Qwen 1024px 全常驻 cal3 分段剖析、固定 20% 轨迹 Dense/Direct/Ready/SparseGraph、5 图 × 4 指令 L3 缓存对照 | Qwen 变长正负分支的 B2 Joint-CFG、在线 K+B。L3 缓存视觉塔特征和 VAE latent，语言模型按当前指令运行。 |
| H2 | FLUX 768/1024/1536px 的 Dense、Direct、Standard、Full-v4、Joint-CFG、参考段 INT8。cal3 涵盖三档预算。较高预算的 Joint-CFG 显式请求 B2，并核对实际回放。 | 若 1536px Full-v4 无法生成配对轨迹，允许分开采集 INT8 的容量探针。探针为非配对结果。 |
| H3 | FLUX 768px 的 Dense、Dense Graph、TeaCache、ToCa、RAS、RegionE、三个在线 SparseEdit 路径。FLUX 1024px 的非路由基线。LongCat 两个 Dense 对照。Qwen 全常驻 Dense，以及明示的固定 20% 补充臂。 | CFG4 SpotEdit、FLUX 1024px 在线路由，以及 LongCat、Qwen 缺少同协议移植的基线和在线配置。 |
| H4 | 对同时具备三个冻结校准来源、多预算观测和 48 来源在线数据的同模型、同分辨率、同运行时配置做样本外预测。 | 缺任一数据的配置标为 BLOCKED。仅用 Hopper 的 cal3 数据估计系数，测试请求只提供预算、已执行步数和图捕获试运行次数等工作量特征。 |

这些入口通过了本地 CPU 契约验证。CUDA Graph、模型放置、效果和性能由内网运行的原始收据判断。全链默认可能返回 `PARTIAL`，上表的覆盖缺口会出现在报告中。已有少量请求、OOM、数值失败和启动失败都会保留实际收据。

## 查看和回传

运行中的阶段状态写入 `session_progress.json`，各阶段的原始日志位于 `H1.log` 至 `H4.log`。结束后查看 `report.md` 和总 `completion.json`，以及对应阶段目录下的逐请求收据。结果归档包含报告、日志、收据、动作轨迹与输出 PNG。执行结束时会打印回传文件路径。

回传运行目录旁的 `sparseedit-track-h-run-01_receipts.tar.gz` 和其 `.sha256` 文件。退出码 0 表示所有所选阶段通过，2 表示部分覆盖或部分结果，1 表示启动或执行错误，130 表示操作员中断。复跑使用新输出目录以便保留每次原始结果。
