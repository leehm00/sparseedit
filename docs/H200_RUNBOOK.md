# H200 冻结环境与运行手册

[仓库首页](../README.md)给出从克隆到执行的命令。本手册说明模型布局、协议和各阶段的验收条件。

## 冻结模型和环境

实验使用 Hugging Face 的 **`black-forest-labs/FLUX.1-Kontext-dev`**。按模型卡准备访问授权和本地模型文件。运行需要与归档一致的 Diffusers 权重布局，包括 `model_index.json`、scheduler、两个文本编码器和 tokenizer、VAE、三片 Transformer 权重，以及同一目录下的其他绑定文件。每个文件的完整路径和 SHA-256 见清单。 [模型卡](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) · [29 个模型文件](../config/model.sha256)

默认目录关系如下。可以选用既有存储映射，`sparseedit-h200` 与 `flux_a40_kontext_28step_20260912` 须位于同一父目录。克隆路径、Python 和模型目录的实际位置须与所用父目录一致。

```text
/mnt/sda1/experiments/
├── sparseedit-h200/                     Git checkout
│   ├── backend_reference/current_v4/backend_bundle/
│   └── studies/concurrency_feasibility_v1/
└── flux_a40_kontext_28step_20260912/
    ├── envs/flux/bin/python             Python 3.12.14
    └── models/FLUX.1-Kontext-dev/       local checkpoint
```

使用随项目准备的冻结 Python 环境。预检严格验证 Python 3.12.14、PyTorch 2.11.0、Triton 3.6.0 和 vLLM 0.26.0 并记录驱动版本。归档使用 PyTorch CUDA 13.0 构建，节点驱动需支持该构建。复用冻结环境时保留环境依赖和 CUDA 扩展包。完整运行逐文件核对模型 SHA-256。可在启动 GPU 任务前单独复核模型和环境。 [归档依赖版本](../config/environment.json)

```bash
exp=/mnt/sda1/experiments
repo="$exp/sparseedit-h200"
study="$repo/studies/concurrency_feasibility_v1"
flux_root="$exp/flux_a40_kontext_28step_20260912"
flux_python="$flux_root/envs/flux/bin/python"

(
  cd "$flux_root/models/FLUX.1-Kontext-dev"
  sha256sum -c "$repo/config/model.sha256"
)
"$flux_python" -c 'import sys; from importlib.metadata import version; print(sys.version.split()[0], *(f"{p}={version(p)}" for p in ("torch", "triton", "vllm")))'
export PYTHONPATH="$study/source" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$flux_python" "$study/scripts/package_h200.py" verify-extracted "$repo" --assets-and-runtime
"$flux_python" -m pytest "$study/tests" --ignore="$study/tests/test_baseline_contract.py" -q
```

环境测试使用新 F0 锚点。归档历史测试检查缺失的同例历史 full 计时，源收据状态为 `BLOCKED`。真实 F2 收据测试在采集后单独启用。 [F0 锚点测试](../studies/concurrency_feasibility_v1/tests/test_f0_anchor.py) · [F2 收据测试](../studies/concurrency_feasibility_v1/tests/test_bench_contract.py)

## 固定工作负载

| 设置 | 值 |
|---|---|
| 请求 | 文本编辑指令和一张参考图，单请求 batch=1 |
| 图像 | 768×768，BF16，生成 token 和参考 token 各 2304 |
| 去噪 | 28 步，索引从 0 开始。0–7 为 warmup，第 23 步 refresh |
| 稀疏 | K=460，20%，重放各自冻结 mask 和 KV action |
| CFG | true CFG 4.0，guidance 1.0，seed 110 |
| 附加功能 | Velocity reuse 和在线 routing 关闭 |
| 双请求 | 两个独立请求分别保存 scheduler、latent、缓存和 stream。dense true-batch 作为单列对照 |

运行时预检绑定源码身份、校准图与轨迹、设备 UUID 以及环境。模型加载、输入 resize、独立 warmup、PNG 写盘和评分单独记录。F2 主指标覆盖编码、去噪、VAE 解码和运行时图像后处理；设备时间用 CUDA events，批次吞吐使用同步后的墙钟。 [执行协议](../studies/concurrency_feasibility_v1/protocol.json)

## 吞吐模式

仓库首页使用 `--throughput-only`。该模式执行 F0 参照复核、单请求逐步等价、F1 隔离和 F2 配对计时。F0、单请求等价和 F1 检查运行时正确性。F2 仍保存 PNG 哈希、显存峰值和 GPU 时间线，图像质量打分、F3 独立容量扫描和 F4 模板探针不在此轮运行范围。回传端可以从保存的 PNG 补充质量分析。

收据标记 `scope=throughput`。完整 Phase A 可使用同一入口的常规模式，将 F3 和 F4 纳入 GPU 实验。以下表格列出完整阶段定义。

## 执行阶段与接受条件

| 阶段 | 入口与接受条件 |
|---|---|
| F0 | `finalize_f0.py` 复核两轮 A40 原始收据，三对 PNG 相同且逐例时差 ≤2%。新基线为 fresh-run anchor，历史同例 full 计时标为 `UNAVAILABLE`。 |
| Task 2 | `run_isolation.py --equivalence` 在本 H200 比较原生 v4 graph 与请求执行器；三例逐步核对 latent、velocity、索引和 KV 动作以及最终 PNG。另记录独立 warmup、同步墙钟和 graph 采集成本。 |
| F1 | `run_isolation.py` 验收三对不同输入。先私有 graph，必要时用私有 eager。每对对照各自串行结果，记录 KV 地址区间、私有对象和 28 步交错顺序。 |
| F4 | `probe_template_concurrency.py` 探测私有 eager/graph、同输入重复、不同输入同形状、graph 捕获、首个与稳态稀疏步。不同 K 签名通过 host 端模板键检查，GPU 请求固定 K=460。 |
| F2 | `bench_concurrency.py --matrix` 采集三组校准配对、serial/concurrent N=1/2、dense 对照和 GPU 时间线。至少两次顺序平衡重复，统计有疑义时自动追加两次。 |
| F3 | `analyze_capacity.py` 采集 N=1 拟合、独立 N=1、N=2 及逐 N 显存扫描，默认上限 16。N>3 按顺序循环校准输入，并保留独立请求 ID。扫描在第一次实际 OOM 或指定上限结束。 |
| F5 | 使用配套包中的 A40 AsyncOmni 和 FlashPS 审计收据。H200 的强对照在 F2 单独记录。 |

F1 以逐位相同为验收。出现差异时，运行器保存原始张量、差异和输出，阶段必须先通过数值审查。如果私有 graph 输出正确性或资源条件受限，失败收据保留，私有 eager 在相同协议下重新验收。F1 单路径默认 1,800 秒，含模型哈希和加载。冷存储需要更久时，在主命令中设置 `--f1-timeout-s`。

## 阻塞状态和处理

| 现象 | 查看与处理 |
|---|---|
| 素材校验失败 | 对照素材包 SHA-256，检查 checkout 与素材包是否配套。选择新的干净目录重新解包。 |
| Python 或模型校验失败 | 查看资源盘点或阶段日志，按上面的冻结环境和权重路径核对版本及文件 SHA-256。 |
| 设备非 H200 或已有任务 | 读取 `resource_inventory/completion.json`，选择空闲的完整 H200 UUID。 |
| F1 不通过或超时 | 保留 `isolation_v1/` 和 `logs/`。定位首个失败阶段，按原协议重新核验。F2 和 F4 等待同设备 F1 通过。 |
| 容量 OOM | 查看 `capacity_v1/` 的首个 OOM 收据与实测峰值。保留原协议和 OOM 记录。 |
| 主脚本退出 1 | 打开 `completion.json` 及 `results/phase_a_report.md`。已有阶段可能通过，结论由各阶段收据给出。重新执行用新输出目录。 |

[如何阅读和回传结果](RESULTS_AND_ANALYSIS.md)
