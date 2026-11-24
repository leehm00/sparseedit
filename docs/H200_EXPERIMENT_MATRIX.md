# H200 实验覆盖清单

同一个 H200 窗口依次收集三模型原生 dense 基线、FLUX 配对实验和扩展数据。F1 和 F2 的结论约束对应结果可支持的性能主张。独立模型、容量探测和 formal FLUX 各自执行并保存收据。窗口开始前可用 `run_h200_window.py --list` 核对实际入口及本地素材。 [机器可读清单](../config/h200_experiment_matrix.json) · [H200 运行手册](H200_RUNBOOK.md)

## 窗口中的可运行项目

| 阶段 | 覆盖范围 | 收据和用法 |
|---|---|---|
| 三模型 dense | FLUX 768px、LongCat 1024px 和 Qwen 1024px 原生单请求 | 三模型分别报告已同步的端到端吞吐、延迟及峰值显存。LongCat 和 Qwen 的模型路径分别由操作者传入。 |
| FLUX F0、F1、F2 | 历史 F0 读取、H200 串行等价、隔离及 N=1/2 配对 | 根据同设备、同执行路径的 F1 和 F2 收据判定。 |
| FLUX 768px 扩展 | 私有 eager、私有 graph 分开测 N=1/2/4/8 | 保存每个点的结果、失败及 OOM。三例校准重复结果用于扩展趋势和容量观察。正式 p99 需要至少 24 个独立来源和两次重复。 |
| FLUX 1024px dense | 原生单请求与实际 Transformer N=2 dense 批处理、相同校准输入 | 记录 1024px 时延和显存的 dense 对照。 |
| FLUX 原生服务 | 原版 vLLM-Omni AsyncOmni 路径 | 从请求和 worker 时间线判断两个请求实际串行或并发执行。 |
| FLUX F3 容量 | 768px 私有请求随并发增长的峰值、容量预测和准入 | 扫描至 `--capacity-max` 或实际容量边界，保留 allocated 和 reserved 口径。 |
| FLUX F4 模板 | 私有 eager、私有 graph、串行重复和 K 签名 | 该路径读取匹配的 F1 收据。收据缺失时仍以本地串行对照采集诊断数据，并标记证据范围。分别记录冷启动、命中和捕获信息，CUDA Graph 的请求私有地址仍独立持有。 |
| FLUX F6 packing | K=460、hidden=3072、MLP=12288，N=1/2/4 的循环和合并 GEMM | 独立微基准，报告真实形状的计算收益。 |
| FLUX 48 来源 | 768px 固定 20% 轨迹，Direct、校准选出的 Standard、Full 各两次 | 安装可选 formal 素材后运行。先完成该 runner 自己的三例后端验收和校准，再测 288 个正式请求。 |

## 48 来源 FLUX 的可选素材

`sparseedit_h200_flux_formal_48_v1.tar.gz` 含冻结 v4 runner、runtime 和 51 组输入与轨迹，覆盖三例校准及 48 个正式来源。代码和素材都由冻结指纹校验，模型权重继续使用本地已准备的 FLUX.1 Kontext Dev。文件约 44 MB，需经内网素材传输渠道带入，SHA-256 校验值随包交付。基础实验继续使用 v2 素材。 [机器可读文件身份](../config/h200_experiment_matrix.json)

```bash
cd "$asset_dir"
sha256sum -c sparseedit_h200_flux_formal_48_v1.tar.gz.sha256
tar -xzf sparseedit_h200_flux_formal_48_v1.tar.gz
formal_assets="$asset_dir/sparseedit_h200_flux_formal_48_v1"
```

将 `--formal-flux-assets "$formal_assets"` 加到窗口命令末尾。单独重试时加 `--stage flux_formal_48` 并使用新的 `--output-root`。正式 FLUX 默认使用单独的较长阶段时限，安排在独立短项之后。完整阶段生成 `flux_formal_48/completion.json`、`flux_formal_48/native/` 和 `flux_formal_48/formal_summary.json`。分析在同一节点验证原始清单及 PNG 后生成。传回完整的阶段目录，即可从收据复核中断位置与已完成的来源和重复。

## 原计划的后续覆盖

| 原计划 | 本窗口对应结论 | 其余实现或输入 |
|---|---|---|
| Task 8 正式 serving 集成 | native AsyncOmni 基线、私有 executor 报告各自的执行边界 | 在 diffusion engine active request set 接入私有请求、容量准入和安全回退。验证 worker 内实际 step 并发、结果一致和关闭/异常无泄漏。 |
| Task 9 FLUX N=4/8、1024px | FLUX 768px 私有 scaling、原生 dense 1024px | 1024px 稀疏需冻结新的 token 几何与逐例轨迹，并执行独立隔离、并发和容量检查。 |
| E1 单请求强后端 | native dense 和 formal 48 来源 Direct、Standard、Full | Standard 的 fastops eager、FF compile 与 sparse graph 先本地验收。额外的 dense fastops、compile 或 graph 变体采用独立入口和同口径校验。 |
| E4 多请求 | FLUX F2 与 N=1/2/4/8 scaling | 正式 48 来源配对和 p99 使用足够独立来源，同设备复现可用的 dense/standard 多请求对照。 |
| E5 容量 | 768px 容量扫描、eager/graph scaling 峰值和 1024px dense 峰值 | 覆盖稀疏 1024px 及受控模板开关后，核对固定 N 误差、最大 N 和预算回退。 |
| E6 敏感性 | 基础协议的 FLUX 20% 稀疏点，dense 分辨率两点 | 10%、35%、50%、稀疏 1024px 和必要的 CFG、warmup、refresh、步数点逐项冻结并重采匹配轨迹。 |
| E7 跨模型 | LongCat 和 Qwen 1024px 原生 dense，分别记录 H200 常驻情况 | LongCat 归档已有 1024px Direct/优化串行及三档预算。正式 driver 接受显式本地模型、源码、输入和轨迹路径，另需完整参考素材和本设备校准，才能归入 H200 矩阵。Qwen 已有双设备稀疏归档，单卡 H200 稀疏路径和多请求隔离需分别移植与验证。 |
| E8 模板摊销 | F4 冷/暖、命中、首请求与稳态，graph/eager 扩展 | 实际异 K 和不同池容量需匹配的多预算轨迹，结构签名检查单列。 |
| E9 质量 | 保存 PNG、哈希和已采集的逐步数值凭据 | 按相同模型、来源、重复和协议配对 dense 与优化输出，在回传端计算 LPIPS、PSNR、SSIM、CLIP。精确路径与算术变化路径各自报告。 |
| E10 Artifact | 窗口汇总、阶段日志及 JSON 收据 | 回传同一窗口的完整目录与 SHA-256，收据中的失败和跳过点随同成功项保留。 |
| Task 11 F6、C4c、E11 | 独立 F6 GEMM、有关的阶段时间线和固定轨迹 | 端到端运行时 packing 及步类型调度需要各自实现和并发校验。外部 mask 的跨接口验证需额外策略和边界测试。 |

原生基线、校准扩展、算子微基准和正式来源矩阵保持各自的请求边界及样本口径。论文表图按模型、分辨率、并发度、设备和协议逐项关联原始收据。 [评估指标定义](METRIC_ALIGNMENT.md) · [分析结果与回传](RESULTS_AND_ANALYSIS.md)
