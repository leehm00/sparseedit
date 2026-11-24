# 图像编辑延迟与服务吞吐的指标对齐

H200 并发主表报告 **images/min**。JSON 同时保留 **images/s**，方便直接与 requests/s 对照。本实验每个请求输出一张图，按相同时间单位换算时，图像数和完成请求数相等。N=1 单图表报告完整请求平均延迟及 p50（s），并列同条件加速比。

## 已发表工作怎么报

| 工作 | 工作负载 | 论文或官方 artifact 的指标 | 对齐方式 |
|---|---|---|---|
| EEdit，ICCV 2025 | 单图提示词、拖拽和参考图编辑 | 表 1 用 `Time (s)`，表 3 用 `Inference (s)`，配合编辑质量指标 | 单图性能表报 N=1 实际请求时间和同模型提速。 [论文](https://doi.org/10.1109/ICCV51701.2025.01623) · [开放全文](https://arxiv.org/pdf/2503.10270) |
| DistriFusion，CVPR 2024 | 多 GPU 合作生成一张图 | 表 1 用 `Latency Value (s)` 与 `Speedup`，正文另述端到端延迟 | 适合对照单图延迟的报告形式，GPU 数量及计时范围需随数字注明。 [正式论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_DistriFusion_Distributed_Parallel_Inference_for_High-Resolution_Diffusion_Models_CVPR_2024_paper.pdf) |
| FlashPS，EuroSys 2026 | 多请求生成式图像编辑服务 | 论文分别报告吞吐提升和平均延迟。官方 artifact 用外部到达率 RPS（requests/s）作负载横轴，统计客户端平均延迟和 p99 | 并发吞吐与延迟分列。正式在线对照还需要对齐外部请求到达、排队和测量窗口。 [论文](https://doi.org/10.1145/3767295.3769379) · [负载与延迟分析代码](https://github.com/Sylvia-16/FlashPS/blob/58d89eb0936f7f0ca62ab833bb9568c5ff48dda7/scheduler/scripts/parse_end2end.py) |
| MoDM，ASPLOS 2025 | 多请求扩散图像生成服务 | 分别测试最大吞吐和在线延迟/SLO。图 10 及 artifact 用 `reqs/min`，图 14 另用 `1/Throughput` | 并发主表报 images/min，旁列摊销 s/image 和请求延迟。 [论文](https://doi.org/10.1145/3760250.3762220) · [开放全文](https://arxiv.org/pdf/2503.11972) |
| Argus，Middleware 2025 | 文生图推理服务 | 图 13 的吞吐轴标注 `Images per min`，端到端实验也报告延迟 SLO 违约 | 主表沿用 images/min，在线评估可随到达率报告延迟与 SLO。 [论文](https://doi.org/10.1145/3721462.3770765) · [开放全文](https://arxiv.org/pdf/2511.06724) |
| RegionE，arXiv 预印本 | FLUX.1 Kontext、Qwen-Image-Edit、Step1X-Edit 单图编辑 | §5.1 明确每次编辑一张图。表 1 报 `Latency (s)` 和相对 vanilla 的 `Speedup` | 在相同模型和工作负载重跑时，可作为 N=1 编辑基线。 [原文](https://arxiv.org/pdf/2510.25590) |

## 当前 F2 对应的口径

| 列 | 定义 |
|---|---|
| 吞吐（images/min） | 所有合格批次完成图数之和，除以同步批次墙钟之和，再乘 60。JSON 也保留未乘 60 的 images/s。 |
| 摊销（s/image） | 同步批次墙钟之和除以完成图数。与 images/s 互为倒数。 |
| 请求延迟平均 / p50（s） | 从本地批次开始到该请求完成。串行按执行顺序累加服务区间，后续请求包括批内等待。并发使用各请求的完成时刻，dense true-batch 的每张图随整个批次返回。 |
| 配对吞吐比 | 相同输入、相同重复、相同 N 的串行批次墙钟除以并发批次墙钟，按原协议汇总配对及 bootstrap 区间。 |

例如，一个并发批次的两张图各在第 10 秒完成，该批次吞吐为 **12 images/min**，摊销为 **5 s/image**，每个请求延迟为 **10 s**。F2 包含编码、去噪、VAE 解码和运行时后处理，模型加载、输入 resize、独立预热、PNG 写盘和评分分别统计。F2 工作负载为固定 N 的离线批次。FlashPS 的请求到达率与在线 SLO 需要单独的服务负载才能复现。

## 表格和基线怎么对应

- **单图表**按模型报告 N=1 完整请求延迟。对照同模型、尺寸、步数、CFG、GPU 和计时边界下的 dense、direct sparse 与 full；复现外部编辑方法时按同一口径重跑。文献数字保留原设备和协议。
- **并发表**按模型报告相同 N、相同输入及相同 GPU 下的 Full 串行、Full 并发和可用的 dense true-batch。表中并列吞吐、摊销和请求延迟。LongCat 和 Qwen 完成各自的 F1 后沿用该表。
- **服务表**若开展在线评估，再固定到达过程、负载率和测量窗口，分别记录到达 RPS、完成吞吐、客户端延迟、尾延迟及 SLO。F2 三例校准用于可行性判断。

更新脚本后可从旧版 F2 原始收据重算上述表格。汇总 JSON 的 `schema_version=2` 记录新的统一请求延迟定义。 [离线分析命令](RESULTS_AND_ANALYSIS.md#对回传目录重新分析)
