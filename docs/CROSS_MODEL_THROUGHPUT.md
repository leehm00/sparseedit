# FLUX、LongCat 和 Qwen 的 H200 吞吐计划

主指标是 H200 上同步后的完整请求 **images/min**。JSON 同时保留 images/s，并记录每请求平均完成时间和 p50、去噪时间和显存峰值。完整墙钟覆盖文本与参考图编码、去噪、VAE 解码及运行时后处理。权重加载、输入 resize、独立预热、PNG 写盘和质量评分分别记录。图像质量可以在回传端复算。吞吐前的逐步等价与双请求隔离用于确认计时的是正确执行的请求。

| 模型 | 同模型对照 | 执行条件 |
|---|---|---|
| FLUX.1 Kontext Dev | Full serial N=1/2、Full concurrent N=1/2、dense true-batch | 已有 H200 运行入口。冻结 768×768、BF16、28 步、warmup 8、refresh 23、true CFG 4.0、K=460、seed 110。 |
| LongCat-Image-Edit | 本窗口执行原生 Dense N=1。统一移植后补 Direct Sparse、Full 和通过本模型隔离验收的 N=2。 | 优先沿用已验证的精确上下文与 graph 路径。基础协议为 1024px、BF16、28 步、warmup 8、refresh 23、CFG 3.5、seed 110。首轮固定 20% 预算、输入与逐例 mask/action。 |
| Qwen-Image-Edit | 本窗口执行原生 Dense N=1。单卡常驻适配后补 Direct Sparse、Full 和通过本模型隔离验收的 N=2。 | 使用原版 Qwen-Image-Edit，以 1024px、BF16、28 步、warmup 8、refresh 23、true CFG 4.0、guidance 1.0、seed 110 和 20% 预算为起点。先验证单张 H200 常驻工作负载，单独保留发生卸载或容量不足的记录。 |

## 跨模型运行次序

1. [H200 窗口入口](../README.md)先分别加载三模型，运行同卡原生基线。各模型使用同一组冻结校准图片和指令，并按自己的分辨率、步数和 CFG 记录协议。
2. 评估 LongCat/Qwen 的 SparseEdit 版本时，分别冻结模型文件哈希、输入、逐例轨迹和源码提交，再运行 Dense、Direct、Full 同输入配对。
3. 在各自的多请求入口上验证 scheduler、latent、条件、KV 和 stream 的隔离。隔离通过后，以独立 batch=1 的两个请求执行顺序平衡的 serial/concurrent N=2。
4. 按模型分表报告绝对 images/min、Direct→Full 优化比和 Full serial→concurrent 并发比。Qwen 的单卡常驻数据独立呈现，历史双卡卸载数据保留原设备和计时口径。

三模型原生阶段独立采集。各 SparseEdit 多请求性能结论使用同模型、同设备、同协议的实测配对。 [窗口覆盖表](H200_EXPERIMENT_MATRIX.md) · [结果分析与验收](RESULTS_AND_ANALYSIS.md) · [文献指标口径](METRIC_ALIGNMENT.md)
