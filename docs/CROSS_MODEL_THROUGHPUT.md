# FLUX、LongCat 和 Qwen 的 H200 吞吐计划

主指标是 H200 上同步后的完整请求 **images/min**。JSON 同时保留 images/s，并记录每请求平均完成时间和 p50、去噪时间和显存峰值。完整墙钟覆盖文本与参考图编码、去噪、VAE 解码及运行时后处理。权重加载、输入 resize、独立预热、PNG 写盘和质量评分分别记录。图像质量可以在回传端复算。吞吐前的逐步等价与双请求隔离用于确认计时的是正确执行的请求。

| 模型 | 同模型对照 | 执行条件 |
|---|---|---|
| FLUX.1 Kontext Dev | Full serial N=1/2、Full concurrent N=1/2、dense true-batch | 已有 H200 运行入口。冻结 768×768、BF16、28 步、warmup 8、refresh 23、true CFG 4.0、K=460、seed 110。 |
| LongCat-Image-Edit | Dense、Direct Sparse、Full 的单请求端到端吞吐。具备同语义多请求入口后测 Full serial/concurrent N=2。 | 优先沿用已验证的精确上下文与 graph 路径。基础协议为 1024px、BF16、28 步、warmup 8、refresh 23、CFG 3.5、seed 110。首轮固定 20% 预算、输入与逐例 mask/action。 |
| Qwen-Image-Edit | Dense、Direct Sparse、Full 的单请求端到端吞吐。具备同语义多请求入口后测 Full serial/concurrent N=2。 | 使用原版 Qwen-Image-Edit，以 1024px、BF16、28 步、warmup 8、refresh 23、true CFG 4.0、guidance 1.0、seed 110 和 20% 预算为起点。先验证单张 H200 常驻工作负载，单独保留发生卸载或容量不足的记录。 |

## 跨模型运行次序

1. 完成 [FLUX 吞吐入口](../README.md) 的三例校准，回传 F1/F2 原始收据并复核判断。正式跨模型矩阵在 FLUX F2 获得 `GO` 并经审阅后展开。
2. 为 LongCat-Image-Edit 和 Qwen-Image-Edit 各冻结本地权重文件哈希、输入、请求设置、逐例轨迹和源码提交。各模型先运行同设备同输入的 Dense、Direct、Full 单请求配对，对照验证主计时边界。
3. 在各自的多请求入口上验证 scheduler、latent、条件、KV 和 stream 的隔离。同一配对和运行路径均通过后，以两个独立 batch=1 请求执行顺序平衡的 serial/concurrent N=2。
4. 按模型分表报告绝对 images/min，以及同模型的 Direct→Full 优化比和 Full serial→concurrent 并发比。Qwen 的单卡常驻结果独立呈现，历史双卡卸载测量仅保留原设备和计时口径。

单请求优化和双请求并发回答两个不同的性能问题，每个比值绑定各自的对照。LongCat 与 Qwen 当前属于跨模型扩展计划，FLUX 的 H200 入口只接受冻结 FLUX 配置。正式表使用实际完成且通过验收的配对收据。 [结果分析与验收](RESULTS_AND_ANALYSIS.md) · [文献指标口径](METRIC_ALIGNMENT.md)
