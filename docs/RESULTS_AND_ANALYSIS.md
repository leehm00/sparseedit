# 结果、统计判定与回传

## 运行产物

以下路径相对于主命令的 `--output-root`。吞吐模式的汇总标记 `scope=throughput`，独立 F3 和 F4 阶段标记为 `NOT_REQUESTED`。请从该目录的 `completion.json` 开始，核对设备 UUID、源码树 SHA-256、协议、子命令和退出码，再查对应阶段的原始收据。

| 路径 | 内容 |
|---|---|
| `resource_inventory/completion.json` | 预检设备型号、UUID、总显存、空闲显存、已有计算进程、驱动及环境版本 |
| `f0_anchor_v1/completion.json` | 两轮冻结 A40 参考的验证结果及原始证据哈希 |
| `executor_equivalence_v1/` | 三例逐步等价收据、原生与执行器 PNG、数值差异、同步时间及 graph 捕获成本 |
| `isolation_v1/completion.json` | F1 总状态及所用路径，`pair_receipt_directory` 指明成功或最后尝试的路径 |
| `isolation_v1/<路径>/` | 每对的 56 次交错入队、28 步比较、KV 地址区间、显存和 PNG。串行参考 PNG 在该路径下的 `serial_reference/` |
| `template_probe_v1/completion.json` | F4 私有 graph/eager 的正确性、模板命中、冷启动及复用结果 |
| `concurrency_v1/` | F2 逐单元原始收据和 PNG，批次与每请求时间、显存、吞吐、执行路径及 GPU 时间线 |
| `capacity_v1/` | F3 原始拟合、独立验证、N 扫描、首次 OOM 或扫描终点，以及固定 N 与最大 N 汇总 |
| `results/phase_a_summary.json`、`results/phase_a_report.md` | 全阶段状态、证据位置和整体判定 |
| `results/feasibility_summary.json`、`results/feasibility_report.md` | F2 矩阵、重复、吞吐、延迟、对照、bootstrap 和决策 |
| `logs/` | 按子命令保留的 stdout/stderr。启动和失败的诊断也写在此处 |

F5 参考位于外挂素材目录的 `$study/artifacts/serving_baseline_v1/`，包含 A40 AsyncOmni 与 FlashPS 审计结果。F2 的 H200 dense 对照以本次同设备实测为准，所有图表归因于其原始运行目录。

## 先核对正确性与覆盖率

- F1 检查逐步 latent、velocity 和最终 PNG 的一致性，同时检查 KV 地址区间、对象身份及按步交错。F2 要求三对通过，且使用相同设备、源码、执行路径和冻结输入。
- F2 确认 `full` 的串行和并发以相同配对与重复比较。dense true-batch 通过实际 Transformer batch 维度验收。standard sparse 并发的不可用路径以 `UNSUPPORTED` 收据记录。
- 吞吐按请求数除以同步批次墙钟计算。同一输入配对和重复的并发吞吐比等于串行墙钟除以并发墙钟。每组先汇总重复，再以三个配对为单位做 5,000 次 percentile bootstrap，随机种子为 110。
- 单请求延迟读取端到端墙钟和设备事件，稀疏步与 kernel 时间用于解释。三例校准数据用于可行性决策，正式尾延迟需更大样本。
- F3 分别读取 allocated 与 reserved 的预测误差。固定 N 验收需要成功实测点，目标相对误差 ≤5%。最大 N 单独核对最后成功点和首个真实 OOM。扫描达到上限时只能确认下界。

| F2 判定 | 规则 |
|---|---|
| `GO` | 配对吞吐点估计 ≥1.15×，且 95% 区间下界 >1.05× |
| `WEAK` | 有效吞吐证据达到 1.05×，且完整统计结果尚未满足 GO 条件 |
| `NO_GO` | 点估计 <1.05× |
| `REPEAT_REQUIRED` | 首两次重复后区间跨 1.05× 或结果为 WEAK，入口加跑两次顺序平衡重复 |
| `BLOCKED` / `INCOMPLETE` | 查看缺失的阶段、原始收据或对照。判定字段此时是空值 |

Phase B 需在 F2 为 GO 后由作者审阅再启动。

## 对回传目录重新分析

在同一 Bash 会话内设置仓库路径和冻结 Python，将 `run` 指向解包的实际运行目录，`analysis` 指向新的输出目录。分析命令读取磁盘证据，可以在无需占用 GPU 的机器运行。F2 分析器会按 SHA-256 重新定位 F1 收据，并核对原始串行和并发 PNG。

```bash
exp=/mnt/sda1/experiments
repo="$exp/sparseedit-h200"
study="$repo/studies/concurrency_feasibility_v1"
flux_python="$exp/flux_a40_kontext_28step_20260912/envs/flux/bin/python"
run=/path/to/returned/h200_phase_a_01
analysis="$run/reanalysis"

export PYTHONPATH="$study/source" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
lane=$("$flux_python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("pair_receipt_directory", "private_graph"))' \
  "$run/isolation_v1/completion.json")

"$flux_python" "$study/scripts/analyze_concurrency.py" \
  --input-root "$run/concurrency_v1" --isolation-root "$run/isolation_v1/$lane" \
  --results-root "$analysis/f2" --skip-quality-scoring

"$flux_python" "$study/scripts/summarize_phase_a.py" \
  --run-root "$run" --results-root "$analysis" \
  --receipt "F0=$run/f0_anchor_v1/completion.json" \
  --receipt "F2=$analysis/f2/feasibility_summary.json"
```

F2 分析器退出 1 时，仍会生成原因和状态 JSON，随后可单独运行 Phase A 汇总命令。形成 F2 完整判定后，可执行真实收据契约检查。如果要在回传端补充 Full 相对 Dense 的像素 MSE 和 PSNR，可在新输出目录重跑分析器并省略 `--skip-quality-scoring`。

```bash
SPARSEEDIT_VERIFY_F2=1 SPARSEEDIT_F2_ROOT="$run/concurrency_v1" \
  "$flux_python" -m pytest "$study/tests/test_bench_contract.py" -q
```

## 回传整个运行目录

保留原始 Markdown、JSON、PNG、日志、GPU 时间线及数值差异。出错、OOM、中断及后续重跑各用自己的目录。下面两条命令生成供回传的 archive 和 SHA-256 文件。

```bash
result_archive="$(basename "$run")_receipts.tar.gz"
tar -czf "$result_archive" -C "$(dirname "$run")" "$(basename "$run")"
sha256sum "$result_archive" > "$result_archive.sha256"
```

接收端先运行 `sha256sum -c`，解包后重建汇总。一起提供 Git 提交 ID 和 `resource_inventory/completion.json`，便于复查环境和源码身份。
