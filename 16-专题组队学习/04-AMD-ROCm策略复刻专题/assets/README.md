# 资产说明

本目录保存 AMD ROCm 策略复刻专题中使用的小体积教学图。图表和关键帧序列由 `../code/generate_tutorial_assets.py` 生成，便于大家核对教程中的成功率和 rollout 行为。

重新生成图表时，先准备自己的实验输出目录，再运行：

```bash
python code/generate_tutorial_assets.py --source-root "$OUTPUT_ROOT"
```

其中 `$OUTPUT_ROOT` 应包含批量评估 JSONL/TSV 和代表性 rollout 视频。若不传 `--source-root`，脚本会使用内置的示例指标生成图表，并为缺失视频生成占位图。

| 文件 | 用途 |
| --- | --- |
| `model_status_summary.png` | 专题当前复刻状态总览 |
| `smolvla_red_blue_success.png` | SmolVLA 红杯/蓝杯固定指令成功率对比 |
| `act_dagger_progress_curve.png` | ACT 从基线到 DAgger 纠偏的成功率变化 |
| `smolvla_blue_failure_sequence.jpg` | SmolVLA baseline 蓝杯失败关键帧 |
| `smolvla_blue_success_sequence.jpg` | SmolVLA 加权采样后蓝杯成功关键帧 |
| `act_failure_sequence.jpg` | ACT DAgger 典型失败关键帧 |
| `act_success_sequence.jpg` | ACT DAgger 物理成功关键帧 |
| `metrics_snapshot.json` | 小体积指标快照，便于核对图表数字 |
