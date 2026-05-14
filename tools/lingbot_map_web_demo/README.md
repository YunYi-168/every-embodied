# LingBot-Map Web Demo

这是 LingBot-Map 的轻量本地 Web 包装器，用于教程 smoke test。它接收一个连续视频，调用本地 LingBot-Map 的 `demo_render/batch_demo.py`，然后展示点云渲染结果。

## 使用方式

```bash
export LINGBOT_MAP_ROOT=/path/to/lingbot-map
export LINGBOT_MAP_CKPT=/path/to/lingbot-map/checkpoints/lingbot-map-long.pt
export LINGBOT_MAP_PYTHON=/path/to/mamba/envs/lingbot-map/bin/python
export LINGBOT_MAP_RUN_ROOT=/path/to/lingbot_map_runs

pip install -r requirements.txt
python app.py
```

打开：

```text
http://127.0.0.1:7860
```

运行结果默认写入 `LINGBOT_MAP_RUN_ROOT`。不要把权重、原始视频、`.npz` 预测文件或长视频输出提交到教程仓库。
