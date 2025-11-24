# Course Quality Detector (pipeline)

## Quick start

1) 创建虚拟环境并安装依赖（需 GPU/CUDA 才能跑得动）：
```bash
pip install -r requirements.txt
```

2) 生成人脸库（默认读取 `data/picture`，输出到 `outputs/face_gallery`）：
```bash
python scripts/face_gallery.py
```

3) 运行端到端流水线（默认视频 `data/video`，模型 `yolo11x.pt`，跟踪使用 ByteTrack）：
```bash
python scripts/run_pipeline.py --video data/video/南通职大主楼509室047zk_20251029072500_20251029091707_2.mp4
```

输出文件：
- `outputs/face_gallery/`：人脸 embeddings + 元数据
- `outputs/track/exp/`：YOLO 跟踪结果与裁剪图
- `outputs/attendance.json`：学生识别与出勤信息
- `outputs/metrics.json`：汇总指标
- `outputs/report.md`：Markdown 报告

## 依赖的 GitHub 项目
- `ultralytics/ultralytics`（YOLOv11/RT-DETR v2 + ByteTrack 跟踪）
- `insightface/insightface`（SCRFD 检测 + AdaFace/MagFace 人脸识别）
- 可扩展对话/ASR：`pyannote/pyannote-audio`、`openai/whisper`
