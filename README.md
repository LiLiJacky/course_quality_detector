# Course Quality Detector (pipeline)

## Quick start

1) 创建虚拟环境并安装依赖（需 GPU/CUDA 才能跑得动；本机为 CPU 会较慢）：
```bash
pip install -r requirements.txt
```

2) 编辑配置（班级、专业、Excel、视频路径等）：`configs/config.yaml`

3) 一键运行（自动按配置完成名册过滤、人脸库、出勤/报告）：
```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

输出文件：
- `outputs/face_gallery/`：人脸 embeddings + 元数据
- `outputs/track/exp/`：YOLO 跟踪结果与裁剪图（仅在非 quick 模式）
- `outputs/attendance.json`：学生识别与出勤信息
- `outputs/metrics.json`：汇总指标
- `outputs/report.md`：Markdown 报告

## 依赖的 GitHub 项目
- `ultralytics/ultralytics`（YOLOv11/RT-DETR v2 + ByteTrack 跟踪）
- `insightface/insightface`（SCRFD 检测 + AdaFace/MagFace 人脸识别）
- 可扩展对话/ASR：`pyannote/pyannote-audio`、`openai/whisper`
