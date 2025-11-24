# Course Quality Detector (pipeline)

## Quick start

1) 创建虚拟环境并安装依赖（需 GPU/CUDA 才能跑得动；本机为 CPU 会较慢）：
```bash
pip install -r requirements.txt
```

2) 从 Excel 名单过滤班级/专业，生成允许的学号列表（示例：英语246，商务英语）：
```bash
python scripts/build_roster.py \
  --excel "data/picture/2024级学生名单.xlsx" \
  --class_name "英语246" \
  --major_name "商务英语" \
  --output rosters/english246.txt
```

3) 生成人脸库，仅嵌入名册中的学生（输出到 `outputs/face_gallery`）：
```bash
python scripts/face_gallery.py --allowed_ids rosters/english246.txt
```

4) 快速出勤（帧采样人脸识别，推荐 CPU 环境使用；下例采样 240 帧，步长 20，阈值 0.28，最少出现 3 次才计入出勤）：
```bash
python scripts/run_pipeline.py \
  --video "data/video/南通职大主楼509室047zk_20251029072500_20251029091707_2.mp4" \
  --allowed_ids rosters/english246.txt \
  --quick_attendance_only \
  --frame_stride 20 \
  --max_frames 240 \
  --match_threshold 0.28 \
  --min_count 3
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
