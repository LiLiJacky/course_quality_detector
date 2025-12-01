# Course Quality Detector (pipeline)

## Quick start

1) 创建 Conda 虚拟环境并安装依赖：
```bash
conda create -y -n cqdetector python=3.10
conda activate cqdetector
pip install -r requirements.txt
```
   - mac/CPU：直接运行，速度较慢。
   - Linux + NVIDIA GPU（推荐）：确保已安装 CUDA 驱动，脚本会自动选择 CUDAExecutionProvider。
   - Windows 11 + NVIDIA GPU：同样自动选择 CUDAExecutionProvider；若无 GPU 则回落 CPU。

2) 编辑配置：`configs/config.yaml`（班级、专业、Excel、视频路径、采样/阈值等）。

3) 一键运行（自动完成名册过滤、人脸库、出勤/报告）：
```bash
# 读取 configs/config.yaml 执行全流程
python scripts/run_pipeline.py --config configs/config.yaml

# 如果仅跑小规模切片数据，添加 --sample_mode（需提前用 scripts/slice_videos.py 生成 sample）
python scripts/run_pipeline.py --config configs/config.yaml --sample_mode

# 多摄像头全量（默认阈值 0.3，frame_stride=10，max_frames=null 全扫）
python scripts/multi_video_attendance.py --config configs/config.yaml \
  --attendance_output outputs/attendance_multi.json \
  --metrics_output outputs/metrics_multi.json \
  --report_output outputs/report.md

# 多摄像头小规模（使用 sample 目录，不影响全量配置）
python scripts/multi_video_attendance.py --config configs/config.yaml --sample_mode \
  --attendance_output outputs/attendance_multi.json \
  --metrics_output outputs/metrics_multi.json \
  --report_output outputs/report.md

# 如需早停（仅当识别到名册/GT 人数且达到最少帧数后才停）
python scripts/multi_video_attendance.py --config configs/config.yaml \
  --early_stop --early_stop_min_frames 200

摄像头分组与权重（配置可调）
- 通过文件名/路径关键字区分：后摄包含 `back_cam_keywords`（默认 ["B","后"]），前摄包含 `front_cam_keywords`（默认 ["A","前"]）。同一摄像头分段视频会自动合并统计。
- 权重默认：后摄 70 分、前摄补分 30 分，出勤阈值 60；仅前摄补漏时需满足更高计数/阈值，权重 60。
- 计数门槛默认：后摄 min_count_back=5，前摄补充 min_count_front=3，前摄补漏 min_count_front_only=8、min_recognized_counts_front_only=12；全局计数过滤 min_recognized_counts=10；相似度阈值 0.32（前摄补漏阈值 0.34）。
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
