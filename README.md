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
