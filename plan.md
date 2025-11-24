# 课堂质量分析系统技术规格文档（Markdown版）

## 1. 系统总体架构

**目标：**

- **输入**：课堂视频（教师与学生）
- **中间结果**：
  - 视觉检测与跟踪（教师/学生）
  - 学生身份识别（基于学号–照片库）
  - 多模态特征提取：视觉行为、参与度、语音、文本
  - 自动计算课堂质量指标
- **输出**：
  1. 结构化课堂质量指标（JSON）
  2. 基于大模型的课堂质量自然语言报告

**模块流程：**

1. 数据输入与同步
2. 视频检测与跟踪
3. 学生身份识别（使用人脸库）
4. 行为与参与度识别
5. 语音与对话分析
6. 质量指标计算
7. 质量报告生成（大模型）

---

## 2. 输入与数据规范

### 2.1 视频输入规范

- 分辨率 ≥ 1280×720
- 25–30 FPS
- 摄像头固定机位，覆盖多数学生面部
- 录制整节课

### 2.2 学生照片库

每位学生需提供：

- `student_id`（必填）
- 正脸照片 3–5 张，分辨率 ≥ 224×224
- 作为人脸特征库（gallery），用于课堂视频匹配

---

## 3. 模块详细设计

## 3.1 数据输入与同步

- 使用 FFmpeg 解封装视频
- 分离音频流、视频流
- 保持统一时间轴，用于多模态对齐

---

## 3.2 视频检测与跟踪模块

### 3.2.1 人体/人脸检测

- 使用 GitHub 最新实现：
  - `ultralytics/ultralytics` 中的 YOLOv11（人/人脸/姿态全套，支持 RT-DETR v2）
  - `insightface/insightface` 中的 SCRFD 2.5G（高精度人脸检测，适合课堂场景）
- 输出：学生/教师 bounding boxes

### 3.2.2 多目标跟踪（MOT）

- `ifzhang/ByteTrack` v0.5（兼容 YOLOv11 输出）
- 可选：`NirAharon/BoT-SORT`（遮挡场景更稳）
- 输出：
  - 每个个体的 `track_id`
  - 在整堂课的轨迹轨道
  - 用于后续身份识别绑定

---

## 3.3 学生身份识别模块

### 3.3.1 人脸特征库构建

- 使用 `insightface/recognition`（MagFace/AdaFace）提取 512 维特征向量
- 存储在向量数据库（Faiss）

### 3.3.2 视频中的身份识别

流程：

1. 从学生轨迹中检测人脸
2. 提取特征向量
3. 与特征库相似度比对（cosine similarity）
4. 若超过阈值（如 0.6），绑定为该 `student_id`

**输出：**

- 到课学生列表
- 每个学生入场/离场时间
- 出勤率、迟到/早退

---

## 3.4 行为与参与度识别模块

### 3.4.1 学生行为分类

- 使用 `ultralytics/ultralytics` 的 YOLOv11-Pose 或 `open-mmlab/mmpose` 的 RTMPose
- 结合 `facebookresearch/pytorchvideo` 中的 SlowFast/VideoMAE 做短片段动作判别
- 行为类别：
  - 专注
  - 低头写作业
  - 举手
  - 东张西望
  - 使用手机等

### 3.4.2 参与度估计

参与度基于：

- 行为分布
- 表情分析（困惑/无聊/专注）
- 头部方向（是否注视教师）

输出：

- 每位学生 0–100 的参与度评分
- 班级参与度时间曲线

---

## 3.5 教师行为识别模块

### 3.5.1 教师行为分类

识别以下教师行为：

- 讲授
- 提问
- 巡视
- 板书
- 操作多媒体
- 方法：复用 YOLOv11/RT-DETR v2 检测教师轨迹，结合 `facebookresearch/pytorchvideo` 的 SlowFast 或 `OpenGVLab/InternVideo2` 做长时序动作识别

### 3.5.2 课堂活动分段

基于教师行为 + 学生行为，自动划分课堂阶段：

- 讲授
- 讨论
- 小组活动
- 总结

---

## 3.6 语音与对话分析模块

### 3.6.1 语音分离与说话人分类

- `pyannote/pyannote-audio` 3.x：VAD + 说话人分离（teacher vs student）

### 3.6.2 语音转文本（ASR）

- `openai/whisper` large-v3 或 `alibaba-damo-academy/FunASR` 的 Paraformer-v2
- 输出逐句文本 + 时间戳

### 3.6.3 对话质量分析

识别：

- 提问次数
- 高阶 vs 低阶问题（基于 Bloom 分类）
- 学生发言次数
- 平均对话轮次
- 高质量讲解片段

---

## 4. 课堂质量指标设计

### 4.1 出勤与参与

| 指标 | 含义 |
|------|------|
| 出勤率 | 到课人数 / 总人数 |
| 迟到率 | 首次出现时间判定 |
| 平均参与度 | 所有学生参与度均值 |
| 低参与学生比例 | 参与度低于阈值者所占比例 |

---

### 4.2 教师教学指标

- 教学活动分布（讲授/讨论/巡视）
- 高阶提问比例
- 教师走动热力图

---

### 4.3 课堂互动指标

- 教师提问次数
- 学生发言次数
- 对话平均轮数
- 学生间互动次数

---

### 4.4 综合评分（可自定义权重）

\[
Q = 0.3A + 0.3E + 0.2I + 0.2T
\]

- A：到课与参与  
- E：课堂情绪与氛围  
- I：互动质量  
- T：教师教学行为多样性  

---

## 5. 报告生成（大模型）

### 5.1 输入给 LLM 的 JSON 示例

```json
{
  "attendance": { "present": 55, "total": 60 },
  "engagement": { "avg": 72.5 },
  "interaction": {
    "teacher_questions": 25,
    "student_responses": 40,
    "high_level_ratio": 0.36
  },
  "teacher_behavior": {
    "lecture_ratio": 0.55,
    "discussion_ratio": 0.30
  }
}
```

### 5.2 提示词（Prompt）

> “请根据以下课堂结构化指标，生成一份正式的课堂质量报告，包含：整体评价、亮点、不足、改进建议。”

### 5.3 推荐 LLM

- 开源可私有化：`Qwen2.5-32B-Instruct`（多语言，支持长上下文）、`THUDM/glm-4-9b-chat`（轻量推理）、`deepseek-ai/DeepSeek-V2-Lite`
- 云端（如可用）：GPT-4o / GPT-4.1，用于更流畅的生成与自检

---

## 6. 隐私与合规

- 学生照片仅存储在校内服务器
- 报告中不展示学生人脸或姓名，可匿名
- 支持数据删除与学生/家长知情权

---

## 7. 实施路线（里程碑）

### 阶段 1（MVP）  
- 完成学生身份识别  
- 基础行为统计（举手/发言）  

### 阶段 2  
- 引入参与度模型  
- 引入 ASR 和对话分析  
- 生成基础质量报告  

### 阶段 3  
- 多模态联合模型  
- 高级课堂质量指标  
- 学校部署 + 专家校准评分

---

## 8. GitHub 执行步骤（输入位于 `./data/picture` 和 `./data/video`）

1. **环境准备**  
   - Python 3.10+/CUDA 11.8；拉取代码：`ultralytics/ultralytics`、`ifzhang/ByteTrack`、`insightface/insightface`、`open-mmlab/mmpose`、`pyannote/pyannote-audio`、`openai/whisper`。  
   - 为每个仓库创建虚拟环境或统一 `conda env`，安装 `requirements.txt` 与 `pip install -e .`（ultralytics、mmpose、bytetrack 需要）。

2. **人脸库构建（./data/picture）**  
   - 使用 insightface 提取特征：`python tools/face_feature.py --input ./data/picture --output ./outputs/face_gallery --model adaface_ir101`。  
   - 将 embeddings 索引进 Faiss：`python tools/build_faiss.py --feat_dir ./outputs/face_gallery --index ./outputs/faiss.index`。

3. **检测 + 跟踪（./data/video）**  
   - YOLOv11/RT-DETR v2 预测：`yolo task=detect model=yolo11x.pt source=./data/video save_txt=True save_crop=True`。  
   - ByteTrack 关联：`python tools/track.py --det ./runs/detect/predict/ --output ./outputs/tracks.json`。

4. **身份绑定**  
   - 对跟踪到的人脸 crop 调用 insightface 特征；使用 Faiss 最近邻匹配阈值 0.6，生成出勤表：`python tools/id_assign.py --tracks ./outputs/tracks.json --faiss ./outputs/faiss.index --save ./outputs/attendance.json`。

5. **姿态/行为识别**  
   - RTMPose/YOLOv11-Pose：`python tools/pose_infer.py --tracks ./outputs/tracks.json --video ./data/video --save ./outputs/pose.pkl`。  
   - SlowFast/VideoMAE 对 2–4s 片段分类：`python tools/action_clip.py --video ./data/video --tracks ./outputs/tracks.json --save ./outputs/action.json`。

6. **语音与文本**  
   - pyannote 说话人分离：`python tools/diarize.py --audio ./data/video --save ./outputs/diarization.rttm`。  
   - Whisper large-v3 转写：`python tools/asr.py --audio ./data/video --save ./outputs/transcript.json`。  
   - 结合 RTTM + ASR 对齐，得到教师/学生发言统计：`python tools/dialog_stats.py --rttm ./outputs/diarization.rttm --asr ./outputs/transcript.json --save ./outputs/dialog.json`。

7. **指标计算与可视化**  
   - 汇总出勤、行为、参与度、对话：`python tools/metrics.py --attendance ./outputs/attendance.json --pose ./outputs/pose.pkl --action ./outputs/action.json --dialog ./outputs/dialog.json --save ./outputs/metrics.json`。  
   - 生成可视化（热力图、时间轴、分布图）：`python tools/plots.py --metrics ./outputs/metrics.json --out_dir ./outputs/plots`。

8. **报告生成**  
   - 将 `metrics.json` 作为结构化输入，调用选定 LLM（如 Qwen2.5-32B-Instruct 或 GPT-4o）生成报告：`python tools/report_llm.py --metrics ./outputs/metrics.json --model qwen2.5-32b-instruct --save ./outputs/report.md`。

---

## 9. 文档结束

如需进一步生成“项目计划书 / PPT / 架构图”，可继续告诉我。
