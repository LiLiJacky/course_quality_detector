"""
Create a small test video set by slicing the first N seconds of each video
while preserving the directory structure.

Example:
python scripts/slice_videos.py \
  --input_dir data/video/509教室-0750-0915-英语246 \
  --output_dir data/video_sample/509教室-0750-0915-英语246 \
  --duration_sec 5
"""

import argparse
import shutil
from pathlib import Path

import cv2


def slice_video(src: Path, dst: Path, duration_sec: int) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"[warn] failed to open {src}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frames = int(fps * duration_sec)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))

    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        count += 1

    cap.release()
    writer.release()
    print(f"[ok] sliced {src} -> {dst} ({count} frames, fps={fps:.2f})")
    return True


def slice_videos(input_dir: Path, output_dir: Path, duration_sec: int):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    videos = sorted(input_dir.glob("**/*.mp4"))
    if not videos:
        raise SystemExit(f"No mp4 files under {input_dir}")
    for vid in videos:
        rel = vid.relative_to(input_dir)
        dst = output_dir / rel
        slice_video(vid, dst, duration_sec)


def parse_args():
    parser = argparse.ArgumentParser(description="Slice videos for small-scale testing")
    parser.add_argument("--input_dir", type=Path, required=True, help="Root dir of original videos")
    parser.add_argument("--output_dir", type=Path, required=True, help="Root dir for sliced videos")
    parser.add_argument(
        "--duration_sec",
        type=int,
        default=5,
        help="Duration in seconds to keep from the start of each video",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    slice_videos(args.input_dir, args.output_dir, args.duration_sec)
