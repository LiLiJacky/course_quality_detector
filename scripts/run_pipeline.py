"""
End-to-end pipeline orchestrator following plan.md.

Steps:
1) Build face gallery from ./data/picture
2) Run detection + tracking on ./data/video using ultralytics YOLO (ByteTrack)
3) Assign identities to crops using InsightFace gallery
4) Compute metrics and generate a markdown report
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run_cmd(cmd: List[str], workdir: Path | None = None):
    print(f"[cmd] {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=workdir, check=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Command not found: {cmd[0]}. Please install required package."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed with code {exc.returncode}: {cmd}") from exc


def step_face_gallery(args):
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/face_gallery.py"),
            "--picture_dir",
            str(args.picture_dir),
            "--output_dir",
            str(args.gallery_dir),
            "--det_name",
            args.det_name,
            "--rec_name",
            args.rec_name,
            *(
                ["--max_images", str(args.max_images)]
                if args.max_images is not None
                else []
            ),
        ]
    )


def step_track(args):
    """
    Use ultralytics YOLOv11 tracking with ByteTrack. Requires `pip install ultralytics`.
    """
    out_dir = args.track_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    yolo_args = [
        "yolo",
        "task=track",
        "mode=predict",
        f"model={args.det_model}",
        f"source={args.video}",
        "tracker=bytetrack.yaml",
        "save=True",
        "save_txt=True",
        "save_crop=True",
        "classes=0",  # only person class to reduce noise crops
        f"project={out_dir}",
        "name=exp",
    ]
    run_cmd(yolo_args)


def step_assign(args):
    crops_dir = args.track_dir / "exp" / "crops"
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/assign_identity.py"),
            "--crops_dir",
            str(crops_dir),
            "--gallery_dir",
            str(args.gallery_dir),
            "--det_name",
            args.det_name,
            "--rec_name",
            args.rec_name,
            "--threshold",
            str(args.match_threshold),
            "--output",
            str(args.attendance_path),
            *(
                ["--max_crops", str(args.max_crops)]
                if args.max_crops is not None
                else []
            ),
        ]
    )


def step_quick_attendance(args):
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/quick_attendance.py"),
            "--video",
            str(args.video),
            "--gallery_dir",
            str(args.gallery_dir),
            "--det_name",
            args.det_name,
            "--threshold",
            str(args.match_threshold),
            "--frame_stride",
            str(args.frame_stride),
            "--max_frames",
            str(args.max_frames) if args.max_frames is not None else "0",
            "--min_count",
            str(args.min_count),
            *(
                ["--allowed_ids", str(args.allowed_ids)]
                if args.allowed_ids is not None
                else []
            ),
            "--output",
            str(args.attendance_path),
        ]
    )


def step_metrics(args):
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/metrics.py"),
            "--attendance",
            str(args.attendance_path),
            "--output",
            str(args.metrics_path),
        ]
    )
    run_cmd(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts/report.py"),
            "--metrics",
            str(args.metrics_path),
            "--output",
            str(args.report_path),
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run end-to-end classroom pipeline")
    parser.add_argument(
        "--picture_dir",
        type=Path,
        default=Path("data/picture"),
        help="Directory with student images",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/video",
        help="Path to video file or directory",
    )
    parser.add_argument(
        "--gallery_dir",
        type=Path,
        default=Path("outputs/face_gallery"),
        help="Directory to save gallery embeddings",
    )
    parser.add_argument(
        "--track_dir",
        type=Path,
        default=Path("outputs/track"),
        help="Directory to store tracking outputs",
    )
    parser.add_argument(
        "--det_model",
        type=str,
        default="yolo11x.pt",
        help="Ultralytics model path or name",
    )
    parser.add_argument(
        "--det_name",
        type=str,
        default="buffalo_l",
        help="InsightFace detector name",
    )
    parser.add_argument(
        "--rec_name",
        type=str,
        default="adaface_ir101",
        help="InsightFace recognizer name",
    )
    parser.add_argument(
        "--match_threshold",
        type=float,
        default=0.3,
        help="Cosine similarity threshold for identity assignment",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=10,
        help="Sample every N frames for quick attendance",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=200,
        help="Max sampled frames for quick attendance (0 means no limit)",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum matched occurrences to accept an ID (quick attendance)",
    )
    parser.add_argument(
        "--allowed_ids",
        type=Path,
        default=None,
        help="Optional text file with allowed IDs (one per line); others ignored",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Limit images when building gallery (for quick tests)",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=None,
        help="Limit number of crops during ID assignment (for quick tests)",
    )
    parser.add_argument(
        "--attendance_path",
        type=Path,
        default=Path("outputs/attendance.json"),
        help="Path for attendance JSON",
    )
    parser.add_argument(
        "--metrics_path",
        type=Path,
        default=Path("outputs/metrics.json"),
        help="Path for metrics JSON",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=Path("outputs/report.md"),
        help="Path for markdown report",
    )
    parser.add_argument(
        "--quick_attendance_only",
        action="store_true",
        help="Skip detection/tracking and use frame-sampling attendance only",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    step_face_gallery(args)
    if args.quick_attendance_only:
        step_quick_attendance(args)
    else:
        step_track(args)
        step_assign(args)
    step_metrics(args)
    print("Pipeline completed. Check outputs directory.")


if __name__ == "__main__":
    main()
