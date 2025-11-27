"""
Multi-camera attendance pipeline with early exit and ground-truth evaluation.

Workflow:
1) Build roster (allowed IDs) from Excel if provided and roster file missing.
2) Build face gallery using allowed IDs.
3) Iterate videos under a directory (multiple cameras), run quick_attendance per video,
   accumulate recognized counts, and early-stop when target_count reached.
4) Save aggregated attendance.json and metrics.json (precision/recall/F1 vs ground truth if given).

Config:
- Can be driven by YAML (reuse configs/config.yaml keys).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
import yaml

from build_roster import build_roster
from face_gallery import build_gallery
from quick_attendance import quick_attendance


def load_yaml(path: Path) -> dict:
    if path and path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def load_ids(path: Path) -> Set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def ensure_roster(excel: Path | None, class_name: str | None, major_name: str | None, roster_path: Path) -> Path:
    if roster_path.exists():
        return roster_path
    if not excel or not class_name or not major_name:
        raise SystemExit("Roster file missing and Excel/class/major not provided.")
    roster_path.parent.mkdir(parents=True, exist_ok=True)
    build_roster(excel, class_name, major_name, roster_path)
    return roster_path


def load_ground_truth(path: Path | None) -> Set[str]:
    if not path or not path.exists():
        return set()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
        col = df.columns[0]
        return set(df[col].astype(str).tolist())
    return load_ids(path)


def evaluate(pred_ids: Set[str], gt_ids: Set[str]) -> Dict[str, float]:
    if not gt_ids:
        return {}
    tp = len(pred_ids & gt_ids)
    prec = tp / len(pred_ids) if pred_ids else 0.0
    rec = tp / len(gt_ids) if gt_ids else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": tp,
        "gt": len(gt_ids),
        "pred": len(pred_ids),
        "false_pos": sorted(list(pred_ids - gt_ids)),
        "miss": sorted(list(gt_ids - pred_ids)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-camera attendance with early exit")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="YAML config")
    parser.add_argument("--video_dir", type=Path, help="Directory containing camera subfolders/videos")
    parser.add_argument("--video_glob", type=str, default="**/*.mp4", help="Glob pattern under video_dir")
    parser.add_argument("--sample_mode", action="store_true", help="Use sample_video_dir instead of full video_dir")
    parser.add_argument("--sample_video_dir", type=Path, help="Directory containing sample camera videos")
    parser.add_argument("--allowed_ids", type=Path, help="Roster text file (one ID per line)")
    parser.add_argument("--excel", type=Path, help="Excel roster path")
    parser.add_argument("--class_name", type=str, help="Class name filter")
    parser.add_argument("--major_name", type=str, help="Major name filter")
    parser.add_argument("--roster_output", type=Path, default=Path("rosters/roster.txt"), help="Where to write roster if built from Excel")
    parser.add_argument("--picture_dir", type=Path, default=Path("data/picture"))
    parser.add_argument("--gallery_dir", type=Path, default=Path("outputs/face_gallery"))
    parser.add_argument("--det_name", type=str, default="buffalo_l")
    parser.add_argument("--match_threshold", type=float, default=0.28)
    parser.add_argument("--frame_stride", type=int, default=20)
    parser.add_argument("--max_frames", type=int, default=None, help="If None, process full video")
    parser.add_argument("--min_count", type=int, default=3)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--target_count", type=int, default=None, help="Early stop count; defaults to roster size or ground truth size")
    parser.add_argument("--early_stop", action="store_true", help="Enable early stop when roster/GT size is reached")
    parser.add_argument("--early_stop_min_frames", type=int, default=200, help="Minimum sampled frames before early stop can trigger")
    parser.add_argument("--ground_truth", type=Path, help="Ground truth attendance file (xlsx or txt)")
    parser.add_argument("--attendance_output", type=Path, default=Path("outputs/attendance.json"))
    parser.add_argument("--metrics_output", type=Path, default=Path("outputs/metrics.json"))
    parser.add_argument("--report_output", type=Path, default=Path("outputs/report.md"), help="Optional report output (uses metrics.py + report.py)")
    return parser.parse_args()


def apply_config(args):
    cfg = load_yaml(args.config)
    path_keys = {
        "video_dir",
        "sample_video_dir",
        "allowed_ids",
        "excel",
        "roster_output",
        "picture_dir",
        "gallery_dir",
        "ground_truth",
        "attendance_output",
        "metrics_output",
        "report_output",
    }
    for k, v in cfg.items():
        if hasattr(args, k):
            if k in path_keys and v is not None:
                setattr(args, k, Path(v))
            else:
                setattr(args, k, v)
    return args


def main():
    args = apply_config(parse_args())

    if args.sample_mode and args.sample_video_dir:
        args.video_dir = args.sample_video_dir

    roster_path = ensure_roster(args.excel, args.class_name, args.major_name, args.roster_output) if args.allowed_ids is None else args.allowed_ids
    allowed_ids = load_ids(roster_path)
    print(f"Loaded roster IDs: {len(allowed_ids)} from {roster_path}")

    # Build gallery limited to allowed IDs
    build_gallery(args.picture_dir, args.gallery_dir, args.det_name, "unused_rec", args.max_images, allowed_ids)

    videos = sorted(Path(args.video_dir).glob(args.video_glob)) if args.video_dir else []
    if not videos:
        raise SystemExit(f"No videos found under {args.video_dir} with pattern {args.video_glob}")

    agg_counts: Dict[str, int] = {}
    per_video: List[dict] = []
    total_frames_processed = 0
    gt_ids = load_ground_truth(args.ground_truth)
    target_count = args.target_count or (len(gt_ids) if gt_ids else len(allowed_ids))
    for vid in videos:
        tmp_out = args.attendance_output.parent / f"tmp_{vid.stem}.json"
        tmp_out.parent.mkdir(parents=True, exist_ok=True)
        result = quick_attendance(
            vid,
            args.gallery_dir,
            args.det_name,
            args.match_threshold,
            args.frame_stride,
            args.max_frames,
            args.min_count,
            allowed_ids,
            tmp_out,
        )
        for sid, cnt in result["recognized_counts"].items():
            agg_counts[sid] = agg_counts.get(sid, 0) + cnt
        per_video.append({"video": str(vid), "detected": len(result["attendance_ids"]), "faces": result.get("faces_detected", 0)})
        total_frames_processed += result.get("frames_processed", 0)

        current_ids = {sid for sid, cnt in agg_counts.items() if cnt >= args.min_count}
        print(f"[progress] attendance so far: {len(current_ids)}/{target_count}, frames={total_frames_processed}")
        if args.early_stop and len(current_ids) >= target_count and total_frames_processed >= args.early_stop_min_frames:
            print("Early stop: all roster/GT IDs recognized with threshold applied.")
            break

    attendance_ids = sorted([sid for sid, cnt in agg_counts.items() if cnt >= args.min_count])
    output = {
        "attendance_ids": attendance_ids,
        "recognized_counts": agg_counts,
        "per_video": per_video,
        "threshold": args.match_threshold,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "min_count": args.min_count,
        "videos_processed": len(per_video),
    }
    args.attendance_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.attendance_output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Saved aggregated attendance to {args.attendance_output}")

    metrics = {"attendance": {"present": len(attendance_ids), "unique_ids": attendance_ids}}
    if gt_ids:
        metrics["evaluation"] = evaluate(set(attendance_ids), gt_ids)
        metrics["ground_truth"] = sorted(list(gt_ids))
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {args.metrics_output}")

    # Optional: generate report.md using existing scripts
    if args.report_output:
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "report.py"),
                    "--metrics",
                    str(args.metrics_output),
                    "--output",
                    str(args.report_output),
                ],
                check=True,
            )
            print(f"Saved report to {args.report_output}")
        except subprocess.CalledProcessError as exc:
            print(f"[warn] report generation failed: {exc}")


if __name__ == "__main__":
    main()
