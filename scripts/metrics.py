"""
Compute simple classroom metrics from attendance and optional action/dialog files.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(
    attendance_path: Path,
    action_path: Path | None,
    dialog_path: Path | None,
    output_path: Path,
) -> None:
    attendance = load_json(attendance_path)
    attendance_ids = attendance.get("attendance_ids", [])
    recognized_counts = attendance.get("recognized_counts", {})

    metrics: Dict[str, Any] = {
        "attendance": {
            "present": len(attendance_ids),
            "unique_ids": attendance_ids,
            "crop_counts": recognized_counts,
        }
    }

    if action_path and action_path.exists():
        metrics["actions"] = load_json(action_path)
    if dialog_path and dialog_path.exists():
        metrics["dialog"] = load_json(dialog_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute classroom metrics")
    parser.add_argument(
        "--attendance",
        type=Path,
        default=Path("outputs/attendance.json"),
        help="Attendance JSON path from assign_identity.py",
    )
    parser.add_argument(
        "--action",
        type=Path,
        default=Path("outputs/action.json"),
        help="Optional action recognition results",
    )
    parser.add_argument(
        "--dialog",
        type=Path,
        default=Path("outputs/dialog.json"),
        help="Optional dialog stats",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics.json"),
        help="Path to save metrics",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_metrics(args.attendance, args.action, args.dialog, args.output)
