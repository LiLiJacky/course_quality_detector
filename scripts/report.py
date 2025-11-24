"""
Generate a simple markdown report from metrics.json.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_report(metrics: dict) -> str:
    attendance = metrics.get("attendance", {})
    present = attendance.get("present", 0)
    ids = attendance.get("unique_ids", [])

    lines = [
        "# 课堂质量报告（简版）",
        "",
        "## 出勤与参与",
        f"- 到课人数：{present}",
        f"- 学生ID列表：{', '.join(ids)}" if ids else "- 学生ID列表：无",
        "- 参与度：未计算（需行为/姿态模块补充）",
    ]

    if "actions" in metrics:
        lines.extend(["", "## 行为统计", f"- 数据：{metrics['actions']}"])
    if "dialog" in metrics:
        lines.extend(["", "## 对话统计", f"- 数据：{metrics['dialog']}"])

    lines.extend(
        [
            "",
            "## 教师教学指标",
            "- 暂未计算（需教师行为识别与课堂分段）",
            "",
            "## 课堂互动指标",
            "- 暂未计算（需对话与提问分析）",
        ]
    )

    lines.append("")
    lines.append("## 备注")
    lines.append("- 本报告基于本地推理结果自动生成，可按需补充。")
    return "\n".join(lines)


def write_report(metrics_path: Path, output_path: Path) -> None:
    metrics = load_json(metrics_path)
    report = render_report(metrics)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Saved report to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render markdown report from metrics")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/metrics.json"),
        help="Metrics JSON path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/report.md"),
        help="Path to save markdown report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_report(args.metrics, args.output)
