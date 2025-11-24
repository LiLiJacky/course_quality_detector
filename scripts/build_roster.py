"""
Extract student roster IDs from the provided Excel list.

Usage example:
python scripts/build_roster.py \
  --excel data/picture/2024级学生名单.xlsx \
  --class_name 英语246 \
  --major_name 商务英语 \
  --output rosters/english246.txt
"""

import argparse
from pathlib import Path

import pandas as pd


def build_roster(excel: Path, class_name: str, major_name: str, output: Path):
    if not excel.exists():
        raise SystemExit(f"Excel file not found: {excel}")

    df = pd.read_excel(excel)
    if "班级" not in df.columns or "专业名称" not in df.columns or "学号" not in df.columns:
        raise SystemExit("Excel must contain columns: 班级, 专业名称, 学号")

    sub = df[(df["班级"] == class_name) & (df["专业名称"] == major_name)]
    if sub.empty:
        raise SystemExit(f"No students found for 班级={class_name}, 专业={major_name}")

    output.parent.mkdir(parents=True, exist_ok=True)
    ids = sub["学号"].astype(str).tolist()
    output.write_text("\n".join(ids), encoding="utf-8")
    print(f"Saved {len(ids)} IDs to {output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build roster from Excel file")
    parser.add_argument("--excel", type=Path, required=True, help="Path to Excel roster")
    parser.add_argument("--class_name", type=str, required=True, help="班级名称，如 英语246")
    parser.add_argument("--major_name", type=str, required=True, help="专业名称，如 商务英语")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rosters/roster.txt"),
        help="Output text file of student IDs (one per line)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_roster(args.excel, args.class_name, args.major_name, args.output)
