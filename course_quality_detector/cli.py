"""Command line entrypoint for course-quality-detector."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List, Mapping

from .analysis import evaluate_feedback, format_report, load_feedback


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate course quality from feedback data.")
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to a JSON/JSONL file with feedback entries.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Inline feedback text; can be provided multiple times.",
    )
    parser.add_argument(
        "--rating",
        action="append",
        type=float,
        default=[],
        help="Optional rating (0-5) aligned with each --text argument.",
    )
    parser.add_argument(
        "--completed",
        action="append",
        choices=["true", "false"],
        default=[],
        help="Optional completion flag aligned with each --text argument.",
    )
    return parser.parse_args(list(argv))


def _build_inline_entries(args: argparse.Namespace) -> List[Mapping[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, text in enumerate(args.text):
        rating = args.rating[idx] if idx < len(args.rating) else None
        completed_raw = args.completed[idx] if idx < len(args.completed) else None
        completed: bool | None = None
        if completed_raw is not None:
            completed = completed_raw.lower() == "true"

        entries.append({"text": text, "rating": rating, "completed": completed})
    return entries


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    feedback_entries: List[Mapping[str, Any]] = []
    if args.path:
        try:
            feedback_entries.extend(load_feedback(args.path))
        except Exception as exc:  # pragma: no cover - surfaced to user
            sys.stderr.write(f"Failed to load feedback: {exc}\n")
            return 1

    if args.text:
        feedback_entries.extend(_build_inline_entries(args))

    if not feedback_entries:
        sys.stderr.write("No feedback provided. Pass a file path or --text entries.\n")
        return 1

    summary = evaluate_feedback(feedback_entries)
    report = format_report(summary)
    sys.stdout.write(report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
