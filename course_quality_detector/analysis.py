"""Analysis utilities for course quality feedback."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping

from .heuristics import ScoreResult, compute_score


def load_feedback(path: str | Path) -> List[Mapping[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Feedback file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fh:
        text = fh.read().strip()
        if not text:
            return []

        if file_path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]

        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError("Feedback JSON must be a list of entries or JSONL lines")


def evaluate_feedback(entries: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    feedback_list = list(entries)
    if not feedback_list:
        return {
            "samples": 0,
            "overall_score": 0.0,
            "average_rating": None,
            "completion_rate": None,
            "by_entry": [],
        }

    scored: List[Dict[str, Any]] = []
    scores: List[float] = []
    ratings: List[float] = []
    completion_flags: List[bool] = []

    for entry in feedback_list:
        result: ScoreResult = compute_score(entry)
        scores.append(result.score)

        rating_value = entry.get("rating")
        if isinstance(rating_value, (int, float)) and 0 <= rating_value <= 5:
            ratings.append(float(rating_value))

        completed_value = entry.get("completed")
        if isinstance(completed_value, bool):
            completion_flags.append(completed_value)

        scored.append(
            {
                "score": result.score,
                "signals": result.signals,
                "text": entry.get("text", ""),
                "rating": rating_value,
                "completed": completed_value,
            }
        )

    summary = {
        "samples": len(feedback_list),
        "overall_score": round(mean(scores), 2),
        "average_rating": round(mean(ratings), 2) if ratings else None,
        "completion_rate": round(sum(completion_flags) / len(completion_flags), 2)
        if completion_flags
        else None,
        "by_entry": scored,
    }
    return summary


def format_report(summary: Mapping[str, Any]) -> str:
    lines = []
    lines.append("Course Quality Report")
    lines.append("----------------------")
    lines.append(f"Samples: {summary.get('samples', 0)}")
    lines.append(f"Overall score: {summary.get('overall_score', 0):.2f} / 100")

    avg_rating = summary.get("average_rating")
    lines.append(f"Average rating: {avg_rating:.2f}" if avg_rating is not None else "Average rating: n/a")

    completion_rate = summary.get("completion_rate")
    if completion_rate is not None:
        lines.append(f"Completion rate: {completion_rate * 100:.1f}%")
    else:
        lines.append("Completion rate: n/a")

    lines.append("")
    lines.append("Entry breakdown:")
    for idx, item in enumerate(summary.get("by_entry", []), start=1):
        lines.append(f"  #{idx}: score={item['score']:.2f} rating={item.get('rating')} completed={item.get('completed')}")
        lines.append(f"      text: {str(item.get('text', ''))[:200]}")

    return "\n".join(lines)
