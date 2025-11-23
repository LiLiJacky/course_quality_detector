"""Lightweight heuristics for course quality estimation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence

POSITIVE_KEYWORDS: Sequence[str] = (
    "engaging",
    "clear",
    "structured",
    "practical",
    "hands-on",
    "interactive",
    "helpful",
)

NEGATIVE_KEYWORDS: Sequence[str] = (
    "boring",
    "confusing",
    "unclear",
    "too fast",
    "too slow",
    "busywork",
    "disorganized",
)


@dataclass
class ScoreResult:
    score: float
    signals: Dict[str, float]

    def clamp(self) -> "ScoreResult":
        clamped = max(0.0, min(self.score, 100.0))
        return ScoreResult(score=clamped, signals=self.signals)


def keyword_signal(text: str, keywords: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for kw in keywords if kw in lowered)


def compute_score(entry: Mapping[str, object]) -> ScoreResult:
    text = str(entry.get("text", "") or "")
    rating = entry.get("rating")
    completed = entry.get("completed")

    length_score = min(len(text) / 200 * 25, 25)  # reward richer feedback up to a cap

    rating_score = 20.0
    if isinstance(rating, (int, float)) and 0 <= rating <= 5:
        rating_score = (float(rating) / 5.0) * 40  # up to 40 points from explicit rating

    positive_hits = keyword_signal(text, POSITIVE_KEYWORDS)
    negative_hits = keyword_signal(text, NEGATIVE_KEYWORDS)
    sentiment_score = (positive_hits * 6) - (negative_hits * 8)

    engagement_score = 0.0
    if "project" in text.lower() or "assignment" in text.lower():
        engagement_score += 5
    if "discussion" in text.lower():
        engagement_score += 5

    completion_score = 0.0
    if isinstance(completed, bool):
        completion_score = 8 if completed else -4

    raw_score = length_score + rating_score + sentiment_score + engagement_score + completion_score

    signals = {
        "length_score": round(length_score, 2),
        "rating_score": round(rating_score, 2),
        "sentiment_score": round(sentiment_score, 2),
        "engagement_score": round(engagement_score, 2),
        "completion_score": round(completion_score, 2),
    }

    return ScoreResult(score=round(raw_score, 2), signals=signals).clamp()
