"""Course quality detector package."""

from .analysis import evaluate_feedback
from .cli import main

__all__ = ["evaluate_feedback", "main"]
