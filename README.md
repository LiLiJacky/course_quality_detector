# course-quality-detector

A minimal Python CLI scaffold for experimenting with course quality detection. The tool demonstrates a simple scoring pipeline over course feedback data and provides spots to extend with richer models.

## Features
- CLI that ingests feedback JSON/JSONL or a list of text snippets.
- Baseline heuristic scoring (length, sentiment proxy via keywords, engagement hints).
- Summary report printed to stdout.
- Simple tests to validate the pipeline shape.

## Quickstart
1. Ensure Python 3.10+ is available.
2. Install the package in editable mode with dev extras:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

3. Run the CLI on a JSONL file (one feedback entry per line):

   ```bash
   course-quality-detector examples/feedback.jsonl
   ```

4. Run tests:

   ```bash
   pytest
   ```

## Data format
- JSONL: each line is an object with at least a `text` field, optional `rating` (0-5) and `completed` (bool).
- JSON array is also supported with the same object shape.

## Extending
- Replace `heuristics.py` with a model-backed scorer.
- Add persistence or API integration around `evaluate_feedback`.
- Wire a vectorizer/embedding pipeline for richer signals.

## License
MIT
