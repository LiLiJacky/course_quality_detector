"""
Build a face embedding gallery from a picture directory using InsightFace.

The file names in picture_dir are treated as student ids (stem without suffix).
Embeddings are stored in output_dir as:
- embeddings.npy: float32 array of shape (N, D)
- meta.json: list of {student_id, image, embedding_index}
"""

import argparse
import json
import platform
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _select_providers() -> list:
    """Prefer CUDA on Windows/Linux if available; fallback to CPU."""
    try:
        import torch

        if torch.cuda.is_available() and platform.system().lower() in ("windows", "linux"):
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    except Exception:
        pass
    return ["CPUExecutionProvider"]


def _init_insightface(det_name: str):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit(
            "insightface is required. Install with `pip install insightface`."
        ) from exc

    providers = _select_providers()
    app = FaceAnalysis(
        name=det_name,
        providers=providers,
        allowed_modules=["detection", "recognition"],
    )
    # Current InsightFace auto-loads the bundled recognition model in the pack (e.g., buffalo_l).
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _extract_embedding(app, image_path: Path) -> Tuple[np.ndarray, dict]:
    import cv2

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    faces = app.get(bgr)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")
    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    face = faces[0]
    return face.embedding.astype(np.float32), {
        "bbox": face.bbox.tolist(),
        "det_score": float(face.det_score),
    }


def build_gallery(
    picture_dir: Path,
    output_dir: Path,
    det_name: str,
    rec_name: str,
    max_images: int | None = None,
    allowed_ids: set[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    app = _init_insightface(det_name)

    embeddings: List[np.ndarray] = []
    meta: List[dict] = []

    for idx, image_path in enumerate(sorted(picture_dir.glob("*.jpg"))):
        if max_images is not None and idx >= max_images:
            break
        student_id = image_path.stem
        if allowed_ids is not None and student_id not in allowed_ids:
            continue
        try:
            emb, face_info = _extract_embedding(app, image_path)
        except ValueError as err:
            print(f"[skip] {student_id}: {err}")
            continue
        meta.append(
            {
                "student_id": student_id,
                "image": str(image_path),
                "embedding_index": len(embeddings),
                "bbox": face_info["bbox"],
                "det_score": face_info["det_score"],
            }
        )
        embeddings.append(emb)

    if not embeddings:
        raise SystemExit("No embeddings were generated.")

    emb_array = np.stack(embeddings, axis=0)
    np.save(output_dir / "embeddings.npy", emb_array)
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(meta)} embeddings to {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build face gallery with InsightFace")
    parser.add_argument(
        "--picture_dir",
        type=Path,
        default=Path("data/picture"),
        help="Directory containing student images",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/face_gallery"),
        help="Directory to save embeddings and metadata",
    )
    parser.add_argument(
        "--det_name",
        type=str,
        default="buffalo_l",
        help="InsightFace detection model tag (e.g., buffalo_l or scrfd_2.5g)",
    )
    parser.add_argument(
        "--rec_name",
        type=str,
        default="adaface_ir101",
        help="InsightFace recognition model tag (e.g., adaface_ir101 or magface_r100)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Optional limit on number of images to process (for quick tests)",
    )
    parser.add_argument(
        "--allowed_ids",
        type=Path,
        default=None,
        help="Optional text file of allowed IDs (one per line); if provided, only these IDs are embedded",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    roster = None
    if args.allowed_ids:
        roster = {
            line.strip()
            for line in args.allowed_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        print(f"Loaded {len(roster)} allowed IDs.")
    build_gallery(
        args.picture_dir,
        args.output_dir,
        args.det_name,
        args.rec_name,
        args.max_images,
        roster,
    )
