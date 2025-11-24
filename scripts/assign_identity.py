"""
Assign identities to tracked face crops using a pre-built gallery.

Expected inputs:
- crops_dir: directory with cropped face images (e.g., runs/track/exp/crops/person/)
- gallery_dir: produced by face_gallery.py (embeddings.npy + meta.json)
Outputs:
- attendance.json with recognized ids and per-crop matches
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_gallery(gallery_dir: Path) -> Tuple[np.ndarray, List[dict]]:
    emb_path = gallery_dir / "embeddings.npy"
    meta_path = gallery_dir / "meta.json"
    if not emb_path.exists() or not meta_path.exists():
        raise SystemExit(f"Gallery files missing in {gallery_dir}")
    embeddings = np.load(emb_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if embeddings.shape[0] != len(meta):
        raise SystemExit("Embedding count does not match meta length.")
    # normalize once for cosine similarity
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    embeddings = embeddings / norm
    return embeddings.astype(np.float32), meta


def _init_insightface(det_name: str):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as exc:
        raise SystemExit(
            "insightface is required. Install with `pip install insightface`."
        ) from exc
    app = FaceAnalysis(
        name=det_name,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _embed_crop(app, image_path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    faces = app.get(bgr)
    if not faces:
        raise ValueError(f"No face detected in {image_path}")
    faces.sort(key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    return faces[0].embedding.astype(np.float32)


def match_crops(
    crops_dir: Path,
    gallery_dir: Path,
    det_name: str,
    rec_name: str,
    threshold: float,
    output_path: Path,
    max_crops: int | None = None,
) -> None:
    embeddings, meta = load_gallery(gallery_dir)
    gallery_ids = [item["student_id"] for item in meta]

    app = _init_insightface(det_name)

    matches: List[dict] = []
    recognized: Dict[str, int] = {}

    crop_paths = sorted([p for p in crops_dir.rglob("*.jpg") if p.is_file()])
    if not crop_paths:
        raise SystemExit(f"No crop images found in {crops_dir}")

    gallery_norm = embeddings
    for idx, crop_path in enumerate(crop_paths):
        if max_crops is not None and idx >= max_crops:
            break
        try:
            emb = _embed_crop(app, crop_path)
        except ValueError as err:
            print(f"[skip] {crop_path.name}: {err}")
            continue
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        sims = np.dot(gallery_norm, emb_norm)
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_id = gallery_ids[best_idx] if best_sim >= threshold else None
        if best_id:
            recognized[best_id] = recognized.get(best_id, 0) + 1
        matches.append(
            {
                "crop": str(crop_path),
                "best_id": best_id,
                "similarity": best_sim,
            }
        )

    attendance = sorted(recognized.keys())
    result = {
        "attendance_ids": attendance,
        "recognized_counts": recognized,
        "matches": matches,
        "threshold": threshold,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(
        f"Saved attendance for {len(attendance)} students "
        f"from {len(matches)} crops to {output_path}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Assign IDs to face crops")
    parser.add_argument(
        "--crops_dir",
        type=Path,
        default=Path("runs/track/exp/crops"),
        help="Directory containing cropped faces from tracking",
    )
    parser.add_argument(
        "--gallery_dir",
        type=Path,
        default=Path("outputs/face_gallery"),
        help="Directory containing embeddings.npy and meta.json",
    )
    parser.add_argument(
        "--det_name",
        type=str,
        default="buffalo_l",
        help="InsightFace detection model tag",
    )
    parser.add_argument(
        "--rec_name",
        type=str,
        default="adaface_ir101",
        help="InsightFace recognition model tag",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.35,
        help="Cosine similarity threshold to accept a match",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/attendance.json"),
        help="Path to save attendance JSON",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=None,
        help="Optional limit on number of crops to process",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    match_crops(
        args.crops_dir,
        args.gallery_dir,
        args.det_name,
        args.rec_name,
        args.threshold,
        args.output,
        args.max_crops,
    )
