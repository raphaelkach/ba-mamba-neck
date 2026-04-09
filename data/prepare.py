"""VisDrone-DET 2019 → COCO + SAHI-Slices.

Pipeline:
    1. Download VisDrone-DET train + val (Google Drive, via gdown).
    2. Convert VisDrone TXT annotations → COCO JSON.
       Drop categories 0 (ignored) and 11 (others); keep 10 valid
       classes (pedestrian, people, bicycle, car, van, truck, tricycle,
       awning-tricycle, bus, motor) remapped to COCO IDs 1-10.
    3. SAHI-Slicing: 640x640 tiles, 20% overlap.
       Train: sliced only (used for training).
       Val:   sliced + original kept (original used for SAHI-Inference
              at evaluation time).
    4. Print dataset statistics (#images, #annotations, COCO size
       buckets: small < 32x32, medium 32-96, large > 96).

Reference: github.com/fcakyon/small-object-detection-benchmark

Usage:
    python data/prepare.py --output /content/visdrone
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from mmengine.logging import MMLogger
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: VisDrone-DET Google Drive file IDs (from the official VisDrone-Dataset
#: GitHub repo). Overridable via CLI flags.
VISDRONE_GDRIVE_IDS: Dict[str, str] = {
    "train": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
    "val": "1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59",
}

#: 10 valid VisDrone classes. Indexed by the original VisDrone category
#: id (1-10). IDs 0 (ignored) and 11 (others) are dropped.
VISDRONE_CLASSES: List[str] = [
    "pedestrian",       # 1
    "people",           # 2
    "bicycle",          # 3
    "car",              # 4
    "van",              # 5
    "truck",            # 6
    "tricycle",         # 7
    "awning-tricycle",  # 8
    "bus",              # 9
    "motor",            # 10
]

#: COCO area thresholds for size buckets (pixels^2).
COCO_SMALL_MAX = 32 * 32
COCO_MEDIUM_MAX = 96 * 96

#: SAHI slicing parameters (shared with training config).
SLICE_SIZE = 640
SLICE_OVERLAP = 0.2

logger = MMLogger.get_instance("prepare", log_level="INFO")


# ---------------------------------------------------------------------------
# Step 1 — Download
# ---------------------------------------------------------------------------

def download_visdrone(output_dir: Path, gdrive_ids: Dict[str, str]) -> Dict[str, Path]:
    """Download VisDrone train/val zips via gdown and extract them.

    Args:
        output_dir: Directory that will hold the raw VisDrone splits.
        gdrive_ids: Mapping ``split -> google drive file id``.

    Returns:
        Mapping ``split -> path to extracted split directory``.
    """
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    split_dirs: Dict[str, Path] = {}
    for split, file_id in gdrive_ids.items():
        zip_path = raw_dir / f"VisDrone2019-DET-{split}.zip"
        split_dir = raw_dir / f"VisDrone2019-DET-{split}"
        split_dirs[split] = split_dir

        if split_dir.exists() and any(split_dir.iterdir()):
            logger.info(f"[{split}] already extracted at {split_dir} — skip")
            continue

        if not zip_path.exists():
            logger.info(f"[{split}] downloading via gdown (id={file_id})")
            subprocess.run(
                [
                    sys.executable, "-m", "gdown",
                    "--id", file_id,
                    "--output", str(zip_path),
                ],
                check=True,
            )

        logger.info(f"[{split}] extracting {zip_path.name}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    return split_dirs


# ---------------------------------------------------------------------------
# Step 2 — VisDrone TXT → COCO JSON
# ---------------------------------------------------------------------------

def visdrone_to_coco(split_dir: Path, split: str) -> Dict:
    """Convert a VisDrone split (images/ + annotations/) to a COCO dict.

    VisDrone TXT format (one object per line):
        bbox_left, bbox_top, bbox_width, bbox_height,
        score, object_category, truncation, occlusion

    We drop:
        * category 0 (ignored regions) and 11 (others)
        * objects with score == 0 (VisDrone marks them as ignore-regions)

    Args:
        split_dir: Path to a VisDrone2019-DET-{split} directory.
        split: Name of the split (``"train"`` / ``"val"``), used in logs.

    Returns:
        COCO-format dictionary with ``images``, ``annotations``, ``categories``.
    """
    images_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    if not images_dir.exists() or not ann_dir.exists():
        raise FileNotFoundError(
            f"Expected {images_dir} and {ann_dir} to exist"
        )

    categories = [
        {"id": i + 1, "name": name, "supercategory": "visdrone"}
        for i, name in enumerate(VISDRONE_CLASSES)
    ]

    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1
    dropped = 0

    for img_id, img_path in enumerate(sorted(images_dir.glob("*.jpg")), start=1):
        with Image.open(img_path) as im:
            width, height = im.size

        images.append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        txt_path = ann_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue

        for line in txt_path.read_text().strip().splitlines():
            parts = line.strip().rstrip(",").split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = map(int, parts[:4])
            score = int(parts[4])
            cat = int(parts[5])

            # Drop ignored / others / zero-score objects.
            if cat == 0 or cat == 11 or score == 0:
                dropped += 1
                continue
            if w <= 0 or h <= 0:
                dropped += 1
                continue

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat,  # 1..10 — already aligned with VISDRONE_CLASSES
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    logger.info(
        f"[{split}] images={len(images)} ann={len(annotations)} dropped={dropped}"
    )
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# Step 3 — SAHI slicing
# ---------------------------------------------------------------------------

def slice_split(
    coco_dict: Dict,
    images_dir: Path,
    out_dir: Path,
    split: str,
) -> Tuple[Path, Path]:
    """Slice a COCO split into 640x640 tiles with 20% overlap.

    Args:
        coco_dict: COCO-format annotations for the unsliced split.
        images_dir: Directory with the original images referenced by ``coco_dict``.
        out_dir: Base output directory for the sliced split.
        split: Name of the split (used in filenames).

    Returns:
        Tuple ``(sliced_json_path, sliced_images_dir)``.
    """
    from sahi.slicing import slice_coco

    out_dir.mkdir(parents=True, exist_ok=True)
    src_json = out_dir / f"{split}_unsliced.json"
    with src_json.open("w") as f:
        json.dump(coco_dict, f)

    sliced_images_dir = out_dir / f"{split}_sliced_images"
    sliced_images_dir.mkdir(exist_ok=True)

    logger.info(f"[{split}] SAHI slicing → {SLICE_SIZE}x{SLICE_SIZE} @ {SLICE_OVERLAP}")
    sliced_coco, sliced_json_path = slice_coco(
        coco_annotation_file_path=str(src_json),
        image_dir=str(images_dir),
        output_coco_annotation_file_name=f"{split}_sliced",
        output_dir=str(sliced_images_dir),
        ignore_negative_samples=True,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=SLICE_OVERLAP,
        overlap_width_ratio=SLICE_OVERLAP,
        min_area_ratio=0.1,
        verbose=False,
    )
    logger.info(
        f"[{split}] sliced: images={len(sliced_coco['images'])} "
        f"ann={len(sliced_coco['annotations'])}"
    )
    return Path(sliced_json_path), sliced_images_dir


# ---------------------------------------------------------------------------
# Step 4 — Statistics
# ---------------------------------------------------------------------------

def size_buckets(coco_dict: Dict) -> Dict[str, int]:
    """Count annotations by COCO size bucket (small/medium/large).

    Args:
        coco_dict: Loaded COCO dictionary with an ``annotations`` list.

    Returns:
        Dict with keys ``small``, ``medium``, ``large``.
    """
    buckets = {"small": 0, "medium": 0, "large": 0}
    for ann in coco_dict["annotations"]:
        area = ann.get("area") or (ann["bbox"][2] * ann["bbox"][3])
        if area < COCO_SMALL_MAX:
            buckets["small"] += 1
        elif area < COCO_MEDIUM_MAX:
            buckets["medium"] += 1
        else:
            buckets["large"] += 1
    return buckets


def class_distribution(coco_dict: Dict) -> Dict[str, int]:
    """Count annotations per class name."""
    id_to_name = {c["id"]: c["name"] for c in coco_dict["categories"]}
    counts: Dict[str, int] = {name: 0 for name in id_to_name.values()}
    for ann in coco_dict["annotations"]:
        counts[id_to_name[ann["category_id"]]] += 1
    return counts


def log_stats(label: str, coco_dict: Dict) -> Dict:
    """Log and return a statistics dict for a COCO split."""
    buckets = size_buckets(coco_dict)
    total = sum(buckets.values()) or 1
    classes = class_distribution(coco_dict)
    stats = {
        "label": label,
        "num_images": len(coco_dict["images"]),
        "num_annotations": len(coco_dict["annotations"]),
        "size_buckets": buckets,
        "size_buckets_pct": {
            k: round(100 * v / total, 2) for k, v in buckets.items()
        },
        "classes": classes,
    }
    logger.info(
        f"[{label}] imgs={stats['num_images']} ann={stats['num_annotations']} "
        f"small={buckets['small']} medium={buckets['medium']} large={buckets['large']}"
    )
    return stats


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def prepare(output_dir: Path, gdrive_ids: Dict[str, str]) -> Dict:
    """Run the full prepare pipeline and return a stats summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    split_dirs = download_visdrone(output_dir, gdrive_ids)

    annotations_dir = output_dir / "annotations"
    annotations_dir.mkdir(exist_ok=True)

    summary: Dict[str, Dict] = {}

    for split, split_dir in split_dirs.items():
        # 2) convert
        coco = visdrone_to_coco(split_dir, split)
        unsliced_json = annotations_dir / f"{split}_unsliced.json"
        with unsliced_json.open("w") as f:
            json.dump(coco, f)

        # 4a) stats — unsliced
        unsliced_stats = log_stats(f"{split}/unsliced", coco)

        # 3) slice (train: sliced only; val: sliced + keep unsliced)
        sliced_json, _ = slice_split(
            coco, split_dir / "images", output_dir / "sliced", split
        )
        # Publish the sliced json next to the unsliced one for convenience.
        final_sliced_json = annotations_dir / f"{split}_sliced.json"
        shutil.copy(sliced_json, final_sliced_json)

        with final_sliced_json.open() as f:
            sliced_coco = json.load(f)
        sliced_stats = log_stats(f"{split}/sliced", sliced_coco)

        summary[split] = {
            "unsliced": unsliced_stats,
            "sliced": sliced_stats,
        }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote summary → {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/content/visdrone"),
        help="Output directory (default: /content/visdrone)",
    )
    parser.add_argument("--train-id", default=VISDRONE_GDRIVE_IDS["train"])
    parser.add_argument("--val-id", default=VISDRONE_GDRIVE_IDS["val"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare(
        output_dir=args.output,
        gdrive_ids={"train": args.train_id, "val": args.val_id},
    )


if __name__ == "__main__":
    main()
