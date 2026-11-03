"""Audit class balance, human box sizes, and split leakage in the YOLO dataset."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "mydata"
OUT_DIR = ROOT / "runs" / "dataset_audit"
SPLITS = {"train": "train", "val": "valid", "test": "test"}
CLASS_NAMES = {0: "fish", 1: "human"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$", re.IGNORECASE)


def source_id(path: Path) -> str:
    """Remove the Roboflow export hash so augmented copies share one source id."""
    return RF_SUFFIX.sub("", path.stem).lower()


def image_fingerprint(path: Path) -> str:
    """Hash decoded pixels to detect byte-different files with identical content."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        digest = hashlib.sha256()
        digest.update(f"{rgb.width}x{rgb.height}".encode("ascii"))
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def projected_area(width: int, height: int, box_w: float, box_h: float, imgsz: int = 640) -> float:
    scale = min(imgsz / width, imgsz / height)
    return box_w * width * scale * box_h * height * scale


def size_group(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_split(split: str, folder: str) -> tuple[dict, list[dict], dict[str, set[str]], list[str]]:
    image_dir = DATA_ROOT / folder / "images"
    label_dir = DATA_ROOT / folder / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    stats = Counter()
    human_areas: list[float] = []
    human_aspects: list[float] = []
    instances: list[dict] = []
    source_ids: set[str] = set()
    fingerprints: set[str] = set()
    examples: list[str] = []

    stats["images"] = len(images)
    for image_path in images:
        source_ids.add(source_id(image_path))
        try:
            fingerprints.add(image_fingerprint(image_path))
            with Image.open(image_path) as image:
                width, height = image.size
        except Exception as exc:
            stats["unreadable_images"] += 1
            examples.append(f"unreadable image: {image_path.name}: {exc}")
            continue

        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            stats["missing_labels"] += 1
            continue

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            stats["empty_images"] += 1
            continue

        classes_in_image: set[int] = set()
        for line_no, line in enumerate(lines, start=1):
            fields = line.split()
            if len(fields) != 5:
                stats["invalid_labels"] += 1
                examples.append(f"invalid fields: {label_path.name}:{line_no}: {line}")
                continue
            try:
                class_id = int(float(fields[0]))
                x, y, box_w, box_h = map(float, fields[1:])
            except ValueError:
                stats["invalid_labels"] += 1
                examples.append(f"invalid values: {label_path.name}:{line_no}: {line}")
                continue

            if class_id not in CLASS_NAMES or box_w <= 0 or box_h <= 0:
                stats["invalid_labels"] += 1
                examples.append(f"invalid class/size: {label_path.name}:{line_no}: {line}")
                continue
            if not all(0 <= value <= 1 for value in (x, y, box_w, box_h)):
                stats["out_of_range_labels"] += 1
            if x - box_w / 2 < 0 or x + box_w / 2 > 1 or y - box_h / 2 < 0 or y + box_h / 2 > 1:
                stats["boxes_outside_image"] += 1

            classes_in_image.add(class_id)
            class_name = CLASS_NAMES[class_id]
            stats[f"{class_name}_instances"] += 1
            area = projected_area(width, height, box_w, box_h)
            group = size_group(area)
            if class_name == "human":
                stats[f"human_{group}"] += 1
                human_areas.append(area)
                human_aspects.append(box_w * width / (box_h * height))
            instances.append(
                {
                    "split": split,
                    "image": image_path.name,
                    "source_id": source_id(image_path),
                    "class": class_name,
                    "x": x,
                    "y": y,
                    "width_norm": box_w,
                    "height_norm": box_h,
                    "area_at_640": area,
                    "size_group": group,
                }
            )

        for class_id in classes_in_image:
            stats[f"{CLASS_NAMES[class_id]}_images"] += 1

    summary = {
        "split": split,
        "images": stats["images"],
        "human_images": stats["human_images"],
        "human_image_pct": 100 * stats["human_images"] / stats["images"] if stats["images"] else 0,
        "fish_images": stats["fish_images"],
        "human_instances": stats["human_instances"],
        "fish_instances": stats["fish_instances"],
        "human_small": stats["human_small"],
        "human_medium": stats["human_medium"],
        "human_large": stats["human_large"],
        "human_area_p25": percentile(human_areas, 25),
        "human_area_median": percentile(human_areas, 50),
        "human_area_p75": percentile(human_areas, 75),
        "human_aspect_median": percentile(human_aspects, 50),
        "empty_images": stats["empty_images"],
        "missing_labels": stats["missing_labels"],
        "invalid_labels": stats["invalid_labels"],
        "out_of_range_labels": stats["out_of_range_labels"],
        "boxes_outside_image": stats["boxes_outside_image"],
        "unreadable_images": stats["unreadable_images"],
    }
    identities = {"source_ids": source_ids, "fingerprints": fingerprints}
    return summary, instances, identities, examples


def overlap_rows(identities: dict[str, dict[str, set[str]]]) -> list[dict]:
    rows = []
    split_names = list(SPLITS)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            source_overlap = identities[left]["source_ids"] & identities[right]["source_ids"]
            exact_overlap = identities[left]["fingerprints"] & identities[right]["fingerprints"]
            rows.append(
                {
                    "split_pair": f"{left}-{right}",
                    "shared_source_ids": len(source_overlap),
                    "identical_decoded_images": len(exact_overlap),
                    "source_examples": "; ".join(sorted(source_overlap)[:10]),
                }
            )
    return rows


def format_num(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_report(summaries: list[dict], overlaps: list[dict], examples: list[str]) -> None:
    lines = ["# Human Dataset Audit", "", "Human size groups use COCO area thresholds after letterbox scaling to 640: small < 32^2, medium < 96^2, large >= 96^2.", ""]
    columns = [
        "split", "images", "human_images", "human_image_pct", "human_instances", "fish_instances",
        "human_small", "human_medium", "human_large", "human_area_median",
    ]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(columns) - 1)) + "|")
    for row in summaries:
        lines.append("| " + " | ".join(format_num(row[column]) for column in columns) + " |")

    lines.extend(["", "## Split Leakage", "", "Roboflow source ids are filenames with the `.rf.<hash>` suffix removed.", ""])
    lines.append("| split_pair | shared_source_ids | identical_decoded_images | source_examples |")
    lines.append("|---|---:|---:|---|")
    for row in overlaps:
        lines.append("| " + " | ".join(str(row[key]) for key in row) + " |")

    quality_columns = [
        "split", "empty_images", "missing_labels", "invalid_labels", "out_of_range_labels",
        "boxes_outside_image", "unreadable_images",
    ]
    lines.extend(["", "## Label Quality", ""])
    lines.append("| " + " | ".join(quality_columns) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(quality_columns) - 1)) + "|")
    for row in summaries:
        lines.append("| " + " | ".join(format_num(row[column]) for column in quality_columns) + " |")

    if examples:
        lines.extend(["", "## Quality Examples", ""])
        lines.extend(f"- `{example}`" for example in examples[:20])
    (OUT_DIR / "human_dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    instances: list[dict] = []
    identities: dict[str, dict[str, set[str]]] = {}
    examples: list[str] = []

    for split, folder in SPLITS.items():
        summary, split_instances, split_identities, split_examples = audit_split(split, folder)
        summaries.append(summary)
        instances.extend(split_instances)
        identities[split] = split_identities
        examples.extend(split_examples)

    overlaps = overlap_rows(identities)
    write_csv(OUT_DIR / "split_summary.csv", summaries)
    write_csv(OUT_DIR / "instances.csv", instances)
    write_csv(OUT_DIR / "split_overlap.csv", overlaps)
    write_report(summaries, overlaps, examples)

    for row in summaries:
        print(
            f"{row['split']:<5} images={row['images']:<5} human_images={row['human_images']:<5} "
            f"human={row['human_instances']:<5} fish={row['fish_instances']:<5} "
            f"human S/M/L={row['human_small']}/{row['human_medium']}/{row['human_large']}"
        )
    for row in overlaps:
        print(
            f"{row['split_pair']:<10} shared_source_ids={row['shared_source_ids']:<4} "
            f"identical_images={row['identical_decoded_images']}"
        )
    print(f"Saved audit to: {OUT_DIR}")


if __name__ == "__main__":
    main()
