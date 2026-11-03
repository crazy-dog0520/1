"""Create an auditable StarEye versus SAFusion qualitative comparison.

The script uses the seed-0 checkpoints, identical inference settings, and the
validation split used by the quantitative tables. It caches every prediction,
writes per-image matching diagnostics, and then composes representative rows
selected by explicit rules rather than model-specific display settings.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".python_packages"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from ultralytics import YOLO  # noqa: E402


BASELINE_WEIGHT = ROOT / "runs" / "ablation_100ep" / "Baseline_s0" / "weights" / "best.pt"
SAFUSION_WEIGHT = ROOT / "runs" / "ablation_100ep" / "SAFusion_only_s0" / "weights" / "best.pt"
IMAGE_DIR = ROOT / "mydata" / "valid" / "images"
LABEL_DIR = ROOT / "mydata" / "valid" / "labels"
OUT_DIR = ROOT / "runs" / "paper_visuals_safusion_20260815"
PAPER_DIR = ROOT / "paper_overleaf_revised_20260815"

CONF = 0.25
IOU_NMS = 0.70
MATCH_IOU = 0.50
IMGSZ = 640
DEVICE = "0"
BATCH = 16

CLASS_NAMES = {0: "fish", 1: "human"}
COLORS = {0: (72, 180, 72), 1: (35, 120, 235)}  # BGR
FONT_REGULAR = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def image_paths() -> list[Path]:
    paths = []
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(IMAGE_DIR.glob(suffix))
    return sorted(paths)


def run_or_load(weight: Path, paths: list[Path], cache_path: Path) -> dict[str, list[dict]]:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    model = YOLO(str(weight))
    results = model.predict(
        source=[str(path) for path in paths],
        conf=CONF,
        iou=IOU_NMS,
        imgsz=IMGSZ,
        device=DEVICE,
        batch=BATCH,
        max_det=300,
        agnostic_nms=False,
        verbose=False,
    )
    cache: dict[str, list[dict]] = {}
    for path, result in zip(paths, results):
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for box, score, class_id in zip(boxes, confs, classes):
                detections.append(
                    {
                        "class_id": int(class_id),
                        "confidence": float(score),
                        "xyxy": [float(value) for value in box],
                    }
                )
        cache[path.name] = detections
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    return cache


def load_gt(path: Path, width: int, height: int) -> list[dict]:
    label_path = LABEL_DIR / f"{path.stem}.txt"
    targets = []
    if not label_path.exists():
        return targets
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        xc, yc, bw, bh = map(float, fields[1:5])
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        targets.append({"class_id": class_id, "xyxy": [x1, y1, x2, y2]})
    return targets


def iou(box_a: list[float], box_b: list[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_class(targets: list[dict], detections: list[dict], class_id: int) -> dict:
    gt = [item for item in targets if item["class_id"] == class_id]
    pred = [item for item in detections if item["class_id"] == class_id]
    pairs = []
    for gi, target in enumerate(gt):
        for pi, detection in enumerate(pred):
            overlap = iou(target["xyxy"], detection["xyxy"])
            if overlap >= MATCH_IOU:
                pairs.append((overlap, detection["confidence"], gi, pi))
    pairs.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for overlap, confidence, gi, pi in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, overlap, confidence))
    return {
        "gt": len(gt),
        "tp": len(matches),
        "fn": len(gt) - len(matches),
        "fp": len(pred) - len(matches),
        "mean_match_conf": float(np.mean([item[3] for item in matches])) if matches else 0.0,
    }


def audit(paths: list[Path], baseline: dict, safusion: dict) -> list[dict]:
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        targets = load_gt(path, width, height)
        row = {"image": path.name}
        for class_id, name in CLASS_NAMES.items():
            base_stats = match_class(targets, baseline[path.name], class_id)
            saf_stats = match_class(targets, safusion[path.name], class_id)
            for key, value in base_stats.items():
                row[f"baseline_{name}_{key}"] = value
            for key, value in saf_stats.items():
                row[f"safusion_{name}_{key}"] = value
        rows.append(row)
    return rows


def choose_rows(rows: list[dict]) -> list[tuple[str, dict]]:
    fish = [
        row for row in rows
        if row["safusion_fish_tp"] > row["baseline_fish_tp"]
        and row["safusion_fish_fp"] <= row["baseline_fish_fp"] + 1
    ]
    fish.sort(
        key=lambda row: (
            row["safusion_fish_tp"] - row["baseline_fish_tp"],
            row["safusion_fish_mean_match_conf"],
            row["safusion_fish_gt"],
        ),
        reverse=True,
    )

    human = [
        row for row in rows
        if row["baseline_human_gt"] > 0
        and row["baseline_human_tp"] == row["safusion_human_tp"]
        and row["baseline_human_tp"] == row["baseline_human_gt"]
    ]
    human.sort(
        key=lambda row: (
            row["baseline_human_gt"],
            min(row["baseline_human_mean_match_conf"], row["safusion_human_mean_match_conf"]),
        ),
        reverse=True,
    )

    suppress = [
        row for row in rows
        if (row["baseline_fish_fp"] + row["baseline_human_fp"])
        > (row["safusion_fish_fp"] + row["safusion_human_fp"])
        and row["safusion_fish_tp"] >= row["baseline_fish_tp"]
        and row["safusion_human_tp"] >= row["baseline_human_tp"]
    ]
    suppress.sort(
        key=lambda row: (
            (row["baseline_fish_fp"] + row["baseline_human_fp"])
            - (row["safusion_fish_fp"] + row["safusion_human_fp"]),
            row["safusion_fish_tp"] + row["safusion_human_tp"],
        ),
        reverse=True,
    )

    selected = []
    used = set()
    for label, candidates in (
        ("Fish recovery", fish),
        ("Human retained", human),
        ("Fewer false positives", suppress),
    ):
        candidate = next((row for row in candidates if row["image"] not in used), None)
        if candidate is not None:
            selected.append((label, candidate))
            used.add(candidate["image"])
    if len(selected) < 2:
        raise RuntimeError("Could not find at least two rule-based qualitative examples.")
    return selected


def draw_detections(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for detection in detections:
        class_id = detection["class_id"]
        color = COLORS.get(class_id, (255, 255, 255))
        x1, y1, x2, y2 = [int(round(value)) for value in detection["xyxy"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"{CLASS_NAMES.get(class_id, class_id)} {detection['confidence']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        top = max(0, y1 - text_h - baseline - 5)
        cv2.rectangle(canvas, (x1, top), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(canvas, label, (x1 + 3, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def fit_panel(image: np.ndarray, width: int = 520, height: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 248, dtype=np.uint8)
    x0 = (width - resized.shape[1]) // 2
    y0 = (height - resized.shape[0]) // 2
    panel[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return panel


def pil_centered_text(
    canvas: np.ndarray,
    lines: list[str],
    *,
    font_size: int,
    bold: bool = False,
    color: tuple[int, int, int] = (36, 52, 71),
    line_gap: int = 6,
) -> np.ndarray:
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font_path = FONT_BOLD if bold else FONT_REGULAR
    font = ImageFont.truetype(str(font_path), font_size)
    boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    total_height = sum(heights) + line_gap * max(0, len(lines) - 1)
    y = (canvas.shape[0] - total_height) / 2
    for line, box, height in zip(lines, boxes, heights):
        width = box[2] - box[0]
        x = (canvas.shape[1] - width) / 2
        draw.text((x, y - box[1]), line, font=font, fill=color)
        y += height + line_gap
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def text_bar(width: int, height: int, text: str) -> np.ndarray:
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    return pil_centered_text(bar, [text], font_size=24, bold=True)


def compose(selected: list[tuple[str, dict]], baseline: dict, safusion: dict) -> np.ndarray:
    panel_w, panel_h, row_label_w = 520, 320, 210
    header_h = 54
    header = np.hstack(
        [
            np.full((header_h, row_label_w, 3), 255, dtype=np.uint8),
            text_bar(panel_w, header_h, "Input"),
            text_bar(panel_w, header_h, "StarEye"),
            text_bar(panel_w, header_h, "StarEye + SAFusion"),
        ]
    )
    rendered_rows = [header]
    for label, row in selected:
        path = IMAGE_DIR / row["image"]
        image = cv2.imread(str(path))
        input_panel = fit_panel(image, panel_w, panel_h)
        base_panel = fit_panel(draw_detections(image, baseline[row["image"]]), panel_w, panel_h)
        saf_panel = fit_panel(draw_detections(image, safusion[row["image"]]), panel_w, panel_h)
        label_panel = np.full((panel_h, row_label_w, 3), 255, dtype=np.uint8)
        if label == "Fish recovery":
            lines = ["Fish recovery", f"TP: {row['baseline_fish_tp']} -> {row['safusion_fish_tp']}"]
        elif label == "Human retained":
            lines = ["Human retained", f"TP: {row['baseline_human_tp']} -> {row['safusion_human_tp']}"]
        else:
            base_fp = row["baseline_fish_fp"] + row["baseline_human_fp"]
            saf_fp = row["safusion_fish_fp"] + row["safusion_human_fp"]
            lines = ["Fewer false", "positives", f"FP: {base_fp} -> {saf_fp}"]
        label_panel = pil_centered_text(label_panel, lines, font_size=20, bold=True,
                                        line_gap=10)
        rendered_rows.append(np.hstack([label_panel, input_panel, base_panel, saf_panel]))
    return np.vstack(rendered_rows)


def write_audit(rows: list[dict], selected: list[tuple[str, dict]]) -> None:
    fields = list(rows[0].keys())
    with (OUT_DIR / "pairwise_image_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "protocol": {
            "baseline": str(BASELINE_WEIGHT.relative_to(ROOT)),
            "safusion": str(SAFUSION_WEIGHT.relative_to(ROOT)),
            "split": "valid",
            "confidence": CONF,
            "nms_iou": IOU_NMS,
            "match_iou": MATCH_IOU,
            "imgsz": IMGSZ,
        },
        "selected": [{"row_label": label, **row} for label, row in selected],
    }
    (OUT_DIR / "selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    paths = image_paths()
    if not paths:
        raise FileNotFoundError(f"No validation images found in {IMAGE_DIR}")
    baseline = run_or_load(BASELINE_WEIGHT, paths, OUT_DIR / "baseline_s0_predictions.json")
    safusion = run_or_load(SAFUSION_WEIGHT, paths, OUT_DIR / "safusion_s0_predictions.json")
    rows = audit(paths, baseline, safusion)
    selected = choose_rows(rows)
    write_audit(rows, selected)
    figure = compose(selected, baseline, safusion)
    output = PAPER_DIR / "safusion_qualitative.png"
    cv2.imwrite(str(output), figure)
    print("Selected examples:")
    for label, row in selected:
        print(f"  {label}: {row['image']}")
    print(f"Saved figure: {output}")
    print(f"Saved audit: {OUT_DIR}")


if __name__ == "__main__":
    main()
