"""Create the TP/FP-labelled qualitative figure as SVG, PDF, and PNG.

The photographs remain raster data, while the layout, boxes, labels, and text
are emitted as vector objects in the SVG/PDF outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".python_packages"))

import matplotlib as mpl  # noqa: E402
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import fitz  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402


RUN_DIR = ROOT / "runs" / "paper_visuals_joint_replacement_20260901"
IMAGE_DIR = ROOT / "mydata" / "valid" / "images"
LABEL_DIR = ROOT / "mydata" / "valid" / "labels"
OUTPUT_DIR = ROOT / "picture"

BASELINE_CACHE = RUN_DIR / "baseline_s0_predictions.json"
JOINT_CACHE = RUN_DIR / "joint_mscaa_safusion_s0_predictions.json"
FISH_RECORD = RUN_DIR / "selected_fish_candidate.json"
HUMAN_RECORD = RUN_DIR / "selected_example.json"
MATCH_IOU = 0.50

CLASS_NAMES = {0: "fish", 1: "human"}
TP_COLOR = "#16a34a"
FP_COLOR = "#dc2626"
TEXT_COLOR = "#26364a"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_targets(image_name: str, width: int, height: int) -> list[dict]:
    label_path = LABEL_DIR / f"{Path(image_name).stem}.txt"
    targets: list[dict] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        class_id = int(float(fields[0]))
        xc, yc, bw, bh = map(float, fields[1:5])
        targets.append(
            {
                "class_id": class_id,
                "xyxy": [
                    (xc - bw / 2) * width,
                    (yc - bh / 2) * height,
                    (xc + bw / 2) * width,
                    (yc + bh / 2) * height,
                ],
            }
        )
    return targets


def iou(box_a: list[float], box_b: list[float]) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def classify_detections(targets: list[dict], detections: list[dict]) -> list[str]:
    """Apply the paper's class-consistent greedy IoU matching protocol."""
    status = ["FP"] * len(detections)
    for class_id in CLASS_NAMES:
        gt_ids = [i for i, target in enumerate(targets) if target["class_id"] == class_id]
        pred_ids = [i for i, pred in enumerate(detections) if pred["class_id"] == class_id]
        pairs = []
        for gi in gt_ids:
            for pi in pred_ids:
                overlap = iou(targets[gi]["xyxy"], detections[pi]["xyxy"])
                if overlap >= MATCH_IOU:
                    pairs.append((overlap, detections[pi]["confidence"], gi, pi))
        pairs.sort(reverse=True)
        used_gt, used_pred = set(), set()
        for _, _, gi, pi in pairs:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            status[pi] = "TP"
    return status


def draw_predictions(ax, detections: list[dict], targets: list[dict]) -> None:
    statuses = classify_detections(targets, detections)
    # High-confidence labels are drawn first; small vertical offsets reduce
    # collisions when several duplicate boxes start at the image boundary.
    label_slots: dict[int, int] = {}
    top_slot = 0
    for detection, status in zip(detections, statuses):
        x1, y1, x2, y2 = detection["xyxy"]
        color = TP_COLOR if status == "TP" else FP_COLOR
        ax.add_patch(
            Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=color, linewidth=2.2, zorder=3,
            )
        )
        if y1 < 55.0:
            label_y = 5.0 + 25.0 * top_slot
            top_slot += 1
        else:
            slot_key = int(round(x1 / 45.0))
            slot = label_slots.get(slot_key, 0)
            label_slots[slot_key] = slot + 1
            label_y = y1 + 3.0 + 25.0 * slot
        label = (
            f"{status}  {CLASS_NAMES.get(detection['class_id'], detection['class_id'])} "
            f"{detection['confidence']:.2f}"
        )
        right_aligned = x1 > 475.0
        label_x = min(635.0, x2 - 4.0) if right_aligned else max(5.0, x1 + 4.0)
        ax.text(
            label_x,
            label_y,
            label,
            ha="right" if right_aligned else "left",
            va="top",
            fontsize=7.2,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.20", "facecolor": color, "edgecolor": color},
            clip_on=True,
            zorder=4,
        )


def configure_image_axis(ax, image: Image.Image) -> None:
    ax.imshow(image)
    ax.set_xlim(0, image.width)
    ax.set_ylim(image.height, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    baseline = load_json(BASELINE_CACHE)
    joint = load_json(JOINT_CACHE)
    fish = load_json(FISH_RECORD)["selected"]
    human = load_json(HUMAN_RECORD)["selected"]
    rows = [
        ("Fish retained", "TP: 2 → 2", "FP: 2 → 1", fish["image"]),
        ("Human recovered", "TP: 1 → 2", "FP: 2 → 0", human["image"]),
    ]

    fig = plt.figure(figsize=(13.3, 8.15), facecolor="white")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=[0.44, 1.0, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        left=0.015,
        right=0.985,
        bottom=0.035,
        top=0.925,
        wspace=0.035,
        hspace=0.14,
    )

    titles = ["Input", "StarEye", "StarEye + MSCAA + SAFusion"]
    for col, title in enumerate(titles, start=1):
        pos = grid[0, col].get_position(fig)
        fig.text(
            (pos.x0 + pos.x1) / 2,
            0.955,
            title,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=TEXT_COLOR,
        )

    legend = fig.legend(
        handles=[
            Patch(facecolor=TP_COLOR, edgecolor=TP_COLOR, label="TP (IoU ≥ 0.50)"),
            Patch(facecolor=FP_COLOR, edgecolor=FP_COLOR, label="FP"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.012, 0.995),
        ncol=2,
        frameon=False,
        fontsize=9,
        handlelength=1.25,
        columnspacing=1.0,
    )
    legend.set_zorder(10)

    for row_index, (row_title, tp_text, fp_text, image_name) in enumerate(rows):
        label_ax = fig.add_subplot(grid[row_index, 0])
        label_ax.axis("off")
        label_ax.text(
            0.5,
            0.57,
            f"{row_title}\n{tp_text}\n{fp_text}",
            ha="center",
            va="center",
            fontsize=12.2,
            fontweight="bold",
            color=TEXT_COLOR,
            linespacing=1.35,
        )

        image = Image.open(IMAGE_DIR / image_name).convert("RGB")
        targets = load_targets(image_name, image.width, image.height)
        panels = [None, baseline[image_name], joint[image_name]]
        for col, detections in enumerate(panels, start=1):
            ax = fig.add_subplot(grid[row_index, col])
            configure_image_axis(ax, image)
            if detections is not None:
                draw_predictions(ax, detections, targets)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    svg_output = OUTPUT_DIR / "joint_qualitative.svg"
    png_output = OUTPUT_DIR / "joint_qualitative.png"
    pdf_output = OUTPUT_DIR / "joint_qualitative.pdf"
    fig.savefig(svg_output, facecolor="white")
    fig.savefig(png_output, facecolor="white", dpi=220)
    plt.close(fig)

    # Matplotlib's PDF backend in this environment can emit an image stream
    # that some PDF consumers reject. Converting the valid SVG with MuPDF
    # keeps text and annotation paths vector-based and is robust in pdfLaTeX.
    svg_document = fitz.open("svg", svg_output.read_bytes())
    pdf_bytes = svg_document.convert_to_pdf()
    pdf_document = fitz.open("pdf", pdf_bytes)
    pdf_document.save(pdf_output)
    pdf_document.close()
    svg_document.close()

    print(f"Saved {svg_output}")
    print(f"Saved {pdf_output}")
    print(f"Saved {png_output}")


if __name__ == "__main__":
    main()
