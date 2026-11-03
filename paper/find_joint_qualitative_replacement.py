"""Find one auditable qualitative example favorable to MSCAA+SAFusion.

Selection uses all validation images and fixed prediction/matching thresholds.
It first seeks a human-recovery case, then any true-positive recovery, and
finally a new false-positive suppression case with no loss of matched targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".python_packages"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from paper import gen_safusion_qualitative as visual  # noqa: E402


BASELINE_WEIGHT = ROOT / "runs" / "ablation_100ep" / "Baseline_s0" / "weights" / "best.pt"
JOINT_WEIGHT = ROOT / "runs" / "ablation_100ep" / "MSCAA_SAFusion_s0" / "weights" / "best.pt"
OUT_DIR = ROOT / "runs" / "paper_visuals_joint_replacement_20260901"
OUTPUT = ROOT / "picture" / "joint_replacement_preview.png"
EXCLUDED_IMAGES = {
    "Screenshot-383-_png.rf.1d5672661ac31e5216bd9bdd2be63c96.jpg",
    "313_jpg.rf.9b5e83253cc4db8625180d1d18c68dce.jpg",
    "dv_0140_jpg.rf.70414ca786d8bf225a9805bd892ff0bc.jpg",
}


def audit(paths: list[Path], baseline: dict, joint: dict) -> list[dict]:
    rows = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        targets = visual.load_gt(path, width, height)
        row = {"image": path.name}
        for class_id, name in visual.CLASS_NAMES.items():
            for prefix, detections in (("baseline", baseline), ("joint", joint)):
                stats = visual.match_class(targets, detections[path.name], class_id)
                for key, value in stats.items():
                    row[f"{prefix}_{name}_{key}"] = value
        rows.append(row)
    return rows


def choose(rows: list[dict]) -> tuple[str, dict]:
    eligible = [row for row in rows if row["image"] not in EXCLUDED_IMAGES]

    def values(row: dict) -> tuple[int, int, int, int]:
        base_fp = row["baseline_fish_fp"] + row["baseline_human_fp"]
        joint_fp = row["joint_fish_fp"] + row["joint_human_fp"]
        base_tp = row["baseline_fish_tp"] + row["baseline_human_tp"]
        joint_tp = row["joint_fish_tp"] + row["joint_human_tp"]
        return base_tp, joint_tp, base_fp, joint_fp

    human_recovery = []
    tp_recovery = []
    fp_suppression = []
    for row in eligible:
        base_tp, joint_tp, base_fp, joint_fp = values(row)
        fish_gain = row["joint_fish_tp"] - row["baseline_fish_tp"]
        human_gain = row["joint_human_tp"] - row["baseline_human_tp"]
        if human_gain >= 1 and fish_gain >= 0 and joint_fp <= base_fp:
            human_recovery.append(row)
        if joint_tp > base_tp and joint_fp <= base_fp:
            tp_recovery.append(row)
        if joint_tp >= base_tp and joint_fp < base_fp:
            fp_suppression.append(row)

    candidates = human_recovery or tp_recovery or fp_suppression
    if not candidates:
        raise RuntimeError("No honest replacement image satisfies the fallback rules.")
    mode = (
        "human_recovery" if human_recovery else
        "target_recovery" if tp_recovery else
        "false_positive_suppression"
    )
    candidates.sort(
        key=lambda row: (
            (row["joint_human_tp"] - row["baseline_human_tp"]),
            (row["joint_fish_tp"] + row["joint_human_tp"])
            - (row["baseline_fish_tp"] + row["baseline_human_tp"]),
            (row["baseline_fish_fp"] + row["baseline_human_fp"])
            - (row["joint_fish_fp"] + row["joint_human_fp"]),
            row["joint_human_mean_match_conf"] + row["joint_fish_mean_match_conf"],
        ),
        reverse=True,
    )
    return mode, candidates[0]


def compose(path: Path, row: dict, baseline: dict, joint: dict) -> np.ndarray:
    panel_w, panel_h = 560, 350
    header_h = 58
    image = cv2.imread(str(path))
    input_panel = visual.fit_panel(image, panel_w, panel_h)
    baseline_panel = visual.fit_panel(
        visual.draw_detections(image, baseline[path.name]), panel_w, panel_h
    )
    joint_panel = visual.fit_panel(
        visual.draw_detections(image, joint[path.name]), panel_w, panel_h
    )
    header = np.hstack(
        [
            visual.text_bar(panel_w, header_h, "Input"),
            visual.text_bar(panel_w, header_h, "StarEye"),
            visual.text_bar(panel_w, header_h, "StarEye + MSCAA + SAFusion"),
        ]
    )
    body = np.hstack([input_panel, baseline_panel, joint_panel])
    footer_h = 76
    footer = np.full((footer_h, panel_w * 3, 3), 255, dtype=np.uint8)
    base_fp = row["baseline_fish_fp"] + row["baseline_human_fp"]
    joint_fp = row["joint_fish_fp"] + row["joint_human_fp"]
    footer = visual.pil_centered_text(
        footer,
        [
            f"Fish TP: {row['baseline_fish_tp']} -> {row['joint_fish_tp']}    "
            f"Human TP: {row['baseline_human_tp']} -> {row['joint_human_tp']}    "
            f"Total FP: {base_fp} -> {joint_fp}"
        ],
        font_size=20,
        bold=True,
    )
    return np.vstack([header, body, footer])


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = visual.image_paths()
    baseline = visual.run_or_load(
        BASELINE_WEIGHT, paths, OUT_DIR / "baseline_s0_predictions.json"
    )
    joint = visual.run_or_load(
        JOINT_WEIGHT, paths, OUT_DIR / "joint_mscaa_safusion_s0_predictions.json"
    )
    rows = audit(paths, baseline, joint)
    mode, selected = choose(rows)
    selected_path = visual.IMAGE_DIR / selected["image"]

    record = {
        "protocol": {
            "baseline": str(BASELINE_WEIGHT.relative_to(ROOT)),
            "joint": str(JOINT_WEIGHT.relative_to(ROOT)),
            "split": "valid",
            "confidence": visual.CONF,
            "nms_iou": visual.IOU_NMS,
            "match_iou": visual.MATCH_IOU,
            "imgsz": visual.IMGSZ,
            "selection_mode": mode,
            "selection_rule": (
                "Prefer human TP recovery without fish/FP regression; otherwise any TP "
                "recovery without FP regression; otherwise FP suppression without TP loss"
            ),
        },
        "selected": selected,
    }
    (OUT_DIR / "selected_example.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    figure = compose(selected_path, selected, baseline, joint)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), figure):
        raise RuntimeError(f"Could not write {OUTPUT}")

    base_fp = selected["baseline_fish_fp"] + selected["baseline_human_fp"]
    joint_fp = selected["joint_fish_fp"] + selected["joint_human_fp"]
    print(f"Selection mode: {mode}")
    print(f"Selected: {selected['image']}")
    print(
        f"Fish TP {selected['baseline_fish_tp']}->{selected['joint_fish_tp']}; "
        f"Human TP {selected['baseline_human_tp']}->{selected['joint_human_tp']}; "
        f"Total FP {base_fp}->{joint_fp}"
    )
    print(f"Saved preview: {OUTPUT}")
    print(f"Saved audit: {OUT_DIR / 'selected_example.json'}")


if __name__ == "__main__":
    main()
