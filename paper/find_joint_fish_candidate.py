"""Select the best honest fish-recovery example for MSCAA+SAFusion.

The full validation prediction caches are reused. Selection first seeks fish
GT recovery, then fish-scene false-positive suppression with unchanged recall,
and finally confidence improvement with unchanged TP/FP counts.
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

from paper import find_joint_qualitative_replacement as search  # noqa: E402
from paper import gen_safusion_qualitative as visual  # noqa: E402


CACHE_DIR = ROOT / "runs" / "paper_visuals_joint_replacement_20260901"
OUTPUT = ROOT / "picture" / "joint_fish_candidate_preview.png"
AUDIT_OUTPUT = CACHE_DIR / "selected_fish_candidate.json"
EXCLUDED = {
    "Screenshot-383-_png.rf.1d5672661ac31e5216bd9bdd2be63c96.jpg",
}


def rank_candidates(rows: list[dict]) -> tuple[str, list[dict]]:
    recovery = []
    suppression = []
    confidence = []
    for row in rows:
        if row["image"] in EXCLUDED:
            continue
        fish_gain = row["joint_fish_tp"] - row["baseline_fish_tp"]
        human_gain = row["joint_human_tp"] - row["baseline_human_tp"]
        if row["baseline_fish_gt"] < 1 or human_gain < 0:
            continue
        base_fp = row["baseline_fish_fp"] + row["baseline_human_fp"]
        joint_fp = row["joint_fish_fp"] + row["joint_human_fp"]
        row = dict(row)
        row["fish_tp_gain"] = fish_gain
        row["total_fp_change"] = joint_fp - base_fp
        row["fish_confidence_change"] = (
            row["joint_fish_mean_match_conf"] - row["baseline_fish_mean_match_conf"]
        )
        if fish_gain >= 1:
            recovery.append(row)
        elif fish_gain == 0 and joint_fp < base_fp:
            suppression.append(row)
        elif fish_gain == 0 and joint_fp == base_fp and row["fish_confidence_change"] > 0:
            confidence.append(row)

    candidates = recovery or suppression or confidence
    mode = (
        "fish_recovery" if recovery else
        "fish_fp_suppression" if suppression else
        "fish_confidence_gain"
    )
    candidates.sort(
        key=lambda row: (
            row["total_fp_change"],
            -row["fish_tp_gain"],
            -row["fish_confidence_change"],
            -row["joint_fish_gt"],
        )
    )
    return mode, candidates


def compose(path: Path, row: dict, baseline: dict, joint: dict) -> np.ndarray:
    panel_w, panel_h = 560, 350
    header_h, footer_h = 58, 76
    image = cv2.imread(str(path))
    panels = [
        visual.fit_panel(image, panel_w, panel_h),
        visual.fit_panel(
            visual.draw_detections(image, baseline[path.name]), panel_w, panel_h
        ),
        visual.fit_panel(
            visual.draw_detections(image, joint[path.name]), panel_w, panel_h
        ),
    ]
    header = np.hstack(
        [
            visual.text_bar(panel_w, header_h, "Input"),
            visual.text_bar(panel_w, header_h, "StarEye"),
            visual.text_bar(panel_w, header_h, "StarEye + MSCAA + SAFusion"),
        ]
    )
    footer = np.full((footer_h, panel_w * 3, 3), 255, dtype=np.uint8)
    base_fp = row["baseline_fish_fp"] + row["baseline_human_fp"]
    joint_fp = row["joint_fish_fp"] + row["joint_human_fp"]
    footer = visual.pil_centered_text(
        footer,
        [
            f"Fish TP: {row['baseline_fish_tp']} -> {row['joint_fish_tp']}    "
            f"Fish FN: {row['baseline_fish_fn']} -> {row['joint_fish_fn']}    "
            f"Total FP: {base_fp} -> {joint_fp}"
        ],
        font_size=20,
        bold=True,
    )
    return np.vstack([header, np.hstack(panels), footer])


def main() -> None:
    paths = visual.image_paths()
    baseline = visual.run_or_load(
        search.BASELINE_WEIGHT, paths, CACHE_DIR / "baseline_s0_predictions.json"
    )
    joint = visual.run_or_load(
        search.JOINT_WEIGHT, paths, CACHE_DIR / "joint_mscaa_safusion_s0_predictions.json"
    )
    rows = search.audit(paths, baseline, joint)
    mode, candidates = rank_candidates(rows)
    if not candidates:
        raise RuntimeError("No alternative fish-recovery candidate was found.")
    selected = candidates[0]
    selected_path = visual.IMAGE_DIR / selected["image"]

    record = {
        "protocol": {
            "baseline": str(search.BASELINE_WEIGHT.relative_to(ROOT)),
            "joint": str(search.JOINT_WEIGHT.relative_to(ROOT)),
            "split": "valid",
            "confidence": visual.CONF,
            "nms_iou": visual.IOU_NMS,
            "match_iou": visual.MATCH_IOU,
            "imgsz": visual.IMGSZ,
            "selection_mode": mode,
            "selection_rule": (
                "Prefer fish TP recovery; otherwise preserve fish/human TP and reduce "
                "total FP; otherwise preserve TP/FP and improve fish match confidence"
            ),
        },
        "candidate_count": len(candidates),
        "selected": selected,
        "top_five": candidates[:5],
    }
    AUDIT_OUTPUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    figure = compose(selected_path, selected, baseline, joint)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(OUTPUT), figure):
        raise RuntimeError(f"Could not write {OUTPUT}")

    base_fp = selected["baseline_fish_fp"] + selected["baseline_human_fp"]
    joint_fp = selected["joint_fish_fp"] + selected["joint_human_fp"]
    print(f"Selection mode: {mode}")
    print(f"Eligible candidates in selected mode: {len(candidates)}")
    print(f"Selected: {selected['image']}")
    print(
        f"Fish TP {selected['baseline_fish_tp']}->{selected['joint_fish_tp']}; "
        f"Fish FN {selected['baseline_fish_fn']}->{selected['joint_fish_fn']}; "
        f"Total FP {base_fp}->{joint_fp}"
    )
    print(f"Saved preview: {OUTPUT}")
    print(f"Saved audit: {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()
