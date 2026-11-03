"""
Build the human-focused journal table from existing checkpoints.

This evaluates two-class checkpoints on mydata/data.yaml and extracts only the
human class metrics, avoiding the class-index mismatch that happens when a
two-class model is evaluated on mydata_human/data.yaml.

Run with the project environment:
    D:\Anaconda3\envs\Stareye\python.exe eval_human_table.py

Outputs:
    runs/human_metrics_table_reproduction/human_metrics_val_raw.csv
    runs/human_metrics_table_reproduction/human_metrics_test_raw.csv
    runs/human_metrics_table_reproduction/human_metrics_summary.md
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mydata" / "data.yaml"
OUT_DIR = ROOT / "runs" / "human_metrics_table_reproduction"

# Only the four configurations with checkpoints present in this delivery copy.
MODEL_GROUPS = [
    (
        "StarEye",
        [ROOT / "runs" / "ablation_100ep" / f"Baseline_s{s}" / "weights" / "best.pt" for s in (0, 42, 123)],
    ),
    (
        "StarEye+MSCAA",
        [ROOT / "runs" / "ablation_100ep" / f"MSCAA_only_s{s}" / "weights" / "best.pt" for s in (0, 42, 123)],
    ),
    (
        "StarEye+SAFusion",
        [ROOT / "runs" / "ablation_100ep" / f"SAFusion_only_s{s}" / "weights" / "best.pt" for s in (0, 42, 123)],
    ),
    (
        "StarEye+MSCAA+SAFusion",
        [ROOT / "runs" / "ablation_100ep" / f"MSCAA_SAFusion_s{s}" / "weights" / "best.pt" for s in (0, 42, 123)],
    ),
]

IOU_LABELS = ["AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95"]
AP_INDEX = {label: idx for idx, label in enumerate(IOU_LABELS)}


def as_float(value: float) -> float:
    """Convert numpy scalars to plain floats and keep missing values explicit."""
    try:
        return float(value)
    except Exception:
        return math.nan


def find_class_index(metrics, target_name: str) -> int:
    """Return the row index inside metrics.box.all_ap for a named class."""
    names = metrics.names
    class_id = None
    for idx, name in names.items():
        if target_name.lower() in str(name).lower():
            class_id = int(idx)
            break
    if class_id is None:
        raise RuntimeError(f"Could not find class containing '{target_name}' in names={names!r}")

    ap_class_index = list(map(int, metrics.box.ap_class_index))
    if class_id not in ap_class_index:
        raise RuntimeError(
            f"Class id {class_id} ({target_name}) has no AP row. "
            f"ap_class_index={ap_class_index}, names={names!r}"
        )
    return ap_class_index.index(class_id)


def evaluate_one(
    name: str, weight: Path, data: Path, split: str, imgsz: int, batch: int, device: str
) -> dict:
    if not weight.exists():
        raise FileNotFoundError(weight)

    model = YOLO(str(weight))
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        workers=0,
        device=device,
        verbose=False,
    )

    row_idx = find_class_index(metrics, "human")
    all_ap = np.asarray(metrics.box.all_ap, dtype=float)
    human_ap = all_ap[row_idx]

    row = {
        "Method": name,
        "Run": weight.parent.parent.name,
        "Weight": str(weight.relative_to(ROOT)),
        "Split": split,
        "Human_mAP50_95": as_float(human_ap.mean()),
        "Human_AP50": as_float(human_ap[AP_INDEX["AP50"]]),
        "Human_AP75": as_float(human_ap[AP_INDEX["AP75"]]),
        "Human_AP90": as_float(human_ap[AP_INDEX["AP90"]]),
        "Human_AP95": as_float(human_ap[AP_INDEX["AP95"]]),
        "Overall_mAP50": as_float(metrics.box.map50),
        "Overall_mAP50_95": as_float(metrics.box.map),
    }
    for label, value in zip(IOU_LABELS, human_ap):
        row[f"Human_{label}"] = as_float(value)
    return row


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return "nan" if math.isnan(value) else f"{value:.4f}"


def mean_std(rows: list[dict], key: str) -> tuple[float, float]:
    arr = np.array([r[key] for r in rows], dtype=float)
    return as_float(arr.mean()), as_float(arr.std(ddof=1))


def summarize_rows(rows: list[dict]) -> list[dict]:
    out = []
    methods = []
    for row in rows:
        if row["Method"] not in methods:
            methods.append(row["Method"])
    for method in methods:
        group = [r for r in rows if r["Method"] == method]
        summary = {"Method": method, "N": len(group)}
        for key in (
            "Human_mAP50_95",
            "Human_AP50",
            "Human_AP75",
            "Human_AP90",
            "Human_AP95",
            "Overall_mAP50",
            "Overall_mAP50_95",
        ):
            mu, sd = mean_std(group, key)
            summary[f"{key}_mean"] = mu
            summary[f"{key}_std"] = sd
        out.append(summary)
    return out


def write_markdown(path: Path, data: Path, by_split: dict[str, list[dict]]) -> None:
    lines = []
    lines.append("# Human-Focused Evaluation")
    lines.append("")
    lines.append(f"Data: `{data}`")
    lines.append("")
    for split, rows in by_split.items():
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| Method | Human mAP50-95 | Human AP50 | Human AP75 | Human AP90 | Human AP95 | Overall mAP50 | Overall mAP50-95 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in summarize_rows(rows):
            lines.append(
                "| "
                + " | ".join(
                    [
                        r["Method"],
                        fmt_pm(r["Human_mAP50_95_mean"], r["Human_mAP50_95_std"], r["N"]),
                        fmt_pm(r["Human_AP50_mean"], r["Human_AP50_std"], r["N"]),
                        fmt_pm(r["Human_AP75_mean"], r["Human_AP75_std"], r["N"]),
                        fmt_pm(r["Human_AP90_mean"], r["Human_AP90_std"], r["N"]),
                        fmt_pm(r["Human_AP95_mean"], r["Human_AP95_std"], r["N"]),
                        fmt_pm(r["Overall_mAP50_mean"], r["Overall_mAP50_std"], r["N"]),
                        fmt_pm(r["Overall_mAP50_95_mean"], r["Overall_mAP50_95_std"], r["N"]),
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test", "both"], default="both")
    parser.add_argument("--data", type=Path, default=DATA, help="Dataset YAML to evaluate.")
    parser.add_argument("--output", type=Path, default=OUT_DIR, help="Directory for CSV and Markdown outputs.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto", help="Use 'auto', 'cpu', or a CUDA index such as '0'.")
    parser.add_argument("--include-missing", action="store_true")
    parser.add_argument(
        "--methods",
        default="all",
        help="Comma-separated method names to evaluate, or 'all'. Example: StarEye,StarEye+MSCAA",
    )
    return parser.parse_args()


def fmt_pm(mean: float, std: float, n: int) -> str:
    if n <= 1:
        return fmt(mean)
    return f"{fmt(mean)} +/- {fmt(std)}"


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        import torch

        args.device = "0" if torch.cuda.is_available() else "cpu"

    data = args.data.resolve()
    output_dir = args.output.resolve()
    if not data.exists():
        raise FileNotFoundError(data)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["val", "test"] if args.split == "both" else [args.split]
    by_split: dict[str, list[dict]] = {}
    wanted = None if args.methods == "all" else {x.strip() for x in args.methods.split(",") if x.strip()}

    for split in splits:
        rows = []
        print(f"\n=== Evaluating split={split} ===")
        for name, weights in MODEL_GROUPS:
            if wanted is not None and name not in wanted:
                continue
            for weight in weights:
                if not weight.exists():
                    message = f"Missing checkpoint: {weight}"
                    if args.include_missing:
                        print(message)
                        continue
                    raise FileNotFoundError(message)

                print(f"{name:<24} {weight.relative_to(ROOT)}")
                row = evaluate_one(name, weight, data, split, args.imgsz, args.batch, args.device)
                rows.append(row)
                print(
                    f"  Human mAP50-95={row['Human_mAP50_95']:.4f} "
                    f"AP75={row['Human_AP75']:.4f} "
                    f"AP95={row['Human_AP95']:.4f}"
                )

        by_split[split] = rows
        write_csv(output_dir / f"human_metrics_{split}_raw.csv", rows)
        write_csv(output_dir / f"human_metrics_{split}_summary.csv", summarize_rows(rows))

    write_markdown(output_dir / "human_metrics_summary.md", data, by_split)
    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
