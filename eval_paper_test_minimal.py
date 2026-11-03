"""Evaluate frozen StarEye checkpoints on the held-out test split.

The default scope is the minimum final-paper comparison:
  - StarEye baseline, seeds 0/42/123
  - StarEye + SAFusion, seeds 0/42/123

No training or test-time tuning is performed. Every invocation creates a new
timestamped output directory so previous validation and test results remain
untouched.

Examples:
    D:\Anaconda3\envs\Stareye\python.exe eval_paper_test_minimal.py --check-only
    D:\Anaconda3\envs\Stareye\python.exe eval_paper_test_minimal.py
    D:\Anaconda3\envs\Stareye\python.exe eval_paper_test_minimal.py --scope full
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "mydata" / "data.yaml"
SEEDS = (0, 42, 123)

MODEL_GROUPS = {
    "StarEye": (
        "Baseline",
        [ROOT / "runs" / "ablation_100ep" / f"Baseline_s{seed}" / "weights" / "best.pt"
         for seed in SEEDS],
    ),
    "StarEye+SAFusion": (
        "SAFusion_only",
        [ROOT / "runs" / "ablation_100ep" / f"SAFusion_only_s{seed}" / "weights" / "best.pt"
         for seed in SEEDS],
    ),
    "StarEye+MSCAA": (
        "MSCAA_only",
        [ROOT / "runs" / "ablation_100ep" / f"MSCAA_only_s{seed}" / "weights" / "best.pt"
         for seed in SEEDS],
    ),
    "StarEye+MSCAA+SAFusion": (
        "MSCAA_SAFusion",
        [ROOT / "runs" / "ablation_100ep" / f"MSCAA_SAFusion_s{seed}" / "weights" / "best.pt"
         for seed in SEEDS],
    ),
}

MINIMAL_METHODS = ("StarEye", "StarEye+SAFusion")
FULL_METHODS = tuple(MODEL_GROUPS)
IOU_LABELS = ("AP50", "AP55", "AP60", "AP65", "AP70", "AP75", "AP80", "AP85", "AP90", "AP95")
IOU_INDEX = {label: idx for idx, label in enumerate(IOU_LABELS)}

RAW_FIELDS = (
    "Method",
    "Condition",
    "Seed",
    "Weight",
    "Split",
    "Overall_mAP50_95",
    "Overall_mAP50",
    "Human_mAP50_95",
    "Human_AP50",
    "Human_AP75",
    "Human_AP90",
    "Human_AP95",
    "Fish_mAP50_95",
    "Fish_AP50",
    "Ultralytics_SaveDir",
)

SUMMARY_METRICS = (
    "Overall_mAP50_95",
    "Overall_mAP50",
    "Human_mAP50_95",
    "Human_AP50",
    "Human_AP75",
    "Human_AP90",
    "Human_AP95",
    "Fish_mAP50_95",
    "Fish_AP50",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("minimal", "full"), default="minimal")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--plots", action="store_true", help="Save Ultralytics plots for every checkpoint.")
    parser.add_argument("--check-only", action="store_true", help="Validate inputs and CUDA without inference.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs" / "paper_test_eval",
        help="Parent directory for timestamped test results.",
    )
    return parser.parse_args()


def selected_methods(scope: str) -> tuple[str, ...]:
    return MINIMAL_METHODS if scope == "minimal" else FULL_METHODS


def checkpoint_rows(methods: tuple[str, ...]) -> list[tuple[str, str, int, Path]]:
    rows = []
    for method in methods:
        condition, weights = MODEL_GROUPS[method]
        for seed, weight in zip(SEEDS, weights):
            rows.append((method, condition, seed, weight))
    return rows


def validate_inputs(data: Path, checkpoints: list[tuple[str, str, int, Path]], device: str) -> dict[str, Any]:
    if not data.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data}")

    missing = [weight for _, _, _, weight in checkpoints if not weight.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required checkpoints:\n{formatted}")

    import torch
    import ultralytics

    use_cuda = str(device).lower() != "cpu"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")

    cuda_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": cuda_name,
    }


def class_name_items(names: Any) -> list[tuple[int, str]]:
    if isinstance(names, dict):
        return [(int(class_id), str(name)) for class_id, name in names.items()]
    return [(idx, str(name)) for idx, name in enumerate(names)]


def class_ap(metrics: Any, target: str) -> np.ndarray:
    class_id = None
    for candidate_id, name in class_name_items(metrics.names):
        if target.lower() in name.lower():
            class_id = candidate_id
            break
    if class_id is None:
        raise RuntimeError(f"Class '{target}' was not found in names={metrics.names!r}")

    evaluated_ids = [int(value) for value in metrics.box.ap_class_index]
    if class_id not in evaluated_ids:
        raise RuntimeError(
            f"Class '{target}' (id={class_id}) has no AP row on the test split; "
            f"evaluated class ids are {evaluated_ids}."
        )
    row = evaluated_ids.index(class_id)
    values = np.asarray(metrics.box.all_ap, dtype=float)[row]
    if values.shape != (10,):
        raise RuntimeError(f"Expected 10 IoU AP values for '{target}', got shape={values.shape}")
    return values


def as_float(value: Any) -> float:
    return float(value)


def evaluate_one(
    method: str,
    condition: str,
    seed: int,
    weight: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    model = YOLO(str(weight))
    run_name = f"{condition}_s{seed}"
    metrics = model.val(
        data=str(args.data.resolve()),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        verbose=False,
        plots=args.plots,
        project=str(output_dir / "ultralytics"),
        name=run_name,
        exist_ok=False,
    )

    human = class_ap(metrics, "human")
    fish = class_ap(metrics, "fish")
    return {
        "Method": method,
        "Condition": condition,
        "Seed": seed,
        "Weight": str(weight.relative_to(ROOT)),
        "Split": "test",
        "Overall_mAP50_95": as_float(metrics.box.map),
        "Overall_mAP50": as_float(metrics.box.map50),
        "Human_mAP50_95": as_float(human.mean()),
        "Human_AP50": as_float(human[IOU_INDEX["AP50"]]),
        "Human_AP75": as_float(human[IOU_INDEX["AP75"]]),
        "Human_AP90": as_float(human[IOU_INDEX["AP90"]]),
        "Human_AP95": as_float(human[IOU_INDEX["AP95"]]),
        "Fish_mAP50_95": as_float(fish.mean()),
        "Fish_AP50": as_float(fish[IOU_INDEX["AP50"]]),
        "Ultralytics_SaveDir": str(metrics.save_dir),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_stats(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else math.nan
    return mean, std


def summarize(raw_rows: list[dict[str, Any]], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    summaries = []
    for method in methods:
        group = [row for row in raw_rows if row["Method"] == method]
        if not group:
            continue
        summary: dict[str, Any] = {"Method": method, "N": len(group)}
        for metric in SUMMARY_METRICS:
            mean, std = sample_stats([float(row[metric]) for row in group])
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std_sample"] = std
        summaries.append(summary)
    return summaries


def paired_deltas(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["Method"], int(row["Seed"])): row for row in raw_rows}
    deltas = []
    for seed in SEEDS:
        baseline = by_key.get(("StarEye", seed))
        safusion = by_key.get(("StarEye+SAFusion", seed))
        if baseline is None or safusion is None:
            continue
        row: dict[str, Any] = {"Seed": seed}
        for metric in SUMMARY_METRICS:
            row[f"Delta_{metric}"] = float(safusion[metric]) - float(baseline[metric])
        deltas.append(row)
    return deltas


def fmt_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} +/- {std:.4f}" if not math.isnan(std) else f"{mean:.4f}"


def write_markdown(
    path: Path,
    summaries: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    lines = [
        "# Frozen-Checkpoint Test Evaluation",
        "",
        f"- Data: `{config['data']}`",
        f"- Split: `test`",
        f"- Seeds: `{', '.join(map(str, SEEDS))}`",
        "- Values are mean +/- sample standard deviation across seeds.",
        "",
        "## Summary",
        "",
        "| Method | Overall mAP50-95 | Overall mAP50 | Human mAP50-95 | Fish mAP50-95 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['Method']} | "
            f"{fmt_pm(row['Overall_mAP50_95_mean'], row['Overall_mAP50_95_std_sample'])} | "
            f"{fmt_pm(row['Overall_mAP50_mean'], row['Overall_mAP50_std_sample'])} | "
            f"{fmt_pm(row['Human_mAP50_95_mean'], row['Human_mAP50_95_std_sample'])} | "
            f"{fmt_pm(row['Fish_mAP50_95_mean'], row['Fish_mAP50_95_std_sample'])} |"
        )

    if deltas:
        lines.extend([
            "",
            "## Paired SAFusion Minus Baseline",
            "",
            "| Seed | Overall mAP50-95 | Overall mAP50 | Human mAP50-95 | Fish mAP50-95 |",
            "|---:|---:|---:|---:|---:|",
        ])
        for row in deltas:
            lines.append(
                f"| {row['Seed']} | "
                f"{row['Delta_Overall_mAP50_95']:+.4f} | "
                f"{row['Delta_Overall_mAP50']:+.4f} | "
                f"{row['Delta_Human_mAP50_95']:+.4f} | "
                f"{row['Delta_Fish_mAP50_95']:+.4f} |"
            )
        lines.extend([
            "",
            "Positive-seed counts:",
            "",
            f"- Overall mAP50-95: `{sum(row['Delta_Overall_mAP50_95'] > 0 for row in deltas)}/{len(deltas)}`",
            f"- Human mAP50-95: `{sum(row['Delta_Human_mAP50_95'] > 0 for row in deltas)}/{len(deltas)}`",
            f"- Fish mAP50-95: `{sum(row['Delta_Fish_mAP50_95'] > 0 for row in deltas)}/{len(deltas)}`",
        ])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_fields() -> tuple[str, ...]:
    fields = ["Method", "N"]
    for metric in SUMMARY_METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_std_sample"))
    return tuple(fields)


def main() -> None:
    args = parse_args()
    args.data = args.data.resolve()
    args.output_root = args.output_root.resolve()
    methods = selected_methods(args.scope)
    checkpoints = checkpoint_rows(methods)
    environment = validate_inputs(args.data, checkpoints, args.device)

    print("\nPreflight passed")
    print(f"  scope       : {args.scope} ({len(checkpoints)} checkpoints)")
    print(f"  data        : {args.data}")
    print(f"  split       : test")
    print(f"  imgsz/batch : {args.imgsz}/{args.batch}")
    print(f"  device      : {args.device} ({environment['cuda_device']})")
    print(f"  torch       : {environment['torch']}")
    if args.check_only:
        print("\nCheck-only mode: no inference was run and no result directory was created.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_root / f"{args.scope}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scope": args.scope,
        "data": str(args.data),
        "split": "test",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": str(args.device),
        "workers": args.workers,
        "plots": args.plots,
        "seeds": list(SEEDS),
        "environment": environment,
        "checkpoints": [
            {"method": method, "condition": condition, "seed": seed, "weight": str(weight)}
            for method, condition, seed, weight in checkpoints
        ],
    }
    write_json(output_dir / "run_config.json", config)

    raw_rows: list[dict[str, Any]] = []
    total = len(checkpoints)
    for index, (method, condition, seed, weight) in enumerate(checkpoints, start=1):
        print(f"\n[{index}/{total}] {method}, seed={seed}")
        print(f"  weight: {weight.relative_to(ROOT)}")
        row = evaluate_one(method, condition, seed, weight, args, output_dir)
        raw_rows.append(row)
        write_csv(output_dir / "per_seed_results.csv", raw_rows, RAW_FIELDS)
        write_json(output_dir / "per_seed_results.json", raw_rows)
        print(
            f"  Overall AP={row['Overall_mAP50_95']:.4f}, "
            f"Human AP={row['Human_mAP50_95']:.4f}, "
            f"Fish AP={row['Fish_mAP50_95']:.4f}"
        )

    summaries = summarize(raw_rows, methods)
    deltas = paired_deltas(raw_rows)
    write_csv(output_dir / "summary_results.csv", summaries, summary_fields())
    write_json(output_dir / "paired_deltas.json", deltas)
    write_markdown(output_dir / "TEST_RESULTS.md", summaries, deltas, config)

    print("\nTest evaluation complete")
    print(f"  Results: {output_dir}")
    print(f"  Read first: {output_dir / 'TEST_RESULTS.md'}")


if __name__ == "__main__":
    main()
