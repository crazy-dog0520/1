"""Smoke-test a forward pass for each of the four saved ablation models."""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "ablation_100ep"
CONDITIONS = (
    ("StarEye", "Baseline"),
    ("StarEye + MSCAA", "MSCAA_only"),
    ("StarEye + SAFusion", "SAFusion_only"),
    ("StarEye + MSCAA + SAFusion", "MSCAA_SAFusion"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(0, 42, 123), default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto", help="auto, cpu, or a torch device such as cuda:0")
    parser.add_argument("--check-only", action="store_true", help="Check paths without inference.")
    args = parser.parse_args()

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    checkpoints = [(name, RUNS / f"{folder}_s{args.seed}" / "weights" / "best.pt")
                   for name, folder in CONDITIONS]
    missing = [path for _, path in checkpoints if not path.is_file()]
    if missing:
        parser.error(f"Missing checkpoint: {missing[0]}")
    if args.check_only:
        print(f"Preflight OK: {len(checkpoints)} checkpoints for seed {args.seed}")
        return

    device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(f"Forward test: seed={args.seed}, input=1x3x{args.imgsz}x{args.imgsz}, device={device}")
    for name, path in checkpoints:
        network = YOLO(str(path), task="detect").model.to(device).eval()
        sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
        with torch.inference_mode():
            output = network(sample)
        if output is None:
            raise RuntimeError(f"No output from {path}")
        print(f"{name:<30} OK ({type(output).__name__})")


if __name__ == "__main__":
    main()
