"""Report fused-inference parameter counts and GFLOPs for four ablation models."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


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
    parser.add_argument("--check-only", action="store_true", help="Check paths without profiling.")
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

    print(f"Fused inference; input: 1 x 3 x {args.imgsz} x {args.imgsz}; GFLOPs = 2 x THOP MACs")
    for name, path in checkpoints:
        network = YOLO(str(path), task="detect").model.fuse(verbose=False).eval()
        params_m = sum(p.numel() for p in network.parameters()) / 1e6
        gflops = get_flops(network, imgsz=args.imgsz)
        if gflops <= 0:
            raise RuntimeError(f"FLOP profiling failed for {path}; check that thop is installed")
        print(f"{name:<30} {params_m:7.3f} M params   {gflops:7.2f} GFLOPs")


if __name__ == "__main__":
    main()
