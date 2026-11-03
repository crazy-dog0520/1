"""Batch-1 CUDA inference benchmark. Run --help or read INFERENCE_BENCHMARK.md."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED0 = ROOT / "runs" / "ablation_100ep"
MODES = ("forward", "pipeline")
SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=SEED0 / "Baseline_s0/weights/best.pt")
    parser.add_argument("--joint", type=Path, default=SEED0 / "MSCAA_SAFusion_s0/weights/best.pt")
    parser.add_argument("--images", type=Path, default=ROOT / "mydata/valid/images")
    parser.add_argument("--imgsz", type=int, default=640, help="Fixed square tensor; batch is always 1")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index; no CPU fallback")
    parser.add_argument("--warmup", type=int, default=100, help="Per model, mode, and round")
    parser.add_argument("--iterations", type=int, default=1000, help="Timed frames per mode and round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--image-count", type=int, default=100, help="Predecoded real images; 0 selects all")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--threads", type=int, default=1, help="CPU threads for torch and OpenCV")
    parser.add_argument("--output", type=Path, help="New directory only; defaults to a timestamped run")
    parser.add_argument("--check", action="store_true", help="Check files/imports/CUDA; no model inference")
    parser.add_argument("--summarize", type=Path, help="Rebuild a run's summaries without inference")
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", choices=("StarEye", "Joint"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-round", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,temperature.gpu,power.draw,"
             "power.limit,clocks.current.sm,memory.used,utilization.gpu", "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"unavailable": str(error)}


def runtime_modules(settings_dir):
    # Keep local custom modules and project binary packages; do not install/download anything.
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    import ultralytics
    import cv2
    import numpy as np
    import torch
    import torchvision
    from ultralytics.data.augment import LetterBox
    from ultralytics.nn.tasks import attempt_load_weights
    from ultralytics.utils import ops
    return ultralytics, cv2, np, torch, torchvision, LetterBox, attempt_load_weights, ops


def validate_inputs(args):
    for name in ("warmup", "iterations", "rounds", "max_det", "threads"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.imgsz < 32 or args.imgsz % 32:
        raise ValueError("--imgsz must be a positive multiple of 32")
    if args.image_count < 0 or args.device < 0:
        raise ValueError("--image-count and --device must be nonnegative")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be in [0, 1]")
    for weight in (args.baseline, args.joint):
        if not weight.is_file() or weight.suffix != ".pt":
            raise FileNotFoundError(f"Local .pt checkpoint required: {weight}")
    if args.baseline.resolve() == args.joint.resolve():
        raise ValueError("Baseline and Joint must be different checkpoint files")
    if not args.images.is_dir():
        raise FileNotFoundError(args.images)
    images = sorted(p.resolve() for p in args.images.rglob("*") if p.is_file() and p.suffix.lower() in SUFFIXES)
    if not images:
        raise ValueError(f"No images found in {args.images}")
    if args.image_count and len(images) > args.image_count:
        images = sorted(random.Random(args.sample_seed).sample(images, args.image_count))
    return images


def run_worker(args):
    config = json.loads(args.worker_config.read_text(encoding="utf-8"))
    out = args.worker_config.parent
    ultra, cv2, np, torch, vision, LetterBox, load_weights, ops = runtime_modules(out / "ultralytics_settings")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this benchmark does not fall back to CPU")
    device = torch.device(f"cuda:{config['device']}")
    torch.cuda.set_device(device)
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(config["threads"])
    cv2.setNumThreads(config["threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["sample_seed"])
    before = gpu_snapshot()

    # Decoding and model load happen before any timed operation.
    images = []
    for item in config["image_manifest"]:
        image_path = Path(item["path"])
        if sha256(image_path) != item["sha256"]:
            raise RuntimeError(f"Image changed after the manifest was written: {image_path}")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot decode image: {image_path}")
        images.append(image)
    weight_info = config["models"][args.worker_model]
    if sha256(Path(weight_info["path"])) != weight_info["sha256"]:
        raise RuntimeError("Checkpoint changed after the manifest was written")
    model = load_weights(weight_info["path"], device=device, fuse=True).eval()
    model.requires_grad_(False)
    model.half() if config["precision"] == "fp16" else model.float()
    stride = int(model.stride.max())
    if config["imgsz"] % stride:
        raise ValueError(f"Input size must be divisible by model stride {stride}")
    letterbox = LetterBox(new_shape=(config["imgsz"], config["imgsz"]), auto=False, stride=stride)

    def preprocess(image):
        padded = letterbox(image=image)
        array = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1)[None])
        tensor = torch.from_numpy(array).to(device)
        tensor = tensor.half() if config["precision"] == "fp16" else tensor.float()
        return tensor.div_(255.0)

    def pipeline(image):
        tensor = preprocess(image)
        prediction = model(tensor, augment=False, visualize=False)
        detections = ops.non_max_suppression(
            prediction, conf_thres=config["conf"], iou_thres=config["iou"],
            max_det=config["max_det"], nc=len(model.names), agnostic=False, multi_label=False,
        )[0]
        detections[:, :4] = ops.scale_boxes(tensor.shape[2:], detections[:, :4], image.shape)
        # The pipeline ends with usable CPU detections; includes the return transfer.
        return detections.cpu().numpy()

    environment = {
        "python": sys.version, "python_executable": sys.executable, "os": platform.platform(),
        "cpu": platform.processor(), "torch": torch.__version__, "torchvision": vision.__version__,
        "opencv": cv2.__version__, "numpy": np.__version__, "ultralytics": ultra.__version__,
        "ultralytics_file": ultra.__file__, "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        "tf32_matmul": False, "tf32_cudnn": False, "cudnn_benchmark": False,
        "cudnn_deterministic": True, "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(), "opencv_threads": cv2.getNumThreads(),
        "model_stride": stride, "parameter_count_after_fusion": sum(p.numel() for p in model.parameters()),
        "class_names": model.names, "model_training": model.training,
        "model_dtype": str(next(model.parameters()).dtype),
    }
    measured = []
    raw = []
    with torch.inference_mode():
        # Forward-only uses one real-image tensor already resident on the GPU.
        resident = preprocess(images[0])
        assert tuple(resident.shape) == (1, 3, config["imgsz"], config["imgsz"])
        mode_order = MODES if args.worker_round % 2 else tuple(reversed(MODES))
        for mode in mode_order:
            print(f"[{args.worker_model}, round {args.worker_round}, {mode}] warmup={config['warmup']}", flush=True)
            for index in range(config["warmup"]):
                result = model(resident, augment=False, visualize=False) if mode == "forward" else pipeline(images[index % len(images)])
                del result
            torch.cuda.synchronize(device)
            # Preserve warmed allocator state; peak allocated is the meaningful model/tensor metric.
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
            latencies = []
            counts = []
            loop_start = time.perf_counter()
            for index in range(config["iterations"]):
                image_index = index % len(images)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                result = model(resident, augment=False, visualize=False) if mode == "forward" else pipeline(images[image_index])
                torch.cuda.synchronize(device)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                latencies.append(elapsed_ms)
                counts.append(None if mode == "forward" else len(result))
                del result
            loop_seconds = time.perf_counter() - loop_start
            assert all(math.isfinite(t) and t > 0 for t in latencies)
            row = {
                "model": args.worker_model, "mode": mode, "round": args.worker_round,
                "frames": len(latencies), "mean_latency_ms": statistics.mean(latencies),
                "p50_latency_ms": percentile(latencies, 0.5), "p95_latency_ms": percentile(latencies, 0.95),
                "serial_loop_seconds": loop_seconds, "throughput_fps": len(latencies) / loop_seconds,
                "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
                "allocated_before_mib": allocated_before / 2**20,
                "latencies_ms": latencies,
            }
            measured.append(row)
            raw.extend({"model": args.worker_model, "mode": mode, "round": args.worker_round,
                        "iteration": i + 1, "image_index": 0 if mode == "forward" else i % len(images),
                        "latency_ms": value, "detections": counts[i]}
                       for i, value in enumerate(latencies))
            print(f"  mean={row['mean_latency_ms']:.3f} ms, P95={row['p95_latency_ms']:.3f} ms, "
                  f"serial FPS={row['throughput_fps']:.2f}, peak allocated={row['peak_allocated_mib']:.1f} MiB", flush=True)
    stem = f"{args.worker_model}_round{args.worker_round:02d}"
    write_csv(out / f"{stem}_samples.csv", raw)
    write_json(out / f"{stem}.json", {"environment": environment, "gpu_before": before,
                                      "gpu_after": gpu_snapshot(), "measurements": measured})


def summarize(out):
    config = json.loads((out / "protocol.json").read_text(encoding="utf-8"))
    jobs = [json.loads(p.read_text(encoding="utf-8"))
            for label in ("StarEye", "Joint") for p in sorted(out.glob(f"{label}_round[0-9][0-9]*.json"))]
    measurements = [row for job in jobs for row in job["measurements"]]
    if not measurements:
        print(f"No completed worker records yet: {out}")
        return
    summary = []
    for label in ("StarEye", "Joint"):
        for mode in MODES:
            rows = sorted((r for r in measurements if r["model"] == label and r["mode"] == mode), key=lambda r: r["round"])
            if not rows:
                continue
            means = [r["mean_latency_ms"] for r in rows]
            rates = [r["throughput_fps"] for r in rows]
            pooled = [value for r in rows for value in r["latencies_ms"]]
            summary.append({
                "model": label, "mode": mode, "completed_rounds": len(rows),
                "latency_mean_ms": statistics.mean(means),
                "latency_sample_sd_ms": statistics.stdev(means) if len(means) > 1 else None,
                "pooled_p50_ms": percentile(pooled, 0.5), "pooled_p95_ms": percentile(pooled, 0.95),
                "throughput_mean_fps": statistics.mean(rates),
                "throughput_sample_sd_fps": statistics.stdev(rates) if len(rates) > 1 else None,
                "peak_allocated_mib": max(r["peak_allocated_mib"] for r in rows),
                "total_frames": len(pooled),
            })
    complete = len(summary) == 4 and all(r["completed_rounds"] == config["rounds"] for r in summary)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "rounds.csv", [{k: v for k, v in r.items() if k != "latencies_ms"} for r in measurements])
    write_json(out / "summary.json", {"complete": complete, "summary": summary})
    gpu_names = sorted({j["environment"]["gpu"] for j in jobs})
    lines = ["# StarEye / Joint inference efficiency", "",
             f"Status: {'COMPLETE' if complete else 'PARTIAL; do not report as the final experiment'}.", "",
             f"Hardware: {', '.join(gpu_names)}. Backend: PyTorch CUDA eager; precision: {config['precision']}; "
             f"batch=1; fixed {config['imgsz']} x {config['imgsz']}; native Conv/BN fusion enabled.", "",
             "Forward: GPU-resident real-image tensor -> native detector output (including box decoding). "
             "Excludes preprocessing, transfers, NMS and CPU result conversion.", "",
             "Pipeline: predecoded BGR CPU image -> fixed-square letterbox, RGB/CHW, host-to-device transfer, "
             "normalization, detector forward, class-aware NMS, box rescaling, CPU numpy detections. "
             "Excludes file reading/decoding, camera acquisition, drawing, saving and high-level predict callbacks.", "",
             f"NMS: conf={config['conf']}, IoU={config['iou']}, max_det={config['max_det']}, "
             "multi_label=False, agnostic=False. These are timing settings, not AP evaluation settings.", "",
             f"{config['rounds']} rounds; {config['warmup']} warmups + {config['iterations']} measured frames per mode/round; "
             f"{len(config['image_manifest'])} real images, cycled in the same order for both models. "
             "Fresh process per model/round; model order and measurement-mode order alternate.", "",
             "Latency uses perf_counter with CUDA synchronization before and after each frame. Mean +/- SD uses "
             "round mean latencies, with sample SD (rounds-1); P50/P95 use pooled measured frames. "
             "FPS uses frames / total serial loop wall time (including loop/synchronization overhead), not mean instantaneous FPS.", "",
             "Peak memory is maximum torch.cuda.max_memory_allocated across rounds, including weights/input/live output tensors. "
             "It excludes cached reserved memory, CUDA context and other processes; the forward resident tensor is retained "
             "in both modes. It is not whole-device nvidia-smi usage.", "",
             "| Model | Scope | Rounds | Latency mean +/- sample SD (ms) | P50 (ms) | P95 (ms) | Serial FPS | Peak allocated (MiB) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        sd = "N/A" if row["latency_sample_sd_ms"] is None else f"{row['latency_sample_sd_ms']:.3f}"
        lines.append(f"| {row['model']} | {row['mode']} | {row['completed_rounds']} | "
                     f"{row['latency_mean_ms']:.3f} +/- {sd} | {row['pooled_p50_ms']:.3f} | "
                     f"{row['pooled_p95_ms']:.3f} | {row['throughput_mean_fps']:.2f} | {row['peak_allocated_mib']:.1f} |")
    if complete:
        for mode in MODES:
            values = {r["model"]: r for r in summary if r["mode"] == mode}
            delta = values["Joint"]["latency_mean_ms"] - values["StarEye"]["latency_mean_ms"]
            percent = 100 * delta / values["StarEye"]["latency_mean_ms"]
            lines += ["", f"Joint minus StarEye, {mode}: {delta:+.3f} ms ({percent:+.2f}% latency)."]
    lines += ["", "These measurements characterize the recorded hardware/software conditions. "
              "They do not validate underwater robot or embedded-device deployment.", ""]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[-(len(summary) + 9):]), flush=True)
    print(f"Results: {out}", flush=True)


def main():
    args = parse_args()
    if args.worker_config:
        run_worker(args)
        return
    if args.summarize:
        summarize(args.summarize.resolve())
        return
    images = validate_inputs(args)
    if args.check:
        _, cv2, _, torch, _, _, _, _ = runtime_modules(ROOT / "runs/inference_efficiency/ultralytics_settings")
        if not torch.cuda.is_available() or args.device >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {args.device} is not available")
        if cv2.imread(str(images[0])) is None:
            raise ValueError(f"Cannot decode {images[0]}")
        print(f"CHECK PASSED. Python: {sys.executable}; torch={torch.__version__}; CUDA build={torch.version.cuda}")
        print(f"Baseline: {args.baseline.resolve()}\nJoint: {args.joint.resolve()}\nImages: {len(images)}")
        print("No checkpoint was loaded and no model inference was performed.")
        return
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = (args.output or ROOT / "runs/inference_efficiency" / f"{stamp}_{args.precision}").resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = {k: getattr(args, k) for k in ("imgsz", "precision", "device", "warmup", "iterations", "rounds",
                                            "image_count", "sample_seed", "conf", "iou", "max_det", "threads")}
    config.update({
        "batch": 1, "backend": "PyTorch CUDA eager", "fusion": True, "created_at_asia_shanghai": stamp,
        "script_sha256": sha256(Path(__file__)), "image_directory": str(args.images.resolve()),
        "models": {label: {"path": str(path.resolve()), "sha256": sha256(path)}
                   for label, path in (("StarEye", args.baseline), ("Joint", args.joint))},
        "image_manifest": [{"path": str(path), "sha256": sha256(path)} for path in images],
    })
    write_json(out / "protocol.json", config)
    print(f"Output: {out}\nDo not run other GPU workloads during this benchmark.", flush=True)
    try:
        for round_index in range(1, args.rounds + 1):
            order = ("StarEye", "Joint") if round_index % 2 else ("Joint", "StarEye")
            for label in order:
                command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker-config",
                           str(out / "protocol.json"), "--worker-model", label, "--worker-round", str(round_index)]
                log = out / f"{label}_round{round_index:02d}.log"
                with log.open("w", encoding="utf-8") as handle:
                    process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                               env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            handle.write(line)
                            handle.flush()
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        process.wait()
                        raise
                if code:
                    raise RuntimeError(f"Worker failed (exit={code}); see {log}")
                summarize(out)
    except BaseException:
        print(f"Incomplete run kept at {out}; completed rounds can be summarized with --summarize.", flush=True)
        raise
    print("BENCHMARK COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
