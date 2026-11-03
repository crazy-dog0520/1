# StarEye / Joint inference efficiency

Status: COMPLETE.

Hardware: NVIDIA GeForce RTX 4060 Laptop GPU. Backend: PyTorch CUDA eager; precision: fp32; batch=1; fixed 640 x 640; native Conv/BN fusion enabled.

Forward: GPU-resident real-image tensor -> native detector output (including box decoding). Excludes preprocessing, transfers, NMS and CPU result conversion.

Pipeline: predecoded BGR CPU image -> fixed-square letterbox, RGB/CHW, host-to-device transfer, normalization, detector forward, class-aware NMS, box rescaling, CPU numpy detections. Excludes file reading/decoding, camera acquisition, drawing, saving and high-level predict callbacks.

NMS: conf=0.25, IoU=0.7, max_det=300, multi_label=False, agnostic=False. These are timing settings, not AP evaluation settings.

5 rounds; 100 warmups + 1000 measured frames per mode/round; 100 real images, cycled in the same order for both models. Fresh process per model/round; model order and measurement-mode order alternate.

Latency uses perf_counter with CUDA synchronization before and after each frame. Mean +/- SD uses round mean latencies, with sample SD (rounds-1); P50/P95 use pooled measured frames. FPS uses frames / total serial loop wall time (including loop/synchronization overhead), not mean instantaneous FPS.

Peak memory is maximum torch.cuda.max_memory_allocated across rounds, including weights/input/live output tensors. It excludes cached reserved memory, CUDA context and other processes; the forward resident tensor is retained in both modes. It is not whole-device nvidia-smi usage.

| Model | Scope | Rounds | Latency mean +/- sample SD (ms) | P50 (ms) | P95 (ms) | Serial FPS | Peak allocated (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|
| StarEye | forward | 5 | 6.546 +/- 0.069 | 6.428 | 7.469 | 152.40 | 95.6 |
| StarEye | pipeline | 5 | 8.947 +/- 0.123 | 8.871 | 9.954 | 111.69 | 98.7 |
| Joint | forward | 5 | 7.487 +/- 0.113 | 7.373 | 8.500 | 133.30 | 98.2 |
| Joint | pipeline | 5 | 9.936 +/- 0.168 | 9.799 | 11.219 | 100.59 | 104.4 |

Joint minus StarEye, forward: +0.940 ms (+14.36% latency).

Joint minus StarEye, pipeline: +0.989 ms (+11.05% latency).

These measurements characterize the recorded hardware/software conditions. They do not validate underwater robot or embedded-device deployment.
