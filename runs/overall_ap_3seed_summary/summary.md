# Four configurations x three seeds: validation overall AP

Metric: overall mAP50:95, in the 0-1 scale. Joint = StarEye + MSCAA + SAFusion (the paper_0912 fusion-to-refinement architecture, runs/ablation_100ep/MSCAA_SAFusion_s*/weights/best.pt).

Source: runs/premscaa_safusion_pilot_v1/multiseed_comparison.json; all 12 values exactly match runs/human_metrics_table/human_metrics_val_raw.csv.

Each checkpoint is the native-fitness-selected best.pt. These are saved unified re-evaluation results, not a new evaluation or training-log maxima. Native fitness = 0.1 * AP50 + 0.9 * mAP50:95; only mAP50:95 is reported below.

Shared saved evaluation protocol: mydata/data.yaml, split=val, imgsz=640, batch=16, workers=0, device=0, native Ultralytics evaluator. The source records all checkpoint paths.

Mean and sample SD are computed from full-precision source values; sample SD uses n-1=2. Display rounding only: four decimal places.

GFLOPs were profiled from the supplied `best.pt` checkpoints with `get_gflops.py --imgsz 640`: fused inference, input 1 x 3 x 640 x 640, GFLOPs = 2 x THOP MACs. All three seeds give the same displayed GFLOPs within each configuration; this is model complexity, not a three-seed AP statistic.

| Configuration | Seed 0 | Seed 42 | Seed 123 | Mean +/- SD | GFLOPs |
|---|---:|---:|---:|---:|---:|
| StarEye | 0.5887 | 0.6480 | 0.6061 | 0.6143 +/- 0.0304 | 24.64 |
| StarEye + MSCAA | 0.6247 | 0.6308 | 0.6062 | 0.6206 +/- 0.0128 | 25.62 |
| StarEye + SAFusion | 0.6195 | 0.6367 | 0.6472 | 0.6345 +/- 0.0140 | 24.65 |
| Joint | 0.6144 | 0.6432 | 0.6324 | 0.6300 +/- 0.0146 | 25.63 |



