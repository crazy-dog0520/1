# Frozen-Checkpoint Test Evaluation

- Data: `D:\Deep_learning\StarEye improve\mydata\data.yaml`
- Split: `test`
- Seeds: `0, 42, 123`
- Values are mean +/- sample standard deviation across seeds.

## Summary

| Method | Overall mAP50-95 | Overall mAP50 | Human mAP50-95 | Fish mAP50-95 |
|---|---:|---:|---:|---:|
| StarEye | 0.4276 +/- 0.0147 | 0.5854 +/- 0.0337 | 0.7626 +/- 0.0087 | 0.0925 +/- 0.0330 |
| StarEye+SAFusion | 0.4275 +/- 0.0095 | 0.5675 +/- 0.0057 | 0.7733 +/- 0.0099 | 0.0817 +/- 0.0092 |

## Paired SAFusion Minus Baseline

| Seed | Overall mAP50-95 | Overall mAP50 | Human mAP50-95 | Fish mAP50-95 |
|---:|---:|---:|---:|---:|
| 0 | -0.0178 | -0.0499 | +0.0160 | -0.0516 |
| 42 | +0.0077 | +0.0017 | +0.0040 | +0.0113 |
| 123 | +0.0100 | -0.0053 | +0.0121 | +0.0079 |

Positive-seed counts:

- Overall mAP50-95: `2/3`
- Human mAP50-95: `3/3`
- Fish mAP50-95: `2/3`
