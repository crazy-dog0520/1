# Four-Condition Three-Seed Comparison

All checkpoints were re-evaluated with the same Ultralytics evaluator on the validation split.
Values are mean +/- sample standard deviation over seeds 0, 42, and 123.

## Three-Seed Summary

| Method | Overall AP | AP50 | Human AP | Fish AP |
|---|---:|---:|---:|---:|
| StarEye | 0.6143 +/- 0.0304 | 0.9080 +/- 0.0307 | 0.7764 +/- 0.0072 | 0.4521 +/- 0.0551 |
| MSCAA only | 0.6206 +/- 0.0128 | 0.9186 +/- 0.0128 | 0.7742 +/- 0.0070 | 0.4670 +/- 0.0223 |
| SAFusion only | 0.6345 +/- 0.0140 | 0.9320 +/- 0.0059 | 0.7732 +/- 0.0146 | 0.4958 +/- 0.0289 |
| Post-fusion MSCAA+SAFusion | 0.6300 +/- 0.0146 | 0.9268 +/- 0.0111 | 0.7712 +/- 0.0152 | 0.4888 +/- 0.0328 |

## Raw Per-Seed Results

| Method | Seed | Overall AP | AP50 | Human AP | Fish AP |
|---|---:|---:|---:|---:|---:|
| StarEye | 0 | 0.5887 | 0.8725 | 0.7681 | 0.4093 |
| StarEye | 42 | 0.6480 | 0.9261 | 0.7816 | 0.5143 |
| StarEye | 123 | 0.6061 | 0.9253 | 0.7795 | 0.4328 |
| MSCAA only | 0 | 0.6247 | 0.9306 | 0.7692 | 0.4802 |
| MSCAA only | 42 | 0.6308 | 0.9051 | 0.7822 | 0.4794 |
| MSCAA only | 123 | 0.6062 | 0.9201 | 0.7711 | 0.4413 |
| SAFusion only | 0 | 0.6195 | 0.9252 | 0.7763 | 0.4628 |
| SAFusion only | 42 | 0.6367 | 0.9353 | 0.7573 | 0.5162 |
| SAFusion only | 123 | 0.6472 | 0.9356 | 0.7860 | 0.5084 |
| Post-fusion MSCAA+SAFusion | 0 | 0.6144 | 0.9237 | 0.7777 | 0.4511 |
| Post-fusion MSCAA+SAFusion | 42 | 0.6432 | 0.9392 | 0.7820 | 0.5044 |
| Post-fusion MSCAA+SAFusion | 123 | 0.6324 | 0.9176 | 0.7538 | 0.5109 |
