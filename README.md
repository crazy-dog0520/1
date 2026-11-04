# Four-condition StarEye experiment: delivery guide

This folder contains the code, data, and checkpoints needed for the core
100-epoch, three-seed comparison. It uses the included modified `ultralytics/` source,
not a pip-installed stock Ultralytics release. Run commands from this folder.

## Experiment files

| Condition | Model configuration | Saved checkpoints |
| --- | --- | --- |
| StarEye baseline | `models/StarEye.yaml` | `runs/ablation_100ep/Baseline_s{0,42,123}/weights/best.pt` |
| MSCAA only | `models/StarEye_MSCAA.yaml` | `runs/ablation_100ep/MSCAA_only_s{0,42,123}/weights/best.pt` |
| SAFusion only | `models/StarEye_CAA_SANeck.yaml` | `runs/ablation_100ep/SAFusion_only_s{0,42,123}/weights/best.pt` |
| MSCAA + SAFusion | `models/StarEye_MSCAA_SANeck.yaml` | `runs/ablation_100ep/MSCAA_SAFusion_s{0,42,123}/weights/best.pt` |

`train_ablation_100ep.py` is the matching training entry point. The four YAMLs
now declare the two classes used in this copy; the historical checkpoints were
also trained/evaluated with two classes (the trainer overrode the older YAML
`nc: 5` from the dataset at training time). Do not mix this 100-epoch protocol
with the separate historical 150-epoch `train_ablation_3seed.py` protocol.

## Dataset and provenance

The actual dataset is `mydata/`, configured by `mydata/data.yaml`:
`fish` (class 0) and `human` (class 1). The image/label counts are 5630/5630
train, 531/531 validation (`valid/`), and 266/266 test. Paths in the YAML are
relative to the YAML location, so this copy does not read the original project.

The upstream source(s), license, and collection/merging history of these images
are not established by the files in this copy. Do not describe them as the
original StarEye paper's private dataset or claim a verified public source.
Before external redistribution, confirm dataset rights and provenance.

## Environment

The saved runs used Python 3.9.25 and the included modified Ultralytics 8.2.50
source. This machine has PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128 with
CUDA available, but its existing `Stareye` conda environment currently lacks
some project packages (including `timm`). Do not assume that environment, or
the bundled `.codex_deps/` cache, is a complete portable environment.

For a fresh Windows machine with an NVIDIA GPU compatible with the CUDA 12.8
PyTorch wheel, run the following in an Anaconda PowerShell Prompt from this
folder. PyTorch and torchvision are installed separately because their CUDA
builds must match; `requirements.txt` covers the remaining core dependencies.

```powershell
conda create -n stareye-delivery python=3.9.25 -y
conda activate stareye-delivery
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch, timm, thop, psutil, seaborn, ultralytics; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print('ultralytics:', ultralytics.__file__)"
python -B train_ablation_100ep.py --check-only
```

The import check must print this folder's `ultralytics/__init__.py`, not a
site-packages copy. The training and ablation-evaluation scripts specify CUDA
device 0, so their supplied commands need an available NVIDIA GPU; `test.py`
can instead run a CPU forward check with `--device cpu`. On a different GPU or
driver, select a compatible PyTorch build using the official installation
instructions. Installing these dependencies does not guarantee identical
floating-point results across hardware. The old `thop==2.0.20` requirement
was an incorrect package/version combination; `requirements.txt` now uses the
available `thop==0.1.1.post2209072238` package. Its `timm==1.0.26` pin is
valid, while the bundled temporary `.codex_deps/` cache happens to contain
`timm` 1.0.28; that cache is not evidence of the historical training version.

## Preflight, evaluation, and reproduction

```powershell
cd 'D:\Deep_learning\StarEye improve - 1'
python -B train_ablation_100ep.py --check-only
python -B eval_paper_test_minimal.py --scope full --check-only
python -B test.py
python -B get_gflops.py
python -B eval_ablation_100ep.py
python -B eval_ablation_100ep_test.py
python -B eval_human_table.py --split val
```

The two `eval_ablation_100ep*` scripts evaluate all 12 saved checkpoints on
validation/test respectively and print overall AP50, overall AP50-95, fish
AP50-95, and human AP50-95. Their `+/-` uses sample SD (`ddof=1`) over three
seeds. `eval_paper_test_minimal.py --scope full` produces timestamped test
records under `runs/paper_test_eval/`; `--check-only` performs no inference.
`test.py` runs a dummy forward pass for all four seed-0 checkpoints, and
`get_gflops.py` reports their fused-inference parameters and GFLOPs using the
local Ultralytics/THOP convention. Both accept `--seed 0|42|123` and
`--check-only`; `get_gflops.py` also accepts `--imgsz`.
`eval_human_table.py` includes only the four present configurations. Its
default and `--methods all` both select those four and write new output under
`runs/human_metrics_table_reproduction/`.

To retrain, run `python -B train_ablation_100ep.py`. It writes to
`runs/ablation_100ep_reproduction/` by default and refuses an existing run
name. Use `--project` for another new output directory. It does not resume or
overwrite the 12 supplied runs. Full retraining was not part of this delivery
check.

## Qualitative figure reproduction

`paper/find_joint_qualitative_replacement.py` and
`paper/find_joint_fish_candidate.py` select the human and fish examples.
Their prediction helper first reads
`runs/paper_visuals_joint_replacement_20260901/baseline_s0_predictions.json`
and `joint_mscaa_safusion_s0_predictions.json` when those files exist.
Running the scripts with these caches present reselects examples from saved
predictions; it does **not** run the Baseline and MSCAA+SAFusion checkpoints.

To reproduce predictions from the supplied weights, first preserve both JSON
caches under different names, leaving the cache directory in place. Run
`python -B paper/find_joint_qualitative_replacement.py` to regenerate both
caches and the human selection, then run
`python -B paper/find_joint_fish_candidate.py` for the fish selection. Finally,
run `python -B paper/compose_joint_qualitative_vector.py` to create
`picture/joint_qualitative.{svg,png,pdf}`. The two selection previews also go
to `picture/`; the regenerated prediction and selection JSON files remain
under `runs/paper_visuals_joint_replacement_20260901/`. This sequence rewrites
the selection records and picture outputs, so archive those too if the previous
selection must be retained. The vector composer additionally needs PyMuPDF
(`python -m pip install PyMuPDF==1.26.5`); it is not needed for the four-condition
training and evaluation. The optional legacy `heatmap.py` needs `grad-cam`
separately and is not part of the core reproduction commands.

## Saved evidence and interpretation

`runs/mscaa_safusion_pilot_v1/multiseed_comparison.json` contains the saved
validation results for all four conditions and three seeds: overall AP50-95,
overall AP50, human AP50-95, and fish AP50-95, with checkpoint paths. The CSV
in that directory contains the per-seed records; the Markdown file summarizes
them using sample SD (`ddof=1`). The directory name is historical: the
PreMSCAA-SAFusion placement experiment has been removed from these three files.
`runs/human_metrics_table/human_metrics_val_raw.csv` contains additional
per-seed human AP thresholds. Its historical summary uses population SD
(`ddof=0`), so do not compare the displayed SDs without noting the convention.
Saved records retain absolute paths from the original machine; new evaluations
use this copy's paths and sample SD.

`runs/paper_test_eval/minimal_20260901_103252/` is an earlier test-set
evaluation of **baseline and SAFusion only**. It does not contain MSCAA-only or
joint test results, so it cannot support a four-condition test-set comparison.
The copied test files also retain their original absolute path strings.

The saved 100-epoch validation overall AP means are baseline 0.6143, MSCAA
0.6206, SAFusion 0.6345, and joint 0.6300. The joint mean is above baseline
but below SAFusion alone. The saved human AP50-95 means are 0.7764, 0.7742,
0.7732, and 0.7712 in the same order. These records do not support a claim
that the joint model has the best overall or human AP; reconcile any paper
table or prose against raw records before sending it to a teacher or journal.

## Inherited StarEye runners

The five inherited StarEye-paper runners are `train.py`, `detect.py`, `val.py`,
`get_FPS.py`, and `heatmap.py`. They are baseline utilities, **not** the
controlled four-condition protocol above. `train.py` is the older 350-epoch
runner and still has its original paths; do not use it to reproduce this
paper's 100-epoch baseline. The other four now default to this copy's
`runs/ablation_100ep/Baseline_s0/weights/best.pt` and local dataset paths.
That checkpoint is from this experiment, not the original paper's weights.
`heatmap.py` additionally needs `pytorch_grad_cam`, which was not available in
the checked environment; its output directory is never silently replaced.

The remaining `train_yolo_compare.py` and `train_compare_extra.py` are optional
external-detector experiments with old absolute data paths. `main.py` also
points to a missing root-level YAML. Do not use these as four-condition entry
points. Historical `runs/*/args.yaml` files retain the original absolute
paths as evidence; use the commands above for reproduction.
