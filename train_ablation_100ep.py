"""
4 ablation conditions × 3 seeds × 100 epochs — sequential batch training.

Original training protocol (100 epoch) + reviewer Issue 4 (3 seeds, mean±std).
Separate output dir from the 150-epoch run — does NOT overwrite it.

Run:   python train_ablation_100ep.py
Then:  python eval_ablation_100ep.py
"""
import warnings
warnings.filterwarnings('ignore')
import argparse
import time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'mydata' / 'data.yaml'
OUT = ROOT / 'runs' / 'ablation_100ep_reproduction'

# All four conditions get 3 seeds → every Table II row has mean±std
MODELS = {
    'Baseline':        ROOT / 'models' / 'StarEye.yaml',
    'SAFusion_only':   ROOT / 'models' / 'StarEye_CAA_SANeck.yaml',
    'MSCAA_only':      ROOT / 'models' / 'StarEye_MSCAA.yaml',
    'MSCAA_SAFusion':  ROOT / 'models' / 'StarEye_MSCAA_SANeck.yaml',
}
SEEDS  = [0, 42, 123]
EPOCHS = 100

TRAIN_KWARGS = dict(
    data            = str(DATA),
    epochs          = EPOCHS,
    imgsz           = 640,
    batch           = 8,
    workers         = 0,
    device          = '0',
    optimizer       = 'SGD',
    lr0             = 0.01,
    lrf             = 0.01,
    momentum        = 0.937,
    weight_decay    = 5e-4,
    warmup_epochs   = 3,
    warmup_momentum = 0.8,
    mosaic          = 1.0,
    fliplr          = 0.5,
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    close_mosaic    = 10,
    cache           = False,
    exist_ok        = False,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the four-condition, three-seed 100-epoch ablation.')
    parser.add_argument('--project', type=Path, default=OUT,
                        help='New output directory; existing run names are never reused.')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate inputs and planned output names without training.')
    args = parser.parse_args()
    project = args.project.resolve()
    jobs = [(cond, yaml, seed)
            for cond, yaml in MODELS.items()
            for seed in SEEDS]
    for path in [DATA, *MODELS.values()]:
        if not path.is_file():
            parser.error(f'Missing required file: {path}')
    existing = [project / f'{cond}_s{seed}' for cond, _, seed in jobs
                if (project / f'{cond}_s{seed}').exists()]
    if existing:
        parser.error(f'Output run already exists: {existing[0]}; choose a new --project')
    if args.check_only:
        print(f'Preflight OK: {len(jobs)} jobs; data={DATA}; output={project}')
        raise SystemExit(0)

    total = len(jobs)
    done  = 0
    t0    = time.time()

    for cond, yaml, seed in jobs:
        done += 1
        name = f'{cond}_s{seed}'
        print(f'\n{"="*60}')
        print(f'[{done}/{total}]  {name}  (epoch={EPOCHS})')
        print(f'{"="*60}')

        model = YOLO(str(yaml), task='detect')
        model.train(
            **TRAIN_KWARGS,
            project = str(project),
            name = name,
            seed = seed,
            deterministic = True,
        )

        elapsed = (time.time() - t0) / 3600
        remaining = elapsed / done * (total - done)
        print(f'  elapsed {elapsed:.1f}h  |  est. remaining {remaining:.1f}h')

    print(f'\nAll {total} runs done in {(time.time()-t0)/3600:.1f}h')
    print('Now run:  python eval_ablation_100ep.py')
