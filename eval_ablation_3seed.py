"""
Evaluate all 12 runs from train_ablation_3seed.py.
Prints:
  - per-run raw numbers
  - Table II: mean ± std for each condition
  - Best MSCAA_SAFusion run → LaTeX row for Table I

Run:  python eval_ablation_3seed.py
"""
import warnings
warnings.filterwarnings('ignore')
import os, numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'mydata' / 'data.yaml'
OUT = ROOT / 'runs' / 'ablation_3seed'

CONDITIONS_SINGLE = ['Baseline', 'SAFusion_only']   # 1 seed only
CONDITIONS_MULTI  = ['MSCAA_only', 'MSCAA_SAFusion'] # 3 seeds
CONDITIONS = CONDITIONS_SINGLE + CONDITIONS_MULTI
SEEDS      = [0, 42, 123]
missing = [OUT / f'{cond}_s{seed}' / 'weights' / 'best.pt'
           for cond in CONDITIONS
           for seed in (SEEDS if cond in CONDITIONS_MULTI else [0])
           if not (OUT / f'{cond}_s{seed}' / 'weights' / 'best.pt').is_file()]
if missing:
    raise FileNotFoundError(f'Incomplete historical 8-run evaluation: {missing[0]}')

def eval_model(pt_path):
    model   = YOLO(pt_path)
    metrics = model.val(data=str(DATA), imgsz=640, batch=16,
                        workers=0, device='0', verbose=False,
                        project=str(ROOT / 'runs' / 'ablation_3seed_eval'), plots=False)
    names     = metrics.names
    ap_cls    = metrics.box.ap
    ap_index  = list(metrics.box.ap_class_index)

    fish_idx  = next((ap_index.index(k) for k,v in names.items()
                      if 'fish' in v.lower() and k in ap_index), None)
    human_idx = next((ap_index.index(k) for k,v in names.items()
                      if 'human' in v.lower() and k in ap_index), None)

    return {
        'mAP50':    metrics.box.map50,
        'mAP5095':  metrics.box.map,
        'Fish':     ap_cls[fish_idx]  if fish_idx  is not None else float('nan'),
        'Human_mAP5095': ap_cls[human_idx] if human_idx is not None else float('nan'),
    }

# ── collect results ──────────────────────────────────────────────────────────
results = {}   # {condition: [dict, dict, dict]}

for cond in CONDITIONS:
    results[cond] = []
    seed_list = SEEDS if cond in CONDITIONS_MULTI else [0]
    for seed in seed_list:
        pt = os.path.join(OUT, f'{cond}_s{seed}', 'weights', 'best.pt')
        if not os.path.exists(pt):
            print(f'  MISSING: {pt}')
            continue
        print(f'  eval  {cond} seed={seed} ...', end=' ', flush=True)
        r = eval_model(pt)
        results[cond].append({'seed': seed, 'pt': pt, **r})
        print(f"mAP50={r['mAP50']:.4f}")

# ── raw numbers ──────────────────────────────────────────────────────────────
print('\n' + '='*75)
print(f"{'Run':<26} {'mAP50':>7} {'mAP5095':>8} {'Fish':>7} {'Human mAP5095':>14}")
print('-'*75)
for cond in CONDITIONS:
    for r in results[cond]:
        name = f"{cond}_s{r['seed']}"
        print(f"{name:<26} {r['mAP50']:>7.4f} {r['mAP5095']:>8.4f} "
              f"{r['Fish']:>7.4f} {r['Human_mAP5095']:>14.4f}")
    print()

# ── Table II: mean ± std ─────────────────────────────────────────────────────
METRICS = ['mAP50', 'mAP5095', 'Fish', 'Human_mAP5095']
LABELS  = {'Baseline':       ('',  ''),
           'MSCAA_only':     (r'\checkmark', ''),
           'SAFusion_only':  ('',  r'\checkmark'),
           'MSCAA_SAFusion': (r'\checkmark', r'\checkmark')}

print('='*75)
print('TABLE II  (mean ± std over 3 seeds)')
print('='*75)
print(f"{'MSCAA':<12} {'SAF':<12} {'mAP50':>14} {'mAP50-95':>14} "
      f"{'Fish':>14} {'Human mAP5095':>14}")
print('-'*75)

table2_latex = []
for cond in CONDITIONS:
    vals = results[cond]
    if not vals:
        continue
    ms, ss = LABELS[cond]
    row_latex = f'{ms} & {ss}'
    for m in METRICS:
        arr = np.array([r[m] for r in vals])
        mu, sd = arr.mean(), arr.std(ddof=1) if len(arr) > 1 else float('nan')
        print(f"  {m}: {mu:.4f} ± {sd:.4f}", end='   ')
        row_latex += f' & ${mu:.3f}\\pm{sd:.3f}$'
    print()
    row_latex += r' \\'
    table2_latex.append((cond, row_latex))

# ── LaTeX Table II ───────────────────────────────────────────────────────────
print('\nLaTeX rows for Table II:')
for cond, row in table2_latex:
    print(f'  % {cond}')
    print(f'  {row}')

# ── best Ours run → Table I ──────────────────────────────────────────────────
ours = results.get('MSCAA_SAFusion', [])
if ours:
    best = max(ours, key=lambda r: r['mAP50'])
    try:
        from thop import profile
        import torch
        model = YOLO(best['pt'])
        dummy = torch.zeros(1, 3, 640, 640).to(
            next(model.model.parameters()).device)
        macs, params = profile(model.model, inputs=(dummy,), verbose=False)
        params_M = params / 1e6
        macs_G   = macs   / 1e9
    except Exception:
        params_M, macs_G = float('nan'), float('nan')

    print('\n' + '='*75)
    print(f'Best Ours run: MSCAA_SAFusion seed={best["seed"]}')
    print(f'  mAP50={best["mAP50"]:.4f}  mAP5095={best["mAP5095"]:.4f}  '
          f'Fish={best["Fish"]:.4f}  Human mAP50-95={best["Human_mAP5095"]:.4f}')
    print(f'  Params={params_M:.2f}M  MACs={macs_G:.1f}G')
    print('\nLaTeX row for Table I:')
    print(f'\\textbf{{Ours}} & \\textbf{{{best["mAP50"]:.3f}}} & '
          f'\\textbf{{{best["mAP5095"]:.3f}}} & '
          f'\\textbf{{{best["Fish"]:.3f}}} & '
          f'\\textbf{{{best["Human_mAP5095"]:.3f}}} & '
          f'{params_M:.2f}\\,M & {macs_G:.1f}\\,G \\\\')
