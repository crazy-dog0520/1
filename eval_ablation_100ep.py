"""
Evaluate all 12 runs from train_ablation_100ep.py (4 conditions × 3 seeds).
Prints:
  - per-run raw numbers
  - Table II: mean ± std for every condition (all four rows have std)
  - Joint-model mean and sample SD (seed 0 is used for size only)
  - Human mAP50-95 summary

Run:  python eval_ablation_100ep.py
"""
import warnings
warnings.filterwarnings('ignore')
import os, numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'mydata' / 'data.yaml'
OUT = ROOT / 'runs' / 'ablation_100ep'

# All four conditions have 3 seeds
CONDITIONS = ['Baseline', 'SAFusion_only', 'MSCAA_only', 'MSCAA_SAFusion']
SEEDS      = [0, 42, 123]
missing = [OUT / f'{cond}_s{seed}' / 'weights' / 'best.pt'
           for cond in CONDITIONS for seed in SEEDS
           if not (OUT / f'{cond}_s{seed}' / 'weights' / 'best.pt').is_file()]
if missing:
    raise FileNotFoundError(f'Incomplete 12-run evaluation; first missing checkpoint: {missing[0]}')

def eval_model(pt_path):
    model   = YOLO(pt_path)
    metrics = model.val(data=str(DATA), imgsz=640, batch=16,
                        workers=0, device='0', verbose=False,
                        project=str(ROOT / 'runs' / 'ablation_100ep_eval'), plots=False)
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
results = {}
for cond in CONDITIONS:
    results[cond] = []
    for seed in SEEDS:
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
print(f"{'Run':<26} {'mAP50':>7} {'mAP5095':>8} {'Fish AP':>7} {'Human AP':>14}")
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
           'SAFusion_only':  ('',  r'\checkmark'),
           'MSCAA_only':     (r'\checkmark', ''),
           'MSCAA_SAFusion': (r'\checkmark', r'\checkmark')}

def stats(cond):
    """Return {metric: (mean, std)} for a condition."""
    vals = results[cond]
    return {m: (np.mean([r[m] for r in vals]),
                np.std([r[m] for r in vals], ddof=1) if len(vals) > 1 else float('nan'))
            for m in METRICS}

print('='*75)
print('TABLE II  (mean ± std over 3 seeds, all rows)')
print('='*75)

table2_latex = []
for cond in CONDITIONS:
    if not results[cond]:
        continue
    s = stats(cond)
    ms, ss = LABELS[cond]
    print(f'{cond:<16} ' + '  '.join(
        f"{m}={s[m][0]:.3f}±{s[m][1]:.3f}" for m in METRICS))
    row = f'{ms} & {ss}' + ''.join(
        f' & ${s[m][0]:.3f}{{\\pm}}{s[m][1]:.3f}$' for m in METRICS) + r' \\'
    table2_latex.append((cond, row))

print('\nLaTeX rows for Table II:')
for cond, row in table2_latex:
    print(f'  % {cond}')
    print(f'  {row}')

# Human AP50-95 summary
print('\n' + '='*75)
print('HUMAN mAP50-95 SUMMARY')
print('='*75)
for cond in CONDITIONS:
    if not results[cond]:
        continue
    mu, sd = stats(cond)['Human_mAP5095']
    print(f'  {cond:<16} {mu:.4f} ± {sd:.4f}')
# Descriptive ranking across methods.
if all(results[c] for c in CONDITIONS):
    h = {c: stats(c)['Human_mAP5095'] for c in CONDITIONS}
    best_mean = max(h, key=lambda c: h[c][0])
    low_std   = min(h, key=lambda c: h[c][1])
    print(f'\n  highest mean Human mAP50-95 : {best_mean}')
    print(f'  lowest  std  Human mAP50-95 : {low_std}')
    print('  Descriptive rankings only; use the complete three-seed results.')

# ── Table I "Ours": mean±std and best ────────────────────────────────────────
ours = results.get('MSCAA_SAFusion', [])
base = results.get('Baseline', [])
if ours:
    s = stats('MSCAA_SAFusion')
    representative = ours[0]
    try:
        from thop import profile
        import torch
        model = YOLO(representative['pt'])
        dummy = torch.zeros(1, 3, 640, 640).to(
            next(model.model.parameters()).device)
        macs, params = profile(model.model, inputs=(dummy,), verbose=False)
        params_M, macs_G = params/1e6, macs/1e9
    except Exception:
        params_M, macs_G = float('nan'), float('nan')

    print('\n' + '='*75)
    print('TABLE I  "Ours" row')
    print('='*75)
    print(f'  mean±std : mAP50={s["mAP50"][0]:.3f}±{s["mAP50"][1]:.3f}  '
          f'Human mAP50-95={s["Human_mAP5095"][0]:.3f}±{s["Human_mAP5095"][1]:.3f}')
    print(f'  Params={params_M:.2f}M  MACs={macs_G:.1f}G')
    if base:
        bmu = stats('Baseline')['mAP50'][0]
        print(f'\n  mAP50 gain over baseline (mean): '
              f'{(s["mAP50"][0]-bmu)*100:+.1f} pp')
