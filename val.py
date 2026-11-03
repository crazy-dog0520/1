import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'runs' / 'ablation_100ep' / 'Baseline_s0' / 'weights' / 'best.pt'

if __name__ == '__main__':
    model = YOLO(str(BASELINE))
    model.val(data=str(ROOT / 'mydata' / 'data.yaml'),
              split='val',
              imgsz=640,
              batch=16,
              # iou=0.7,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project=str(ROOT / 'runs' / 'val'),
              name='baseline_s0',
              )
