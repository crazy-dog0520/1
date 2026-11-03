import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / 'runs' / 'ablation_100ep' / 'Baseline_s0' / 'weights' / 'best.pt'

# 推理参数官方详解链接：https://docs.ultralytics.com/modes/predict/#inference-sources:~:text=of%20Results%20objects-,Inference%20Arguments,-model.predict()

if __name__ == '__main__':
    model = YOLO(str(BASELINE))
    model.predict(source=str(ROOT / 'mydata' / 'test' / 'images'),
                  imgsz=640,
                  project=str(ROOT / 'runs' / 'detect'),
                  name='baseline_s0',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  line_width=4, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )
